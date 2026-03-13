#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "Operators.hpp"
#include "GpuColumnStore.hpp"
#include "EngineError.hpp"
#include "EngineConfig.hpp"

#include <iostream>
#include <vector>
#include <set>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include "Logger.hpp"

namespace engine {

// Forward declarations (implemented in ProjectHelpers.cpp)
bool projectSubstring(const FunctionCall& func,
    EvalContext& ctx, TableResult& out, const std::string& outName,
    const std::string& posName, bool debug);
bool projectExtractYear(const FunctionCall& func,
    const std::string& funcLower,
    EvalContext& ctx, TableResult& out, const std::string& outName,
    const std::string& posName, bool debug);
bool projectCrossTableLookup(const std::string& lookupCol,
    EvalContext& ctx, TableResult& out, const std::string& outName,
    const std::string& posName, size_t& projectedRowCount, bool& rowCountInitialized,
    std::unordered_map<std::string, EvalContext>* tableContexts, bool debug);
bool projectComputedExpression(
    const TypedExprPtr& expr,
    size_t exprIndex,
    const std::string& outName,
    const std::string& posName,
    EvalContext& ctx,
    TableResult& out,
    const std::function<void(size_t)>& updateRowCount,
    bool debug);

// -- Extracted: fuzzyFindColumn --
// Fuzzy column resolution for join aliases, aggregate expressions, and TPC-H key suffixes.
static std::string fuzzyFindColumn(const std::string& name, const EvalContext& ctx, bool debug) {
    // 1. Try Suffix match for TPC-H keys (e.g. l_suppkey for ps_suppkey) - WITH SIZE CHECK
    if (name.find("_suppkey") != std::string::npos) {
        for (const auto& [n, vec] : ctx.u32Cols) {
            if (n.find("_suppkey") != std::string::npos && vec.size() == ctx.rowCount) return n;
        }
    }
    if (name.find("_partkey") != std::string::npos) {
        for (const auto& [n, vec] : ctx.u32Cols) {
            if (n.find("_partkey") != std::string::npos && vec.size() == ctx.rowCount) return n;
        }
    }

    for (const auto& [n, vec] : ctx.f32Cols) {
        if (n.size() > name.size() && n.rfind(name, 0) == 0 && n.find("_rhs_") != std::string::npos && vec.size() == ctx.rowCount) return n;
    }
    for (const auto& [n, vec] : ctx.u32Cols) {
        if (n.size() > name.size() && n.rfind(name, 0) == 0 && n.find("_rhs_") != std::string::npos && vec.size() == ctx.rowCount) return n;
    }

    // 2. Fuzzy match for aggregate columns (e.g. "sum(x * y)" vs "sum((x*cast(yasdecimal...)))")
    auto extractAggPrefix = [](const std::string& s) -> std::pair<std::string, std::string> {
        static const std::vector<std::string> aggFuncs = {"sum(", "avg(", "min(", "max(", "count("};
        std::string lower = s;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        for (const auto& func : aggFuncs) {
            if (lower.rfind(func, 0) == 0) {
                std::string rest = s.substr(func.size());
                std::string firstCol;
                bool inCol = false;
                for (char c : rest) {
                    if (std::isalpha(c) || c == '_') {
                        inCol = true;
                        firstCol += c;
                    } else if (inCol && std::isdigit(c)) {
                        firstCol += c;
                    } else if (inCol) {
                        break;
                    }
                }
                return {func, firstCol};
            }
        }
        return {"", ""};
    };

    auto [aggPrefix, firstCol] = extractAggPrefix(name);
    if (!aggPrefix.empty() && !firstCol.empty()) {
        std::string firstMatch;
        std::string varyingMatch;

        for (const auto& [n, vec] : ctx.f32Cols) {
            std::string lowerN = n;
            std::transform(lowerN.begin(), lowerN.end(), lowerN.begin(), ::tolower);
            if (lowerN.rfind(aggPrefix, 0) == 0 && n.find(firstCol) != std::string::npos) {
                if (firstMatch.empty()) firstMatch = n;
                if (vec.size() > 1) {
                    float first = vec[0];
                    bool varying = false;
                    for (size_t i = 1; i < std::min(vec.size(), engine::config::kColumnSampleSize); ++i) {
                        if (vec[i] != first) { varying = true; break; }
                    }
                    if (varying && varyingMatch.empty()) {
                        varyingMatch = n;
                        LOG_DEBUG("Exec", "Project: aggregate fuzzy match (varying) '" << name << "' -> '" << n << "'\n");
                    }
                }
            }
        }

        if (varyingMatch.empty()) {
            for (const auto& [n, buf] : ctx.f32ColsGPU) {
                if (!buf) continue;
                std::string lowerN = n;
                std::transform(lowerN.begin(), lowerN.end(), lowerN.begin(), ::tolower);
                if (lowerN.rfind(aggPrefix, 0) == 0 && n.find(firstCol) != std::string::npos) {
                    if (firstMatch.empty()) firstMatch = n;
                    size_t cnt = buf->length() / sizeof(float);
                    if (cnt > 1) {
                        float* ptr = static_cast<float*>(buf->contents());
                        bool varying = false;
                        for (size_t i = 1; i < std::min(cnt, engine::config::kColumnSampleSize); ++i) {
                            if (ptr[i] != ptr[0]) { varying = true; break; }
                        }
                        if (varying) {
                            varyingMatch = n;
                            LOG_DEBUG("Exec", "Project: aggregate fuzzy match (varying GPU) '" << name << "' -> '" << n << "'\n");
                            break;
                        }
                    }
                }
            }
        }

        if (!varyingMatch.empty()) {
            if (debug && varyingMatch != firstMatch) 
                LOG_INFO("Exec", "Project: preferring varying column '" << varyingMatch << "' over '" << firstMatch << "'\n");
            return varyingMatch;
        }
        if (!firstMatch.empty()) {
            LOG_DEBUG("Exec", "Project: aggregate fuzzy match '" << name << "' -> '" << firstMatch << "'\n");
            return firstMatch;
        }
    }

    return "";
}

// -- Extracted: projectSubstring --

// -- Extracted: projectStringColumn --
// Resolves a string column by name (with alias/prefix/dict fallback), applies
// activeRows compaction (GPU flat-string gather or CPU fallback), and pushes to output.
static bool projectStringColumn(
    const std::string& col, const std::string& outName, const std::string& posName,
    EvalContext& ctx, TableResult& out,
    size_t& projectedRowCount, bool& rowCountInitialized, bool /*debug*/)
{
    std::string strLookupCol = col;

    // Resolve alias for string lookup
    if (ctx.stringCols.find(strLookupCol) == ctx.stringCols.end()) {
         if (ctx.columnAliases.count(strLookupCol)) strLookupCol = ctx.columnAliases[strLookupCol];

         if (ctx.stringCols.find(strLookupCol) == ctx.stringCols.end()) {
             size_t dot = strLookupCol.find('.');
             if (dot != std::string::npos) {
                 std::string suffix = strLookupCol.substr(dot + 1);
                 if (ctx.stringCols.count(suffix)) strLookupCol = suffix;
             }
         }

         if (ctx.stringCols.find(strLookupCol) == ctx.stringCols.end()) {
              for (const auto& [n, _] : ctx.stringCols) {
                  if (n.size() > strLookupCol.size() && n.rfind(strLookupCol, 0) == 0) {
                      char sep = n[strLookupCol.size()];
                      if (sep == '_' || sep == '.') { strLookupCol = n; break; }
                  }
              }
         }
         // Also try dictCols (stringCols may have been invalidated by dict migration)
         if (ctx.stringCols.find(strLookupCol) == ctx.stringCols.end()) {
             if (ctx.hasDictCol(strLookupCol)) {
                 ctx.ensureStringCol(strLookupCol);
             } else {
                 for (const auto& [n, _] : ctx.dictCols) {
                     if (n.size() > strLookupCol.size() && n.rfind(strLookupCol, 0) == 0) {
                         char sep = n[strLookupCol.size()];
                         if (sep == '_' || sep == '.') {
                             ctx.ensureStringCol(n);
                             strLookupCol = n;
                             break;
                         }
                     }
                 }
             }
         }
    }

    // Materialize from dict if needed
    ctx.ensureStringCol(strLookupCol);
    if (!ctx.stringCols.count(strLookupCol)) return false;

    LOG_DEBUG("Exec", "Project: pass-through string col " << col << " (as " << strLookupCol << ") -> " << (outName.empty() ? col : outName));
    std::vector<std::string> sub;
    if (ctx.rowCount == 0) {
         sub = {};
    } else if (ctx.activeRows.empty() || ctx.stringCols[strLookupCol].size() == ctx.activeRows.size()) {
         sub = ctx.stringCols[strLookupCol];
    } else {
         bool gpuDone = false;
         if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0) {
             ctx.ensureFlatStringCol(strLookupCol);
             auto fit = ctx.flatStringCols.find(strLookupCol);
             if (fit != ctx.flatStringCols.end() && fit->second.chars) {
                 auto r = GpuOps::gatherFlatString(
                     fit->second.chars, fit->second.offsets, fit->second.lengths,
                     ctx.activeRowsGPU, ctx.activeRowsCountGPU, true);
                 if (r.chars) {
                     const uint32_t* offs = static_cast<const uint32_t*>(r.offsets->contents());
                     const uint32_t* lens = static_cast<const uint32_t*>(r.lengths->contents());
                     const char* ch = static_cast<const char*>(r.chars->contents());
                     sub.resize(r.rowCount);
                     for (uint32_t i = 0; i < r.rowCount; ++i) sub[i].assign(ch + offs[i], lens[i]);
                     gpuDone = true;
                 }
             }
         }
         if (!gpuDone) {
             ctx.ensureActiveRowsCPU();
             sub.reserve(ctx.activeRows.size());
             for(auto idx : ctx.activeRows) {
                 if (idx < ctx.stringCols[strLookupCol].size()) sub.push_back(ctx.stringCols[strLookupCol][idx]);
                 else sub.push_back("");
             }
         }
    }

    LOG_DEBUG("Exec", "Project: string col size " << sub.size());

    if (!rowCountInitialized) { projectedRowCount = sub.size(); rowCountInitialized = true; }
    else if (sub.size() > 0 && projectedRowCount != sub.size()) {
        if (sub.size() > projectedRowCount) projectedRowCount = sub.size();
    }

    if (!outName.empty()) {
        ctx.stringCols[outName] = sub;
        if (outName != col) ctx.columnAliases[col] = outName;
        if (ctx.hasDictCol(strLookupCol) && !outName.empty() && outName != strLookupCol) {
            ctx.dictCols[outName] = ctx.dictCols[strLookupCol];
        }
    }
    ctx.stringCols[posName] = sub;
    out.stringCols.push_back(std::move(sub));
    out.stringNames.push_back(outName.empty() ? col : outName);
    return true;
}

// -- Extracted: resolveProjectColumn --
// Resolves a column name through multi-instance suffixed variants (col_1..col_9),
// column aliases, CTE alias fallback, and fuzzy matching.
static std::string resolveProjectColumn(
    const std::string& col, const std::string& outName,
    EvalContext& ctx, const std::set<std::string>& usedColumns, bool debug)
{
    std::string lookupCol = col;
    bool baseMissing = (ctx.u32Cols.find(col) == ctx.u32Cols.end() &&
                        ctx.f32Cols.find(col) == ctx.f32Cols.end() &&
                        ctx.stringCols.find(col) == ctx.stringCols.end() &&
                        ctx.dictCols.find(col) == ctx.dictCols.end());

    if (baseMissing || usedColumns.count(col) > 0) {
        for (int suffix = 1; suffix <= 9; ++suffix) {
            std::string suffixedCol = col + "_" + std::to_string(suffix);
            if (ctx.u32Cols.count(suffixedCol) > 0 || ctx.f32Cols.count(suffixedCol) > 0 ||
                ctx.stringCols.count(suffixedCol) > 0 || ctx.dictCols.count(suffixedCol) > 0) {
                if (usedColumns.count(suffixedCol) == 0) {
                    lookupCol = suffixedCol;
                    LOG_DEBUG("Exec", "Project: multi-instance column " << col << " -> " << lookupCol);
                    break;
                }
            }
        }
    }

    // Check for column alias (e.g., supplier_no -> l_suppkey from CTE)
    if (ctx.u32Cols.find(lookupCol) == ctx.u32Cols.end() &&
        ctx.f32Cols.find(lookupCol) == ctx.f32Cols.end()) {
        auto aliasIt = ctx.columnAliases.find(lookupCol);
        if (aliasIt != ctx.columnAliases.end()) {
            LOG_DEBUG("Exec", "Project: alias resolution " << lookupCol << " -> " << aliasIt->second);
            lookupCol = aliasIt->second;
        }
        else if (!outName.empty() && outName != col &&
                 (ctx.u32Cols.find(outName) != ctx.u32Cols.end() ||
                  ctx.f32Cols.find(outName) != ctx.f32Cols.end())) {
            LOG_DEBUG("Exec", "Project: CTE alias fallback " << lookupCol << " -> " << outName);
            lookupCol = outName;
            ctx.columnAliases[col] = outName;
        }
    }

    // Fuzzy match for join aliases (e.g., min(x) vs min(x)_rhs_29)
    if (ctx.u32Cols.find(lookupCol) == ctx.u32Cols.end() &&
        ctx.f32Cols.find(lookupCol) == ctx.f32Cols.end()) {
         std::string found = fuzzyFindColumn(lookupCol, ctx, debug);
         if (!found.empty()) {
              LOG_DEBUG("Exec", "Project: fuzzy match " << lookupCol << " -> " << found);
              lookupCol = found;
         }
    }

    LOG_DEBUG("Exec", "Project: lookup " << col << " as " << lookupCol);
    return lookupCol;
}

// -- Extracted: projectU32Column --
// Downloads a u32 column from GPU if missing on CPU, applies activeRows compaction
// (GPU gather or CPU fallback), handles alias tracking, and pushes to output.
static bool projectU32Column(
    const std::string& col, const std::string& lookupCol,
    const std::string& outName, const std::string& posName,
    EvalContext& ctx, TableResult& out, std::set<std::string>& usedColumns,
    size_t& projectedRowCount, bool& rowCountInitialized, bool debug)
{
    // Try to download from GPU if missing on CPU OR if CPU column has wrong size
    auto itDirect = ctx.u32Cols.find(lookupCol);
    bool missingCPU = (itDirect == ctx.u32Cols.end());
    if (!missingCPU && ctx.rowCount > 0 && itDirect->second.empty()) missingCPU = true;
    if (!missingCPU && ctx.rowCount > 0 && itDirect->second.size() != ctx.rowCount) {
        if (ctx.u32ColsGPU.count(lookupCol)) {
            LOG_DEBUG("Exec", "Project: CPU col " << lookupCol << " size=" << itDirect->second.size() << " != ctx.rowCount=" << ctx.rowCount << ", re-downloading from GPU\n");
            missingCPU = true;
        }
    }

    if (missingCPU) {
        MTL::Buffer* buf = nullptr;
        if (ctx.u32ColsGPU.count(lookupCol)) buf = ctx.u32ColsGPU[lookupCol];
        if (buf) {
             LOG_DEBUG("Exec", "Project: downloading GPU column " << lookupCol);
             uint32_t cnt = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
             if (cnt == 0 && ctx.rowCount > 0 && !ctx.activeRowsGPU) cnt = ctx.rowCount;
             std::vector<uint32_t> down;
             if (cnt > 0) {
                 if (ctx.activeRowsGPU) {
                     down = gatherToVector<uint32_t>(buf, ctx.activeRowsGPU, cnt);
                 } else {
                     down.resize(cnt);
                     std::memcpy(down.data(), buf->contents(), cnt * sizeof(uint32_t));
                 }
             }
             ctx.u32Cols[lookupCol] = std::move(down);
        }
    }

    auto itU = ctx.u32Cols.find(lookupCol);
    if (itU == ctx.u32Cols.end()) return false;

    usedColumns.insert(lookupCol);
    std::vector<uint32_t> colData;
    if (ctx.rowCount == 0) {
        colData = {};
    } else if (ctx.activeRows.empty() || itU->second.size() == ctx.activeRows.size()) {
        colData = itU->second;
    } else if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0) {
        auto itGpu = ctx.u32ColsGPU.find(lookupCol);
        if (itGpu != ctx.u32ColsGPU.end() && itGpu->second) {
            colData = gatherToVector<uint32_t>(itGpu->second, ctx.activeRowsGPU, ctx.activeRowsCountGPU);
        } else {
            auto& s = GpuColumnStore::instance();
            MTL::Buffer* src = s.device()->newBuffer(itU->second.data(), itU->second.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            colData = gatherToVector<uint32_t>(src, ctx.activeRowsGPU, ctx.activeRowsCountGPU);
            src->release();
        }
    } else {
        ctx.ensureActiveRowsCPU();
        colData.reserve(ctx.activeRows.size());
        for (uint32_t idx : ctx.activeRows) {
            colData.push_back(idx < itU->second.size() ? itU->second[idx] : 0);
        }
    }

    if (debug) {
        LOG_INFO("Exec", "Project: column " << lookupCol << " size=" << colData.size());
        if (!colData.empty()) std::cerr << " first=" << colData[0];
        if (colData.size() > 1) std::cerr << " second=" << colData[1];
        std::set<uint32_t> uniq(colData.begin(), colData.end());
        LOG_INFO("PROJECT", " distinct=" << uniq.size());
    }

    if (!rowCountInitialized) { projectedRowCount = colData.size(); rowCountInitialized = true; }
    else if (colData.size() > 0 && projectedRowCount != colData.size()) {
        if (colData.size() > projectedRowCount) projectedRowCount = colData.size();
    }

    if (!outName.empty() && outName != lookupCol && outName != col) {
        ctx.u32Cols[outName] = colData;
        if (col != outName && col != lookupCol) {
            ctx.columnAliases[col] = outName;
            LOG_DEBUG("Exec", "Project: tracking alias " << col << " -> " << outName);
        }
    }
    if (col != lookupCol && col != outName) {
        ctx.u32Cols[col] = colData;
        LOG_DEBUG("Exec", "Project: also storing as CTE alias " << col);
    }
    ctx.u32Cols[posName] = colData;
    out.u32Cols.push_back(std::move(colData));
    out.u32Names.push_back(outName.empty() ? lookupCol : outName);
    LOG_DEBUG("Exec", "Project: Pushing U32 col " << (outName.empty()?lookupCol:outName));
    return true;
}

// -- Extracted: projectF32Column --
// Downloads a f32 column from GPU if missing on CPU, applies activeRows compaction,
// handles CTE aliasing, and pushes to output.
static bool projectF32Column(
    const std::string& col, const std::string& lookupCol,
    const std::string& outName, const std::string& posName,
    EvalContext& ctx, TableResult& out, std::set<std::string>& usedColumns,
    size_t& projectedRowCount, bool& rowCountInitialized, bool debug)
{
    // Check if F32 data is on GPU and needs downloading
    bool missingCPU_F32 = (ctx.f32Cols.find(lookupCol) == ctx.f32Cols.end());
    if (!missingCPU_F32 && ctx.rowCount > 0 && ctx.f32Cols[lookupCol].empty()) missingCPU_F32 = true;
    if (!missingCPU_F32 && ctx.rowCount > 0 && ctx.f32Cols[lookupCol].size() != ctx.rowCount) {
        if (ctx.f32ColsGPU.count(lookupCol)) {
            LOG_DEBUG("Exec", "Project: CPU f32 col " << lookupCol << " size=" << ctx.f32Cols[lookupCol].size() << " != ctx.rowCount=" << ctx.rowCount << ", re-downloading from GPU\n");
            missingCPU_F32 = true;
        }
    }

    if (missingCPU_F32) {
        MTL::Buffer* buf = nullptr;
        if (ctx.f32ColsGPU.count(lookupCol)) buf = ctx.f32ColsGPU[lookupCol];
        if (buf) {
             LOG_DEBUG("Exec", "Project: downloading GPU f32 column " << lookupCol);
             uint32_t cnt = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
             if (cnt == 0 && ctx.rowCount > 0 && !ctx.activeRowsGPU) cnt = ctx.rowCount;
             if (cnt > 0) {
                 std::vector<float> down(cnt);
                 if (ctx.activeRowsGPU) {
                     down = gatherToVector<float>(buf, ctx.activeRowsGPU, cnt);
                 } else {
                     std::memcpy(down.data(), buf->contents(), cnt * sizeof(float));
                 }
                 ctx.f32Cols[lookupCol] = std::move(down);
             }
        }
    }

    auto itF = ctx.f32Cols.find(lookupCol);
    if (itF == ctx.f32Cols.end()) return false;

    usedColumns.insert(lookupCol);
    std::vector<float> colData;
    if (ctx.rowCount == 0) {
        colData = {};
    } else if (!ctx.activeRows.empty()) {
        if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0 && itF->second.size() > ctx.activeRows.size()) {
            auto itGpu = ctx.f32ColsGPU.find(lookupCol);
            if (itGpu != ctx.f32ColsGPU.end() && itGpu->second) {
                colData = gatherToVector<float>(itGpu->second, ctx.activeRowsGPU, ctx.activeRowsCountGPU);
            } else {
                auto& s = GpuColumnStore::instance();
                MTL::Buffer* src = s.device()->newBuffer(itF->second.data(), itF->second.size() * sizeof(float), MTL::ResourceStorageModeShared);
                colData = gatherToVector<float>(src, ctx.activeRowsGPU, ctx.activeRowsCountGPU);
                src->release();
            }
        } else {
            ctx.ensureActiveRowsCPU();
            colData.reserve(ctx.activeRows.size());
            for (uint32_t idx : ctx.activeRows) {
                colData.push_back(idx < itF->second.size() ? itF->second[idx] : 0.0f);
            }
        }
    } else {
        colData = itF->second;
    }

    if (debug) {
        LOG_INFO("Exec", "Project: f32 column " << lookupCol << " size=" << colData.size());
        if (!colData.empty()) std::cerr << " first=" << colData[0];
        if (colData.size() > 1) std::cerr << " second=" << colData[1];
        float minV = colData.empty() ? 0 : colData[0], maxV = minV;
        for (float v : colData) { minV = std::min(minV, v); maxV = std::max(maxV, v); }
        LOG_INFO("PROJECT", " min=" << minV << " max=" << maxV);
    }

    if (!rowCountInitialized) { projectedRowCount = colData.size(); rowCountInitialized = true; }
    else if (colData.size() > 0 && projectedRowCount != colData.size()) {
        if (colData.size() > projectedRowCount) projectedRowCount = colData.size();
    }

    if (!outName.empty() && outName != col) {
        ctx.f32Cols[outName] = colData;
    }
    if (col != lookupCol && col != outName) {
        ctx.f32Cols[col] = colData;
        LOG_DEBUG("Exec", "Project: f32 also storing as CTE alias " << col);
    }
    ctx.f32Cols[posName] = colData;
    out.f32Cols.push_back(std::move(colData));
    out.f32Names.push_back(outName.empty() ? col : outName);
    return true;
}

// ── Copy prior output columns into context for chained projections ──
// When a Project follows another Project (no table context), columns from
// the prior output need to be available in the EvalContext.
static void copyPriorOutputToCtx(TableResult& out, EvalContext& ctx,
                                  bool hasExistingOutput, bool /*debug*/) {
    if (!hasExistingOutput) return;

    bool shouldCopy = ctx.currentTable.empty();
    if (shouldCopy) {
        for (size_t i = 0; i < out.u32Names.size() && i < out.u32Cols.size(); ++i) {
            if (ctx.u32Cols.find(out.u32Names[i]) == ctx.u32Cols.end())
                ctx.u32Cols[out.u32Names[i]] = out.u32Cols[i];
        }
        for (size_t i = 0; i < out.f32Names.size() && i < out.f32Cols.size(); ++i) {
            if (ctx.f32Cols.find(out.f32Names[i]) == ctx.f32Cols.end())
                ctx.f32Cols[out.f32Names[i]] = out.f32Cols[i];
        }
        for (size_t i = 0; i < out.stringNames.size() && i < out.stringCols.size(); ++i) {
            if (ctx.stringCols.find(out.stringNames[i]) == ctx.stringCols.end())
                ctx.stringCols[out.stringNames[i]] = out.stringCols[i];
        }
    }
    out.u32Cols.clear();  out.u32Names.clear();
    out.f32Cols.clear();  out.f32Names.clear();
    out.stringCols.clear(); out.stringNames.clear();
    out.order.clear();
}

bool GpuExecutor::executeProject(const IRProject& project, EvalContext& ctx, TableResult& out, std::unordered_map<std::string, EvalContext>* tableContexts) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    if (debug) {
        LOG_INFO("Exec", "Project START: currentTable=" << ctx.currentTable << " ctx.u32Cols=");
        for (const auto& [k, v] : ctx.u32Cols) std::cerr << k << " ";
        LOG_INFO("PROJECT", "\n");
    }

    const size_t originalRowCount = ctx.rowCount;
    size_t projectedRowCount = ctx.rowCount;
    bool rowCountInitialized = false;
    auto updateRowCount = [&](size_t size) {
        if (!rowCountInitialized) {
            projectedRowCount = size;
            rowCountInitialized = true;
        } else if (size > 0 && projectedRowCount != size) {
            // Prefer the new size when encountering differing column lengths (e.g., scalar aggregates)
            if (size > projectedRowCount) projectedRowCount = size;
        }
    };
    
    bool hasExistingOutput = !out.u32Cols.empty() || !out.f32Cols.empty();
    
    if (debug && hasExistingOutput) {
        LOG_INFO("Exec", "Project: hasExistingOutput=true, out.u32Names=");
        for (const auto& n : out.u32Names) std::cerr << n << " ";
        LOG_INFO("PROJECT", "\n");
    }
    
    // Copy prior output columns into context (for chained projections)
    copyPriorOutputToCtx(out, ctx, hasExistingOutput, debug);
    
    std::set<std::string> usedColumns;
    
    for (size_t i = 0; i < project.exprs.size(); ++i) {
        const auto& expr = project.exprs[i];
        std::string outName = i < project.outputNames.size() ? project.outputNames[i] : "";
        
        // Generate a name for positional reference (#N)
        std::string posName = "#" + std::to_string(i);
        
        if (debug) {
            LOG_INFO("Exec", "Project: expr[" << i << "] outName=" << outName);
            if (expr) {
                LOG_INFO("PROJECT", " kind=" << static_cast<int>(expr->kind));
                if (expr->kind == TypedExpr::Kind::Column) {
                    LOG_DEBUG("PROJECT", " col=" << expr->asColumn().column);
                }
            }
            LOG_DEBUG("PROJECT", "\n");
        }
        
        if (!expr) continue;
        
        // Handle DuckDB internal functions like __internal_decompress_string(#0)
        // These are essentially passthrough - extract the inner column
        // BUT NOT for computation functions like EXTRACT or SUBSTRING
        if (expr->kind == TypedExpr::Kind::Function) {
            const auto& func = expr->asFunction();
            std::string funcLower = func.name;
            std::transform(funcLower.begin(), funcLower.end(), funcLower.begin(), ::tolower);
            
            if (debug) {
                LOG_INFO("Exec", "Project: function '" << func.name << "' (lower: '" << funcLower  << "') args=" << func.args.size());
            }
            
            // Handle SUBSTRING/SUBSTR as a string computation function
            if ((funcLower == "substring" || funcLower == "substr") && func.args.size() >= 1) {
                if (projectSubstring(func, ctx, out, outName, posName, debug)) continue;
            }
            
            if (projectExtractYear(func, funcLower, ctx, out, outName, posName, debug)) continue;

            // Skip passthrough for computation functions that need actual evaluation
            bool isComputation = (funcLower == "extract" || funcLower == "year" ||
                                  funcLower == "month" || funcLower == "day" ||
                                  funcLower == "substring" || funcLower == "substr");
            
            // One-arg column function -> treat as column reference (unless computation).
            if (!isComputation && func.args.size() == 1 && func.args[0] && 
                func.args[0]->kind == TypedExpr::Kind::Column) {
                std::string col = func.args[0]->asColumn().column;
                // Try to find the column or its equivalent (e.g., #0 -> first group key)
                auto itU = ctx.u32Cols.find(col);
                if (itU != ctx.u32Cols.end()) {
                    // Lazy-fetch from GPU if CPU vector is empty sentinel
                    if (itU->second.empty() && ctx.u32ColsGPU.count(col) && ctx.u32ColsGPU[col]) {
                        uint32_t rc = ctx.rowCount;
                        itU->second.resize(rc);
                        std::memcpy(itU->second.data(), ctx.u32ColsGPU[col]->contents(), rc * sizeof(uint32_t));
                    }
                    LOG_DEBUG("Exec", "Project: function passthrough " << col);
                    ctx.u32Cols[posName] = itU->second;
                    out.u32Cols.push_back(itU->second);
                    out.u32Names.push_back(outName.empty() ? col : outName);
                    continue;
                }
                // For #N positional references, look up directly (they should exist in context)
                if (col.size() >= 2 && col[0] == '#' && std::isdigit(static_cast<unsigned char>(col[1]))) {
                    auto itU2 = ctx.u32Cols.find(col);
                    if (itU2 != ctx.u32Cols.end()) {
                        LOG_DEBUG("Exec", "Project: function passthrough positional " << col);
                        ctx.u32Cols[posName] = itU2->second;
                        out.u32Cols.push_back(itU2->second);
                        out.u32Names.push_back(outName.empty() ? col : outName);
                        continue;
                    }
                    // Also try f32
                    auto itF = ctx.f32Cols.find(col);
                    if (itF != ctx.f32Cols.end()) {
                        LOG_DEBUG("Exec", "Project: function passthrough positional " << col << " (f32)\n");
                        ctx.f32Cols[posName] = itF->second;
                        out.f32Cols.push_back(itF->second);
                        out.f32Names.push_back(outName.empty() ? col : outName);
                        continue;
                    }
                }
            }
        }
        
        if (expr->kind == TypedExpr::Kind::Column) {
            // Simple column reference - copy to context with new name if needed
            std::string col = expr->asColumn().column;
            
            if (debug) {
                 LOG_INFO("Exec", "Project: Looking for col '" << col << "'\n");
            }
            
            // String column pass-through (alias/prefix/dict resolution + GPU/CPU compaction)
            if (projectStringColumn(col, outName, posName, ctx, out, projectedRowCount, rowCountInitialized, debug)) continue;

            
            // Handle post-GroupBy positional references: #N might be SUM_#N or COUNT_#N
            if (col.size() >= 2 && col[0] == '#') {
                // Try SUM_#N first for aggregate outputs
                std::string sumName = "SUM_" + col;
                auto itSum = ctx.f32Cols.find(sumName);
                if (itSum != ctx.f32Cols.end()) {
                    LOG_DEBUG("Exec", "Project: mapping " << col << " -> " << sumName);
                    ctx.f32Cols[posName] = itSum->second;
                    if (!outName.empty()) ctx.f32Cols[outName] = itSum->second;
                    out.f32Cols.push_back(itSum->second);
                    out.f32Names.push_back(outName.empty() ? col : outName);
                    continue;
                }
                // Try COUNT_#N
                std::string countName = "COUNT_" + col;
                auto itCount = ctx.f32Cols.find(countName);
                if (itCount != ctx.f32Cols.end()) {
                    LOG_DEBUG("Exec", "Project: mapping " << col << " -> " << countName);
                    ctx.f32Cols[posName] = itCount->second;
                    if (!outName.empty()) ctx.f32Cols[outName] = itCount->second;
                    out.f32Cols.push_back(itCount->second);
                    out.f32Names.push_back(outName.empty() ? col : outName);
                    continue;
                }
            }
            
            // Resolve column name through multi-instance, alias, CTE fallback, fuzzy matching
            std::string lookupCol = resolveProjectColumn(col, outName, ctx, usedColumns, debug);

            // U32 column download/compact/output
            if (projectU32Column(col, lookupCol, outName, posName, ctx, out, usedColumns, projectedRowCount, rowCountInitialized, debug)) continue;

            // F32 column download/compact/output
            if (projectF32Column(col, lookupCol, outName, posName, ctx, out, usedColumns, projectedRowCount, rowCountInitialized, debug)) continue;

            if (projectCrossTableLookup(lookupCol, ctx, out, outName, posName, projectedRowCount, rowCountInitialized, tableContexts, debug)) continue;
            
            // Column not found in context - might be an alias for aggregate output
            // Try to find aggregate column by positional key (#N) first, then by name pattern
            bool foundAggregate = false;
            
            // First: try positional match using the projection index (#0, #1, etc.)
            if (ctx.f32Cols.count(posName)) {
                LOG_DEBUG("Exec", "Project: mapping unknown alias '" << col << "' to positional aggregate " << posName);
                auto& aggData = ctx.f32Cols[posName];
                ctx.f32Cols[col] = aggData;
                if (!outName.empty()) ctx.f32Cols[outName] = aggData;
                out.f32Cols.push_back(aggData);
                out.f32Names.push_back(outName.empty() ? col : outName);
                foundAggregate = true;
            }
            
            // Fallback: try aggregate-prefixed keys matching the projection index
            if (!foundAggregate) {
                std::string idxSuffix = posName; // "#N"
                for (const auto& [aggName, aggData] : ctx.f32Cols) {
                    if ((aggName.find("COUNT_") == 0 || aggName.find("SUM_") == 0 || 
                         aggName.find("AVG_") == 0 || aggName.find("MIN_") == 0 || aggName.find("MAX_") == 0) &&
                        aggName.find(idxSuffix) != std::string::npos) {
                        LOG_DEBUG("Exec", "Project: mapping unknown alias '" << col << "' to aggregate " << aggName);
                        ctx.f32Cols[col] = aggData;
                        ctx.f32Cols[posName] = aggData;
                        if (!outName.empty()) ctx.f32Cols[outName] = aggData;
                        out.f32Cols.push_back(aggData);
                        out.f32Names.push_back(outName.empty() ? col : outName);
                        foundAggregate = true;
                        break;
                    }
                }
            }
            if (foundAggregate) continue;
        } else {
            if (projectComputedExpression(expr, i, outName, posName, ctx, out, updateRowCount, debug)) continue;
        }
    }
    
    if (rowCountInitialized) {
        out.rowCount = projectedRowCount;
        ctx.rowCount = projectedRowCount;
    } else {
        // Prefer GPU activeRows count if available
        size_t fallbackCount = originalRowCount;
        if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0) {
            fallbackCount = ctx.activeRowsCountGPU;
        } else if (!ctx.activeRows.empty()) {
            fallbackCount = ctx.activeRows.size();
        }
        out.rowCount = fallbackCount;
        ctx.rowCount = fallbackCount;
    }

    // Populate GPU buffer mirrors in TableResult for downstream operators
    out.u32ColsGPU.resize(out.u32Names.size());
    for (size_t i = 0; i < out.u32Names.size(); ++i) {
        if (ctx.u32ColsGPU.count(out.u32Names[i])) {
            out.u32ColsGPU[i] = ctx.u32ColsGPU[out.u32Names[i]];
        }
    }
    out.f32ColsGPU.resize(out.f32Names.size());
    for (size_t i = 0; i < out.f32Names.size(); ++i) {
        if (ctx.f32ColsGPU.count(out.f32Names[i])) {
            out.f32ColsGPU[i] = ctx.f32ColsGPU[out.f32Names[i]];
        }
    }

    return true;
}

} // namespace engine
