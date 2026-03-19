#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "Operators.hpp"
#include "GpuColumnStore.hpp"
#include "EngineConfig.hpp"

#include <iostream>
#include <vector>
#include <set>
#include <cstring>
#include <algorithm>
#include <cctype>
#include "Logger.hpp"

namespace engine {

// ============================================================================
// Column helpers extracted from Project.cpp
// ============================================================================

std::string fuzzyFindColumn(const std::string& name, const EvalContext& ctx, bool debug) {
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

bool projectStringColumn(
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
                 ctx.ensureFlatStringCol(strLookupCol);
             } else {
                 for (const auto& [n, _] : ctx.dictCols) {
                     if (n.size() > strLookupCol.size() && n.rfind(strLookupCol, 0) == 0) {
                         char sep = n[strLookupCol.size()];
                         if (sep == '_' || sep == '.') {
                             ctx.ensureFlatStringCol(n);
                             strLookupCol = n;
                             break;
                         }
                     }
                 }
             }
         }
    }

    // Try to build flat buffers from dict if not already available
    if (!ctx.flatStringCols.count(strLookupCol) || ctx.flatStringCols[strLookupCol].rowCount == 0) {
        ctx.ensureFlatStringCol(strLookupCol);
    }

    // Check if flat buffers available (avoids expensive string materialization for GPU path)
    bool hasFlatBuffers = ctx.flatStringCols.count(strLookupCol) &&
                          ctx.flatStringCols[strLookupCol].rowCount > 0;

    LOG_DEBUG("Exec", "Project: pass-through string col " << col << " (as " << strLookupCol << ") -> " << (outName.empty() ? col : outName));

    // --- DEFERRED MATERIALIZATION PATH ---
    if (hasFlatBuffers && ctx.rowCount > 0) {
        auto& flat = ctx.flatStringCols[strLookupCol];
        bool canDefer = false;
        FlatStringCol deferredFlat;

        if (ctx.activeRows.empty()) {
            deferredFlat = flat;
            canDefer = true;
        } else if (flat.rowCount == ctx.activeRows.size()) {
            deferredFlat = flat;
            canDefer = true;
        } else if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0) {
            auto r = GpuOps::gatherFlatString(
                flat.chars, flat.offsets, flat.lengths,
                ctx.activeRowsGPU, ctx.activeRowsCountGPU, true);
            if (r.chars) {
                deferredFlat.takeFrom(std::move(r.chars), std::move(r.offsets),
                                     std::move(r.lengths), r.rowCount, r.totalBytes);
                canDefer = true;
            }
        }

        if (canDefer) {
            std::string key = outName.empty() ? col : outName;
            uint32_t deferredRows = deferredFlat.rowCount;

            out.flatStringResults[key] = deferredFlat;
            if (ctx.hasDictCol(strLookupCol)) {
                out.dictStringResults[key] = ctx.dictCols[strLookupCol];
                if (key != strLookupCol) ctx.dictCols[key] = ctx.dictCols[strLookupCol];
            }
            if (key != strLookupCol) ctx.flatStringCols[key] = deferredFlat;
            ctx.flatStringCols[posName] = deferredFlat;

            out.stringCols.push_back({});
            out.stringNames.push_back(key);

            if (!outName.empty() && outName != col)
                ctx.columnAliases[col] = outName;

            if (!rowCountInitialized) { projectedRowCount = deferredRows; rowCountInitialized = true; }
            else if (deferredRows > 0 && projectedRowCount != deferredRows) {
                if (deferredRows > projectedRowCount) projectedRowCount = deferredRows;
            }

            LOG_DEBUG("Exec", "Project: DEFERRED string col " << key << " (" << deferredRows << " rows, flat pass-through)");
            return true;
        }
    }

    // --- MATERIALIZATION PATH (fallback) ---
    std::vector<std::string> sub;
    if (ctx.rowCount == 0) {
         sub = {};
    } else if (!ctx.activeRows.empty() && ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0 && hasFlatBuffers) {
         auto& flat = ctx.flatStringCols[strLookupCol];
         if (flat.rowCount == ctx.activeRows.size()) {
             const uint32_t* offs = static_cast<const uint32_t*>(flat.offsets->contents());
             const uint32_t* lens = static_cast<const uint32_t*>(flat.lengths->contents());
             const char* ch = static_cast<const char*>(flat.chars->contents());
             sub.resize(flat.rowCount);
             for (uint32_t i = 0; i < flat.rowCount; ++i) sub[i].assign(ch + offs[i], lens[i]);
         } else {
             auto r = GpuOps::gatherFlatString(
                 flat.chars, flat.offsets, flat.lengths,
                 ctx.activeRowsGPU, ctx.activeRowsCountGPU, true);
             if (r.chars) {
                 const uint32_t* offs = static_cast<const uint32_t*>(r.offsets->contents());
                 const uint32_t* lens = static_cast<const uint32_t*>(r.lengths->contents());
                 const char* ch = static_cast<const char*>(r.chars->contents());
                 sub.resize(r.rowCount);
                 for (uint32_t i = 0; i < r.rowCount; ++i) sub[i].assign(ch + offs[i], lens[i]);
             } else {
                 ctx.ensureStringCol(strLookupCol);
                 if (!ctx.stringCols.count(strLookupCol)) return false;
                 ctx.ensureActiveRowsCPU();
                 sub.reserve(ctx.activeRows.size());
                 for(auto idx : ctx.activeRows) {
                     if (idx < ctx.stringCols[strLookupCol].size()) sub.push_back(ctx.stringCols[strLookupCol][idx]);
                     else sub.push_back("");
                 }
             }
         }
    } else {
         bool flatAvail = ctx.flatStringCols.count(strLookupCol) &&
                          ctx.flatStringCols[strLookupCol].rowCount > 0;

         if (flatAvail && (ctx.activeRows.empty() || !ctx.activeRowsGPU)) {
             auto& flat = ctx.flatStringCols[strLookupCol];
             if (ctx.activeRows.empty() || flat.rowCount == ctx.activeRows.size()) {
                 const uint32_t* offs = static_cast<const uint32_t*>(flat.offsets->contents());
                 const uint32_t* lens = static_cast<const uint32_t*>(flat.lengths->contents());
                 const char* ch = static_cast<const char*>(flat.chars->contents());
                 sub.resize(flat.rowCount);
                 for (uint32_t i = 0; i < flat.rowCount; ++i) sub[i].assign(ch + offs[i], lens[i]);
             } else {
                 ctx.ensureStringCol(strLookupCol);
                 if (!ctx.stringCols.count(strLookupCol)) return false;
                 ctx.ensureActiveRowsCPU();
                 sub.reserve(ctx.activeRows.size());
                 for(auto idx : ctx.activeRows) {
                     if (idx < ctx.stringCols[strLookupCol].size()) sub.push_back(ctx.stringCols[strLookupCol][idx]);
                     else sub.push_back("");
                 }
             }
         } else {
         ctx.ensureStringCol(strLookupCol);
         if (!ctx.stringCols.count(strLookupCol)) return false;

         if (ctx.activeRows.empty() || ctx.stringCols[strLookupCol].size() == ctx.activeRows.size()) {
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

std::string resolveProjectColumn(
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

bool projectU32Column(
    const std::string& col, const std::string& lookupCol,
    const std::string& outName, const std::string& posName,
    EvalContext& ctx, TableResult& out, std::set<std::string>& usedColumns,
    size_t& projectedRowCount, bool& rowCountInitialized, bool debug)
{
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

bool projectF32Column(
    const std::string& col, const std::string& lookupCol,
    const std::string& outName, const std::string& posName,
    EvalContext& ctx, TableResult& out, std::set<std::string>& usedColumns,
    size_t& projectedRowCount, bool& rowCountInitialized, bool debug)
{
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

} // namespace engine
