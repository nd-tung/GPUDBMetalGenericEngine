#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "Operators.hpp"
#include "GpuColumnStore.hpp"
#include "EngineError.hpp"

#include <iostream>
#include <vector>
#include <set>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <functional>
#include <map>

namespace engine {

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
                        if (debug) std::cerr << "[Exec] Project: aggregate fuzzy match (varying) '" << name << "' -> '" << n << "'\n";
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
                        float* ptr = (float*)buf->contents();
                        bool varying = false;
                        for (size_t i = 1; i < std::min(cnt, engine::config::kColumnSampleSize); ++i) {
                            if (ptr[i] != ptr[0]) { varying = true; break; }
                        }
                        if (varying) {
                            varyingMatch = n;
                            if (debug) std::cerr << "[Exec] Project: aggregate fuzzy match (varying GPU) '" << name << "' -> '" << n << "'\n";
                            break;
                        }
                    }
                }
            }
        }

        if (!varyingMatch.empty()) {
            if (debug && varyingMatch != firstMatch) 
                std::cerr << "[Exec] Project: preferring varying column '" << varyingMatch << "' over '" << firstMatch << "'\n";
            return varyingMatch;
        }
        if (!firstMatch.empty()) {
            if (debug) std::cerr << "[Exec] Project: aggregate fuzzy match '" << name << "' -> '" << firstMatch << "'\n";
            return firstMatch;
        }
    }

    return "";
}

// -- Extracted: projectSubstring --
// Handles SUBSTRING(column, start, length) projection.
// Returns true if handled (caller should continue).
static bool projectSubstring(const FunctionCall& func,
    EvalContext& ctx, TableResult& out, const std::string& outName,
    const std::string& posName, bool debug) {
    // SUBSTRING(column, start, length)
    if (func.args[0] && func.args[0]->kind == TypedExpr::Kind::Column) {
        std::string colName = func.args[0]->asColumn().column;
        int startPos = 1;  // SQL SUBSTRING is 1-based
        int length = -1;   // -1 means to end

        // Get start position
        if (func.args.size() >= 2 && func.args[1] && 
            func.args[1]->kind == TypedExpr::Kind::Literal) {
            const auto& lit = func.args[1]->asLiteral();
            if (std::holds_alternative<int64_t>(lit.value)) {
                startPos = static_cast<int>(std::get<int64_t>(lit.value));
            }
        }
        // Get length
        if (func.args.size() >= 3 && func.args[2] && 
            func.args[2]->kind == TypedExpr::Kind::Literal) {
            const auto& lit = func.args[2]->asLiteral();
            if (std::holds_alternative<int64_t>(lit.value)) {
                length = static_cast<int>(std::get<int64_t>(lit.value));
            }
        }

        // --- GPU path: use FlatStringCol or flatten on-the-fly from stringCols ---
        std::string flatKey = colName;
        if (!ctx.flatStringCols.count(flatKey)) {
            std::string resolved = ctx.resolveColName(colName);
            if (!resolved.empty() && ctx.flatStringCols.count(resolved)) flatKey = resolved;
        }

        // If no pre-existing flat buffers, try flatten on-the-fly from stringCols
        bool tempFlat = false;
        FlatStringCol flatBuf;
        if (!ctx.flatStringCols.count(flatKey)) {
            // Find CPU strings (materialize from dict if needed)
            std::string strKey = colName;
            ctx.ensureStringCol(strKey);
            auto sIt = ctx.stringCols.find(strKey);
            if (sIt == ctx.stringCols.end()) {
                std::string resolved = ctx.resolveColName(colName);
                if (!resolved.empty()) {
                    ctx.ensureStringCol(resolved);
                    sIt = ctx.stringCols.find(resolved);
                    if (sIt != ctx.stringCols.end()) strKey = sIt->first;
                }
            }
            if (sIt != ctx.stringCols.end() && !sIt->second.empty()) {
                const auto& data = sIt->second;
                uint32_t rc = static_cast<uint32_t>(data.size());
                std::vector<uint32_t> offsets(rc), lengths(rc);
                size_t totalChars = 0;
                for (const auto& s : data) totalChars += s.size();
                std::vector<char> chars;
                chars.reserve(totalChars);
                size_t cur = 0;
                for (size_t i = 0; i < rc; ++i) {
                    offsets[i] = static_cast<uint32_t>(cur);
                    lengths[i] = static_cast<uint32_t>(data[i].size());
                    chars.insert(chars.end(), data[i].begin(), data[i].end());
                    cur += data[i].size();
                }
                flatBuf.rowCount = rc;
                flatBuf.totalBytes = static_cast<uint32_t>(totalChars);
                flatBuf.chars.reset(GpuOps::createBuffer(chars.empty() ? (const void*)"\0" : chars.data(),
                                                      std::max(chars.size(), (size_t)1)));
                flatBuf.offsets.reset(GpuOps::createBuffer(offsets.data(), offsets.size() * sizeof(uint32_t)));
                flatBuf.lengths.reset(GpuOps::createBuffer(lengths.data(), lengths.size() * sizeof(uint32_t)));
                tempFlat = true;
            }
        }

        FlatStringCol* flatPtr = nullptr;
        if (ctx.flatStringCols.count(flatKey))
            flatPtr = &ctx.flatStringCols[flatKey];
        else if (tempFlat)
            flatPtr = &flatBuf;

        if (flatPtr) {
            auto& flat = *flatPtr;
            uint32_t gpuStart = static_cast<uint32_t>(std::max(startPos, 1));
            uint32_t gpuLen = (length >= 0) ? static_cast<uint32_t>(length) : 0xFFFFFFFF;
            uint32_t rc = flat.rowCount;

            // GPU substring: compute new offsets/lengths (zero-copy into same chars)
            auto [subOffsets, subLengths] = GpuOps::substringFlat(
                flat.offsets, flat.lengths, gpuStart, gpuLen, rc);

            if (subOffsets && subLengths) {
                // Release temp input offsets/lengths (no longer needed after substring)
                if (tempFlat) {
                    flatBuf.offsets = nullptr;
                    flatBuf.lengths = nullptr;
                }

                // GPU hash encode for u32 groupby compatibility
                MTL::Buffer* encodedGPU = GpuOps::stringHashEncodeU32(
                    flat.chars, subOffsets, subLengths, rc);

                // Download u32 encoding
                std::vector<uint32_t> encoded(rc);
                if (encodedGPU) {
                    std::memcpy(encoded.data(), encodedGPU->contents(), rc * sizeof(uint32_t));
                    // Keep GPU buffer for downstream operators
                    ctx.u32ColsGPU[outName].reset(encodedGPU);
                    ctx.u32ColsGPU[posName] = ctx.u32ColsGPU[outName]; // GpuBuffer copy retains
                }

                // Reconstruct CPU strings from flat buffers for downstream use
                const uint8_t* charsPtr = static_cast<const uint8_t*>(flat.chars->contents());
                const uint32_t* offPtr = static_cast<const uint32_t*>(subOffsets->contents());
                const uint32_t* lenPtr = static_cast<const uint32_t*>(subLengths->contents());
                std::vector<std::string> substrResults(rc);
                for (uint32_t i = 0; i < rc; ++i) {
                    substrResults[i] = std::string(reinterpret_cast<const char*>(charsPtr + offPtr[i]), lenPtr[i]);
                }

                // Store FlatStringCol for the output under the new name
                FlatStringCol outFlat;
                outFlat.chars     = flat.chars;   // GpuBuffer copy auto-retains
                outFlat.offsets.reset(subOffsets); // takes ownership
                outFlat.lengths.reset(subLengths); // takes ownership
                outFlat.rowCount  = rc;
                outFlat.totalBytes = flat.totalBytes; // conservative
                ctx.flatStringCols[outName] = outFlat;
                ctx.flatStringCols[posName] = outFlat;

                ctx.stringCols[outName] = std::move(substrResults);
                ctx.stringCols[posName] = ctx.stringCols[outName];
                ctx.u32Cols[outName] = encoded;
                ctx.u32Cols[posName] = encoded;
                out.u32Cols.push_back(encoded);
                out.u32Names.push_back(outName);

                if (debug) {
                    std::cerr << "[Exec] Project: GPU SUBSTRING computed " << rc 
                              << " results for " << outName << "\n";
                }
                return true;
            }
            // GPU path failed — clean up temp buffers and fall through to CPU
            if (tempFlat) {
                flatBuf.release();
            }
        }

        // --- CPU fallback: find raw strings ---
        ctx.ensureStringCol(colName);
        auto strIt = ctx.stringCols.find(colName);
        if (strIt == ctx.stringCols.end()) {
            std::string resolved = ctx.resolveColName(colName);
            if (!resolved.empty()) {
                ctx.ensureStringCol(resolved);
                strIt = ctx.stringCols.find(resolved);
            }
        }

        if (strIt != ctx.stringCols.end()) {
            const auto& rawStrings = strIt->second;
            std::vector<std::string> substrResults;
            substrResults.reserve(rawStrings.size());

            for (const auto& str : rawStrings) {
                // SQL SUBSTRING is 1-based
                size_t start = (startPos > 0) ? static_cast<size_t>(startPos - 1) : 0;
                size_t len = (length >= 0) ? static_cast<size_t>(length) : str.size();
                if (start < str.size()) {
                    substrResults.push_back(str.substr(start, len));
                } else {
                    substrResults.push_back("");
                }
            }

            ctx.stringCols[outName] = std::move(substrResults);
            ctx.stringCols[posName] = ctx.stringCols[outName];

            // Build flat GPU buffers + dict + hash for groupby compatibility
            flattenStringCol(ctx, outName);
            buildDictCol(ctx, outName);
            auto flatIt = ctx.flatStringCols.find(outName);
            if (flatIt != ctx.flatStringCols.end() && flatIt->second.rowCount > 0) {
                auto& flat = flatIt->second;
                MTL::Buffer* hashBuf = GpuOps::stringFnv1aU32(flat.chars, flat.offsets, flat.lengths, flat.rowCount);
                if (hashBuf) {
                    std::vector<uint32_t> encoded(flat.rowCount);
                    std::memcpy(encoded.data(), hashBuf->contents(), flat.rowCount * sizeof(uint32_t));
                    ctx.u32Cols[outName] = encoded;
                    ctx.u32Cols[posName] = encoded;
                    ctx.u32ColsGPU[outName].reset(hashBuf);  // GpuBuffer takes ownership
                    ctx.u32ColsGPU[posName] = ctx.u32ColsGPU[outName]; // GpuBuffer copy retains
                    out.u32Cols.push_back(std::move(encoded));
                    out.u32Names.push_back(outName);
                }
            } else {
                // Fallback: CPU hash
                std::vector<uint32_t> encoded;
                encoded.reserve(ctx.stringCols[outName].size());
                for (const auto& s : ctx.stringCols[outName]) {
                    encoded.push_back(GpuOps::fnv1a32(s));
                }
                ctx.u32Cols[outName] = encoded;
                ctx.u32Cols[posName] = encoded;
                out.u32Cols.push_back(std::move(encoded));
                out.u32Names.push_back(outName);
            }

            if (debug) {
                std::cerr << "[Exec] Project: CPU SUBSTRING computed " << ctx.stringCols[outName].size() 
                          << " results for " << outName << "\n";
            }
            return true;
        }
    }
    return false;
}

// -- Extracted: projectExtractYear --
// Handles EXTRACT(YEAR FROM col) and YEAR(col) projection.
// Returns true if handled (caller should continue).
static bool projectExtractYear(const FunctionCall& func,
    const std::string& funcLower,
    EvalContext& ctx, TableResult& out, const std::string& outName,
    const std::string& posName, bool debug) {
    // Handle EXTRACT(YEAR FROM col) or YEAR(col)
    bool isYearFunc = (funcLower == "year" && func.args.size() == 1) || 
                      (funcLower == "extract" && func.args.size() >= 2);

    // Refine extract check: arg[0] should be 'year'
    if (funcLower == "extract" && isYearFunc) {
         if (func.args[0]->kind == TypedExpr::Kind::Literal && 
             std::holds_alternative<std::string>(func.args[0]->asLiteral().value)) {
             std::string part = std::get<std::string>(func.args[0]->asLiteral().value);
             std::transform(part.begin(), part.end(), part.begin(), ::tolower);
             if (part != "year") isYearFunc = false;
         } else {
             isYearFunc = false;
         }
    }

    if (isYearFunc) {
        const auto& colArg = (funcLower == "year") ? func.args[0] : func.args[1];
        if (colArg && colArg->kind == TypedExpr::Kind::Column) {
            std::string colName = colArg->asColumn().column;

            // Look for integer column (u32 or f32)
            std::vector<uint32_t> results;
            bool found = false;

            // ── M11: GPU fast-path for YEAR extraction ──────────────
            // Try to run extractYearU32 directly on GPU without downloading
            auto tryGpuYear = [&](const std::string& target) -> bool {
                MTL::Buffer* gpuBuf = nullptr;
                if (ctx.u32ColsGPU.count(target)) gpuBuf = ctx.u32ColsGPU.at(target);
                if (!gpuBuf && ctx.columnAliases.count(target)) {
                    std::string alias = ctx.columnAliases.at(target);
                    if (ctx.u32ColsGPU.count(alias)) gpuBuf = ctx.u32ColsGPU.at(alias);
                }
                if (!gpuBuf) {
                    // Fuzzy search on GPU keys
                    for (const auto& [k, buf] : ctx.u32ColsGPU) {
                        if (k.size() > target.size() && k.rfind(target, 0) == 0) {
                            char nextChar = k[target.size()];
                            if (nextChar == '_' || k.find("_rhs_") != std::string::npos) {
                                gpuBuf = buf; break;
                            }
                        }
                    }
                }
                if (!gpuBuf || gpuBuf->length() < sizeof(uint32_t)) return false;

                uint32_t gpuCount = (uint32_t)(gpuBuf->length() / sizeof(uint32_t));
                MTL::Buffer* inputBuf = gpuBuf;
                bool ownInput = false;

                // Gather by activeRows if needed
                if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0 && ctx.activeRowsCountGPU != gpuCount) {
                    inputBuf = GpuOps::gatherU32(gpuBuf, ctx.activeRowsGPU, ctx.activeRowsCountGPU);
                    gpuCount = ctx.activeRowsCountGPU;
                    ownInput = true;
                }

                MTL::Buffer* yearBuf = GpuOps::extractYearU32(inputBuf, gpuCount);
                if (ownInput) inputBuf->release();
                if (!yearBuf) return false;

                // Download to CPU
                results.resize(gpuCount);
                std::memcpy(results.data(), yearBuf->contents(), gpuCount * sizeof(uint32_t));
                // Keep GPU buffer for downstream operators
                ctx.u32ColsGPU[outName].reset(yearBuf);
                ctx.u32ColsGPU[posName] = ctx.u32ColsGPU[outName]; // GpuBuffer copy retains

                if (debug) std::cerr << "[Exec] Project: YEAR GPU-computed " << gpuCount << " results for " << outName << "\n";
                return true;
            };

            found = tryGpuYear(colName);
            // ── End GPU fast-path ────────────────────────────────────

            if (!found) {
            // CPU fallback: find column in CPU maps or download from GPU
            auto findKeyAndFetch = [&](const std::string& target) -> std::string {
                // First try CPU
                if (ctx.u32Cols.count(target) && !ctx.u32Cols.at(target).empty()) return target;
                if (ctx.columnAliases.count(target)) {
                    std::string alias = ctx.columnAliases.at(target);
                    if (ctx.u32Cols.count(alias) && !ctx.u32Cols.at(alias).empty()) return alias;
                }
                // Fuzzy search: starts with target + "_" or target + "_rhs"
                for (const auto& [k, v] : ctx.u32Cols) {
                     if (!v.empty() && k.size() > target.size() && k.rfind(target, 0) == 0) {
                          // Prefix match. Check boundary.
                          char nextChar = k[target.size()];
                          if (nextChar == '_' || k.find("_rhs_") != std::string::npos) {
                              return k;
                          }
                     }
                }

                // Check GPU and fetch if found
                auto tryFetchGPU = [&](const std::string& key) -> bool {
                    if (ctx.u32ColsGPU.count(key)) {
                        MTL::Buffer* buf = ctx.u32ColsGPU.at(key);
                        size_t count = buf->length() / sizeof(uint32_t);
                        if (count > 0) {
                            std::vector<uint32_t> down(count);
                            std::memcpy(down.data(), buf->contents(), count * sizeof(uint32_t));
                            ctx.u32Cols[key] = std::move(down);
                            if (debug) std::cerr << "[Exec] Project: YEAR lazy-fetched " << key << " from GPU (" << count << " rows)\n";
                            return true;
                        }
                    }
                    return false;
                };

                // Try direct target from GPU
                if (tryFetchGPU(target)) return target;

                // Try alias from GPU
                if (ctx.columnAliases.count(target)) {
                    std::string alias = ctx.columnAliases.at(target);
                    if (tryFetchGPU(alias)) return alias;
                }

                // Try fuzzy search on GPU keys
                for (const auto& [k, buf] : ctx.u32ColsGPU) {
                    if (k.size() > target.size() && k.rfind(target, 0) == 0) {
                        char nextChar = k[target.size()];
                        if (nextChar == '_' || k.find("_rhs_") != std::string::npos) {
                            if (tryFetchGPU(k)) return k;
                        }
                    }
                }

                return "";
            };

            std::string actualKey = findKeyAndFetch(colName);
            auto itU = (actualKey.empty()) ? ctx.u32Cols.end() : ctx.u32Cols.find(actualKey);


            if (itU != ctx.u32Cols.end() && !itU->second.empty()) {
                found = true;
                const auto& data = itU->second;
                results.reserve(data.size());
                // Respect activeRows if set
                ctx.ensureActiveRowsCPU();
                if (ctx.activeRows.size() == ctx.activeRowsCountGPU && ctx.activeRowsCountGPU > 0 && ctx.activeRows.size() != data.size()) {
                    for (uint32_t idx : ctx.activeRows) {
                        if (idx < data.size()) {
                            uint32_t val = data[idx];
                            if (val > 19000000) results.push_back(val / 10000);
                            else results.push_back(1970 + static_cast<uint32_t>(val / 365.25));
                        } else results.push_back(0);
                    }
                } else {
                    for (uint32_t val : data) {
                        if (val > 19000000) results.push_back(val / 10000);
                        else results.push_back(1970 + static_cast<uint32_t>(val / 365.25));
                    }
                }
            }

            // If not found in U32, could be String "YYYY-MM-DD"
            if (!found) {
                auto itS = ctx.stringCols.find(colName);
                if (itS != ctx.stringCols.end()) {
                    found = true;
                    const auto& data = itS->second;
                    results.reserve(data.size());
                     ctx.ensureActiveRowsCPU();
                     if (ctx.activeRows.size() == ctx.activeRowsCountGPU && ctx.activeRowsCountGPU > 0 && ctx.activeRows.size() != data.size()) {
                        for(uint32_t idx : ctx.activeRows) {
                            if(idx < data.size()) {
                                const auto& s = data[idx];
                                if (s.size() >= 4) { try { results.push_back(std::stoi(s.substr(0, 4))); } catch(...) { results.push_back(0); } }
                                else results.push_back(0);
                            } else results.push_back(0);
                        }
                     } else {
                        for (const auto& s : data) {
                            if (s.size() >= 4) { try { results.push_back(std::stoi(s.substr(0, 4))); } catch(...) { results.push_back(0); } }
                            else results.push_back(0);
                        }
                     }
                }
            }
            } // end if (!found) — CPU fallback

            if (found) {
                if(debug) std::cerr << "[Exec] Project: YEAR computed " << results.size() << " results for " << outName << " (Input table: " << ctx.currentTable << ")\n";
                ctx.u32Cols[outName] = results;
                ctx.u32Cols[posName] = results;
                out.u32Cols.push_back(results);
                out.u32Names.push_back(outName);
                return true;
            }
        }
    }
    return false;
}

// -- Extracted: projectCrossTableLookup --
// Ad-hoc hash join against saved table contexts for dimension columns.
// Returns true if handled (caller should continue).
//
// DESIGN NOTE (E7 — Separation of Concerns):
//   This function hard-codes TPC-H foreign-key relationships (s_suppkey,
//   n_nationkey, r_regionkey, etc.) to perform ad-hoc hash joins that
//   should ideally be handled by the join operator in the IR plan.
//   The planner sometimes omits explicit joins for small dimension lookups,
//   so this fallback resolves missing columns at projection time.
//
//   TODO: Refactor to use a data-driven FK map (e.g., from SchemaRegistry)
//   instead of hard-coded column-name prefixes, so this works for arbitrary
//   schemas beyond TPC-H.
static bool projectCrossTableLookup(const std::string& lookupCol,
    EvalContext& ctx, TableResult& out, const std::string& outName,
    const std::string& posName, size_t& projectedRowCount, bool& rowCountInitialized,
    std::unordered_map<std::string, EvalContext>* tableContexts, bool debug) {
    auto updateRowCount = [&](size_t size) {
        if (!rowCountInitialized) { projectedRowCount = size; rowCountInitialized = true; }
        else if (size > 0 && projectedRowCount != size && size > projectedRowCount) projectedRowCount = size;
    };
    std::string neededCol = lookupCol;
    std::string targetKey;
    std::string currentKey;

    if (neededCol.rfind("s_", 0) == 0) { targetKey = "s_suppkey"; currentKey = "ps_suppkey"; }
    else if (neededCol.rfind("n_", 0) == 0) { targetKey = "n_nationkey"; currentKey = "s_nationkey"; }
    else if (neededCol.rfind("r_", 0) == 0) { targetKey = "r_regionkey"; currentKey = "n_regionkey"; }

    // Check overrides for currentKey if not found
    if (!currentKey.empty() && ctx.u32Cols.find(currentKey) == ctx.u32Cols.end()) {
        if (targetKey == "s_suppkey" && ctx.u32Cols.count("s_suppkey")) currentKey = "s_suppkey";
        if (targetKey == "n_nationkey" && ctx.u32Cols.count("n_nationkey")) currentKey = "n_nationkey";
    }

    if (!currentKey.empty() && ctx.u32Cols.count(currentKey) && tableContexts) {
        const EvalContext* sourceCtx = nullptr;
        for (const auto& [tName, tCtx] : *tableContexts) {
            if ((tCtx.u32Cols.count(neededCol) || tCtx.f32Cols.count(neededCol) || tCtx.stringCols.count(neededCol)) &&
                tCtx.u32Cols.count(targetKey)) {
                sourceCtx = &tCtx;
                break;
            }
        }

        if (sourceCtx) {
            if (debug) std::cerr << "[Exec] Project: performing GPU ad-hoc join for " << neededCol << " on " << currentKey << " -> " << targetKey << "\n";

            // --- GPU hash join: build keys from dimension table, probe from current context ---
            MTL::Buffer* buildKeysGPU = nullptr;
            bool buildKeysOwned = false;
            if (sourceCtx->u32ColsGPU.count(targetKey) && sourceCtx->u32ColsGPU.at(targetKey)) {
                buildKeysGPU = sourceCtx->u32ColsGPU.at(targetKey);
            } else {
                const auto& sKeys = sourceCtx->u32Cols.at(targetKey);
                buildKeysGPU = GpuOps::createBuffer(sKeys.data(), sKeys.size() * sizeof(uint32_t));
                buildKeysOwned = true;
            }
            uint32_t buildCount = static_cast<uint32_t>(buildKeysGPU->length() / sizeof(uint32_t));

            // Get probe keys on GPU (prefer existing GPU buffer, fallback to CPU upload)
            MTL::Buffer* probeKeysGPU = nullptr;
            uint32_t probeCount = 0;
            bool probeOwned = false;

            if (ctx.u32ColsGPU.count(currentKey)) {
                if (ctx.activeRowsGPU) {
                    probeKeysGPU = GpuOps::gatherU32(ctx.u32ColsGPU[currentKey], ctx.activeRowsGPU, ctx.activeRowsCountGPU);
                    probeCount = ctx.activeRowsCountGPU;
                    probeOwned = true;
                } else {
                    probeKeysGPU = ctx.u32ColsGPU[currentKey];
                    probeCount = static_cast<uint32_t>(ctx.rowCount);
                }
            } else {
                const auto& probeKeysFull = ctx.u32Cols.at(currentKey);
                MTL::Buffer* probeSrc = GpuOps::createBuffer(probeKeysFull.data(), probeKeysFull.size() * sizeof(uint32_t));
                if (!ctx.activeRows.empty() || ctx.activeRowsGPU) {
                    if (ctx.activeRowsGPU) {
                        probeKeysGPU = GpuOps::gatherU32(probeSrc, ctx.activeRowsGPU, ctx.activeRowsCountGPU);
                        probeCount = ctx.activeRowsCountGPU;
                    } else {
                        MTL::Buffer* arBuf = GpuOps::createBuffer(ctx.activeRows.data(), ctx.activeRows.size() * sizeof(uint32_t));
                        probeKeysGPU = GpuOps::gatherU32(probeSrc, arBuf, static_cast<uint32_t>(ctx.activeRows.size()));
                        probeCount = static_cast<uint32_t>(ctx.activeRows.size());
                        arBuf->release();
                    }
                    probeSrc->release();
                } else {
                    probeKeysGPU = probeSrc;
                    probeCount = static_cast<uint32_t>(probeKeysFull.size());
                }
                probeOwned = true;
            }

            // GPU hash join
            auto jRes = GpuOps::joinHash(buildKeysGPU, nullptr, buildCount, probeKeysGPU, nullptr, probeCount);
            if (buildKeysOwned) buildKeysGPU->release();
            if (probeOwned) probeKeysGPU->release();

            if (debug) std::cerr << "[Exec] Project: GPU ad-hoc join matched " << jRes.count << "/" << probeCount << " rows\n";

            // joinHash output is in probe order (probe row i → buildIndices[i])
            // For FK joins, jRes.count should equal probeCount

            if (sourceCtx->f32Cols.count(neededCol) || sourceCtx->f32ColsGPU.count(neededCol)) {
                MTL::Buffer* srcValsGPU = nullptr;
                bool srcOwned = false;
                if (sourceCtx->f32ColsGPU.count(neededCol) && sourceCtx->f32ColsGPU.at(neededCol)) {
                    srcValsGPU = sourceCtx->f32ColsGPU.at(neededCol);
                } else {
                    const auto& sVals = sourceCtx->f32Cols.at(neededCol);
                    srcValsGPU = GpuOps::createBuffer(sVals.data(), sVals.size() * sizeof(float));
                    srcOwned = true;
                }
                MTL::Buffer* gathered = GpuOps::gatherF32(srcValsGPU, jRes.buildIndices, jRes.count);
                if (srcOwned) srcValsGPU->release();

                std::vector<float> res(jRes.count);
                std::memcpy(res.data(), gathered->contents(), jRes.count * sizeof(float));

                ctx.f32Cols[posName] = res;
                ctx.f32ColsGPU[posName].reset(gathered);
                if (!outName.empty()) {
                    ctx.f32Cols[outName] = res;
                    ctx.f32ColsGPU[outName] = ctx.f32ColsGPU[posName]; // GpuBuffer copy auto-retains
                }
                out.f32Cols.push_back(res);
                out.f32ColsGPU.push_back(ctx.f32ColsGPU[posName]); // GpuBuffer copy auto-retains
                out.f32Names.push_back(outName.empty() ? neededCol : outName);
                updateRowCount(res.size());
                return true;
            } else if (sourceCtx->stringCols.count(neededCol)) {
                // Strings: download buildIndices to CPU and gather strings
                const auto& sVals = sourceCtx->stringCols.at(neededCol);
                std::vector<uint32_t> buildIdx(jRes.count);
                std::memcpy(buildIdx.data(), jRes.buildIndices->contents(), jRes.count * sizeof(uint32_t));

                std::vector<std::string> res;
                res.reserve(jRes.count);
                for (uint32_t bi : buildIdx) {
                    res.push_back(bi < sVals.size() ? sVals[bi] : "");
                }
                ctx.stringCols[posName] = res;
                if (!outName.empty()) ctx.stringCols[outName] = res;
                out.stringCols.push_back(res);
                out.stringNames.push_back(outName.empty() ? neededCol : outName);
                // Dummy u32 encoding
                std::vector<uint32_t> encoded;
                for (const auto& s : res) encoded.push_back(s.empty() ? 0 : (uint32_t)s[0]);
                ctx.u32Cols[posName] = encoded;
                if (!outName.empty()) ctx.u32Cols[outName] = encoded;
                out.u32Cols.push_back(encoded);
                out.u32Names.push_back(outName.empty() ? neededCol : outName);
                updateRowCount(res.size());
                return true;
            } else if (sourceCtx->u32Cols.count(neededCol) || sourceCtx->u32ColsGPU.count(neededCol)) {
                MTL::Buffer* srcValsGPU = nullptr;
                bool srcOwned = false;
                if (sourceCtx->u32ColsGPU.count(neededCol) && sourceCtx->u32ColsGPU.at(neededCol)) {
                    srcValsGPU = sourceCtx->u32ColsGPU.at(neededCol);
                } else {
                    const auto& sVals = sourceCtx->u32Cols.at(neededCol);
                    srcValsGPU = GpuOps::createBuffer(sVals.data(), sVals.size() * sizeof(uint32_t));
                    srcOwned = true;
                }
                MTL::Buffer* gathered = GpuOps::gatherU32(srcValsGPU, jRes.buildIndices, jRes.count);
                if (srcOwned) srcValsGPU->release();

                std::vector<uint32_t> res(jRes.count);
                std::memcpy(res.data(), gathered->contents(), jRes.count * sizeof(uint32_t));

                ctx.u32Cols[posName] = res;
                ctx.u32ColsGPU[posName].reset(gathered); // GpuBuffer takes ownership
                if (!outName.empty()) {
                    ctx.u32Cols[outName] = res;
                    ctx.u32ColsGPU[outName] = ctx.u32ColsGPU[posName]; // GpuBuffer copy retains
                }
                out.u32Cols.push_back(res);
                out.u32ColsGPU.push_back(ctx.u32ColsGPU[posName]); // GpuBuffer copy auto-retains
                out.u32Names.push_back(outName.empty() ? neededCol : outName);
                updateRowCount(res.size());
                return true;
            }

            // No matching value column found — release join result
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Helper: project a non-Column (computed) expression.
// Returns true when the expression was handled (caller should `continue`).
// ---------------------------------------------------------------------------
static bool projectComputedExpression(
    const TypedExprPtr& expr,
    size_t exprIndex,
    const std::string& outName,
    const std::string& posName,
    EvalContext& ctx,
    TableResult& out,
    const std::function<void(size_t)>& updateRowCount,
    bool debug)
{
    // Check if expression output name matches an existing column (e.g., from aggregation)
    if (!outName.empty() && ctx.f32Cols.count(outName)) {
        if (debug) std::cerr << "[Exec] Project: resolving complex expression '" << outName << "' as existing f32 column\n";
        auto& colData = ctx.f32Cols[outName];
        updateRowCount(colData.size());
        ctx.f32Cols[posName] = colData;
        out.f32Cols.push_back(colData);
        out.f32Names.push_back(outName);
        return true;
    }
    if (!outName.empty() && ctx.u32Cols.count(outName)) {
        if (debug) std::cerr << "[Exec] Project: resolving complex expression '" << outName << "' as existing u32 column\n";
        auto& colData = ctx.u32Cols[outName];
        updateRowCount(colData.size());
        ctx.u32Cols[posName] = colData;
        out.u32Cols.push_back(colData);
        out.u32Names.push_back(outName);
        return true;
    }

    // Computed expression — evaluate on GPU and add to context
    ctx.aggregateCounter = 0;

    MTL::Buffer* gpuBuf = GpuExecutor::evaluateExpression(expr, ctx);
    std::vector<float> values;

    if (gpuBuf) {
        if (debug) std::cerr << "[Exec] Project: computed expr[" << exprIndex << "] on GPU\n";
        ctx.f32ColsGPU[posName].reset(gpuBuf);
        if (!outName.empty()) {
            ctx.f32ColsGPU[outName] = ctx.f32ColsGPU[posName];
        }

        uint32_t cnt = (ctx.activeRowsGPU) ? ctx.activeRowsCountGPU : ctx.rowCount;
        if (cnt == 0 && ctx.rowCount > 0 && !ctx.activeRowsGPU) cnt = ctx.rowCount;

        if (cnt > 0) {
            values.resize(cnt);
            std::memcpy(values.data(), gpuBuf->contents(), cnt * sizeof(float));
        }
    } else {
        if (debug) std::cerr << "[Exec] Project: GPU eval failed. Fallback disabled.\n";
        ENGINE_THROW("GPU Project eval failed for expression index " + std::to_string(exprIndex) + " (" + outName + ")");
    }

    if (!values.empty()) {
        if (debug) {
            std::cerr << "[Exec] Project: computed expr[" << exprIndex << "] (" << posName << ") -> "
                      << values.size() << " values\n";
        }
        if (!outName.empty()) {
            ctx.f32Cols[outName] = values;
        }
        ctx.f32Cols[posName] = values;
        out.f32Cols.push_back(std::move(values));
        updateRowCount(out.f32Cols.back().size());
        out.f32Names.push_back(outName.empty() ? posName : outName);
    } else {
        // Expression evaluation produced empty result — look for fallback columns
        bool found = false;

        // Try outName first (e.g., "c_count" for aggregate output)
        if (!outName.empty()) {
            auto itF = ctx.f32Cols.find(outName);
            if (itF != ctx.f32Cols.end()) {
                if (debug) std::cerr << "[Exec] Project: found outName " << outName << " in f32Cols\n";
                ctx.f32Cols[posName] = itF->second;
                out.f32Cols.push_back(itF->second);
                out.f32Names.push_back(outName);
                found = true;
            }
            if (!found) {
                auto itU = ctx.u32Cols.find(outName);
                if (itU != ctx.u32Cols.end()) {
                    if (debug) std::cerr << "[Exec] Project: found outName " << outName << " in u32Cols\n";
                    ctx.u32Cols[posName] = itU->second;
                    out.u32Cols.push_back(itU->second);
                    out.u32Names.push_back(outName);
                    found = true;
                }
            }

            // Fuzzy/Suffix Search for truncated aliases
            if (!found) {
                for (const auto& [key, val] : ctx.f32Cols) {
                    if (outName.size() >= 3 && key.size() >= 3 &&
                        (key.size() >= outName.size() ? key.rfind(outName) == key.size() - outName.size()
                                                      : outName.rfind(key) == outName.size() - key.size())) {
                        if (debug) std::cerr << "[Exec] Project: suffix match f32 '" << outName << "' -> '" << key << "'\n";
                        ctx.f32Cols[posName] = val;
                        out.f32Cols.push_back(val);
                        out.f32Names.push_back(outName);
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    for (const auto& [key, val] : ctx.u32Cols) {
                        if (outName.size() >= 3 && key.size() >= 3 &&
                            (key.size() >= outName.size() ? key.rfind(outName) == key.size() - outName.size()
                                                          : outName.rfind(key) == outName.size() - key.size())) {
                            if (debug) std::cerr << "[Exec] Project: suffix match u32 '" << outName << "' -> '" << key << "'\n";
                            ctx.u32Cols[posName] = val;
                            out.u32Cols.push_back(val);
                            out.u32Names.push_back(outName);
                            found = true;
                            break;
                        }
                    }
                }
            }
        }

        // Try posName (#N) in f32Cols
        if (!found) {
            auto itF = ctx.f32Cols.find(posName);
            if (itF != ctx.f32Cols.end()) {
                if (debug) std::cerr << "[Exec] Project: found " << posName << " in f32Cols\n";
                out.f32Cols.push_back(itF->second);
                updateRowCount(itF->second.size());
                out.f32Names.push_back(outName.empty() ? posName : outName);
                found = true;
            }
        }

        // Try SUM_#N pattern
        if (!found) {
            std::string sumName = "SUM_" + posName;
            auto itF = ctx.f32Cols.find(sumName);
            if (itF != ctx.f32Cols.end()) {
                if (debug) std::cerr << "[Exec] Project: found " << sumName << " in f32Cols\n";
                out.f32Cols.push_back(itF->second);
                updateRowCount(itF->second.size());
                out.f32Names.push_back(outName.empty() ? sumName : outName);
                found = true;
            }
        }

        // Check u32 columns as fallback (only for non-aggregate expressions)
        if (!found && expr->kind != TypedExpr::Kind::Aggregate) {
            auto itU = ctx.u32Cols.find(posName);
            if (itU != ctx.u32Cols.end()) {
                if (debug) std::cerr << "[Exec] Project: found " << posName << " in u32Cols\n";
                out.u32Cols.push_back(itU->second);
                updateRowCount(itU->second.size());
                out.u32Names.push_back(outName.empty() ? posName : outName);
                found = true;
            }
        }

        if (!found && debug) {
            std::cerr << "[Exec] Project: expr[" << exprIndex << "] evaluation failed, no fallback found\n";
        }
    }
    return true;
}

// -- Extracted: projectStringColumn --
// Resolves a string column by name (with alias/prefix/dict fallback), applies
// activeRows compaction (GPU flat-string gather or CPU fallback), and pushes to output.
static bool projectStringColumn(
    const std::string& col, const std::string& outName, const std::string& posName,
    EvalContext& ctx, TableResult& out,
    size_t& projectedRowCount, bool& rowCountInitialized, bool debug)
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

    if (debug) std::cerr << "[Exec] Project: pass-through string col " << col
                         << " (as " << strLookupCol << ") -> "
                         << (outName.empty() ? col : outName) << "\n";
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

    if (debug) std::cerr << "[Exec] Project: string col size " << sub.size() << "\n";

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
                    if (debug) std::cerr << "[Exec] Project: multi-instance column " << col << " -> " << lookupCol << "\n";
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
            if (debug) std::cerr << "[Exec] Project: alias resolution " << lookupCol << " -> " << aliasIt->second << "\n";
            lookupCol = aliasIt->second;
        }
        else if (!outName.empty() && outName != col &&
                 (ctx.u32Cols.find(outName) != ctx.u32Cols.end() ||
                  ctx.f32Cols.find(outName) != ctx.f32Cols.end())) {
            if (debug) std::cerr << "[Exec] Project: CTE alias fallback " << lookupCol << " -> " << outName << "\n";
            lookupCol = outName;
            ctx.columnAliases[col] = outName;
        }
    }

    // Fuzzy match for join aliases (e.g., min(x) vs min(x)_rhs_29)
    if (ctx.u32Cols.find(lookupCol) == ctx.u32Cols.end() &&
        ctx.f32Cols.find(lookupCol) == ctx.f32Cols.end()) {
         std::string found = fuzzyFindColumn(lookupCol, ctx, debug);
         if (!found.empty()) {
              if (debug) std::cerr << "[Exec] Project: fuzzy match " << lookupCol << " -> " << found << "\n";
              lookupCol = found;
         }
    }

    if (debug) std::cerr << "[Exec] Project: lookup " << col << " as " << lookupCol << "\n";
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
            if (debug) std::cerr << "[Exec] Project: CPU col " << lookupCol << " size=" << itDirect->second.size() << " != ctx.rowCount=" << ctx.rowCount << ", re-downloading from GPU\n";
            missingCPU = true;
        }
    }

    if (missingCPU) {
        MTL::Buffer* buf = nullptr;
        if (ctx.u32ColsGPU.count(lookupCol)) buf = ctx.u32ColsGPU[lookupCol];
        if (buf) {
             if (debug) std::cerr << "[Exec] Project: downloading GPU column " << lookupCol << "\n";
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
        std::cerr << "[Exec] Project: column " << lookupCol << " size=" << colData.size();
        if (!colData.empty()) std::cerr << " first=" << colData[0];
        if (colData.size() > 1) std::cerr << " second=" << colData[1];
        std::set<uint32_t> uniq(colData.begin(), colData.end());
        std::cerr << " distinct=" << uniq.size() << "\n";
    }

    if (!rowCountInitialized) { projectedRowCount = colData.size(); rowCountInitialized = true; }
    else if (colData.size() > 0 && projectedRowCount != colData.size()) {
        if (colData.size() > projectedRowCount) projectedRowCount = colData.size();
    }

    if (!outName.empty() && outName != lookupCol && outName != col) {
        ctx.u32Cols[outName] = colData;
        if (col != outName && col != lookupCol) {
            ctx.columnAliases[col] = outName;
            if (debug) std::cerr << "[Exec] Project: tracking alias " << col << " -> " << outName << "\n";
        }
    }
    if (col != lookupCol && col != outName) {
        ctx.u32Cols[col] = colData;
        if (debug) std::cerr << "[Exec] Project: also storing as CTE alias " << col << "\n";
    }
    ctx.u32Cols[posName] = colData;
    out.u32Cols.push_back(std::move(colData));
    out.u32Names.push_back(outName.empty() ? lookupCol : outName);
    if (debug) std::cerr << "[Exec] Project: Pushing U32 col " << (outName.empty()?lookupCol:outName) << "\n";
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
            if (debug) std::cerr << "[Exec] Project: CPU f32 col " << lookupCol << " size=" << ctx.f32Cols[lookupCol].size() << " != ctx.rowCount=" << ctx.rowCount << ", re-downloading from GPU\n";
            missingCPU_F32 = true;
        }
    }

    if (missingCPU_F32) {
        MTL::Buffer* buf = nullptr;
        if (ctx.f32ColsGPU.count(lookupCol)) buf = ctx.f32ColsGPU[lookupCol];
        if (buf) {
             if (debug) std::cerr << "[Exec] Project: downloading GPU f32 column " << lookupCol << "\n";
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
        std::cerr << "[Exec] Project: f32 column " << lookupCol << " size=" << colData.size();
        if (!colData.empty()) std::cerr << " first=" << colData[0];
        if (colData.size() > 1) std::cerr << " second=" << colData[1];
        float minV = colData.empty() ? 0 : colData[0], maxV = minV;
        for (float v : colData) { minV = std::min(minV, v); maxV = std::max(maxV, v); }
        std::cerr << " min=" << minV << " max=" << maxV << "\n";
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
        if (debug) std::cerr << "[Exec] Project: f32 also storing as CTE alias " << col << "\n";
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
        std::cerr << "[Exec] Project START: currentTable=" << ctx.currentTable << " ctx.u32Cols=";
        for (const auto& [k, v] : ctx.u32Cols) std::cerr << k << " ";
        std::cerr << "\n";
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
        std::cerr << "[Exec] Project: hasExistingOutput=true, out.u32Names=";
        for (const auto& n : out.u32Names) std::cerr << n << " ";
        std::cerr << "\n";
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
            std::cerr << "[Exec] Project: expr[" << i << "] outName=" << outName;
            if (expr) {
                std::cerr << " kind=" << static_cast<int>(expr->kind);
                if (expr->kind == TypedExpr::Kind::Column) {
                    if (debug) std::cerr << " col=" << expr->asColumn().column;
                }
            }
            if (debug) std::cerr << "\n";
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
                std::cerr << "[Exec] Project: function '" << func.name << "' (lower: '" << funcLower 
                          << "') args=" << func.args.size() << "\n";
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
                    if (debug) std::cerr << "[Exec] Project: function passthrough " << col << "\n";
                    ctx.u32Cols[posName] = itU->second;
                    out.u32Cols.push_back(itU->second);
                    out.u32Names.push_back(outName.empty() ? col : outName);
                    continue;
                }
                // For #N positional references, look up directly (they should exist in context)
                if (col.size() >= 2 && col[0] == '#' && std::isdigit(static_cast<unsigned char>(col[1]))) {
                    auto itU = ctx.u32Cols.find(col);
                    if (itU != ctx.u32Cols.end()) {
                        if (debug) std::cerr << "[Exec] Project: function passthrough positional " << col << "\n";
                        ctx.u32Cols[posName] = itU->second;
                        out.u32Cols.push_back(itU->second);
                        out.u32Names.push_back(outName.empty() ? col : outName);
                        continue;
                    }
                    // Also try f32
                    auto itF = ctx.f32Cols.find(col);
                    if (itF != ctx.f32Cols.end()) {
                        if (debug) std::cerr << "[Exec] Project: function passthrough positional " << col << " (f32)\n";
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
                 std::cerr << "[Exec] Project: Looking for col '" << col << "'\n";
            }
            
            // String column pass-through (alias/prefix/dict resolution + GPU/CPU compaction)
            if (projectStringColumn(col, outName, posName, ctx, out, projectedRowCount, rowCountInitialized, debug)) continue;

            
            // Handle post-GroupBy positional references: #N might be SUM_#N or COUNT_#N
            if (col.size() >= 2 && col[0] == '#') {
                // Try SUM_#N first for aggregate outputs
                std::string sumName = "SUM_" + col;
                auto itSum = ctx.f32Cols.find(sumName);
                if (itSum != ctx.f32Cols.end()) {
                    if (debug) std::cerr << "[Exec] Project: mapping " << col << " -> " << sumName << "\n";
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
                    if (debug) std::cerr << "[Exec] Project: mapping " << col << " -> " << countName << "\n";
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
                if (debug) std::cerr << "[Exec] Project: mapping unknown alias '" << col << "' to positional aggregate " << posName << "\n";
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
                        if (debug) std::cerr << "[Exec] Project: mapping unknown alias '" << col << "' to aggregate " << aggName << "\n";
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
