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

bool projectSubstring(const FunctionCall& func,
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
                flatBuf.offsets= GpuOps::createBuffer(offsets.data(), offsets.size() * sizeof(uint32_t));
                flatBuf.lengths= GpuOps::createBuffer(lengths.data(), lengths.size() * sizeof(uint32_t));
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
                GpuBuffer encodedGPU = GpuOps::stringHashEncodeU32(
                    flat.chars, subOffsets, subLengths, rc);

                // Keep GPU buffer — skip CPU download (lazy-fetch at output)
                std::vector<uint32_t> encoded;  // empty sentinel
                if (encodedGPU) {
                    ctx.u32ColsGPU[outName].reset(encodedGPU);
                    ctx.u32ColsGPU[posName] = ctx.u32ColsGPU[outName]; // GpuBuffer copy retains
                }

                // Store FlatStringCol for the output under the new name
                // Skip CPU string reconstruction — FlatStringCol is authoritative;
                // downstream operators (filter, groupby) use GPU buffers;
                // output stage lazy-materializes from flatStringCols if needed.
                FlatStringCol outFlat;
                outFlat.chars     = flat.chars;   // GpuBuffer copy auto-retains
                outFlat.offsets.reset(subOffsets); // takes ownership
                outFlat.lengths.reset(subLengths); // takes ownership
                outFlat.rowCount  = rc;
                outFlat.totalBytes = flat.totalBytes; // conservative
                ctx.flatStringCols[outName] = outFlat;
                ctx.flatStringCols[posName] = outFlat;

                ctx.stringCols[outName] = {};  // empty sentinel for column discovery
                ctx.stringCols[posName] = {};
                ctx.u32Cols[outName] = encoded;
                ctx.u32Cols[posName] = encoded;
                out.u32Cols.push_back(encoded);
                out.u32Names.push_back(outName);

                if (debug) {
                    LOG_INFO("Exec", "Project: GPU SUBSTRING computed " << rc  << " results for " << outName);
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
                GpuBuffer hashBuf = GpuOps::stringFnv1aU32(flat.chars, flat.offsets, flat.lengths, flat.rowCount);
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
                LOG_INFO("Exec", "Project: CPU SUBSTRING computed " << ctx.stringCols[outName].size()  << " results for " << outName);
            }
            return true;
        }
    }
    return false;
}

// -- Extracted: projectExtractYear --
// Handles EXTRACT(YEAR FROM col) and YEAR(col) projection.
// Returns true if handled (caller should continue).
bool projectExtractYear(const FunctionCall& func,
    const std::string& funcLower,
    EvalContext& ctx, TableResult& out, const std::string& outName,
    const std::string& posName, bool /*debug*/) {
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
                    inputBuf = GpuOps::gatherU32(gpuBuf, ctx.activeRowsGPU, ctx.activeRowsCountGPU).detach();
                    gpuCount = ctx.activeRowsCountGPU;
                    ownInput = true;
                }

                GpuBuffer yearBuf = GpuOps::extractYearU32(inputBuf, gpuCount);
                if (ownInput) inputBuf->release();
                if (!yearBuf) return false;

                // Download to CPU
                results.resize(gpuCount);
                std::memcpy(results.data(), yearBuf->contents(), gpuCount * sizeof(uint32_t));
                // Keep GPU buffer for downstream operators
                ctx.u32ColsGPU[outName].reset(yearBuf);
                ctx.u32ColsGPU[posName] = ctx.u32ColsGPU[outName]; // GpuBuffer copy retains

                LOG_DEBUG("Exec", "Project: YEAR GPU-computed " << gpuCount << " results for " << outName);
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
                            LOG_DEBUG("Exec", "Project: YEAR lazy-fetched " << key << " from GPU (" << count << " rows)\n");
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
                            if (val > config::kYYYYMMDDThreshold) results.push_back(val / 10000);
                            else results.push_back(config::kEpochYear + static_cast<uint32_t>(val / config::kDaysPerYear));
                        } else results.push_back(0);
                    }
                } else {
                    for (uint32_t val : data) {
                        if (val > config::kYYYYMMDDThreshold) results.push_back(val / 10000);
                        else results.push_back(config::kEpochYear + static_cast<uint32_t>(val / config::kDaysPerYear));
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
                LOG_DEBUG("Exec", "Project: YEAR computed " << results.size() << " results for " << outName << " (Input table: " << ctx.currentTable << ")\n");
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
//   This function uses SchemaRegistry FK metadata to perform ad-hoc hash joins
//   for missing dimension columns that the planner sometimes omits explicit
//   joins for (e.g., small dimension lookups resolved at projection time).
bool projectCrossTableLookup(const std::string& lookupCol,
    EvalContext& ctx, TableResult& out, const std::string& outName,
    const std::string& posName, size_t& projectedRowCount, bool& rowCountInitialized,
    std::unordered_map<std::string, EvalContext>* tableContexts, bool /*debug*/) {
    auto updateRowCount = [&](size_t size) {
        if (!rowCountInitialized) { projectedRowCount = size; rowCountInitialized = true; }
        else if (size > 0 && projectedRowCount != size && size > projectedRowCount) projectedRowCount = size;
    };
    std::string neededCol = lookupCol;
    std::string targetKey;
    std::string currentKey;

    auto fkOpt = SchemaRegistry::instance().findFKForColumn(neededCol);
    if (fkOpt) {
        targetKey = fkOpt->dimKey;
        currentKey = fkOpt->fkColumn;
    }

    // Fallback: if expected FK column is missing, try the PK itself
    if (!currentKey.empty() && ctx.u32Cols.find(currentKey) == ctx.u32Cols.end()) {
        if (ctx.u32Cols.count(targetKey)) currentKey = targetKey;
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
            LOG_DEBUG("Exec", "Project: performing GPU ad-hoc join for " << neededCol << " on " << currentKey << " -> " << targetKey);

            // --- GPU hash join: build keys from dimension table, probe from current context ---
            MTL::Buffer* buildKeysGPU = nullptr;
            bool buildKeysOwned = false;
            if (sourceCtx->u32ColsGPU.count(targetKey) && sourceCtx->u32ColsGPU.at(targetKey)) {
                buildKeysGPU = sourceCtx->u32ColsGPU.at(targetKey);
            } else {
                const auto& sKeys = sourceCtx->u32Cols.at(targetKey);
                buildKeysGPU = GpuOps::createBuffer(sKeys.data(), sKeys.size() * sizeof(uint32_t)).detach();
                buildKeysOwned = true;
            }
            uint32_t buildCount = static_cast<uint32_t>(buildKeysGPU->length() / sizeof(uint32_t));

            // Get probe keys on GPU (prefer existing GPU buffer, fallback to CPU upload)
            MTL::Buffer* probeKeysGPU = nullptr;
            uint32_t probeCount = 0;
            bool probeOwned = false;

            if (ctx.u32ColsGPU.count(currentKey)) {
                if (ctx.activeRowsGPU) {
                    probeKeysGPU = GpuOps::gatherU32(ctx.u32ColsGPU[currentKey], ctx.activeRowsGPU, ctx.activeRowsCountGPU).detach();
                    probeCount = ctx.activeRowsCountGPU;
                    probeOwned = true;
                } else {
                    probeKeysGPU = ctx.u32ColsGPU[currentKey];
                    probeCount = static_cast<uint32_t>(ctx.rowCount);
                }
            } else {
                const auto& probeKeysFull = ctx.u32Cols.at(currentKey);
                GpuBuffer probeSrc = GpuOps::createBuffer(probeKeysFull.data(), probeKeysFull.size() * sizeof(uint32_t));
                if (!ctx.activeRows.empty() || ctx.activeRowsGPU) {
                    if (ctx.activeRowsGPU) {
                        probeKeysGPU = GpuOps::gatherU32(probeSrc, ctx.activeRowsGPU, ctx.activeRowsCountGPU).detach();
                        probeCount = ctx.activeRowsCountGPU;
                    } else {
                        GpuBuffer arBuf = GpuOps::createBuffer(ctx.activeRows.data(), ctx.activeRows.size() * sizeof(uint32_t));
                        probeKeysGPU = GpuOps::gatherU32(probeSrc, arBuf, static_cast<uint32_t>(ctx.activeRows.size())).detach();
                        probeCount = static_cast<uint32_t>(ctx.activeRows.size());
                    }
                } else {
                    probeKeysGPU = probeSrc.detach();
                    probeCount = static_cast<uint32_t>(probeKeysFull.size());
                }
                probeOwned = true;
            }

            // GPU hash join
            auto jRes = GpuOps::joinHash(buildKeysGPU, buildCount, probeKeysGPU, probeCount);
            if (buildKeysOwned) buildKeysGPU->release();
            if (probeOwned) probeKeysGPU->release();

            LOG_DEBUG("Exec", "Project: GPU ad-hoc join matched " << jRes.count << "/" << probeCount << " rows\n");

            // joinHash output is in probe order (probe row i → buildIndices[i])
            // For FK joins, jRes.count should equal probeCount

            if (sourceCtx->f32Cols.count(neededCol) || sourceCtx->f32ColsGPU.count(neededCol)) {
                MTL::Buffer* srcValsGPU = nullptr;
                bool srcOwned = false;
                if (sourceCtx->f32ColsGPU.count(neededCol) && sourceCtx->f32ColsGPU.at(neededCol)) {
                    srcValsGPU = sourceCtx->f32ColsGPU.at(neededCol);
                } else {
                    const auto& sVals = sourceCtx->f32Cols.at(neededCol);
                    srcValsGPU = GpuOps::createBuffer(sVals.data(), sVals.size() * sizeof(float)).detach();
                    srcOwned = true;
                }
                GpuBuffer gathered = GpuOps::gatherF32(srcValsGPU, jRes.buildIndices, jRes.count);
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
                    srcValsGPU = GpuOps::createBuffer(sVals.data(), sVals.size() * sizeof(uint32_t)).detach();
                    srcOwned = true;
                }
                GpuBuffer gathered = GpuOps::gatherU32(srcValsGPU, jRes.buildIndices, jRes.count);
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
bool projectComputedExpression(
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
        auto& colData = ctx.f32Cols[outName];
        // Lazy-fetch from GPU if CPU vector is empty sentinel
        if (colData.empty() && ctx.f32ColsGPU.count(outName) && ctx.f32ColsGPU[outName]) {
            uint32_t rc = ctx.rowCount;
            colData.resize(rc);
            std::memcpy(colData.data(), ctx.f32ColsGPU[outName]->contents(), rc * sizeof(float));
        }
        LOG_DEBUG("Exec", "Project: resolving complex expression '" << outName << "' as existing f32 column\n");
        updateRowCount(colData.size());
        ctx.f32Cols[posName] = colData;
        out.f32Cols.push_back(colData);
        out.f32Names.push_back(outName);
        return true;
    }
    if (!outName.empty() && ctx.u32Cols.count(outName)) {
        auto& colData = ctx.u32Cols[outName];
        // Lazy-fetch from GPU if CPU vector is empty sentinel
        if (colData.empty() && ctx.u32ColsGPU.count(outName) && ctx.u32ColsGPU[outName]) {
            uint32_t rc = ctx.rowCount;
            colData.resize(rc);
            std::memcpy(colData.data(), ctx.u32ColsGPU[outName]->contents(), rc * sizeof(uint32_t));
        }
        LOG_DEBUG("Exec", "Project: resolving complex expression '" << outName << "' as existing u32 column\n");
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
        LOG_DEBUG("Exec", "Project: computed expr[" << exprIndex << "] on GPU\n");
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
        LOG_DEBUG("Exec", "Project: GPU eval failed. Fallback disabled.\n");
        ENGINE_THROW("GPU Project eval failed for expression index " + std::to_string(exprIndex) + " (" + outName + ")");
    }

    if (!values.empty()) {
        if (debug) {
            LOG_INFO("Exec", "Project: computed expr[" << exprIndex << "] (" << posName << ") -> " << values.size() << " values\n");
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
                // Lazy-fetch from GPU if CPU vector is empty sentinel
                if (itF->second.empty() && ctx.f32ColsGPU.count(outName) && ctx.f32ColsGPU[outName]) {
                    uint32_t rc = ctx.rowCount;
                    itF->second.resize(rc);
                    std::memcpy(itF->second.data(), ctx.f32ColsGPU[outName]->contents(), rc * sizeof(float));
                }
                LOG_DEBUG("Exec", "Project: found outName " << outName << " in f32Cols\n");
                ctx.f32Cols[posName] = itF->second;
                out.f32Cols.push_back(itF->second);
                out.f32Names.push_back(outName);
                found = true;
            }
            if (!found) {
                auto itU = ctx.u32Cols.find(outName);
                if (itU != ctx.u32Cols.end()) {
                    // Lazy-fetch from GPU if CPU vector is empty sentinel
                    if (itU->second.empty() && ctx.u32ColsGPU.count(outName) && ctx.u32ColsGPU[outName]) {
                        uint32_t rc = ctx.rowCount;
                        itU->second.resize(rc);
                        std::memcpy(itU->second.data(), ctx.u32ColsGPU[outName]->contents(), rc * sizeof(uint32_t));
                    }
                    LOG_DEBUG("Exec", "Project: found outName " << outName << " in u32Cols\n");
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
                        LOG_DEBUG("Exec", "Project: suffix match f32 '" << outName << "' -> '" << key << "'\n");
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
                            LOG_DEBUG("Exec", "Project: suffix match u32 '" << outName << "' -> '" << key << "'\n");
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
                LOG_DEBUG("Exec", "Project: found " << posName << " in f32Cols\n");
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
                LOG_DEBUG("Exec", "Project: found " << sumName << " in f32Cols\n");
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
                LOG_DEBUG("Exec", "Project: found " << posName << " in u32Cols\n");
                out.u32Cols.push_back(itU->second);
                updateRowCount(itU->second.size());
                out.u32Names.push_back(outName.empty() ? posName : outName);
                found = true;
            }
        }

        if (!found && debug) {
            LOG_WARN("Exec", "Project: expr[" << exprIndex << "] evaluation failed, no fallback found\n");
        }
    }
    return true;
}
} // namespace engine
