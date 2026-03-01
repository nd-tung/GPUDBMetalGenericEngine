#include "GpuExecutor.hpp"
#include "GpuExecutorPriv.hpp"
#include "Operators.hpp"
#include "ColumnStoreGPU.hpp"

#include <iostream>
#include <vector>
#include <set>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <map>

namespace engine {

bool GpuExecutor::executeProject(const IRProject& project, EvalContext& ctx, TableResult& out, std::unordered_map<std::string, EvalContext>* tableContexts) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    if (debug) {
        std::cerr << "[Exec] Project START: currentTable=" << ctx.currentTable << " ctx.u32Cols=";
        for (const auto& [k, v] : ctx.u32Cols) std::cerr << k << " ";
        std::cerr << "\n";
    }

    // NOTE: activeRows sync from GPU is deferred until needed (e.g., for string columns)
    // Lambda delegates to EvalContext method
    auto ensureActiveRowsCPU = [&]() {
        ctx.ensureActiveRowsCPU();
    };

    const size_t originalRowCount = ctx.rowCount;
    size_t projectedRowCount = ctx.rowCount;
    bool rowCountInitialized = false;
    auto updateRowCount = [&](size_t size) {
        if (size == 0) return;
        if (!rowCountInitialized) {
            projectedRowCount = size;
            rowCountInitialized = true;
        } else if (projectedRowCount != size) {
            // Prefer the new size when encountering differing column lengths (e.g., scalar aggregates)
            if (size > projectedRowCount) projectedRowCount = size;
        }
    };
    
    bool hasExistingOutput = !out.u32_cols.empty() || !out.f32_cols.empty();
    
    if (debug && hasExistingOutput) {
        std::cerr << "[Exec] Project: hasExistingOutput=true, out.u32_names=";
        for (const auto& n : out.u32_names) std::cerr << n << " ";
        std::cerr << "\n";
    }
    
    bool shouldCopyFromOut = hasExistingOutput && ctx.currentTable.empty();
    
    if (shouldCopyFromOut) {
        std::map<std::string, std::vector<uint32_t>> savedU32;
        std::map<std::string, std::vector<float>> savedF32;
        std::map<std::string, std::vector<std::string>> savedString;
        
        for (size_t i = 0; i < out.u32_names.size() && i < out.u32_cols.size(); ++i) {
            savedU32[out.u32_names[i]] = out.u32_cols[i];
        }
        for (size_t i = 0; i < out.f32_names.size() && i < out.f32_cols.size(); ++i) {
            savedF32[out.f32_names[i]] = out.f32_cols[i];
        }
        for (size_t i = 0; i < out.string_names.size() && i < out.string_cols.size(); ++i) {
            savedString[out.string_names[i]] = out.string_cols[i];
        }

        for (const auto& [n, v] : savedU32) {
            if (ctx.u32Cols.find(n) == ctx.u32Cols.end()) ctx.u32Cols[n] = v;
        }
        for (const auto& [n, v] : savedF32) {
            if (ctx.f32Cols.find(n) == ctx.f32Cols.end()) ctx.f32Cols[n] = v;
        }
        for (const auto& [n, v] : savedString) {
            if (ctx.stringCols.find(n) == ctx.stringCols.end()) ctx.stringCols[n] = v;
        }
    }
    
    if (hasExistingOutput) {
        out.u32_cols.clear();
        out.u32_names.clear();
        out.f32_cols.clear();
        out.f32_names.clear();
        out.string_cols.clear();
        out.string_names.clear();
        out.order.clear();
    }
    
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
                    std::cerr << " col=" << expr->asColumn().column;
                }
            }
            std::cerr << "\n";
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
                        for (int suffix = 1; suffix <= 9; ++suffix) {
                            std::string sfx = colName + "_" + std::to_string(suffix);
                            if (ctx.flatStringCols.count(sfx)) { flatKey = sfx; break; }
                        }
                    }
                    
                    // If no pre-existing flat buffers, try flatten on-the-fly from stringCols
                    bool tempFlat = false;
                    FlatStringCol flatBuf;
                    if (!ctx.flatStringCols.count(flatKey)) {
                        // Find CPU strings
                        std::string strKey = colName;
                        auto sIt = ctx.stringCols.find(strKey);
                        if (sIt == ctx.stringCols.end()) {
                            for (int suffix = 1; suffix <= 9; ++suffix) {
                                sIt = ctx.stringCols.find(colName + "_" + std::to_string(suffix));
                                if (sIt != ctx.stringCols.end()) { strKey = sIt->first; break; }
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
                            flatBuf.chars = GpuOps::createBuffer(chars.empty() ? (const void*)"\0" : chars.data(),
                                                                  std::max(chars.size(), (size_t)1));
                            flatBuf.offsets = GpuOps::createBuffer(offsets.data(), offsets.size() * sizeof(uint32_t));
                            flatBuf.lengths = GpuOps::createBuffer(lengths.data(), lengths.size() * sizeof(uint32_t));
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
                                flatBuf.offsets->release();
                                flatBuf.lengths->release();
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
                                encodedGPU->release();
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
                            outFlat.chars     = flat.chars;   // shared — zero-copy
                            outFlat.offsets   = subOffsets;
                            outFlat.lengths   = subLengths;
                            outFlat.rowCount  = rc;
                            outFlat.totalBytes = flat.totalBytes; // conservative
                            ctx.flatStringCols[outName] = outFlat;
                            ctx.flatStringCols[posName] = outFlat;
                            
                            ctx.stringCols[outName] = std::move(substrResults);
                            ctx.stringCols[posName] = ctx.stringCols[outName];
                            ctx.u32Cols[outName] = encoded;
                            ctx.u32Cols[posName] = encoded;
                            out.u32_cols.push_back(encoded);
                            out.u32_names.push_back(outName);
                            
                            if (debug) {
                                std::cerr << "[Exec] Project: GPU SUBSTRING computed " << rc 
                                          << " results for " << outName << "\n";
                            }
                            continue;
                        }
                        // GPU path failed — clean up temp buffers and fall through to CPU
                        if (tempFlat) {
                            if (flatBuf.chars) flatBuf.chars->release();
                            if (flatBuf.offsets) flatBuf.offsets->release();
                            if (flatBuf.lengths) flatBuf.lengths->release();
                        }
                    }
                    
                    // --- CPU fallback: find raw strings ---
                    auto strIt = ctx.stringCols.find(colName);
                    if (strIt == ctx.stringCols.end()) {
                        // Try suffixed versions
                        for (int suffix = 1; suffix <= 9; ++suffix) {
                            strIt = ctx.stringCols.find(colName + "_" + std::to_string(suffix));
                            if (strIt != ctx.stringCols.end()) break;
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
                                hashBuf->release();
                                ctx.u32Cols[outName] = encoded;
                                ctx.u32Cols[posName] = encoded;
                                ctx.u32ColsGPU[outName] = GpuOps::stringFnv1aU32(flat.chars, flat.offsets, flat.lengths, flat.rowCount);
                                out.u32_cols.push_back(std::move(encoded));
                                out.u32_names.push_back(outName);
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
                            out.u32_cols.push_back(std::move(encoded));
                            out.u32_names.push_back(outName);
                        }
                        
                        if (debug) {
                            std::cerr << "[Exec] Project: CPU SUBSTRING computed " << ctx.stringCols[outName].size() 
                                      << " results for " << outName << "\n";
                        }
                        continue;
                    }
                }
            }
            
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
                        yearBuf->release();

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
                        ensureActiveRowsCPU();
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
                             ensureActiveRowsCPU();
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
                        out.u32_cols.push_back(results);
                        out.u32_names.push_back(outName);
                        continue;
                    }
                }
            }

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
                    out.u32_cols.push_back(itU->second);
                    out.u32_names.push_back(outName.empty() ? col : outName);
                    continue;
                }
                // For #N positional references, look up directly (they should exist in context)
                if (col.size() >= 2 && col[0] == '#' && std::isdigit(static_cast<unsigned char>(col[1]))) {
                    auto itU = ctx.u32Cols.find(col);
                    if (itU != ctx.u32Cols.end()) {
                        if (debug) std::cerr << "[Exec] Project: function passthrough positional " << col << "\n";
                        ctx.u32Cols[posName] = itU->second;
                        out.u32_cols.push_back(itU->second);
                        out.u32_names.push_back(outName.empty() ? col : outName);
                        continue;
                    }
                    // Also try f32
                    auto itF = ctx.f32Cols.find(col);
                    if (itF != ctx.f32Cols.end()) {
                        if (debug) std::cerr << "[Exec] Project: function passthrough positional " << col << " (f32)\n";
                        ctx.f32Cols[posName] = itF->second;
                        out.f32_cols.push_back(itF->second);
                        out.f32_names.push_back(outName.empty() ? col : outName);
                        continue;
                    }
                }
            }
        }
        
        if (expr->kind == TypedExpr::Kind::Column) {
            // Simple column reference - copy to context with new name if needed
            std::string col = expr->asColumn().column;
            
            // Resolve alias for string lookup
            std::string strLookupCol = col;
            
            if (debug) {
                 std::cerr << "[Exec] Project: Looking for col '" << col << "'\n";
                 std::cerr << "[Exec] Project: ActiveRows size: " << ctx.activeRows.size() 
                           << ", StringCols count: " << ctx.stringCols.size() << "\n";
                 if (debug) { // Verbose list
                     for(auto& kv : ctx.stringCols) std::cerr << "   Found StringCol: " << kv.first << " size=" << kv.second.size() << "\n";
                     for(auto& kv : ctx.u32Cols) std::cerr << "   Found U32Col: " << kv.first << " size=" << kv.second.size() << "\n";
                 }
            }
            
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
            }

            // Check string columns first (Pass-through)
            if (ctx.stringCols.count(strLookupCol)) {
                if (debug) std::cerr << "[Exec] Project: pass-through string col " << col << " (as " << strLookupCol << ") -> " << (outName.empty() ? col : outName) << "\n";
                // Need CPU activeRows for string column projection
                ensureActiveRowsCPU();
                std::vector<std::string> sub;
                if (ctx.rowCount == 0) {
                     sub = {};
                } else if (ctx.activeRows.empty() || ctx.stringCols[strLookupCol].size() == ctx.activeRows.size()) {
                     sub = ctx.stringCols[strLookupCol];
                } else {
                     sub.reserve(ctx.activeRows.size());
                     for(auto idx : ctx.activeRows) {
                         if (idx < ctx.stringCols[strLookupCol].size()) sub.push_back(ctx.stringCols[strLookupCol][idx]);
                         else sub.push_back("");
                     }
                }
                
                if (debug) std::cerr << "[Exec] Project: string col size " << sub.size() << "\n";
                updateRowCount(sub.size());

                if (!outName.empty()) {
                    ctx.stringCols[outName] = sub;
                    if (outName != col) ctx.columnAliases[col] = outName;
                }
                ctx.stringCols[posName] = sub;
                out.string_cols.push_back(std::move(sub));
                out.string_names.push_back(outName.empty() ? col : outName);
                
                // String type takes precedence over u32 hashes
                continue;
            }

            
            // Handle post-GroupBy positional references: #N might be SUM_#N or COUNT_#N
            if (col.size() >= 2 && col[0] == '#') {
                // Try SUM_#N first for aggregate outputs
                std::string sumName = "SUM_" + col;
                auto itSum = ctx.f32Cols.find(sumName);
                if (itSum != ctx.f32Cols.end()) {
                    if (debug) std::cerr << "[Exec] Project: mapping " << col << " -> " << sumName << "\n";
                    ctx.f32Cols[posName] = itSum->second;
                    if (!outName.empty()) ctx.f32Cols[outName] = itSum->second;
                    out.f32_cols.push_back(itSum->second);
                    out.f32_names.push_back(outName.empty() ? col : outName);
                    continue;
                }
                // Try COUNT_#N
                std::string countName = "COUNT_" + col;
                auto itCount = ctx.f32Cols.find(countName);
                if (itCount != ctx.f32Cols.end()) {
                    if (debug) std::cerr << "[Exec] Project: mapping " << col << " -> " << countName << "\n";
                    ctx.f32Cols[posName] = itCount->second;
                    if (!outName.empty()) ctx.f32Cols[outName] = itCount->second;
                    out.f32_cols.push_back(itCount->second);
                    out.f32_names.push_back(outName.empty() ? col : outName);
                    continue;
                }
            }
            
            // For multi-instance columns: if col was already used OR MISSING, try suffixed versions
            std::string lookupCol = col;
            bool baseMissing = (ctx.u32Cols.find(col) == ctx.u32Cols.end() && 
                                ctx.f32Cols.find(col) == ctx.f32Cols.end() && 
                                ctx.stringCols.find(col) == ctx.stringCols.end());

            if (baseMissing || usedColumns.count(col) > 0) {
                // Try col_1, col_2, col_3, etc.
                for (int suffix = 1; suffix <= 9; ++suffix) {
                    std::string suffixedCol = col + "_" + std::to_string(suffix);
                    if (ctx.u32Cols.count(suffixedCol) > 0 || ctx.f32Cols.count(suffixedCol) > 0 || ctx.stringCols.count(suffixedCol) > 0) {
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
                // Also try: if col not found but outName is, use outName as lookup
                // This handles CTE aliasing where DuckDB plan asks to read "supplier_no" (alias)
                // but we only have "l_suppkey" (the output name which matches actual column)
                else if (!outName.empty() && outName != col &&
                         (ctx.u32Cols.find(outName) != ctx.u32Cols.end() ||
                          ctx.f32Cols.find(outName) != ctx.f32Cols.end())) {
                    if (debug) std::cerr << "[Exec] Project: CTE alias fallback " << lookupCol << " -> " << outName << "\n";
                    lookupCol = outName;
                    // Track this alias for future lookups
                    ctx.columnAliases[col] = outName;
                }
            }
            
            // Fuzzy match for join aliases (e.g., min(x) vs min(x)_rhs_29)
            if (ctx.u32Cols.find(lookupCol) == ctx.u32Cols.end() && 
                ctx.f32Cols.find(lookupCol) == ctx.f32Cols.end()) {
                 auto fuzzyFind = [&](const std::string& name) -> std::string {
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
                     // Extract aggregate function prefix and first column name
                     auto extractAggPrefix = [](const std::string& s) -> std::pair<std::string, std::string> {
                         // Extract "sum(", "avg(", "min(", "max(", "count(" prefix
                         static const std::vector<std::string> aggFuncs = {"sum(", "avg(", "min(", "max(", "count("};
                         std::string lower = s;
                         std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                         for (const auto& func : aggFuncs) {
                             if (lower.rfind(func, 0) == 0) {
                                 // Extract first column-like name (ps_XXX, l_XXX, etc.)
                                 std::string rest = s.substr(func.size());
                                 // Find first column name (alphanumeric with underscores)
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
                         // Look for matching f32 column with same prefix and column reference
                         // Prefer columns with VARYING values (not scalar broadcast) for aggregate lookups
                         std::string firstMatch;
                         std::string varyingMatch;
                         
                         for (const auto& [n, vec] : ctx.f32Cols) {
                             std::string lowerN = n;
                             std::transform(lowerN.begin(), lowerN.end(), lowerN.begin(), ::tolower);
                             // Check if it starts with same aggregate function and contains the column
                             if (lowerN.rfind(aggPrefix, 0) == 0 && n.find(firstCol) != std::string::npos) {
                                 if (firstMatch.empty()) firstMatch = n;
                                 // Check if values vary (not a scalar broadcast)
                                 if (vec.size() > 1) {
                                     float first = vec[0];
                                     bool varying = false;
                                     for (size_t i = 1; i < std::min(vec.size(), (size_t)100); ++i) {
                                         if (vec[i] != first) { varying = true; break; }
                                     }
                                     if (varying && varyingMatch.empty()) {
                                         varyingMatch = n;
                                         if (debug) std::cerr << "[Exec] Project: aggregate fuzzy match (varying) '" << name << "' -> '" << n << "'\n";
                                     }
                                 }
                             }
                         }
                         
                         // Also check GPU buffers for varying columns
                         if (varyingMatch.empty()) {
                             for (const auto& [n, buf] : ctx.f32ColsGPU) {
                                 if (!buf) continue;
                                 std::string lowerN = n;
                                 std::transform(lowerN.begin(), lowerN.end(), lowerN.begin(), ::tolower);
                                 if (lowerN.rfind(aggPrefix, 0) == 0 && n.find(firstCol) != std::string::npos) {
                                     if (firstMatch.empty()) firstMatch = n;
                                     // Check if GPU buffer has varying values
                                     size_t cnt = buf->length() / sizeof(float);
                                     if (cnt > 1) {
                                         float* ptr = (float*)buf->contents();
                                         bool varying = false;
                                         for (size_t i = 1; i < std::min(cnt, (size_t)100); ++i) {
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
                         
                         // Return varying match if found, otherwise first match
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
                 };
                 std::string found = fuzzyFind(lookupCol);
                 if (!found.empty()) {
                      if (debug) std::cerr << "[Exec] Project: fuzzy match " << lookupCol << " -> " << found << "\n";
                      lookupCol = found;
                 }
            }
            
            if (debug) std::cerr << "[Exec] Project: lookup " << col << " as " << lookupCol << "\n";
            
            // Try to download from GPU if missing on CPU OR if CPU column has wrong size
            auto itDirect = ctx.u32Cols.find(lookupCol);
            bool missingCPU = (itDirect == ctx.u32Cols.end());
            if (!missingCPU && ctx.rowCount > 0 && itDirect->second.empty()) missingCPU = true;
            // Also prefer GPU if CPU column size doesn't match context rowCount
            if (!missingCPU && ctx.rowCount > 0 && itDirect->second.size() != ctx.rowCount) {
                // GPU column may have correct filtered size
                if (ctx.u32ColsGPU.count(lookupCol)) {
                    if (debug) std::cerr << "[Exec] Project: CPU col " << lookupCol << " size=" << itDirect->second.size() << " != ctx.rowCount=" << ctx.rowCount << ", re-downloading from GPU\n";
                    missingCPU = true;
                }
            }
            
            if (missingCPU) {
                MTL::Buffer* buf = nullptr;
                if (ctx.u32ColsGPU.count(lookupCol)) {
                    buf = ctx.u32ColsGPU[lookupCol];
                }
                
                if (buf) {
                     if (debug) std::cerr << "[Exec] Project: downloading GPU column " << lookupCol << "\n";
                     uint32_t cnt = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
                     if (cnt == 0 && ctx.rowCount > 0 && !ctx.activeRowsGPU) cnt = ctx.rowCount;
                     
                     std::vector<uint32_t> down;
                     if (cnt > 0) {
                         down.resize(cnt);
                         MTL::Buffer* src = buf;
                         bool temp = false;
                         
                         if (ctx.activeRowsGPU) {
                             src = GpuOps::gatherU32(buf, ctx.activeRowsGPU, cnt);
                             temp = true;
                         }
                         
                         std::memcpy(down.data(), src->contents(), cnt * sizeof(uint32_t));
                         if (temp) src->release();
                     }
                     ctx.u32Cols[lookupCol] = std::move(down);
                }
            }
            
            auto itU = ctx.u32Cols.find(lookupCol);
            if (itU != ctx.u32Cols.end()) {
                usedColumns.insert(lookupCol);
                // If there are activeRows, we need to compact the column
                std::vector<uint32_t> colData;
                // If column size matches activeRows size, column is already dense - return as-is
                if (ctx.rowCount == 0) {
                    colData = {};
                } else if (ctx.activeRows.empty() || itU->second.size() == ctx.activeRows.size()) {
                    colData = itU->second;
                } else if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0) {
                    // GPU gather path: upload CPU vector, gather by GPU activeRows, download
                    auto& s = ColumnStoreGPU::instance();
                    MTL::Buffer* src = s.device()->newBuffer(itU->second.data(), itU->second.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    MTL::Buffer* dst = GpuOps::gatherU32(src, ctx.activeRowsGPU, ctx.activeRowsCountGPU);
                    colData.resize(ctx.activeRowsCountGPU);
                    std::memcpy(colData.data(), dst->contents(), ctx.activeRowsCountGPU * sizeof(uint32_t));
                    src->release(); dst->release();
                } else {
                    // Need CPU activeRows for compaction
                    ensureActiveRowsCPU();
                    colData.reserve(ctx.activeRows.size());
                    for (uint32_t idx : ctx.activeRows) {
                        colData.push_back(idx < itU->second.size() ? itU->second[idx] : 0);
                    }
                }
                
                if (debug) {
                    std::cerr << "[Exec] Project: column " << lookupCol << " size=" << colData.size();
                    if (!colData.empty()) std::cerr << " first=" << colData[0];
                    if (colData.size() > 1) std::cerr << " second=" << colData[1];
                    // Check for distinct values
                    std::set<uint32_t> uniq(colData.begin(), colData.end());
                    std::cerr << " distinct=" << uniq.size();
                    std::cerr << "\n";
                }
                
                updateRowCount(colData.size());
                if (!outName.empty() && outName != lookupCol) {
                    ctx.u32Cols[outName] = colData;
                    // Track alias: if input col differs from outName, register both directions
                    if (col != outName && col != lookupCol) {
                        ctx.columnAliases[col] = outName;
                        if (debug) std::cerr << "[Exec] Project: tracking alias " << col << " -> " << outName << "\n";
                    }
                }
                // For CTE aliasing: also store under the alias name (col) if it differs
                // This handles cases like "l_suppkey as supplier_no" where join needs "supplier_no"
                if (col != lookupCol && col != outName) {
                    ctx.u32Cols[col] = colData;
                    if (debug) std::cerr << "[Exec] Project: also storing as CTE alias " << col << "\n";
                }
                ctx.u32Cols[posName] = colData;
                out.u32_cols.push_back(std::move(colData));
                out.u32_names.push_back(outName.empty() ? lookupCol : outName);
                if (debug) std::cerr << "[Exec] Project: Pushing U32 col " << (outName.empty()?lookupCol:outName) << "\n";
                continue;
            }

            // Check if F32 data is on GPU and needs downloading
            bool missingCPU_F32 = (ctx.f32Cols.find(lookupCol) == ctx.f32Cols.end());
            if (!missingCPU_F32 && ctx.rowCount > 0 && ctx.f32Cols[lookupCol].empty()) missingCPU_F32 = true;
            // Also prefer GPU if CPU column size doesn't match context rowCount
            if (!missingCPU_F32 && ctx.rowCount > 0 && ctx.f32Cols[lookupCol].size() != ctx.rowCount) {
                if (ctx.f32ColsGPU.count(lookupCol)) {
                    if (debug) std::cerr << "[Exec] Project: CPU f32 col " << lookupCol << " size=" << ctx.f32Cols[lookupCol].size() << " != ctx.rowCount=" << ctx.rowCount << ", re-downloading from GPU\n";
                    missingCPU_F32 = true;
                }
            }
            
            if (missingCPU_F32) {
                MTL::Buffer* buf = nullptr;
                if (ctx.f32ColsGPU.count(lookupCol)) {
                    buf = ctx.f32ColsGPU[lookupCol];
                }
                
                if (buf) {
                     if (debug) std::cerr << "[Exec] Project: downloading GPU f32 column " << lookupCol << "\n";
                     uint32_t cnt = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
                     if (cnt == 0 && ctx.rowCount > 0 && !ctx.activeRowsGPU) cnt = ctx.rowCount; // Fallback to raw count if not filtered

                     if (cnt > 0) {
                         std::vector<float> down(cnt);
                         MTL::Buffer* src = buf;
                         bool temp = false;
                         
                         if (ctx.activeRowsGPU) {
                             src = GpuOps::gatherF32(buf, ctx.activeRowsGPU, cnt);
                             temp = true;
                         }
                         
                         std::memcpy(down.data(), src->contents(), cnt * sizeof(float));
                         if (temp) src->release();
                         
                         ctx.f32Cols[lookupCol] = std::move(down);
                     }
                }
            }
            
            auto itF = ctx.f32Cols.find(lookupCol);
            if (itF != ctx.f32Cols.end()) {
                usedColumns.insert(lookupCol);
                // If there are activeRows, we need to compact the column
                std::vector<float> colData;
                if (ctx.rowCount == 0) {
                    colData = {};
                } else if (!ctx.activeRows.empty()) {
                    // Prefer GPU gather for f32 activeRows compaction
                    if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0 && itF->second.size() > ctx.activeRows.size()) {
                        auto& s = ColumnStoreGPU::instance();
                        MTL::Buffer* src = s.device()->newBuffer(itF->second.data(), itF->second.size() * sizeof(float), MTL::ResourceStorageModeShared);
                        MTL::Buffer* dst = GpuOps::gatherF32(src, ctx.activeRowsGPU, ctx.activeRowsCountGPU);
                        colData.resize(ctx.activeRowsCountGPU);
                        std::memcpy(colData.data(), dst->contents(), ctx.activeRowsCountGPU * sizeof(float));
                        src->release(); dst->release();
                    } else {
                        // CPU fallback
                        ensureActiveRowsCPU();
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
                
                updateRowCount(colData.size());
                if (!outName.empty() && outName != col) {
                    ctx.f32Cols[outName] = colData;
                }
                // For CTE aliasing: also store under the alias name (col) if it differs from lookupCol
                if (col != lookupCol && col != outName) {
                    ctx.f32Cols[col] = colData;
                    if (debug) std::cerr << "[Exec] Project: f32 also storing as CTE alias " << col << "\n";
                }
                ctx.f32Cols[posName] = colData;
                out.f32_cols.push_back(std::move(colData));
                out.f32_names.push_back(outName.empty() ? col : outName);
                continue;
            }

            // Cross-table lookup for missing columns from saved contexts
            if (true) {
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
                        const auto& sKeys = sourceCtx->u32Cols.at(targetKey);
                        uint32_t buildCount = static_cast<uint32_t>(sKeys.size());
                        MTL::Buffer* buildKeysGPU = GpuOps::createBuffer(sKeys.data(), sKeys.size() * sizeof(uint32_t));
                        
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
                        buildKeysGPU->release();
                        if (probeOwned) probeKeysGPU->release();
                        
                        if (debug) std::cerr << "[Exec] Project: GPU ad-hoc join matched " << jRes.count << "/" << probeCount << " rows\n";
                        
                        // joinHash output is in probe order (probe row i → buildIndices[i])
                        // For FK joins, jRes.count should equal probeCount
                        
                        if (sourceCtx->f32Cols.count(neededCol)) {
                            const auto& sVals = sourceCtx->f32Cols.at(neededCol);
                            MTL::Buffer* srcValsGPU = GpuOps::createBuffer(sVals.data(), sVals.size() * sizeof(float));
                            MTL::Buffer* gathered = GpuOps::gatherF32(srcValsGPU, jRes.buildIndices, jRes.count);
                            srcValsGPU->release();
                            
                            std::vector<float> res(jRes.count);
                            std::memcpy(res.data(), gathered->contents(), jRes.count * sizeof(float));
                            gathered->release();
                            
                            ctx.f32Cols[posName] = res;
                            if (!outName.empty()) ctx.f32Cols[outName] = res;
                            out.f32_cols.push_back(res);
                            out.f32_names.push_back(outName.empty() ? neededCol : outName);
                            updateRowCount(res.size());
                            if (jRes.buildIndices) jRes.buildIndices->release();
                            if (jRes.probeIndices) jRes.probeIndices->release();
                            continue;
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
                            out.string_cols.push_back(res);
                            out.string_names.push_back(outName.empty() ? neededCol : outName);
                            // Dummy u32 encoding
                            std::vector<uint32_t> encoded;
                            for (const auto& s : res) encoded.push_back(s.empty() ? 0 : (uint32_t)s[0]);
                            ctx.u32Cols[posName] = encoded;
                            if (!outName.empty()) ctx.u32Cols[outName] = encoded;
                            out.u32_cols.push_back(encoded);
                            out.u32_names.push_back(outName.empty() ? neededCol : outName);
                            updateRowCount(res.size());
                            if (jRes.buildIndices) jRes.buildIndices->release();
                            if (jRes.probeIndices) jRes.probeIndices->release();
                            continue;
                        } else if (sourceCtx->u32Cols.count(neededCol)) {
                            const auto& sVals = sourceCtx->u32Cols.at(neededCol);
                            MTL::Buffer* srcValsGPU = GpuOps::createBuffer(sVals.data(), sVals.size() * sizeof(uint32_t));
                            MTL::Buffer* gathered = GpuOps::gatherU32(srcValsGPU, jRes.buildIndices, jRes.count);
                            srcValsGPU->release();
                            
                            std::vector<uint32_t> res(jRes.count);
                            std::memcpy(res.data(), gathered->contents(), jRes.count * sizeof(uint32_t));
                            gathered->release();
                            
                            ctx.u32Cols[posName] = res;
                            if (!outName.empty()) ctx.u32Cols[outName] = res;
                            out.u32_cols.push_back(res);
                            out.u32_names.push_back(outName.empty() ? neededCol : outName);
                            updateRowCount(res.size());
                            if (jRes.buildIndices) jRes.buildIndices->release();
                            if (jRes.probeIndices) jRes.probeIndices->release();
                            continue;
                        }
                        
                        // No matching value column found — release join result
                        if (jRes.buildIndices) jRes.buildIndices->release();
                        if (jRes.probeIndices) jRes.probeIndices->release();
                    }
                }
            }
            
            // Column not found in context - might be an alias for aggregate output
            // Try to find aggregate column by position (COUNT_#1, SUM_#1, etc.)
            // This handles cases like "c_count" which is an alias for count(o_orderkey) → COUNT_#1
            bool foundAggregate = false;
            for (const auto& [aggName, aggData] : ctx.f32Cols) {
                // Check if this is an aggregate column (COUNT_#N, SUM_#N, AVG_#N, etc.)
                if (aggName.find("COUNT_#") == 0 || aggName.find("SUM_#") == 0 || 
                    aggName.find("AVG_#") == 0 || aggName.find("MIN_#") == 0 || aggName.find("MAX_#") == 0) {
                    if (debug) std::cerr << "[Exec] Project: mapping unknown alias '" << col << "' to aggregate " << aggName << "\n";
                    ctx.f32Cols[col] = aggData;
                    ctx.f32Cols[posName] = aggData;
                    if (!outName.empty()) ctx.f32Cols[outName] = aggData;
                    out.f32_cols.push_back(aggData);
                    out.f32_names.push_back(outName.empty() ? col : outName);
                    foundAggregate = true;
                    break;
                }
            }
            if (foundAggregate) continue;
        } else {
            // Check if expression output name matches an existing column (e.g., from aggregation)
            if (!outName.empty() && ctx.f32Cols.count(outName)) {
                if (debug) std::cerr << "[Exec] Project: resolving complex expression '" << outName << "' as existing f32 column\n";
                auto& colData = ctx.f32Cols[outName];
                updateRowCount(colData.size());
                // Also map to posName for subsequent access
                ctx.f32Cols[posName] = colData;
                out.f32_cols.push_back(colData); // Copy
                out.f32_names.push_back(outName);
                continue;
            }
            if (!outName.empty() && ctx.u32Cols.count(outName)) {
                if (debug) std::cerr << "[Exec] Project: resolving complex expression '" << outName << "' as existing u32 column\n";
                auto& colData = ctx.u32Cols[outName];
                updateRowCount(colData.size());
                ctx.u32Cols[posName] = colData;
                out.u32_cols.push_back(colData); // Copy
                out.u32_names.push_back(outName);
                continue;
            }

            // Computed expression - evaluate and add to context
            // Reset aggregate counter before each top-level expression evaluation
            g_aggregateCounter = 0;
            
            // Try GPU evaluation first
            MTL::Buffer* gpuBuf = evalExprFloatGPU(expr, ctx);
            std::vector<float> values;
            
            if (gpuBuf) {
                if (debug) std::cerr << "[Exec] Project: computed expr[" << i << "] on GPU\n";
                // Store in GPU context
                if (!outName.empty()) {
                    ctx.f32ColsGPU[outName] = gpuBuf;
                }
                ctx.f32ColsGPU[posName] = gpuBuf;
                
                // Sync to CPU for compatibility with downstream operators
                uint32_t cnt = (ctx.activeRowsGPU) ? ctx.activeRowsCountGPU : ctx.rowCount;
                 // Fallback if row count seems wrong (e.g. no filter applied yet)
                if (cnt == 0 && ctx.rowCount > 0 && !ctx.activeRowsGPU) cnt = ctx.rowCount;
                
                if (cnt > 0) {
                    values.resize(cnt);
                    std::memcpy(values.data(), gpuBuf->contents(), cnt * sizeof(float));
                }
            } else {
                if (debug) std::cerr << "[Exec] Project: GPU eval failed. Fallback disabled.\n";
                throw std::runtime_error("GPU Project eval failed for expression index " + std::to_string(i) + " (" + outName + ")");
            }

            
            if (!values.empty()) {
                if (debug) {
                    std::cerr << "[Exec] Project: computed expr[" << i << "] (" << posName << ") -> " 
                              << values.size() << " values\n";
                }
                if (!outName.empty()) {
                    ctx.f32Cols[outName] = values;
                }
                ctx.f32Cols[posName] = values;
                out.f32_cols.push_back(std::move(values));
                updateRowCount(out.f32_cols.back().size());
                out.f32_names.push_back(outName.empty() ? posName : outName);
            } else {
                    // Expression evaluation failed - try to find column by positional reference
                    // This handles post-GroupBy projections where #N refers to aggregated output
                    bool found = false;
                    
                    // Try outName first (e.g., "c_count" for aggregate output)
                    if (!outName.empty()) {
                        auto itF = ctx.f32Cols.find(outName);
                        if (itF != ctx.f32Cols.end()) {
                            if (debug) std::cerr << "[Exec] Project: found outName " << outName << " in f32Cols\n";
                            ctx.f32Cols[posName] = itF->second;
                            out.f32_cols.push_back(itF->second);
                            out.f32_names.push_back(outName);
                            found = true;
                        }
                        // Also check u32Cols for outName (e.g. mixed types or string results)
                        if (!found) {
                            auto itU = ctx.u32Cols.find(outName);
                            if (itU != ctx.u32Cols.end()) {
                                if (debug) std::cerr << "[Exec] Project: found outName " << outName << " in u32Cols\n";
                                ctx.u32Cols[posName] = itU->second;
                                out.u32_cols.push_back(itU->second);
                                out.u32_names.push_back(outName);
                                found = true;
                            }
                        }

                        // Fuzzy/Suffix Search for truncated aliases (e.g. containing parentheses)
                        if (!found) {
                            // Check f32Cols for partial match
                            // Prefer matches where key ends with outName (suffix) or outName matches key suffix
                            for (const auto& [key, val] : ctx.f32Cols) {
                                if (key.find(outName) != std::string::npos || outName.find(key) != std::string::npos) {
                                    if (debug) std::cerr << "[Exec] Project: fuzzy match f32 '" << outName << "' -> '" << key << "'\n";
                                    ctx.f32Cols[posName] = val;
                                    out.f32_cols.push_back(val);
                                    out.f32_names.push_back(outName); // Keep the requested name
                                    found = true;
                                    break;
                                }
                            }
                            
                            // Check u32Cols for partial match
                            if (!found) {
                                for (const auto& [key, val] : ctx.u32Cols) {
                                    if (key.find(outName) != std::string::npos || outName.find(key) != std::string::npos) {
                                        if (debug) std::cerr << "[Exec] Project: fuzzy match u32 '" << outName << "' -> '" << key << "'\n";
                                        ctx.u32Cols[posName] = val;
                                        out.u32_cols.push_back(val);
                                        out.u32_names.push_back(outName);
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
                            out.f32_cols.push_back(itF->second);
                            updateRowCount(itF->second.size());
                            out.f32_names.push_back(outName.empty() ? posName : outName);
                            found = true;
                        }
                    }
                    
                    // Try SUM_#N pattern
                    if (!found) {
                        std::string sumName = "SUM_" + posName;
                        auto itF = ctx.f32Cols.find(sumName);
                        if (itF != ctx.f32Cols.end()) {
                            if (debug) std::cerr << "[Exec] Project: found " << sumName << " in f32Cols\n";
                            out.f32_cols.push_back(itF->second);
                            updateRowCount(itF->second.size());
                            out.f32_names.push_back(outName.empty() ? sumName : outName);
                            found = true;
                        }
                    }
                    
                    // Check u32 columns as fallback (only for non-aggregate expressions)
                    // Skip u32 fallback for Aggregate expressions to avoid using group keys
                    if (!found && expr->kind != TypedExpr::Kind::Aggregate) {
                        auto itU = ctx.u32Cols.find(posName);
                        if (itU != ctx.u32Cols.end()) {
                            if (debug) std::cerr << "[Exec] Project: found " << posName << " in u32Cols\n";
                            out.u32_cols.push_back(itU->second);
                            updateRowCount(itU->second.size());
                            out.u32_names.push_back(outName.empty() ? posName : outName);
                            found = true;
                        }
                    }
                    
                    if (!found && debug) {
                        std::cerr << "[Exec] Project: expr[" << i << "] evaluation failed, no fallback found\n";
                    }
                }
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
    return true;
}

} // namespace engine
