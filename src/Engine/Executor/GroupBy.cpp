#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "Operators.hpp"
#include "GpuColumnStore.hpp"
#include "KernelTimer.hpp"
#include "Schema.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace engine {

bool GpuExecutor::executeGroupBy(const IRGroupBy& groupBy, EvalContext& ctx, TableResult& out) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    if (debug) {
        std::cerr << "[Exec] GroupBy: ctx.rowCount=" << ctx.rowCount << "\n";
        std::cerr << "[Exec] GroupBy: ctx.u32Cols.size=" << ctx.u32Cols.size() << ":";
        for (const auto& [n,v] : ctx.u32Cols) std::cerr << " " << n << "(" << v.size() << ")";
        std::cerr << "\n";
        std::cerr << "[Exec] GroupBy: ctx.f32Cols.size=" << ctx.f32Cols.size() << ":";
        for (const auto& [n,v] : ctx.f32Cols) std::cerr << " " << n << "(" << v.size() << ")";
        std::cerr << "\n";
        std::cerr << "[Exec] GroupBy: keys.size=" << groupBy.keys.size() << "\n";
        for (size_t i = 0; i < groupBy.keys.size(); ++i) {
            if (groupBy.keys[i] && groupBy.keys[i]->kind == TypedExpr::Kind::Column) {
                std::cerr << "[Exec] GroupBy:   key[" << i << "]=" << groupBy.keys[i]->asColumn().column << "\n";
            }
        }
    }
    
    // Build key vectors
    std::vector<std::vector<uint32_t>> keyVecs;
    std::vector<std::string> keyNames;
    std::vector<std::vector<std::string>> outputStringMaps;
    std::vector<std::unordered_map<uint32_t, std::string>> hashToStringMaps;  // For hash-based lookup
    std::vector<bool> keyFromF32;  // Track which keys were converted from f32
    std::vector<MTL::Buffer*> keyBufsGPU;  // Pre-existing GPU buffers for keys (avoid re-upload)
    
    // Expected row count for this GroupBy - prefer GPU activeRows count
    size_t expectedKeyRows = ctx.rowCount;
    if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0) {
        expectedKeyRows = ctx.activeRowsCountGPU;
    } else if (!ctx.activeRows.empty()) {
        expectedKeyRows = ctx.activeRows.size();
    }
    
    for (size_t i = 0; i < groupBy.keys.size(); ++i) {
        const auto& keyExpr = groupBy.keys[i];
        std::string keyName = i < groupBy.keyNames.size() ? groupBy.keyNames[i] : "";
        
        if (keyExpr && keyExpr->kind == TypedExpr::Kind::Column) {
            const std::string& col = keyExpr->asColumn().column;

            // LAZY FETCH: If vector is empty but on GPU, bring it back
            if ((ctx.u32Cols.find(col) == ctx.u32Cols.end() || ctx.u32Cols[col].empty()) && ctx.u32ColsGPU.count(col)) {
                 MTL::Buffer* buf = ctx.u32ColsGPU.at(col);
                 size_t count = buf->length() / sizeof(uint32_t);
                 if (count > 0) {
                     std::vector<uint32_t> down(count);
                     std::memcpy(down.data(), buf->contents(), count * sizeof(uint32_t));
                     ctx.u32Cols[col] = std::move(down);
                     if(debug) std::cerr << "[Exec] GroupBy: Lazy fetch key " << col << " from GPU (" << count << " rows)\n";
                 }
            }
            
            // Try exact match first
            auto it = ctx.u32Cols.find(col);
            
            // Prefer column with matching row count (in case of duplicates with different sizes)
            if (it != ctx.u32Cols.end() && it->second.size() != expectedKeyRows) {
                if (debug) {
                    std::cerr << "[Exec] GroupBy: key " << col << " has wrong size (" << it->second.size() 
                              << " vs expected " << expectedKeyRows << "), looking for positional ref\n";
                }
                // Look for positional ref (#N) with correct size
                auto origIt = it;
                it = ctx.u32Cols.end();  // Reset to not found
                
                // Try positional references #0, #1, ... that might have the right data
                for (size_t pos = 0; pos < 20 && it == ctx.u32Cols.end(); ++pos) {
                    std::string posKey = "#" + std::to_string(pos);
                    auto posIt = ctx.u32Cols.find(posKey);
                    if (posIt != ctx.u32Cols.end() && posIt->second.size() == expectedKeyRows) {
                        // Check if this positional ref might be the column we want
                        // Compare a sample of values (first and last) as heuristic
                        const auto& origData = origIt->second;
                        const auto& posData = posIt->second;
                        if (!origData.empty() && !posData.empty()) {
                            // Compare first value (respecting activeRows).
                            size_t firstIdx = 0;
                            ctx.ensureActiveRowsCPU();
                            if (!ctx.activeRows.empty()) {
                                firstIdx = ctx.activeRows[0];
                            }
                            
                            if (firstIdx < origData.size() && origData[firstIdx] == posData[0]) {
                                // Double check with last element if enough rows
                                bool matchConfirmed = true;
                                if (posData.size() > 1) {
                                  size_t lastIdx = posData.size() - 1;
                                  size_t origLastIdx = lastIdx;
                                  if (!ctx.activeRows.empty() && lastIdx < ctx.activeRows.size()) {
                                      origLastIdx = ctx.activeRows[lastIdx];
                                  }
                                  if (origLastIdx < origData.size() && origData[origLastIdx] != posData[lastIdx]) {
                                      matchConfirmed = false;
                                  }
                                }
                                
                                if (matchConfirmed) {
                                    it = posIt;
                                    if (debug) {
                                        std::cerr << "[Exec] GroupBy: using positional " << posKey 
                                                  << " for key " << col << " (matched via value sampling)\n";
                                    }
                                }
                            }
                        }
                    }
                }
                
                // If still not found, fall back to original (will trigger size mismatch later)
                if (it == ctx.u32Cols.end()) {
                    it = origIt;
                }
            }
            
            // If col is a positional reference like #N, try to find the Nth column
            if (it == ctx.u32Cols.end() && col.size() >= 2 && col[0] == '#') {
                try {
                    size_t pos = std::stoul(col.substr(1));
                    size_t idx = 0;
                    for (auto& [name, data] : ctx.u32Cols) {
                        if (idx == pos) {
                            it = ctx.u32Cols.find(name);
                            if (keyName.empty()) keyName = name;  // Use actual column name
                            if (debug) {
                                std::cerr << "[Exec] GroupBy: resolved positional " << col << " to " << name << "\n";
                            }
                            break;
                        }
                        idx++;
                    }
                } catch (...) {}
            }
            
            // Try keyName as fallback if col not found
            if (it == ctx.u32Cols.end() && !keyName.empty() && keyName != col) {
                it = ctx.u32Cols.find(keyName);
                if (debug && it != ctx.u32Cols.end()) {
                    std::cerr << "[Exec] GroupBy: found key using keyName " << keyName << "\n";
                }
            }
            
            // Try suffixed versions for multi-instance tables
            if (it == ctx.u32Cols.end()) {
                for (int suffix = 1; suffix <= 9 && it == ctx.u32Cols.end(); ++suffix) {
                    std::string suffixedCol = col + "_" + std::to_string(suffix);
                    it = ctx.u32Cols.find(suffixedCol);
                }
            }
            
            
            bool stringHandled = false;
            
            // (Path 0) Dictionary ID path — collision-free, no hashing or collision check needed.
            // Dict IDs are compact 0-based sequential integers, ideal as groupby keys.
            if (!stringHandled && ctx.dictCols.count(col) && !ctx.dictCols.at(col).dictionary.empty()) {
                auto& dict = ctx.dictCols[col];  // non-const: ensureIdsCPU may mutate
                ctx.ensureActiveRowsCPU();
                
                std::vector<uint32_t> ids;
                if (!ctx.activeRows.empty() && dict.ids.size() != expectedKeyRows) {
                    // Filter by active rows: GPU gather of dict IDs
                    if (dict.idsGPU && ctx.activeRowsGPU) {
                        MTL::Buffer* gathered = GpuOps::gatherU32(dict.idsGPU, ctx.activeRowsGPU, ctx.activeRowsCountGPU);
                        ids.resize(expectedKeyRows);
                        std::memcpy(ids.data(), gathered->contents(), expectedKeyRows * sizeof(uint32_t));
                        gathered->release();
                    } else {
                        dict.ensureIdsCPU();
                        ids.reserve(expectedKeyRows);
                        for (uint32_t r : ctx.activeRows) {
                            ids.push_back(r < dict.ids.size() ? dict.ids[r] : 0);
                        }
                    }
                } else {
                    // Dense data: download GPU IDs if CPU mirror is stale/empty
                    if (dict.idsGPU && dict.ids.size() != expectedKeyRows) {
                        uint32_t gpuRows = (uint32_t)(dict.idsGPU->length() / sizeof(uint32_t));
                        ids.resize(gpuRows);
                        std::memcpy(ids.data(), dict.idsGPU->contents(), gpuRows * sizeof(uint32_t));
                        if (ids.size() > expectedKeyRows) ids.resize(expectedKeyRows);
                    } else {
                        dict.ensureIdsCPU();
                        ids = dict.ids;
                        if (ids.size() > expectedKeyRows) ids.resize(expectedKeyRows);
                    }
                }
                
                // Build ID→string reverse map for output recovery
                std::unordered_map<uint32_t, std::string> idToStr;
                idToStr.reserve(dict.dictionary.size());
                for (uint32_t d = 0; d < static_cast<uint32_t>(dict.dictionary.size()); ++d) {
                    idToStr[d] = dict.dictionary[d];
                }
                
                keyVecs.push_back(std::move(ids));
                keyBufsGPU.push_back(nullptr);
                keyNames.push_back(keyName.empty() ? col : keyName);
                keyFromF32.push_back(false);
                outputStringMaps.push_back({});
                hashToStringMaps.push_back(std::move(idToStr));
                stringHandled = true;
                if (debug) std::cerr << "[Exec] GroupBy: Dict ID key for " << col
                                     << " (" << dict.dictionary.size() << " unique, collision-free)\n";
            }
            
            // (Path 1) String key encoding — GPU FNV1a hash with CPU collision check.
            // If collision-free, uses hash directly as groupby key (skips CPU std::map).
            // Falls back to CPU sequential ID encoding if collisions detected.
            if (!stringHandled && (ctx.stringCols.count(col) || ctx.hasDictCol(col))) {
                // Materialize stringCols from dict if needed
                ctx.ensureStringCol(col);
                if (ctx.stringCols.count(col) && !ctx.stringCols.at(col).empty()) {
                const auto& strData = ctx.stringCols.at(col);
                
                // If activeRowsGPU is set but activeRows (CPU) is empty, download indices
                ctx.ensureActiveRowsCPU();
                
                if (strData.size() == expectedKeyRows || !ctx.activeRows.empty()) {
                     // --- GPU FNV1a hash path ---
                     bool gpuHashOk = false;
                     
                     // Find or create flat string buffers
                     std::string flatKey = col;
                     if (!ctx.flatStringCols.count(flatKey)) {
                         for (int sfx = 1; sfx <= 9; ++sfx) {
                             std::string sfxKey = col + "_" + std::to_string(sfx);
                             if (ctx.flatStringCols.count(sfxKey)) { flatKey = sfxKey; break; }
                         }
                     }
                     
                     MTL::Buffer* hashBuf = nullptr;
                     
                     if (ctx.flatStringCols.count(flatKey)) {
                         auto& flat = ctx.flatStringCols[flatKey];
                         // If activeRows filtering needed, gather offsets/lengths first
                         if (!ctx.activeRows.empty() && strData.size() != expectedKeyRows && ctx.activeRowsGPU) {
                             auto gOff = GpuOps::gatherU32(flat.offsets, ctx.activeRowsGPU, ctx.activeRowsCountGPU);
                             auto gLen = GpuOps::gatherU32(flat.lengths, ctx.activeRowsGPU, ctx.activeRowsCountGPU);
                             hashBuf = GpuOps::stringFnv1aU32(flat.chars, gOff, gLen, expectedKeyRows);
                             gOff->release(); gLen->release();
                         } else {
                             hashBuf = GpuOps::stringFnv1aU32(flat.chars, flat.offsets, flat.lengths, expectedKeyRows);
                         }
                     } else {
                         if (debug) std::cerr << "[Exec] GroupBy: WARN no flatStringCols for " << col << ", skipping GPU hash\n";
                     }
                     
                     if (hashBuf) {
                         // Download hashes and check for collisions
                         std::vector<uint32_t> hashes(expectedKeyRows);
                         std::memcpy(hashes.data(), hashBuf->contents(), expectedKeyRows * sizeof(uint32_t));
                         hashBuf->release();
                         
                         // Build hash→string map (also serves as collision check)
                         std::unordered_map<uint32_t, std::string> hashMap;
                         bool hasCollision = false;
                         const std::vector<std::string>* srcData = &strData;
                         std::vector<std::string> activeFiltered;
                         if (!ctx.activeRows.empty() && strData.size() != expectedKeyRows) {
                             activeFiltered.reserve(expectedKeyRows);
                             for (uint32_t r : ctx.activeRows) {
                                 activeFiltered.push_back(r < strData.size() ? strData[r] : "");
                             }
                             srcData = &activeFiltered;
                         }
                         for (uint32_t i = 0; i < expectedKeyRows && !hasCollision; ++i) {
                             auto [it2, inserted] = hashMap.try_emplace(hashes[i], (*srcData)[i]);
                             if (!inserted && it2->second != (*srcData)[i]) {
                                 hasCollision = true;
                             }
                         }
                         
                         if (!hasCollision) {
                             // Collision-free: use GPU hashes directly as groupby keys
                             gpuHashOk = true;
                             keyVecs.push_back(std::move(hashes));
                             keyBufsGPU.push_back(nullptr);
                             keyNames.push_back(keyName.empty() ? col : keyName);
                             keyFromF32.push_back(false);
                             outputStringMaps.push_back({});  // Empty — using hashToStringMaps
                             hashToStringMaps.push_back(std::move(hashMap));
                             stringHandled = true;
                             if (debug) std::cerr << "[Exec] GroupBy: GPU FNV1a encoded string key " << col
                                                  << " (" << hashToStringMaps.back().size() << " unique, collision-free)\n";
                         } else {
                             if (debug) std::cerr << "[Exec] GroupBy: GPU FNV1a collision for " << col << ", falling back to CPU\n";
                         }
                     }
                     
                     // --- CPU fallback: sequential ID encoding ---
                     if (!gpuHashOk) {
                         std::vector<uint32_t> ids;
                         ids.reserve(expectedKeyRows);
                         std::vector<std::string> reverseMap;
                         std::map<std::string, uint32_t> forwardMap;
                         uint32_t nextId = 1;
                         
                         auto processStr = [&](const std::string& s) {
                             if (forwardMap.find(s) == forwardMap.end()) {
                                 forwardMap[s] = nextId;
                                 reverseMap.push_back(s);
                                 nextId++;
                             }
                             ids.push_back(forwardMap[s]);
                         };
                         
                         if (!ctx.activeRows.empty() && strData.size() != expectedKeyRows) {
                             for (uint32_t r : ctx.activeRows) {
                                 if (r < strData.size()) processStr(strData[r]);
                                 else ids.push_back(0); 
                             }
                         } else {
                             for (const auto& s : strData) processStr(s);
                         }
                         
                         if (debug) {
                             std::cerr << "[Exec] GroupBy: CPU encoded string key " << col << " to u32 IDs (" << reverseMap.size() << " unique)\n";
                         }
                         keyVecs.push_back(std::move(ids));
                         keyBufsGPU.push_back(nullptr);
                         keyNames.push_back(keyName.empty() ? col : keyName);
                         keyFromF32.push_back(false);
                         outputStringMaps.push_back(std::move(reverseMap));
                         hashToStringMaps.push_back({});
                         stringHandled = true;
                     }
                }
                } // end if stringCols non-empty
            }
            
            if (!stringHandled) {
                if (it != ctx.u32Cols.end()) {
                    if (!ctx.activeRows.empty() && it->second.size() != expectedKeyRows) {
                        // GPU gather path: use gatherU32 when GPU buffers available
                        if (ctx.activeRowsGPU && ctx.u32ColsGPU.count(col) && ctx.u32ColsGPU[col]) {
                            MTL::Buffer* gathered = GpuOps::gatherU32(ctx.u32ColsGPU[col], ctx.activeRowsGPU, (uint32_t)expectedKeyRows);
                            std::vector<uint32_t> filtered(expectedKeyRows);
                            std::memcpy(filtered.data(), gathered->contents(), expectedKeyRows * sizeof(uint32_t));
                            keyVecs.push_back(std::move(filtered));
                            keyBufsGPU.push_back(gathered);  // keep GPU buffer, avoid re-upload
                        } else {
                            // CPU fallback
                            std::vector<uint32_t> filtered;
                            filtered.reserve(expectedKeyRows);
                            for (uint32_t r : ctx.activeRows) {
                                if (r < it->second.size()) filtered.push_back(it->second[r]);
                                else filtered.push_back(0);
                            }
                            keyVecs.push_back(std::move(filtered));
                            keyBufsGPU.push_back(nullptr);
                        }
                    } else {
                        keyVecs.push_back(it->second);
                        // Pass through existing GPU buffer to avoid re-upload,
                        // but only if it matches the expected size (reject stale
                        // pre-filter buffers that weren't compacted).
                        if (ctx.u32ColsGPU.count(col) && ctx.u32ColsGPU[col]) {
                            size_t gpuElems = ctx.u32ColsGPU[col]->length() / sizeof(uint32_t);
                            if (gpuElems == it->second.size()) {
                                ctx.u32ColsGPU[col]->retain();
                                keyBufsGPU.push_back(ctx.u32ColsGPU[col]);
                            } else {
                                if (debug) std::cerr << "[Exec] GroupBy: SKIP stale GPU buf for " << col
                                                     << " (gpu=" << gpuElems << " vs cpu=" << it->second.size() << ")\n";
                                keyBufsGPU.push_back(nullptr);
                            }
                        } else {
                            keyBufsGPU.push_back(nullptr);
                        }
                    }
                    keyNames.push_back(keyName.empty() ? col : keyName);
                    keyFromF32.push_back(false);
                    
                    // Build hash->string map for string output
                    ctx.ensureStringCol(col); // materialize from dict if needed
                    if (ctx.stringCols.count(col)) {
                        const auto& strData = ctx.stringCols.at(col);
                        std::unordered_map<uint32_t, std::string> hashToStr;
                        
                        const auto& u32Data = it->second;
                        if (debug) std::cerr << "[Exec] GroupBy: building hash->string map for " << col 
                                             << " u32Data.size=" << u32Data.size() 
                                             << " strData.size=" << strData.size() << "\n";
                        
                        for (size_t r = 0; r < std::min(u32Data.size(), strData.size()); ++r) {
                            uint32_t hash = u32Data[r];
                            if (hashToStr.find(hash) == hashToStr.end()) {
                                hashToStr[hash] = strData[r];
                            }
                        }
                        
                        if (debug) std::cerr << "[Exec] GroupBy: built hash->string map with " << hashToStr.size() << " entries\n";
                        
                        hashToStringMaps.push_back(std::move(hashToStr));
                        outputStringMaps.push_back({});
                    } else {
                        hashToStringMaps.push_back({});
                        outputStringMaps.push_back({});
                    }
                } else {
                    // Try f32Cols - convert to u32 for grouping (e.g., count values)

                    // GPU fast path: if f32 is on GPU, bitcast directly without downloading
                    bool gpuBitcastDone = false;
                    if (ctx.f32ColsGPU.count(col)) {
                        MTL::Buffer* gpuF32 = ctx.f32ColsGPU.at(col);
                        uint32_t gpuCount = (uint32_t)(gpuF32->length() / sizeof(float));
                        if (gpuCount > 0) {
                            MTL::Buffer* gpuU32 = nullptr;
                            if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0 && gpuCount != (uint32_t)expectedKeyRows) {
                                // Gather active rows first, then bitcast
                                MTL::Buffer* gathered = GpuOps::gatherF32(gpuF32, ctx.activeRowsGPU, ctx.activeRowsCountGPU, true);
                                gpuU32 = GpuOps::bitcastF32ToU32(gathered, ctx.activeRowsCountGPU);
                                gathered->release();
                                gpuCount = ctx.activeRowsCountGPU;
                            } else {
                                gpuU32 = GpuOps::bitcastF32ToU32(gpuF32, gpuCount);
                            }
                            std::vector<uint32_t> converted(gpuCount);
                            std::memcpy(converted.data(), gpuU32->contents(), gpuCount * sizeof(uint32_t));
                            if (debug) std::cerr << "[Exec] GroupBy: GPU bitcast f32 key " << col << " to u32 (" << gpuCount << " rows)\n";
                            keyVecs.push_back(std::move(converted));
                            keyBufsGPU.push_back(gpuU32);  // keep GPU buffer, avoid re-upload
                            keyNames.push_back(keyName.empty() ? col : keyName);
                            keyFromF32.push_back(true);
                            outputStringMaps.push_back({});
                            hashToStringMaps.push_back({});
                            gpuBitcastDone = true;
                        }
                    }

                    if (!gpuBitcastDone) {
                        // CPU fallback: download f32, bitcast element-by-element
                        if ((ctx.f32Cols.find(col) == ctx.f32Cols.end() || ctx.f32Cols[col].empty()) && ctx.f32ColsGPU.count(col)) {
                             MTL::Buffer* buf = ctx.f32ColsGPU.at(col);
                             size_t count = buf->length() / sizeof(float);
                             if (count > 0) {
                                 std::vector<float> down(count);
                                 std::memcpy(down.data(), buf->contents(), count * sizeof(float));
                                 ctx.f32Cols[col] = std::move(down);
                                 if(debug) std::cerr << "[Exec] GroupBy: Lazy fetch F32 key " << col << " from GPU\n";
                             }
                        }

                        auto itF = ctx.f32Cols.find(col);
                        if (itF == ctx.f32Cols.end()) {
                            for (int suffix = 1; suffix <= 9 && itF == ctx.f32Cols.end(); ++suffix) {
                                std::string suffixedCol = col + "_" + std::to_string(suffix);
                                itF = ctx.f32Cols.find(suffixedCol);
                            }
                        }
                        if (itF != ctx.f32Cols.end()) {
                            std::vector<uint32_t> converted;
                            converted.reserve(expectedKeyRows);
                            if (!ctx.activeRows.empty() && itF->second.size() != expectedKeyRows) {
                                for(uint32_t r : ctx.activeRows) {
                                    if (r < itF->second.size()) {
                                        uint32_t bits; std::memcpy(&bits, &itF->second[r], sizeof(bits));
                                        converted.push_back(bits);
                                    }
                                    else converted.push_back(0);
                                }
                            } else {
                                for (float f : itF->second) {
                                    uint32_t bits; std::memcpy(&bits, &f, sizeof(bits));
                                    converted.push_back(bits);
                                }
                            }
                            if (debug) std::cerr << "[Exec] GroupBy: converted f32 key " << col << " to u32\n";
                            keyVecs.push_back(std::move(converted));
                            keyBufsGPU.push_back(nullptr);
                            keyNames.push_back(keyName.empty() ? col : keyName);
                            keyFromF32.push_back(true);
                            outputStringMaps.push_back({});
                            hashToStringMaps.push_back({});
                        }
                    }
                }
            }
        }
    }
    
    if (keyVecs.empty()) return false;
    
    // Check if any key vector is empty (0 rows)
    bool hasEmptyKeys = false;
    for (const auto& kv : keyVecs) {
        if (kv.empty()) {
            hasEmptyKeys = true;
            break;
        }
    }
    
    // Empty keys: return 0 groups
    if (hasEmptyKeys) {
        if (debug) {
            std::cerr << "[Exec] GroupBy: empty key vectors, returning 0 groups\n";
        }
        out.rowCount = 0;
        out.u32Cols.clear();
        out.u32Cols.resize(keyVecs.size());
        out.u32Names = keyNames;
        out.f32Cols.clear();
        out.f32Names.clear();
        out.order.clear();
        // Still need to populate aggNames
        for (const auto& spec : groupBy.aggSpecs) {
            std::string name = spec.outputName;
            if (name.empty()) {
                name = aggFuncName(spec.func);
            }
            out.f32Names.push_back(name);
            out.f32Cols.push_back({});
        }
        for (size_t i = 0; i < out.u32Names.size(); ++i) {
            out.order.push_back({TableResult::ColRef::Kind::U32, i, out.u32Names[i]});
        }
        for (size_t i = 0; i < out.f32Names.size(); ++i) {
            out.order.push_back({TableResult::ColRef::Kind::F32, i, out.f32Names[i]});
        }
        return true;
    }
    
    // Build aggregate input vectors
    std::vector<std::vector<float>> aggInputs;
    std::vector<MTL::Buffer*> aggBufsGPU;  // Pre-existing GPU buffers for agg inputs (avoid re-upload)
    std::vector<AggFunc> aggFuncs;
    std::vector<std::string> aggNames;
    
    for (const auto& spec : groupBy.aggSpecs) {
        aggFuncs.push_back(spec.func);
        // Use outputName if provided, otherwise generate one from function and input
        std::string name = spec.outputName;
        if (name.empty()) {
            name = aggFuncName(spec.func);
            if (!spec.inputExpr.empty()) {
                name += "_" + spec.inputExpr;
            }
        }
        aggNames.push_back(name);
        
        if (debug) {
            std::cerr << "[Exec] GroupBy: agg func=" << static_cast<int>(spec.func) 
                      << " outputName=" << name << " inputExpr=" << spec.inputExpr << "\n";
        }
        
        if (spec.func == AggFunc::CountStar) {
            // COUNT(*) doesn't need input - counts all rows
            aggInputs.push_back({});
            aggBufsGPU.push_back(nullptr);
        } else if (spec.func == AggFunc::Count || spec.func == AggFunc::CountDistinct) {
            // COUNT(column) / COUNT(DISTINCT column) - need to track which rows have non-NULL values
            // We'll store the column values so we can check for NULLs (0 = NULL sentinel)
            std::string col = spec.inputExpr;
            while (!col.empty() && col.front() == '(' && col.back() == ')') {
                col = col.substr(1, col.size() - 2);
            }
            col = trim_copy(col);
            
            // Get the u32 column values (most columns are u32) - prefer u32Cols for correct row count
            std::vector<float> input;
            MTL::Buffer* inputGPU = nullptr;  // Track GPU buffer to avoid re-upload
            
             // LAZY FETCH U32: If vector is empty but on GPU, bring it back
            if ((ctx.u32Cols.find(col) == ctx.u32Cols.end() || ctx.u32Cols[col].empty()) && ctx.u32ColsGPU.count(col)) {
                 MTL::Buffer* buf = ctx.u32ColsGPU.at(col);
                 size_t count = buf->length() / sizeof(uint32_t);
                 if (count > 0) {
                     std::vector<uint32_t> down(count);
                     std::memcpy(down.data(), buf->contents(), count * sizeof(uint32_t));
                     ctx.u32Cols[col] = std::move(down);
                 }
            }

            auto itU = ctx.u32Cols.find(col);
            if (itU != ctx.u32Cols.end()) {
                // GPU u32→f32 cast (with optional activeRows gather)
                auto& s = GpuColumnStore::instance();
                MTL::Buffer* u32Buf = ctx.u32ColsGPU.count(col) ? ctx.u32ColsGPU[col]
                    : s.device()->newBuffer(itU->second.data(), itU->second.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                bool ownU32 = !ctx.u32ColsGPU.count(col);
                MTL::Buffer* src = u32Buf;
                bool ownSrc = false;
                if (!ctx.activeRows.empty() && itU->second.size() > expectedKeyRows && ctx.activeRowsGPU) {
                    src = GpuOps::gatherU32(u32Buf, ctx.activeRowsGPU, (uint32_t)expectedKeyRows);
                    ownSrc = true;
                }
                uint32_t castCount = ownSrc ? (uint32_t)expectedKeyRows : (uint32_t)itU->second.size();
                MTL::Buffer* f32Buf = GpuOps::castU32ToF32(src, castCount);
                input.resize(castCount);
                std::memcpy(input.data(), f32Buf->contents(), castCount * sizeof(float));
                inputGPU = f32Buf;  // keep GPU buffer
                if (ownSrc) src->release();
                if (ownU32) u32Buf->release();
                if (debug) {
                    std::cerr << "[Exec] GroupBy: CountDistinct input from u32 col " << col << " size=" << input.size() << " (GPU cast)\n";
                }
            } else {
                 // LAZY FETCH F32: If vector is empty but on GPU, bring it back
                if ((ctx.f32Cols.find(col) == ctx.f32Cols.end() || ctx.f32Cols[col].empty()) && ctx.f32ColsGPU.count(col)) {
                     MTL::Buffer* buf = ctx.f32ColsGPU.at(col);
                     size_t count = buf->length() / sizeof(float);
                     if (count > 0) {
                         std::vector<float> down(count);
                         std::memcpy(down.data(), buf->contents(), count * sizeof(float));
                         ctx.f32Cols[col] = std::move(down);
                     }
                }

                auto itF = ctx.f32Cols.find(col);
                if (itF != ctx.f32Cols.end()) {
                    if (!ctx.activeRows.empty() && itF->second.size() > expectedKeyRows && ctx.activeRowsGPU) {
                         auto& s = GpuColumnStore::instance();
                         MTL::Buffer* src = ctx.f32ColsGPU.count(col) ? ctx.f32ColsGPU[col]
                             : s.device()->newBuffer(itF->second.data(), itF->second.size() * sizeof(float), MTL::ResourceStorageModeShared);
                         bool ownSrc = !ctx.f32ColsGPU.count(col);
                         MTL::Buffer* dst = GpuOps::gatherF32(src, ctx.activeRowsGPU, (uint32_t)expectedKeyRows);
                         input.resize(expectedKeyRows);
                         std::memcpy(input.data(), dst->contents(), expectedKeyRows * sizeof(float));
                         dst->release();
                         if (ownSrc) src->release();
                    } else {
                        input = itF->second;
                    }
                    if (debug) {
                        std::cerr << "[Exec] GroupBy: CountDistinct input from f32 col " << col << " size=" << input.size() << "\n";
                    }
                } else if (debug) {
                    std::cerr << "[Exec] GroupBy: CountDistinct col " << col << " not found!\n";
                }
            }
            aggInputs.push_back(std::move(input));
            aggBufsGPU.push_back(inputGPU);
        } else {
            // Evaluate input expression
            std::vector<float> input;
            MTL::Buffer* inputGPU = nullptr;  // Track GPU buffer to avoid re-upload
            bool inputGPUOwned = false;

            // Use pre-computed column if available (avoids double-gather)
            bool foundPrecomputed = false;
            if (!spec.inputExpr.empty()) {
                // Try f32Cols (CPU) for exact match with correct size
                auto itPreF = ctx.f32Cols.find(spec.inputExpr);
                if (itPreF != ctx.f32Cols.end() && itPreF->second.size() == expectedKeyRows) {
                    input = itPreF->second;
                    foundPrecomputed = true;
                    if (debug) std::cerr << "[Exec] GroupBy: using pre-computed f32Col '" << spec.inputExpr << "' (" << input.size() << " values)\n";
                }
                // Try f32ColsGPU for exact match
                if (!foundPrecomputed && ctx.f32ColsGPU.count(spec.inputExpr)) {
                    MTL::Buffer* preBuf = ctx.f32ColsGPU[spec.inputExpr];
                    size_t bufElems = preBuf->length() / sizeof(float);
                    if (bufElems == expectedKeyRows) {
                        input.resize(expectedKeyRows);
                        std::memcpy(input.data(), preBuf->contents(), expectedKeyRows * sizeof(float));
                        inputGPU = preBuf;  // ctx-owned, don't release
                        inputGPUOwned = false;
                        foundPrecomputed = true;
                        if (debug) std::cerr << "[Exec] GroupBy: using pre-computed f32ColGPU '" << spec.inputExpr << "' (" << input.size() << " values)\n";
                    }
                }
            }

            if (!foundPrecomputed) {
                MTL::Buffer* buf = evaluateExpression(spec.input, ctx);
                if (buf) {
                     uint32_t count = (ctx.activeRowsGPU) ? ctx.activeRowsCountGPU : ctx.rowCount;
                     if (ctx.activeRowsGPU && ctx.activeRowsCountGPU == 0) count = 0;
                     input.resize(count);
                     if (count > 0) std::memcpy(input.data(), buf->contents(), count * sizeof(float));
                     inputGPU = buf;  // keep GPU buffer
                     inputGPUOwned = true;
                }
            }

            if (debug) {
                std::cerr << "[Exec] GroupBy: evaluateExpression returned " << input.size() << " values\n";
                if (!input.empty()) {
                    float sum = 0, minV = input[0], maxV = input[0];
                    size_t zeroCount = 0;
                    for (float v : input) { 
                        sum += v; 
                        minV = std::min(minV, v); 
                        maxV = std::max(maxV, v); 
                        if (v == 0) zeroCount++;
                    }
                    std::cerr << "[Exec] GroupBy: evalExprFloat stats: sum=" << sum << " min=" << minV << " max=" << maxV << " avg=" << (sum/input.size()) << " zeros=" << zeroCount << "\n";
                    // Print first 10 values
                    std::cerr << "[Exec] GroupBy: first 10 values: ";
                    for (size_t i = 0; i < std::min(input.size(), size_t(10)); ++i) {
                        std::cerr << input[i] << " ";
                    }
                    std::cerr << "\n";
                }
            }
            if (input.empty()) {
                // Try from inputExpr string (might be positional ref like #3)
                std::string col = spec.inputExpr;
                // Strip surrounding parens if any
                while (!col.empty() && col.front() == '(' && col.back() == ')') {
                    col = col.substr(1, col.size() - 2);
                }
                col = trim_copy(col);
                
                if (debug) std::cerr << "[Exec] GroupBy: trying col=" << col << "\n";
                
                // If col is still empty and we have f32 columns, try common names first
                if (col.empty()) {
                    // For SUM/AVG, look for columns with matching base names like c_acctbal
                    // Prefer non-suffixed version
                    for (const auto& [name, vals] : ctx.f32Cols) {
                        // Skip positional and aggregate columns
                        if (name[0] == '#' || name == "SUM" || name == "AVG" || 
                            name == "COUNT(*)" || name == "MIN" || name == "MAX") continue;
                        // Prefer non-suffixed version (c_acctbal over c_acctbal_1)
                        bool hasSuffix = name.size() > 2 && name[name.size()-2] == '_' && 
                                        std::isdigit(name[name.size()-1]);
                        if (!hasSuffix) {
                            col = name;
                            break;
                        }
                    }
                    // Fallback to any f32 column
                    if (col.empty() && !ctx.f32Cols.empty()) {
                        col = ctx.f32Cols.begin()->first;
                    }
                    if (debug && !col.empty()) {
                        std::cerr << "[Exec] GroupBy: inferred col=" << col << " for empty inputExpr\n";
                    }
                }
                
                auto itF = ctx.f32Cols.find(col);
                if (itF != ctx.f32Cols.end()) {
                    if (!ctx.activeRows.empty() && itF->second.size() > expectedKeyRows && ctx.activeRowsGPU) {
                         auto& s = GpuColumnStore::instance();
                         MTL::Buffer* src = ctx.f32ColsGPU.count(col) ? ctx.f32ColsGPU[col]
                             : s.device()->newBuffer(itF->second.data(), itF->second.size() * sizeof(float), MTL::ResourceStorageModeShared);
                         bool ownSrc = !ctx.f32ColsGPU.count(col);
                         MTL::Buffer* dst = GpuOps::gatherF32(src, ctx.activeRowsGPU, (uint32_t)expectedKeyRows);
                         input.resize(expectedKeyRows);
                         std::memcpy(input.data(), dst->contents(), expectedKeyRows * sizeof(float));
                         inputGPU = dst;  // keep GPU buffer
                         inputGPUOwned = true;
                         if (ownSrc) src->release();
                    } else {
                        input = itF->second;
                        // Check if GPU buffer available in ctx
                        if (ctx.f32ColsGPU.count(col) && ctx.f32ColsGPU[col] &&
                            ctx.f32ColsGPU[col]->length() >= itF->second.size() * sizeof(float)) {
                            inputGPU = ctx.f32ColsGPU[col];
                            inputGPUOwned = false;
                        }
                    }
                    if (debug) {
                        std::cerr << "[Exec] GroupBy: found f32 col, size=" << input.size() << "\n";
                        if (!input.empty()) {
                            float sum = 0, minV = input[0], maxV = input[0];
                            for (float v : input) { sum += v; minV = std::min(minV, v); maxV = std::max(maxV, v); }
                            std::cerr << "[Exec] GroupBy: col " << col << " stats: sum=" << sum << " min=" << minV << " max=" << maxV << " avg=" << (sum/input.size()) << "\n";
                        }
                    }
                } else {
                    auto itU = ctx.u32Cols.find(col);
                    if (itU != ctx.u32Cols.end()) {
                        // GPU u32→f32 cast (with optional activeRows gather)
                        auto& s = GpuColumnStore::instance();
                        MTL::Buffer* u32Buf = ctx.u32ColsGPU.count(col) ? ctx.u32ColsGPU[col]
                            : s.device()->newBuffer(itU->second.data(), itU->second.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                        bool ownU32 = !ctx.u32ColsGPU.count(col);
                        MTL::Buffer* src = u32Buf;
                        bool ownSrc = false;
                        if (!ctx.activeRows.empty() && itU->second.size() > expectedKeyRows && ctx.activeRowsGPU) {
                            src = GpuOps::gatherU32(u32Buf, ctx.activeRowsGPU, (uint32_t)expectedKeyRows);
                            ownSrc = true;
                        }
                        uint32_t castCount = ownSrc ? (uint32_t)expectedKeyRows : (uint32_t)itU->second.size();
                        MTL::Buffer* f32Buf = GpuOps::castU32ToF32(src, castCount);
                        input.resize(castCount);
                        std::memcpy(input.data(), f32Buf->contents(), castCount * sizeof(float));
                        inputGPU = f32Buf;  // keep GPU buffer
                        inputGPUOwned = true;
                        if (ownSrc) src->release();
                        if (ownU32) u32Buf->release();
                    }
                }
            }
            aggInputs.push_back(std::move(input));
            // Retain ctx-owned GPU buffers so they can be safely released later
            if (inputGPU && !inputGPUOwned) inputGPU->retain();
            aggBufsGPU.push_back(inputGPU);
        }
    }
    
    // --- CountDistinct Handling (2-Stage GPU) ---
    // Check if we have any CountDistinct aggregates
    int countDistinctIdx = -1;
    for (size_t i = 0; i < aggFuncs.size(); ++i) {
        if (aggFuncs[i] == AggFunc::CountDistinct) {
            countDistinctIdx = static_cast<int>(i);
            break;
        }
    }

    if (countDistinctIdx >= 0) {
        if (debug) std::cerr << "[Exec] GroupBy: Detected CountDistinct, attempting 2-stage GPU execution.\n";
        
        const auto& distinctSpec = groupBy.aggSpecs[countDistinctIdx];
        std::string distinctInputStr = distinctSpec.inputExpr;
        
        // Verify multiple CountDistincts
        for (size_t i = 0; i < aggFuncs.size(); ++i) {
            if (aggFuncs[i] == AggFunc::CountDistinct) {
                if (groupBy.aggSpecs[i].inputExpr != distinctInputStr) {
                    throw std::runtime_error("Multiple different CountDistinct columns not supported on GPU yet.");
                }
            }
        }
        
        // Stage 1: Group By {Keys + DistinctCol}
        IRGroupBy stage1Spec;
        stage1Spec.keys = groupBy.keys;
        stage1Spec.keyNames = groupBy.keyNames;
        
        // Add DistinctCol to keys
        if (distinctSpec.input) {
            stage1Spec.keys.push_back(distinctSpec.input);
            // Use inputExpr string as name? or a temp name
            std::string dName = "distinct_col_stage1";
            if (distinctSpec.input->kind == TypedExpr::Kind::Column) {
                 dName = distinctSpec.input->asColumn().column;
            }
            stage1Spec.keyNames.push_back(dName);
        } else {
             throw std::runtime_error("CountDistinct missing input expression node");
        }

        // Stage 1 Aggregates: Add dummy COUNT(*) because GPU kernel requires at least 1 agg
        IRGroupBy::AggSpec dummyAgg;
        dummyAgg.func = AggFunc::CountStar; 
        dummyAgg.outputName = "dummy_cnt";
        stage1Spec.aggSpecs.push_back(dummyAgg);
        
        TableResult stage1Res;
        bool s1Ok = executeGroupBy(stage1Spec, ctx, stage1Res);
        if (!s1Ok) throw std::runtime_error("Stage 1 GroupBy failed (CountDistinct pre-pass)");
        
        if (debug) std::cerr << "[Exec] GroupBy: Stage 1 complete. Rows=" << stage1Res.rowCount << "\n";

        // Stage 2: Group By {Keys} on stage1Res, with COUNT(*)
        EvalContext stage2Ctx;
        stage2Ctx.rowCount = stage1Res.rowCount;
        
        // Populate context from Stage 1 result.
        // Skip u32 columns for string keys so the next GroupBy re-encodes them,
        // preserving string maps in the final result.
        
        std::set<std::string> strColNames;
        for(size_t i=0; i<stage1Res.stringNames.size(); ++i) {
            stage2Ctx.stringCols[stage1Res.stringNames[i]] = stage1Res.stringCols[i];
            strColNames.insert(stage1Res.stringNames[i]);
        }
        
        for(size_t i=0; i<stage1Res.u32Names.size(); ++i) {
             // Only copy as u32 if it's NOT a string column
             if (strColNames.find(stage1Res.u32Names[i]) == strColNames.end()) {
                stage2Ctx.u32Cols[stage1Res.u32Names[i]] = stage1Res.u32Cols[i];
             }
        }

        for(size_t i=0; i<stage1Res.f32Names.size(); ++i) {
            stage2Ctx.f32Cols[stage1Res.f32Names[i]] = stage1Res.f32Cols[i];
        }
        
        IRGroupBy stage2Spec;
        // Reconstruct keys for Stage 2 (Columns referencing stage1 outputs)
        for(const auto& kn : groupBy.keyNames) {
             auto col = std::make_shared<TypedExpr>();
             col->kind = TypedExpr::Kind::Column;
             col->asColumn().column = kn; 
             stage2Spec.keys.push_back(col);
             stage2Spec.keyNames.push_back(kn);
        }
        
        // Reconstruct aggregates
        for(size_t i=0; i<groupBy.aggSpecs.size(); ++i) {
            const auto& spec = groupBy.aggSpecs[i];
            IRGroupBy::AggSpec s2Agg;
            s2Agg.outputName = spec.outputName;
            
            if (spec.func == AggFunc::CountDistinct) {
                s2Agg.func = AggFunc::CountStar; 
            } else {
                 s2Agg.func = spec.func; 
            }
            stage2Spec.aggSpecs.push_back(s2Agg);
        }
        
        return executeGroupBy(stage2Spec, stage2Ctx, out);
    }
    
    // Try GPU GroupBy if all aggregates are GPU-compatible (no CountDistinct)
    bool useGpu = true;
    
    // Also require at most 8 keys (GPU kernel limit)
    if (keyVecs.size() > 8) {
        useGpu = false;
    }
    
    // Check for size consistency - all keyVecs and aggInputs should have same size
    // Prefer GPU activeRows count over CPU
    size_t expectedRowCount = ctx.rowCount;
    if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0) {
        expectedRowCount = ctx.activeRowsCountGPU;
    } else if (!ctx.activeRows.empty()) {
        expectedRowCount = ctx.activeRows.size();
    }

    // Determine consensus row count from keys if possible
    size_t consensusRowCount = expectedRowCount;
    bool keysConsistent = true;
    if (!keyVecs.empty()) {
        size_t firstSize = keyVecs[0].size();
        for (const auto& kv : keyVecs) {
            if (kv.size() != firstSize) {
                keysConsistent = false;
                break;
            }
        }
        if (keysConsistent && firstSize != expectedRowCount) {
            if (debug) {
                std::cerr << "[Exec] GroupBy: Warning: ctx.rowCount (" << expectedRowCount 
                          << ") differs from consistent key size (" << firstSize << "). Using key size.\n";
            }
            consensusRowCount = firstSize;
        }
    }
    
    // Verify key sizes match consensus
    for (size_t i = 0; i < keyVecs.size(); ++i) {
        const auto& kv = keyVecs[i];
        if (kv.size() != consensusRowCount) {
            if (debug) {
                std::cerr << "[Exec] GroupBy: key size mismatch for key index " << i << " (name: " 
                          << (i < keyNames.size() ? keyNames[i] : "?") << "), expected " << consensusRowCount 
                          << " but got " << kv.size() << "\n";
                std::cerr << "[Exec]   ctx.rowCount=" << ctx.rowCount << "\n";
            }
            throw std::runtime_error("GroupBy Key size mismatch. CPU fallback disabled.");
        }
    }
    
    // GPU GroupBy path
    if (useGpu && !keyVecs.empty()) {
        auto& store = GpuColumnStore::instance();
        if (store.device() && store.library() && store.queue()) {
            

            if (debug) {
                std::cerr << "[Exec] GroupBy: Using GPU path with " << keyVecs.size() << " keys and " << aggFuncs.size() << " aggregates\n";
            }
            
            // Determine row count from key vectors - prefer GPU activeRows
            size_t gpuRowCount = ctx.rowCount;
            if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0) {
                gpuRowCount = ctx.activeRowsCountGPU;
            } else if (!ctx.activeRows.empty()) {
                gpuRowCount = ctx.activeRows.size();
            }
            if (!keyVecs.empty() && !keyVecs[0].empty()) {
                gpuRowCount = keyVecs[0].size();
            }
            
            // Create GPU buffers for keys (with +1 bias to avoid 0 as empty marker)
            std::vector<MTL::Buffer*> keyBufs;
            std::vector<MTL::Buffer*> toRelease;
            bool gpuOk = true;
            
            for (size_t k = 0; k < keyVecs.size() && gpuOk; ++k) {
                MTL::Buffer* srcBuf = nullptr;
                bool usedGpuBuf = false;
                // Use pre-existing GPU buffer if available and exactly sized
                if (k < keyBufsGPU.size() && keyBufsGPU[k] &&
                    keyBufsGPU[k]->length() == gpuRowCount * sizeof(uint32_t)) {
                    srcBuf = keyBufsGPU[k];
                    keyBufsGPU[k] = nullptr;  // take ownership
                    usedGpuBuf = true;
                } else {
                    // Upload key vector to GPU, pad with zeros if needed
                    srcBuf = store.device()->newBuffer(gpuRowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    if (!srcBuf) { gpuOk = false; break; }
                    uint32_t* ptr = static_cast<uint32_t*>(srcBuf->contents());
                    size_t copyLen = std::min(keyVecs[k].size(), gpuRowCount);
                    if (copyLen > 0) memcpy(ptr, keyVecs[k].data(), copyLen * sizeof(uint32_t));
                    if (copyLen < gpuRowCount) memset(ptr + copyLen, 0, (gpuRowCount - copyLen) * sizeof(uint32_t));
                }

                // ── Diagnostic: verify srcBuf matches keyVecs[k] BEFORE biasing ──
                if (debug && k == 0) {
                    const uint32_t* sp = static_cast<const uint32_t*>(srcBuf->contents());
                    size_t srcMismatches = 0;
                    for (size_t i = 0; i < std::min(gpuRowCount, keyVecs[k].size()); ++i) {
                        if (sp[i] != keyVecs[k][i]) {
                            if (srcMismatches < 5) {
                                std::cerr << "[Exec] GroupBy: SRC MISMATCH[" << i << "] GPU=" << sp[i] << " CPU=" << keyVecs[k][i] << "\n";
                            }
                            srcMismatches++;
                        }
                    }
                    std::cerr << "[Exec] GroupBy: srcBuf vs keyVecs[0]: " << srcMismatches << " mismatches"
                              << " (usedGpuBuf=" << usedGpuBuf
                              << " srcLen=" << srcBuf->length()/4 
                              << " keyVecLen=" << keyVecs[k].size()
                              << " gpuRowCount=" << gpuRowCount << ")\n";
                }

                // GPU: add +1 bias to avoid 0 as empty marker in hash table
                auto biased = GpuOps::arithAddConstU32(srcBuf, 1, static_cast<uint32_t>(gpuRowCount));
                srcBuf->release();
                if (!biased) { gpuOk = false; break; }

                if (debug) {
                    uint32_t* bp = static_cast<uint32_t*>(biased->contents());
                    std::map<uint32_t, size_t> dist;
                    for (size_t i = 0; i < gpuRowCount; ++i) dist[bp[i]]++;
                    std::cerr << "[Exec] GroupBy: GPU key buf[" << k << "] distribution (biased):";
                    for (auto& [v,c] : dist) std::cerr << " " << v << ":" << c;
                    std::cerr << "\n";
                }
                keyBufs.push_back(biased);
                toRelease.push_back(biased);
            }
            
            // Create GPU buffers for aggregate inputs
            std::vector<MTL::Buffer*> aggBufs;
            std::vector<uint32_t> aggTypesGpu;
            
            for (size_t a = 0; a < aggFuncs.size() && gpuOk; ++a) {
                // Map AggFunc to GPU type: 0=SUM, 1=COUNT, 2=MIN, 3=MAX
                uint32_t gpuType = 0;
                switch (aggFuncs[a]) {
                    case AggFunc::Sum: gpuType = 0; break;
                    case AggFunc::Avg: gpuType = 0; break;  // AVG = SUM / COUNT, computed later
                    case AggFunc::Count:
                        // COUNT(column) should only count non-NULL values.
                        // Use SUM with a non-null indicator (1.0 for non-null, 0.0 for null).
                        // NULL sentinel is 0 for u32 columns from outer joins.
                        gpuType = 0;  // SUM
                        break;
                    case AggFunc::CountStar: gpuType = 1; break;
                    case AggFunc::Min: gpuType = 2; break;
                    case AggFunc::Max: gpuType = 3; break;
                    default: gpuType = 0; break;
                }
                aggTypesGpu.push_back(gpuType);
                
                MTL::Buffer* aggBuf = nullptr;
                if (aggFuncs[a] == AggFunc::Count) {
                    // COUNT(column): GPU non-null indicator (1.0 if non-null, 0.0 if null/zero)
                    MTL::Buffer* srcBuf = nullptr;
                    if (a < aggBufsGPU.size() && aggBufsGPU[a] &&
                        aggBufsGPU[a]->length() >= gpuRowCount * sizeof(float)) {
                        srcBuf = aggBufsGPU[a];
                        aggBufsGPU[a] = nullptr;
                    } else {
                        srcBuf = store.device()->newBuffer(gpuRowCount * sizeof(float), MTL::ResourceStorageModeShared);
                        if (srcBuf) {
                            float* ptr = static_cast<float*>(srcBuf->contents());
                            size_t copyLen = std::min(aggInputs[a].size(), gpuRowCount);
                            if (copyLen > 0) memcpy(ptr, aggInputs[a].data(), copyLen * sizeof(float));
                            if (copyLen < gpuRowCount) memset(ptr + copyLen, 0, (gpuRowCount - copyLen) * sizeof(float));
                        }
                    }
                    if (srcBuf) {
                        aggBuf = GpuOps::nonNullIndicatorF32(srcBuf, static_cast<uint32_t>(gpuRowCount));
                        srcBuf->release();
                        if (aggBuf) toRelease.push_back(aggBuf);
                    }
                } else if (gpuType == 1) {
                    // COUNT doesn't need input data, but kernel needs a buffer
                    aggBuf = store.device()->newBuffer(gpuRowCount * sizeof(float), MTL::ResourceStorageModeShared);
                    if (aggBuf) {
                        std::memset(aggBuf->contents(), 0, gpuRowCount * sizeof(float));
                        toRelease.push_back(aggBuf);
                    }
                } else if (!aggInputs[a].empty()) {
                    // Use pre-existing GPU buffer if available
                    if (a < aggBufsGPU.size() && aggBufsGPU[a] &&
                        aggBufsGPU[a]->length() >= gpuRowCount * sizeof(float)) {
                        aggBuf = aggBufsGPU[a];
                        aggBufsGPU[a] = nullptr;
                        toRelease.push_back(aggBuf);
                    } else {
                        aggBuf = store.device()->newBuffer(gpuRowCount * sizeof(float), MTL::ResourceStorageModeShared);
                        if (aggBuf) {
                            float* ptr = static_cast<float*>(aggBuf->contents());
                            for (size_t i = 0; i < gpuRowCount; ++i) {
                                ptr[i] = (i < aggInputs[a].size()) ? aggInputs[a][i] : 0.0f;
                            }
                            toRelease.push_back(aggBuf);
                        }
                    }
                } else {
                    // No input data - create dummy buffer
                    aggBuf = store.device()->newBuffer(gpuRowCount * sizeof(float), MTL::ResourceStorageModeShared);
                    if (aggBuf) {
                        std::memset(aggBuf->contents(), 0, gpuRowCount * sizeof(float));
                        toRelease.push_back(aggBuf);
                    }
                }
                if (!aggBuf) { gpuOk = false; break; }
                aggBufs.push_back(aggBuf);
            }
            
            // For AVG, we also need a COUNT aggregate
            // Add an extra COUNT aggregate for each AVG
            std::vector<size_t> avgIndices;
            for (size_t a = 0; a < aggFuncs.size(); ++a) {
                if (aggFuncs[a] == AggFunc::Avg) {
                    avgIndices.push_back(a);
                    // Add a COUNT aggregate
                    aggTypesGpu.push_back(1);  // COUNT type
                    MTL::Buffer* countBuf = store.device()->newBuffer(gpuRowCount * sizeof(float), MTL::ResourceStorageModeShared);
                    if (countBuf) {
                        std::memset(countBuf->contents(), 0, gpuRowCount * sizeof(float));
                        toRelease.push_back(countBuf);
                        aggBufs.push_back(countBuf);
                    }
                }
            }
            
            if (gpuOk) {
                auto htOpt = GpuOps::groupByAggMultiKeyTyped(keyBufs, aggBufs, aggTypesGpu, static_cast<uint32_t>(gpuRowCount));
                
                if (htOpt.has_value()) {
                    const uint32_t cap = htOpt->capacity;
                    const auto* keyWords = reinterpret_cast<const uint32_t*>(htOpt->htKeys->contents());
                    const auto* aggWords = reinterpret_cast<const uint32_t*>(htOpt->htAggs->contents());
                    
                    if (debug) {
                        std::cerr << "[Exec] GroupBy: GPU hash table capacity=" << cap << " gpuRowCount=" << gpuRowCount << "\n";
                        // Count non-empty slots
                        size_t nonEmpty = 0;
                        for (uint32_t s = 0; s < cap; ++s) {
                            if (keyWords[s * 8 + 0] != 0) nonEmpty++;
                        }
                        std::cerr << "[Exec] GroupBy: GPU hash table non-empty slots=" << nonEmpty << "\n";
                    }
                    
                    // ── CPU verification: recompute GROUP BY from input data, compare with GPU ──
                    {
                        // Check GPU key buffer vs CPU keyVecs row-by-row
                        if (!keyBufs.empty() && !keyVecs.empty()) {
                            const uint32_t* gpuKeys = static_cast<const uint32_t*>(keyBufs[0]->contents());
                            size_t mismatches = 0;
                            for (size_t i = 0; i < std::min(gpuRowCount, keyVecs[0].size()); ++i) {
                                uint32_t gpuKey = gpuKeys[i];          // biased
                                uint32_t cpuKey = keyVecs[0][i] + 1;   // bias CPU for comparison
                                if (gpuKey != cpuKey) {
                                    if (mismatches < 10) {
                                        std::cerr << "[Exec] GroupBy VERIFY: KEY MISMATCH at row " << i 
                                                  << " GPU=" << gpuKey << " CPU=" << cpuKey << "\n";
                                    }
                                    mismatches++;
                                }
                            }
                            std::cerr << "[Exec] GroupBy VERIFY: key mismatches = " << mismatches << " / " << std::min(gpuRowCount, keyVecs[0].size()) << "\n";
                        }
                        
                        // Check GPU agg buffer vs CPU aggInputs row-by-row
                        if (!aggBufs.empty() && !aggInputs.empty() && !aggInputs[0].empty()) {
                            const float* gpuAggs = static_cast<const float*>(aggBufs[0]->contents());
                            size_t mismatches = 0;
                            for (size_t i = 0; i < std::min(gpuRowCount, aggInputs[0].size()); ++i) {
                                if (gpuAggs[i] != aggInputs[0][i]) {
                                    if (mismatches < 10) {
                                        std::cerr << "[Exec] GroupBy VERIFY: AGG MISMATCH at row " << i 
                                                  << " GPU=" << gpuAggs[i] << " CPU=" << aggInputs[0][i] << "\n";
                                    }
                                    mismatches++;
                                }
                            }
                            std::cerr << "[Exec] GroupBy VERIFY: agg mismatches = " << mismatches << " / " << std::min(gpuRowCount, aggInputs[0].size()) << "\n";
                        }
                        // Compute CPU GROUP BY using double precision for accuracy
                        std::unordered_map<uint64_t, double> cpuGroupSums;
                        size_t numK = keyVecs.size();
                        for (size_t i = 0; i < gpuRowCount; ++i) {
                            uint64_t compositeKey = 0;
                            for (size_t k = 0; k < numK; ++k) {
                                compositeKey ^= (uint64_t(keyVecs[k][i]) + 1) * (2654435761ULL + k * 31); // simple hash combine
                            }
                            double val = (aggFuncs.size() > 0 && !aggInputs[0].empty() && i < aggInputs[0].size())
                                ? static_cast<double>(aggInputs[0][i]) : 0.0;
                            cpuGroupSums[compositeKey] += val;
                        }
                        
                        // Also compute CPU GROUP BY keyed on actual first key value (for finding max supplier)
                        std::unordered_map<uint32_t, double> cpuByKey0;
                        for (size_t i = 0; i < gpuRowCount; ++i) {
                            uint32_t k0 = keyVecs[0][i];
                            double val = (aggFuncs.size() > 0 && !aggInputs[0].empty() && i < aggInputs[0].size())
                                ? static_cast<double>(aggInputs[0][i]) : 0.0;
                            cpuByKey0[k0] += val;
                        }
                        
                        // Find top CPU group
                        uint32_t cpuMaxKey = 0; double cpuMaxVal = -1e30;
                        for (auto& [k, v] : cpuByKey0) {
                            if (v > cpuMaxVal) { cpuMaxVal = v; cpuMaxKey = k; }
                        }
                        
                        // Compute GPU grand total and find top GPU group
                        double gpuGrandTotal = 0.0;
                        uint32_t gpuMaxKey = 0; float gpuMaxVal = -1e30f;
                        for (uint32_t s = 0; s < cap; ++s) {
                            uint32_t k0 = keyWords[s * 8 + 0];
                            if (k0 == 0) continue;
                            uint32_t raw = aggWords[s * 16 + 0];
                            float fval = *reinterpret_cast<const float*>(&raw);
                            gpuGrandTotal += static_cast<double>(fval);
                            if (fval > gpuMaxVal) {
                                gpuMaxVal = fval;
                                gpuMaxKey = k0 - 1; // de-bias
                            }
                        }
                        
                        // Compute CPU grand total
                        double cpuGrandTotal = 0.0;
                        if (!aggInputs.empty() && !aggInputs[0].empty()) {
                            for (size_t i = 0; i < std::min(aggInputs[0].size(), gpuRowCount); ++i) {
                                cpuGrandTotal += static_cast<double>(aggInputs[0][i]);
                            }
                        }
                        
                        std::cerr << "[Exec] GroupBy VERIFY: CPU grand total = " << std::fixed << std::setprecision(2) << cpuGrandTotal << "\n";
                        std::cerr << "[Exec] GroupBy VERIFY: GPU grand total = " << std::fixed << std::setprecision(2) << gpuGrandTotal << "\n";
                        std::cerr << "[Exec] GroupBy VERIFY: difference = " << std::fixed << std::setprecision(2) << (gpuGrandTotal - cpuGrandTotal) << "\n";
                        std::cerr << "[Exec] GroupBy VERIFY: CPU max key=" << cpuMaxKey << " val=" << std::fixed << std::setprecision(2) << cpuMaxVal << "\n";
                        std::cerr << "[Exec] GroupBy VERIFY: GPU max key=" << gpuMaxKey << " val=" << std::fixed << std::setprecision(2) << static_cast<double>(gpuMaxVal) << "\n";
                        
                        // Check GPU group for the CPU max key
                        for (uint32_t s = 0; s < cap; ++s) {
                            uint32_t k0 = keyWords[s * 8 + 0];
                            if (k0 == cpuMaxKey + 1) { // +1 for bias
                                uint32_t raw = aggWords[s * 16 + 0];
                                float fval = *reinterpret_cast<const float*>(&raw);
                                std::cerr << "[Exec] GroupBy VERIFY: GPU value for CPU max key " << cpuMaxKey << " = " << std::fixed << std::setprecision(2) << static_cast<double>(fval) << "\n";
                                break;
                            }
                        }
                        
                        // Count how many GPU groups have HIGHER revenue than the correct max
                        size_t higherCount = 0;
                        for (uint32_t s = 0; s < cap; ++s) {
                            uint32_t k0 = keyWords[s * 8 + 0];
                            if (k0 == 0) continue;
                            uint32_t raw = aggWords[s * 16 + 0];
                            float fval = *reinterpret_cast<const float*>(&raw);
                            if (static_cast<double>(fval) > cpuMaxVal * 1.001) { // >0.1% higher than CPU max
                                higherCount++;
                                std::cerr << "[Exec] GroupBy VERIFY: GPU group key=" << (k0-1) << " val=" << std::fixed << std::setprecision(2) << static_cast<double>(fval) << " EXCEEDS CPU max!\n";
                            }
                        }
                        std::cerr << "[Exec] GroupBy VERIFY: " << higherCount << " GPU groups exceed CPU max revenue\n";
                    }
                    
                    // Clear and resize for fresh extraction
                    out.u32Cols.clear();
                    out.u32Cols.resize(keyVecs.size());
                    out.u32ColsGPU.clear();
                    out.u32ColsGPU.resize(keyVecs.size(), nullptr);
                    out.u32Names = keyNames;
                    out.f32Cols.clear();
                    out.f32Cols.resize(aggFuncs.size());
                    out.f32ColsGPU.clear();
                    out.f32ColsGPU.resize(aggFuncs.size(), nullptr);
                    out.f32Names = aggNames;
                    out.stringCols.clear();
                    out.stringNames.clear();
                    out.rowCount = 0;
                    
                    // GPU Stream Compaction: Mark → Prefix Sum → Compact
                    uint32_t numKeysHT = static_cast<uint32_t>(keyVecs.size());
                    uint32_t numAvgExtra = 0;
                    for (auto& af : aggFuncs) if (af == AggFunc::Avg) numAvgExtra++;
                    uint32_t numAggsTotal = static_cast<uint32_t>(aggFuncs.size()) + numAvgExtra;

                    auto extractResult = GpuOps::extractGroupByHT(*htOpt, numKeysHT, numAggsTotal);
                    if (extractResult && extractResult->rowCount > 0) {
                        out.rowCount = extractResult->rowCount;

                        // Move extracted keys directly into output columns + GPU buffers
                        for (size_t k = 0; k < keyVecs.size(); ++k) {
                            out.u32Cols[k] = std::move(extractResult->keyCols[k]);
                            out.u32ColsGPU[k] = extractResult->keyColsGPU[k];  // transfer ownership
                            extractResult->keyColsGPU[k] = nullptr;
                        }

                        // Process raw aggregate words with correct type conversion
                        for (size_t a = 0; a < aggFuncs.size(); ++a) {
                            out.f32Cols[a].resize(extractResult->rowCount);
                        }
                        size_t avgCount = 0;
                        for (size_t a = 0; a < aggFuncs.size(); ++a) {
                            uint32_t rc = extractResult->rowCount;
                            MTL::Buffer* aggGPU = (a < extractResult->aggColsGPU.size()) ? extractResult->aggColsGPU[a] : nullptr;
                            
                            if (aggFuncs[a] == AggFunc::CountStar) {
                                // Integer count → f32 cast on GPU
                                if (aggGPU) {
                                    MTL::Buffer* f32Buf = GpuOps::castU32ToF32(aggGPU, rc);
                                    if (f32Buf) {
                                        std::memcpy(out.f32Cols[a].data(), f32Buf->contents(), rc * sizeof(float));
                                        out.f32ColsGPU[a] = f32Buf; // keep GPU buffer
                                        extractResult->aggColsGPU[a] = nullptr; // ownership transferred via f32Buf using raw data
                                    }
                                } else {
                                    const auto& rawWords = extractResult->aggWords[a];
                                    for (uint32_t r = 0; r < rc; ++r)
                                        out.f32Cols[a][r] = static_cast<float>(rawWords[r]);
                                }
                            } else if (aggFuncs[a] == AggFunc::Avg) {
                                // AVG = sum / count; sum is f32 bits in u32, count is u32
                                size_t countIdx = aggFuncs.size() + avgCount;
                                MTL::Buffer* countGPU = (countIdx < extractResult->aggColsGPU.size()) ? extractResult->aggColsGPU[countIdx] : nullptr;
                                if (aggGPU && countGPU) {
                                    // GPU path: cast count u32→f32, then divide sum/count
                                    MTL::Buffer* countF32 = GpuOps::castU32ToF32(countGPU, rc);
                                    if (countF32) {
                                        // aggGPU contains f32 bits (sum) stored as u32 — same bits, use as f32 directly
                                        MTL::Buffer* avgBuf = GpuOps::arithDivF32ColCol(aggGPU, countF32, rc);
                                        countF32->release();
                                        if (avgBuf) {
                                            std::memcpy(out.f32Cols[a].data(), avgBuf->contents(), rc * sizeof(float));
                                            out.f32ColsGPU[a] = avgBuf;
                                        }
                                    }
                                } else {
                                    // CPU fallback
                                    const auto& rawWords = extractResult->aggWords[a];
                                    for (uint32_t r = 0; r < rc; ++r) {
                                        uint32_t w = rawWords[r];
                                        float sum = *reinterpret_cast<const float*>(&w);
                                        uint32_t cw = extractResult->aggWords[countIdx][r];
                                        float count = static_cast<float>(cw);
                                        out.f32Cols[a][r] = count > 0 ? sum / count : 0.0f;
                                    }
                                }
                                avgCount++;
                            } else if (aggFuncs[a] == AggFunc::Count) {
                                // Count(column): f32 non-null indicator sum — bits already f32
                                if (aggGPU) {
                                    // Raw bits are f32 — use GPU buffer directly
                                    std::memcpy(out.f32Cols[a].data(), aggGPU->contents(), rc * sizeof(float));
                                    out.f32ColsGPU[a] = aggGPU;
                                    aggGPU->retain();
                                    extractResult->aggColsGPU[a] = nullptr;
                                } else {
                                    const auto& rawWords = extractResult->aggWords[a];
                                    for (uint32_t r = 0; r < rc; ++r) {
                                        uint32_t w = rawWords[r];
                                        out.f32Cols[a][r] = *reinterpret_cast<const float*>(&w);
                                    }
                                }
                            } else {
                                // SUM/MIN/MAX: raw bits are f32 — use GPU buffer directly
                                if (aggGPU) {
                                    std::memcpy(out.f32Cols[a].data(), aggGPU->contents(), rc * sizeof(float));
                                    out.f32ColsGPU[a] = aggGPU;
                                    aggGPU->retain();
                                    extractResult->aggColsGPU[a] = nullptr;
                                } else {
                                    const auto& rawWords = extractResult->aggWords[a];
                                    for (uint32_t r = 0; r < rc; ++r) {
                                        uint32_t w = rawWords[r];
                                        out.f32Cols[a][r] = *reinterpret_cast<const float*>(&w);
                                    }
                                }
                            }
                        }
                        // Create GPU buffers for aggregates that don't already have one
                        for (size_t a = 0; a < aggFuncs.size(); ++a) {
                            if (!out.f32ColsGPU[a] && !out.f32Cols[a].empty()) {
                                out.f32ColsGPU[a] = GpuOps::createBuffer(
                                    out.f32Cols[a].data(),
                                    out.f32Cols[a].size() * sizeof(float));
                            }
                        }
                        // Release unused raw agg GPU buffers from extract
                        for (size_t a = 0; a < extractResult->aggColsGPU.size(); ++a) {
                            if (extractResult->aggColsGPU[a]) {
                                extractResult->aggColsGPU[a]->release();
                                extractResult->aggColsGPU[a] = nullptr;
                            }
                        }
                    }
                    
                    // Post-process string columns
                    for (size_t k = 0; k < keyVecs.size(); ++k) {
                        // Check if we have hash->string mapping (for pre-hashed keys)
                        if (k < hashToStringMaps.size() && !hashToStringMaps[k].empty()) {
                            // Use hash lookup
                            std::vector<std::string> strCol;
                            strCol.reserve(out.rowCount);
                            const auto& hashMap = hashToStringMaps[k];
                            
                            if (debug) std::cerr << "[Exec] GroupBy: Post-proc string col " << k 
                                                 << " via hash lookup, hashMap.size=" << hashMap.size() << "\n";
                            
                            for (uint32_t hashVal : out.u32Cols[k]) {
                                auto it = hashMap.find(hashVal);
                                if (it != hashMap.end()) {
                                    strCol.push_back(it->second);
                                } else {
                                    strCol.push_back("");
                                }
                            }
                            if (debug) std::cerr << "[Exec] GroupBy: Built strCol with " << strCol.size() << " strings via hash lookup\n";
                            out.stringCols.push_back(std::move(strCol));
                            out.stringNames.push_back(out.u32Names[k]);
                        } else if (!outputStringMaps[k].empty()) {
                            // Convert IDs back to strings (1-based index)
                            std::vector<std::string> strCol;
                            strCol.reserve(out.rowCount);
                            const auto& map = outputStringMaps[k];
                            
                            if (debug) std::cerr << "[Exec] GroupBy: Post-proc string col " << k 
                                                 << " u32_cols[k].size=" << out.u32Cols[k].size() 
                                                 << " map.size=" << map.size() << "\n";
                            
                            for (uint32_t val : out.u32Cols[k]) {
                                if (val > 0 && (val - 1) < map.size()) {
                                    strCol.push_back(map[val - 1]);
                                } else {
                                    strCol.push_back(""); 
                                }
                            }
                            if (debug) std::cerr << "[Exec] GroupBy: Built strCol with " << strCol.size() << " strings\n";
                            out.stringCols.push_back(std::move(strCol));
                            out.stringNames.push_back(out.u32Names[k]);
                        }
                    }

                    // Restore f32 keys that were bit-reinterpreted to u32
                    for (size_t k = 0; k < keyVecs.size(); ++k) {
                        if (k < keyFromF32.size() && keyFromF32[k]) {
                            // Convert u32 bits back to float and move to f32_cols
                            std::vector<float> restored(out.u32Cols[k].size());
                            for (size_t j = 0; j < restored.size(); ++j) {
                                std::memcpy(&restored[j], &out.u32Cols[k][j], sizeof(float));
                            }
                            if (debug) std::cerr << "[Exec] GroupBy: restoring f32 key " << out.u32Names[k] 
                                                 << " (" << restored.size() << " values)\n";
                            // Create GPU buffer for the restored f32 key (same bits as u32)
                            MTL::Buffer* restoredGPU = out.u32ColsGPU[k];  // reuse — same bit pattern
                            if (restoredGPU) restoredGPU->retain();  // shared ownership
                            // Add to f32 output (prepend before aggregates)
                            out.f32Names.insert(out.f32Names.begin(), out.u32Names[k]);
                            out.f32Cols.insert(out.f32Cols.begin(), std::move(restored));
                            out.f32ColsGPU.insert(out.f32ColsGPU.begin(), restoredGPU);
                            // Mark u32 slot as converted (will be handled in order building)
                        }
                    }

                    // Build output order - check if any string column was produced
                    out.order.clear();
                    size_t strIdx = 0;
                    // Count how many f32-restored keys were prepended (they shift agg f32 indices)
                    size_t f32KeyCount = 0;
                    for (size_t k = 0; k < keyVecs.size(); ++k) {
                        if (k < keyFromF32.size() && keyFromF32[k]) f32KeyCount++;
                    }
                    for (size_t i = 0; i < out.u32Names.size(); ++i) {
                        bool hasStrings = (!outputStringMaps[i].empty()) || 
                                          (i < hashToStringMaps.size() && !hashToStringMaps[i].empty());
                        bool wasF32 = (i < keyFromF32.size() && keyFromF32[i]);
                        if (hasStrings) {
                            out.order.push_back({TableResult::ColRef::Kind::String, strIdx++, out.u32Names[i]});
                        } else if (wasF32) {
                            // Find the f32 index for this key (prepended before aggregates)
                            size_t f32Idx = 0;
                            for (size_t fi = 0; fi < out.f32Names.size(); ++fi) {
                                if (out.f32Names[fi] == out.u32Names[i]) { f32Idx = fi; break; }
                            }
                            out.order.push_back({TableResult::ColRef::Kind::F32, f32Idx, out.u32Names[i]});
                        } else {
                            out.order.push_back({TableResult::ColRef::Kind::U32, i, out.u32Names[i]});
                        }
                    }
                    for (size_t i = f32KeyCount; i < out.f32Names.size(); ++i) {
                        out.order.push_back({TableResult::ColRef::Kind::F32, i, out.f32Names[i]});
                    }
                    
                    // Mark single-char columns
                    const auto& schema = SchemaRegistry::instance();
                    for (const auto& name : out.u32Names) {
                        std::string table = tableForColumn(name);
                        if (schema.isSingleCharColumn(table, name)) {
                            out.singleCharCols.insert(name);
                        }
                    }
                    
                    // Release GPU resources
                    GpuOps::release(*htOpt);
                    for (auto* buf : toRelease) buf->release();
                    for (auto* buf : keyBufsGPU) { if (buf) buf->release(); }
                    for (size_t i = 0; i < aggBufsGPU.size(); ++i) { if (aggBufsGPU[i]) aggBufsGPU[i]->release(); }
                    
                    if (debug) {
                        std::cerr << "[Exec] GroupBy: GPU completed with " << out.rowCount << " groups\n";
                        std::cerr << "[Exec] GroupBy: GPU output u32_cols.size=" << out.u32Cols.size();
                        for (size_t i = 0; i < out.u32Cols.size(); ++i) {
                            std::cerr << " " << out.u32Names[i] << "(" << out.u32Cols[i].size() << ")";
                        }
                        std::cerr << "\n[Exec] GroupBy: GPU output f32_cols.size=" << out.f32Cols.size();
                        for (size_t i = 0; i < out.f32Cols.size(); ++i) {
                            std::cerr << " " << out.f32Names[i] << "(" << out.f32Cols[i].size() << ")";
                            if (!out.f32Cols[i].empty()) {
                                std::cerr << "[" << out.f32Cols[i][0];
                                if (out.f32Cols[i].size() > 1) std::cerr << "," << out.f32Cols[i][1];
                                std::cerr << "]";
                            }
                        }
                        std::cerr << "\n";
                    }
                    return true;
                }
            }
            
            
            // GPU failed, release buffers and fall through to CPU
            for (auto* buf : toRelease) buf->release();
            for (auto* buf : keyBufsGPU) { if (buf) buf->release(); }
            for (size_t i = 0; i < aggBufsGPU.size(); ++i) { if (aggBufsGPU[i]) aggBufsGPU[i]->release(); }
            if (debug) {
                std::cerr << "[Exec] GroupBy: GPU path failed, falling back to CPU\n";
            }
        }
    }
    
    for (auto* buf : keyBufsGPU) { if (buf) buf->release(); }
    for (size_t i = 0; i < aggBufsGPU.size(); ++i) { if (aggBufsGPU[i]) aggBufsGPU[i]->release(); }
    throw std::runtime_error("GPU GroupBy failed: conditions not met for any kernel (and CPU fallback is disabled).");
}

} // namespace engine
