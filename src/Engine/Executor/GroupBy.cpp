#include "GpuExecutor.hpp"
#include "GpuExecutorPriv.hpp"
#include "Operators.hpp"
#include "ColumnStoreGPU.hpp"
#include "TypedExprEval.hpp"
#include "KernelTimer.hpp"
#include "Schema.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
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
                            if (!ctx.activeRows.empty()) {
                                if (ctx.activeRows.size() > 0) {
                                    firstIdx = ctx.activeRows[0];
                                }
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
            
            
            bool u32Valid = (it != ctx.u32Cols.end() && it->second.size() == expectedKeyRows);
            bool stringHandled = false;
            
            // (Path 1) is collision-free and guarantees correct grouping.
            if (ctx.stringCols.count(col) && !ctx.stringCols.at(col).empty()) {
                const auto& strData = ctx.stringCols.at(col);
                
                // If activeRowsGPU is set but activeRows (CPU) is empty, download indices
                if (strData.size() != expectedKeyRows && ctx.activeRows.empty() 
                    && ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0) {
                    uint32_t* gpuPtr = (uint32_t*)ctx.activeRowsGPU->contents();
                    ctx.activeRows.assign(gpuPtr, gpuPtr + ctx.activeRowsCountGPU);
                    if (debug) std::cerr << "[Exec] GroupBy: Downloaded activeRowsGPU (" << ctx.activeRowsCountGPU << " rows) for string filtering\n";
                }
                
                if (strData.size() == expectedKeyRows || !ctx.activeRows.empty()) {
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
                         std::cerr << "[Exec] GroupBy: encoded string key " << col << " to u32 IDs (" << reverseMap.size() << " unique)\n";
                         std::cerr << "[Exec] GroupBy: strData.size=" << strData.size() << " activeRows.size=" << ctx.activeRows.size() << " expectedKeyRows=" << expectedKeyRows << "\n";
                         std::cerr << "[Exec] GroupBy: reverseMap (ID->string):";
                         for (size_t ri = 0; ri < reverseMap.size(); ++ri) std::cerr << " " << (ri+1) << "=\"" << reverseMap[ri] << "\"";
                         std::cerr << "\n";
                         // Distribution of IDs
                         std::map<uint32_t, size_t> idDist;
                         for (auto v : ids) idDist[v]++;
                         std::cerr << "[Exec] GroupBy: ID distribution:";
                         for (auto& [id,cnt] : idDist) std::cerr << " " << id << ":" << cnt;
                         std::cerr << "\n";
                         // Print first 10 strings from strData
                         std::cerr << "[Exec] GroupBy: first 10 strData:";
                         for (size_t si = 0; si < std::min(strData.size(), size_t(10)); ++si) std::cerr << " \"" << strData[si] << "\"";
                         std::cerr << "\n";
                         if (!ctx.activeRows.empty()) {
                             std::cerr << "[Exec] GroupBy: first 10 activeRows:";
                             for (size_t si = 0; si < std::min(ctx.activeRows.size(), size_t(10)); ++si) std::cerr << " " << ctx.activeRows[si];
                             std::cerr << "\n";
                         }
                     }
                     keyVecs.push_back(std::move(ids));
                     keyNames.push_back(keyName.empty() ? col : keyName);
                     keyFromF32.push_back(false);
                     outputStringMaps.push_back(std::move(reverseMap));
                     hashToStringMaps.push_back({});  // Empty - we use outputStringMaps for this case
                     stringHandled = true;
                }
            }
            
            if (!stringHandled) {
                if (it != ctx.u32Cols.end()) {
                    if (!ctx.activeRows.empty() && it->second.size() != expectedKeyRows) {
                        std::vector<uint32_t> filtered;
                        filtered.reserve(expectedKeyRows);
                        for (uint32_t r : ctx.activeRows) {
                            if (r < it->second.size()) filtered.push_back(it->second[r]);
                            else filtered.push_back(0);
                        }
                        keyVecs.push_back(std::move(filtered));
                    } else {
                        keyVecs.push_back(it->second);
                    }
                    keyNames.push_back(keyName.empty() ? col : keyName);
                    keyFromF32.push_back(false);
                    
                    // Build hash->string map for string output
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
            
                     // LAZY FETCH F32: If vector is empty but on GPU, bring it back
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
                        keyNames.push_back(keyName.empty() ? col : keyName);
                        keyFromF32.push_back(true);
                        outputStringMaps.push_back({});
                        hashToStringMaps.push_back({});
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
        out.u32_cols.clear();
        out.u32_cols.resize(keyVecs.size());
        out.u32_names = keyNames;
        out.f32_cols.clear();
        out.f32_names.clear();
        out.order.clear();
        // Still need to populate aggNames
        for (const auto& spec : groupBy.aggSpecs) {
            std::string name = spec.outputName;
            if (name.empty()) {
                name = aggFuncName(spec.func);
            }
            out.f32_names.push_back(name);
            out.f32_cols.push_back({});
        }
        for (size_t i = 0; i < out.u32_names.size(); ++i) {
            out.order.push_back({TableResult::ColRef::Kind::U32, i, out.u32_names[i]});
        }
        for (size_t i = 0; i < out.f32_names.size(); ++i) {
            out.order.push_back({TableResult::ColRef::Kind::F32, i, out.f32_names[i]});
        }
        return true;
    }
    
    // Build aggregate input vectors
    std::vector<std::vector<float>> aggInputs;
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
                // Store u32 values as float to track NULLs (0 = NULL)
                if (!ctx.activeRows.empty() && itU->second.size() > expectedKeyRows) {
                     input.reserve(expectedKeyRows);
                     for (uint32_t r : ctx.activeRows) {
                         if (r < itU->second.size()) input.push_back(static_cast<float>(itU->second[r]));
                         else input.push_back(0.0f);
                     }
                } else {
                    input.resize(itU->second.size());
                    for (size_t j = 0; j < itU->second.size(); ++j) {
                        input[j] = static_cast<float>(itU->second[j]);
                    }
                }
                if (debug) {
                    std::cerr << "[Exec] GroupBy: CountDistinct input from u32 col " << col << " size=" << input.size() << "\n";
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
                    if (!ctx.activeRows.empty() && itF->second.size() > expectedKeyRows) {
                         input.reserve(expectedKeyRows);
                         for (uint32_t r : ctx.activeRows) {
                             if (r < itF->second.size()) input.push_back(itF->second[r]);
                             else input.push_back(0.0f);
                         }
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
        } else {
            // Evaluate input expression
            std::vector<float> input;

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
                        foundPrecomputed = true;
                        if (debug) std::cerr << "[Exec] GroupBy: using pre-computed f32ColGPU '" << spec.inputExpr << "' (" << input.size() << " values)\n";
                    }
                }
            }

            if (!foundPrecomputed) {
                MTL::Buffer* buf = evalExprFloatGPU(spec.input, ctx);
                if (buf) {
                     uint32_t count = (ctx.activeRowsGPU) ? ctx.activeRowsCountGPU : ctx.rowCount;
                     if (ctx.activeRowsGPU && ctx.activeRowsCountGPU == 0) count = 0;
                     input.resize(count);
                     if (count > 0) std::memcpy(input.data(), buf->contents(), count * sizeof(float));
                     buf->release();
                }
            }

            if (debug) {
                std::cerr << "[Exec] GroupBy: evalExprFloatGPU returned " << input.size() << " values\n";
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
                    if (!ctx.activeRows.empty() && itF->second.size() > expectedKeyRows) {
                         input.reserve(expectedKeyRows);
                         for (uint32_t r : ctx.activeRows) {
                             if (r < itF->second.size()) input.push_back(itF->second[r]);
                             else input.push_back(0.0f);
                         }
                    } else {
                        input = itF->second;
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
                        if (!ctx.activeRows.empty() && itU->second.size() > expectedKeyRows) {
                             input.reserve(expectedKeyRows);
                             for (uint32_t r : ctx.activeRows) {
                                 if (r < itU->second.size()) input.push_back(static_cast<float>(itU->second[r]));
                                 else input.push_back(0.0f);
                             }
                        } else {
                            input.resize(itU->second.size());
                            for (size_t j = 0; j < itU->second.size(); ++j) {
                                input[j] = static_cast<float>(itU->second[j]);
                            }
                        }
                    }
                }
            }
            aggInputs.push_back(std::move(input));
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
        for(size_t i=0; i<stage1Res.string_names.size(); ++i) {
            stage2Ctx.stringCols[stage1Res.string_names[i]] = stage1Res.string_cols[i];
            strColNames.insert(stage1Res.string_names[i]);
        }
        
        for(size_t i=0; i<stage1Res.u32_names.size(); ++i) {
             // Only copy as u32 if it's NOT a string column
             if (strColNames.find(stage1Res.u32_names[i]) == strColNames.end()) {
                stage2Ctx.u32Cols[stage1Res.u32_names[i]] = stage1Res.u32_cols[i];
             }
        }

        for(size_t i=0; i<stage1Res.f32_names.size(); ++i) {
            stage2Ctx.f32Cols[stage1Res.f32_names[i]] = stage1Res.f32_cols[i];
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
    
    // Use consensus count for execution
    size_t executionCount = consensusRowCount;
    
    // GPU GroupBy path
    if (useGpu && !keyVecs.empty()) {
        auto& store = ColumnStoreGPU::instance();
        if (store.device() && store.library() && store.queue()) {
            
            // ... inside GPU path use executionCount ...

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
                auto buf = store.device()->newBuffer(gpuRowCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                if (!buf) { gpuOk = false; break; }
                uint32_t* ptr = static_cast<uint32_t*>(buf->contents());
                for (size_t i = 0; i < gpuRowCount; ++i) {
                    // Add 1 to bias away from 0 (empty marker in hash table)
                    ptr[i] = (i < keyVecs[k].size() ? keyVecs[k][i] : 0) + 1;
                }
                if (debug) {
                    std::map<uint32_t, size_t> dist;
                    for (size_t i = 0; i < gpuRowCount; ++i) dist[ptr[i]]++;
                    std::cerr << "[Exec] GroupBy: GPU key buf[" << k << "] distribution (biased):";
                    for (auto& [v,c] : dist) std::cerr << " " << v << ":" << c;
                    std::cerr << "\n";
                }
                keyBufs.push_back(buf);
                toRelease.push_back(buf);
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
                    // COUNT(column): create non-null indicator (1.0 if non-null, 0.0 if null/zero)
                    aggBuf = store.device()->newBuffer(gpuRowCount * sizeof(float), MTL::ResourceStorageModeShared);
                    if (aggBuf) {
                        float* ptr = static_cast<float*>(aggBuf->contents());
                        for (size_t i = 0; i < gpuRowCount; ++i) {
                            if (i < aggInputs[a].size()) {
                                // Non-null indicator: value != 0 means non-null
                                ptr[i] = (aggInputs[a][i] != 0.0f) ? 1.0f : 0.0f;
                            } else {
                                ptr[i] = 0.0f;
                            }
                        }
                        toRelease.push_back(aggBuf);
                    }
                } else if (gpuType == 1) {
                    // COUNT doesn't need input data, but kernel needs a buffer
                    aggBuf = store.device()->newBuffer(gpuRowCount * sizeof(float), MTL::ResourceStorageModeShared);
                    if (aggBuf) {
                        std::memset(aggBuf->contents(), 0, gpuRowCount * sizeof(float));
                        toRelease.push_back(aggBuf);
                    }
                } else if (!aggInputs[a].empty()) {
                    aggBuf = store.device()->newBuffer(gpuRowCount * sizeof(float), MTL::ResourceStorageModeShared);
                    if (aggBuf) {
                        float* ptr = static_cast<float*>(aggBuf->contents());
                        for (size_t i = 0; i < gpuRowCount; ++i) {
                            ptr[i] = (i < aggInputs[a].size()) ? aggInputs[a][i] : 0.0f;
                        }
                        toRelease.push_back(aggBuf);
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
                    const auto* keyWords = reinterpret_cast<const uint32_t*>(htOpt->ht_keys->contents());
                    const auto* aggWords = reinterpret_cast<const uint32_t*>(htOpt->ht_aggs->contents());
                    
                    if (debug) {
                        std::cerr << "[Exec] GroupBy: GPU hash table capacity=" << cap << " gpuRowCount=" << gpuRowCount << "\n";
                        // Count non-empty slots
                        size_t nonEmpty = 0;
                        for (uint32_t s = 0; s < cap; ++s) {
                            if (keyWords[s * 8 + 0] != 0) nonEmpty++;
                        }
                        std::cerr << "[Exec] GroupBy: GPU hash table non-empty slots=" << nonEmpty << "\n";
                        // Dump all non-empty slots
                        for (uint32_t s = 0; s < cap; ++s) {
                            uint32_t k0 = keyWords[s * 8 + 0];
                            if (k0 == 0) continue;
                            std::cerr << "[Exec] GroupBy: HT slot " << s << ": keys=[";
                            for (size_t ki = 0; ki < keyVecs.size(); ++ki) {
                                if (ki) std::cerr << ",";
                                std::cerr << keyWords[s * 8 + ki];
                            }
                            std::cerr << "] aggs=[";
                            for (size_t ai = 0; ai < aggFuncs.size(); ++ai) {
                                if (ai) std::cerr << ",";
                                uint32_t raw = aggWords[s * 16 + ai];
                                if (aggFuncs[ai] == AggFunc::Count || aggFuncs[ai] == AggFunc::CountStar)
                                    std::cerr << raw;
                                else
                                    std::cerr << *reinterpret_cast<const float*>(&raw);
                            }
                            std::cerr << "]\n";
                        }
                    }
                    
                    // Clear and resize for fresh extraction
                    out.u32_cols.clear();
                    out.u32_cols.resize(keyVecs.size());
                    out.u32_names = keyNames;
                    out.f32_cols.clear();
                    out.f32_cols.resize(aggFuncs.size());
                    out.f32_names = aggNames;
                    out.string_cols.clear();
                    out.string_names.clear();
                    out.rowCount = 0;
                    
                    // GPU Stream Compaction: Mark → Prefix Sum → Compact
                    uint32_t numKeysHT = static_cast<uint32_t>(keyVecs.size());
                    uint32_t numAvgExtra = 0;
                    for (auto& af : aggFuncs) if (af == AggFunc::Avg) numAvgExtra++;
                    uint32_t numAggsTotal = static_cast<uint32_t>(aggFuncs.size()) + numAvgExtra;

                    auto extractResult = GpuOps::extractGroupByHT(*htOpt, numKeysHT, numAggsTotal);
                    if (extractResult && extractResult->rowCount > 0) {
                        out.rowCount = extractResult->rowCount;

                        // Move extracted keys directly into output columns
                        for (size_t k = 0; k < keyVecs.size(); ++k) {
                            out.u32_cols[k] = std::move(extractResult->keyCols[k]);
                        }

                        // Process raw aggregate words with correct type conversion
                        for (size_t a = 0; a < aggFuncs.size(); ++a) {
                            out.f32_cols[a].resize(extractResult->rowCount);
                        }
                        size_t avgCount = 0;
                        for (size_t a = 0; a < aggFuncs.size(); ++a) {
                            const auto& rawWords = extractResult->aggWords[a];
                            for (uint32_t r = 0; r < extractResult->rowCount; ++r) {
                                float val = 0.0f;
                                uint32_t w = rawWords[r];
                                if (aggFuncs[a] == AggFunc::CountStar) {
                                    val = static_cast<float>(w);
                                } else if (aggFuncs[a] == AggFunc::Count) {
                                    val = *reinterpret_cast<const float*>(&w);
                                } else if (aggFuncs[a] == AggFunc::Avg) {
                                    float sum = *reinterpret_cast<const float*>(&w);
                                    uint32_t cw = extractResult->aggWords[aggFuncs.size() + avgCount][r];
                                    float count = static_cast<float>(cw);
                                    val = count > 0 ? sum / count : 0.0f;
                                } else {
                                    val = *reinterpret_cast<const float*>(&w);
                                }
                                out.f32_cols[a][r] = val;
                            }
                            if (aggFuncs[a] == AggFunc::Avg) avgCount++;
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
                            
                            for (uint32_t hashVal : out.u32_cols[k]) {
                                auto it = hashMap.find(hashVal);
                                if (it != hashMap.end()) {
                                    strCol.push_back(it->second);
                                } else {
                                    strCol.push_back("");
                                }
                            }
                            if (debug) std::cerr << "[Exec] GroupBy: Built strCol with " << strCol.size() << " strings via hash lookup\n";
                            out.string_cols.push_back(std::move(strCol));
                            out.string_names.push_back(out.u32_names[k]);
                        } else if (!outputStringMaps[k].empty()) {
                            // Convert IDs back to strings (1-based index)
                            std::vector<std::string> strCol;
                            strCol.reserve(out.rowCount);
                            const auto& map = outputStringMaps[k];
                            
                            if (debug) std::cerr << "[Exec] GroupBy: Post-proc string col " << k 
                                                 << " u32_cols[k].size=" << out.u32_cols[k].size() 
                                                 << " map.size=" << map.size() << "\n";
                            
                            for (uint32_t val : out.u32_cols[k]) {
                                if (val > 0 && (val - 1) < map.size()) {
                                    strCol.push_back(map[val - 1]);
                                } else {
                                    strCol.push_back(""); 
                                }
                            }
                            if (debug) std::cerr << "[Exec] GroupBy: Built strCol with " << strCol.size() << " strings\n";
                            out.string_cols.push_back(std::move(strCol));
                            out.string_names.push_back(out.u32_names[k]);
                        }
                    }

                    // Restore f32 keys that were bit-reinterpreted to u32
                    for (size_t k = 0; k < keyVecs.size(); ++k) {
                        if (k < keyFromF32.size() && keyFromF32[k]) {
                            // Convert u32 bits back to float and move to f32_cols
                            std::vector<float> restored(out.u32_cols[k].size());
                            for (size_t j = 0; j < restored.size(); ++j) {
                                std::memcpy(&restored[j], &out.u32_cols[k][j], sizeof(float));
                            }
                            if (debug) std::cerr << "[Exec] GroupBy: restoring f32 key " << out.u32_names[k] 
                                                 << " (" << restored.size() << " values)\n";
                            // Add to f32 output (prepend before aggregates)
                            out.f32_names.insert(out.f32_names.begin(), out.u32_names[k]);
                            out.f32_cols.insert(out.f32_cols.begin(), std::move(restored));
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
                    for (size_t i = 0; i < out.u32_names.size(); ++i) {
                        bool hasStrings = (!outputStringMaps[i].empty()) || 
                                          (i < hashToStringMaps.size() && !hashToStringMaps[i].empty());
                        bool wasF32 = (i < keyFromF32.size() && keyFromF32[i]);
                        if (hasStrings) {
                            out.order.push_back({TableResult::ColRef::Kind::String, strIdx++, out.u32_names[i]});
                        } else if (wasF32) {
                            // Find the f32 index for this key (prepended before aggregates)
                            size_t f32Idx = 0;
                            for (size_t fi = 0; fi < out.f32_names.size(); ++fi) {
                                if (out.f32_names[fi] == out.u32_names[i]) { f32Idx = fi; break; }
                            }
                            out.order.push_back({TableResult::ColRef::Kind::F32, f32Idx, out.u32_names[i]});
                        } else {
                            out.order.push_back({TableResult::ColRef::Kind::U32, i, out.u32_names[i]});
                        }
                    }
                    for (size_t i = f32KeyCount; i < out.f32_names.size(); ++i) {
                        out.order.push_back({TableResult::ColRef::Kind::F32, i, out.f32_names[i]});
                    }
                    
                    // Mark single-char columns
                    const auto& schema = SchemaRegistry::instance();
                    for (const auto& name : out.u32_names) {
                        std::string table = tableForColumn(name);
                        if (schema.isSingleCharColumn(table, name)) {
                            out.singleCharCols.insert(name);
                        }
                    }
                    
                    // Release GPU resources
                    GpuOps::release(*htOpt);
                    for (auto* buf : toRelease) buf->release();
                    
                    if (debug) {
                        std::cerr << "[Exec] GroupBy: GPU completed with " << out.rowCount << " groups\n";
                        std::cerr << "[Exec] GroupBy: GPU output u32_cols.size=" << out.u32_cols.size();
                        for (size_t i = 0; i < out.u32_cols.size(); ++i) {
                            std::cerr << " " << out.u32_names[i] << "(" << out.u32_cols[i].size() << ")";
                        }
                        std::cerr << "\n[Exec] GroupBy: GPU output f32_cols.size=" << out.f32_cols.size();
                        for (size_t i = 0; i < out.f32_cols.size(); ++i) {
                            std::cerr << " " << out.f32_names[i] << "(" << out.f32_cols[i].size() << ")";
                            if (!out.f32_cols[i].empty()) {
                                std::cerr << "[" << out.f32_cols[i][0];
                                if (out.f32_cols[i].size() > 1) std::cerr << "," << out.f32_cols[i][1];
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
            if (debug) {
                std::cerr << "[Exec] GroupBy: GPU path failed, falling back to CPU\n";
            }
        }
    }
    
    throw std::runtime_error("GPU GroupBy failed: conditions not met for any kernel (and CPU fallback is disabled).");
}

} // namespace engine
