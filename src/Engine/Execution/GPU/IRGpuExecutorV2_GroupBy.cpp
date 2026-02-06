#include "IRGpuExecutorV2.hpp"
#include "IRGpuExecutorV2_Priv.hpp"
#include "OperatorsGPU.hpp"
#include "ColumnStoreGPU.hpp"
#include "TypedExprEval.hpp"
#include "KernelTimer.hpp"
#include "Schema.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <unordered_set>

namespace engine {

bool IRGpuExecutorV2::executeGroupBy(const IRGroupByV2& groupBy, EvalContext& ctx, TableResult& out) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    if (debug) {
        std::cerr << "[V2] GroupBy: ctx.rowCount=" << ctx.rowCount << "\n";
        std::cerr << "[V2] GroupBy: ctx.u32Cols.size=" << ctx.u32Cols.size() << ":";
        for (const auto& [n,v] : ctx.u32Cols) std::cerr << " " << n << "(" << v.size() << ")";
        std::cerr << "\n";
        std::cerr << "[V2] GroupBy: ctx.f32Cols.size=" << ctx.f32Cols.size() << ":";
        for (const auto& [n,v] : ctx.f32Cols) std::cerr << " " << n << "(" << v.size() << ")";
        std::cerr << "\n";
        std::cerr << "[V2] GroupBy: keys.size=" << groupBy.keys.size() << "\n";
        for (size_t i = 0; i < groupBy.keys.size(); ++i) {
            if (groupBy.keys[i] && groupBy.keys[i]->kind == TypedExpr::Kind::Column) {
                std::cerr << "[V2] GroupBy:   key[" << i << "]=" << groupBy.keys[i]->asColumn().column << "\n";
            }
        }
    }
    
    // Build key vectors
    std::vector<std::vector<uint32_t>> keyVecs;
    std::vector<std::string> keyNames;
    std::vector<std::vector<std::string>> outputStringMaps;
    std::vector<std::unordered_map<uint32_t, std::string>> hashToStringMaps;  // For hash-based lookup
    
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
                     if(debug) std::cerr << "[V2] GroupBy: Lazy fetch key " << col << " from GPU (" << count << " rows)\n";
                 }
            }
            
            // Try exact match first
            auto it = ctx.u32Cols.find(col);
            
            // Prefer column with matching row count (in case of duplicates with different sizes)
            if (it != ctx.u32Cols.end() && it->second.size() != expectedKeyRows) {
                if (debug) {
                    std::cerr << "[V2] GroupBy: key " << col << " has wrong size (" << it->second.size() 
                              << " vs expected " << expectedKeyRows << "), looking for positional ref\n";
                }
                // Size Mismatch: Look for positional ref (#N) with correct size
                // (Project might have renamed columns).
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
                                        std::cerr << "[V2] GroupBy: using positional " << posKey 
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
                                std::cerr << "[V2] GroupBy: resolved positional " << col << " to " << name << "\n";
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
                    std::cerr << "[V2] GroupBy: found key using keyName " << keyName << "\n";
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
            
            if (!u32Valid && ctx.stringCols.count(col)) {
                const auto& strData = ctx.stringCols.at(col);
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
                     
                     if (debug) std::cerr << "[V2] GroupBy: encoded string key " << col << " to u32 IDs (" << reverseMap.size() << " unique)\n";
                     keyVecs.push_back(std::move(ids));
                     keyNames.push_back(keyName.empty() ? col : keyName);
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
                    
                    // Build hash->string map for string output
                    if (ctx.stringCols.count(col)) {
                        const auto& strData = ctx.stringCols.at(col);
                        std::unordered_map<uint32_t, std::string> hashToStr;
                        
                        const auto& u32Data = it->second;
                        if (debug) std::cerr << "[V2] GroupBy: building hash->string map for " << col 
                                             << " u32Data.size=" << u32Data.size() 
                                             << " strData.size=" << strData.size() << "\n";
                        
                        for (size_t r = 0; r < std::min(u32Data.size(), strData.size()); ++r) {
                            uint32_t hash = u32Data[r];
                            if (hashToStr.find(hash) == hashToStr.end()) {
                                hashToStr[hash] = strData[r];
                            }
                        }
                        
                        if (debug) std::cerr << "[V2] GroupBy: built hash->string map with " << hashToStr.size() << " entries\n";
                        
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
                             if(debug) std::cerr << "[V2] GroupBy: Lazy fetch F32 key " << col << " from GPU\n";
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
                                if (r < itF->second.size()) converted.push_back(static_cast<uint32_t>(itF->second[r]));
                                else converted.push_back(0);
                            }
                        } else {
                            for (float f : itF->second) {
                                converted.push_back(static_cast<uint32_t>(f));
                            }
                        }
                        if (debug) std::cerr << "[V2] GroupBy: converted f32 key " << col << " to u32\n";
                        keyVecs.push_back(std::move(converted));
                        keyNames.push_back(keyName.empty() ? col : keyName);
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
    
    // If keys are empty but context says there are rows, the context is inconsistent
    // Return true with 0 groups rather than failing
    if (hasEmptyKeys) {
        if (debug) {
            std::cerr << "[V2] GroupBy: empty key vectors, returning 0 groups\n";
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
            std::cerr << "[V2] GroupBy: agg func=" << static_cast<int>(spec.func) 
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
                    std::cerr << "[V2] GroupBy: CountDistinct input from u32 col " << col << " size=" << input.size() << "\n";
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
                        std::cerr << "[V2] GroupBy: CountDistinct input from f32 col " << col << " size=" << input.size() << "\n";
                    }
                } else if (debug) {
                    std::cerr << "[V2] GroupBy: CountDistinct col " << col << " not found!\n";
                }
            }
            aggInputs.push_back(std::move(input));
        } else {
            // Evaluate input expression
            std::vector<float> input;
            MTL::Buffer* buf = evalExprFloatGPU(spec.input, ctx);
            if (buf) {
                 uint32_t count = (ctx.activeRowsGPU) ? ctx.activeRowsCountGPU : ctx.rowCount;
                 if (ctx.activeRowsGPU && ctx.activeRowsCountGPU == 0) count = 0;
                 input.resize(count);
                 if (count > 0) std::memcpy(input.data(), buf->contents(), count * sizeof(float));
                 buf->release();
            }

            if (debug) {
                std::cerr << "[V2] GroupBy: evalExprFloatGPU returned " << input.size() << " values\n";
                if (!input.empty()) {
                    float sum = 0, minV = input[0], maxV = input[0];
                    size_t zeroCount = 0;
                    for (float v : input) { 
                        sum += v; 
                        minV = std::min(minV, v); 
                        maxV = std::max(maxV, v); 
                        if (v == 0) zeroCount++;
                    }
                    std::cerr << "[V2] GroupBy: evalExprFloat stats: sum=" << sum << " min=" << minV << " max=" << maxV << " avg=" << (sum/input.size()) << " zeros=" << zeroCount << "\n";
                    // Print first 10 values
                    std::cerr << "[V2] GroupBy: first 10 values: ";
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
                
                if (debug) std::cerr << "[V2] GroupBy: trying col=" << col << "\n";
                
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
                        std::cerr << "[V2] GroupBy: inferred col=" << col << " for empty inputExpr\n";
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
                        std::cerr << "[V2] GroupBy: found f32 col, size=" << input.size() << "\n";
                        if (!input.empty()) {
                            float sum = 0, minV = input[0], maxV = input[0];
                            for (float v : input) { sum += v; minV = std::min(minV, v); maxV = std::max(maxV, v); }
                            std::cerr << "[V2] GroupBy: col " << col << " stats: sum=" << sum << " min=" << minV << " max=" << maxV << " avg=" << (sum/input.size()) << "\n";
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
        if (debug) std::cerr << "[V2] GroupBy: Detected CountDistinct, attempting 2-stage GPU execution.\n";
        
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
        IRGroupByV2 stage1Spec;
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
        IRGroupByV2::AggSpec dummyAgg;
        dummyAgg.func = AggFunc::CountStar; 
        dummyAgg.outputName = "dummy_cnt";
        stage1Spec.aggSpecs.push_back(dummyAgg);
        
        TableResult stage1Res;
        bool s1Ok = executeGroupBy(stage1Spec, ctx, stage1Res);
        if (!s1Ok) throw std::runtime_error("Stage 1 GroupBy failed (CountDistinct pre-pass)");
        
        if (debug) std::cerr << "[V2] GroupBy: Stage 1 complete. Rows=" << stage1Res.rowCount << "\n";

        // Stage 2: Group By {Keys} on stage1Res, with COUNT(*)
        EvalContext stage2Ctx;
        stage2Ctx.rowCount = stage1Res.rowCount;
        
        // Populate Context from Stage 1 Result
        // CAREFUL: If we blindly copy u32Cols, the next GroupBy will see IDs and treat them as u32 keys,
        // ignoring the string values. We want it to use the strings (and re-encode them) so that
        // the final result includes the string maps.
        
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
        
        IRGroupByV2 stage2Spec;
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
            IRGroupByV2::AggSpec s2Agg;
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
            if (debug || true) {
                std::cerr << "[V2] GroupBy: Warning: ctx.rowCount (" << expectedRowCount 
                          << ") differs from consistent key size (" << firstSize << "). Using key size.\n";
            }
            consensusRowCount = firstSize;
        }
    }
    
    // Verify key sizes match consensus
    for (size_t i = 0; i < keyVecs.size(); ++i) {
        const auto& kv = keyVecs[i];
        if (kv.size() != consensusRowCount) {
            if (debug || true) { // Force print
                std::cerr << "[V2] GroupBy: key size mismatch for key index " << i << " (name: " 
                          << (i < keyNames.size() ? keyNames[i] : "?") << "), expected " << consensusRowCount 
                          << " but got " << kv.size() << "\n";
                // Dump activeRows info
                std::cerr << "[V2]   ctx.rowCount=" << ctx.rowCount << "\n";
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
                std::cerr << "[V2] GroupBy: Using GPU path with " << keyVecs.size() << " keys and " << aggFuncs.size() << " aggregates\n";
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
                    case AggFunc::CountStar: gpuType = 1; break;
                    case AggFunc::Min: gpuType = 2; break;
                    case AggFunc::Max: gpuType = 3; break;
                    default: gpuType = 0; break;
                }
                aggTypesGpu.push_back(gpuType);
                
                MTL::Buffer* aggBuf = nullptr;
                if (gpuType == 1) {
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
                auto htOpt = OperatorsGPU::groupByAggMultiKeyTyped(keyBufs, aggBufs, aggTypesGpu, static_cast<uint32_t>(gpuRowCount));
                
                if (htOpt.has_value()) {
                    const uint32_t cap = htOpt->capacity;
                    const auto* keyWords = reinterpret_cast<const uint32_t*>(htOpt->ht_keys->contents());
                    const auto* aggWords = reinterpret_cast<const uint32_t*>(htOpt->ht_aggs->contents());
                    
                    if (debug) {
                        std::cerr << "[V2] GroupBy: GPU hash table capacity=" << cap << " gpuRowCount=" << gpuRowCount << "\n";
                        // Count non-empty slots
                        size_t nonEmpty = 0;
                        for (uint32_t s = 0; s < cap; ++s) {
                            if (keyWords[s * 8 + 0] != 0) nonEmpty++;
                        }
                        std::cerr << "[V2] GroupBy: GPU hash table non-empty slots=" << nonEmpty << "\n";
                    }
                    
                    // Extract results from GPU hash table
                    // IMPORTANT: Clear and resize to start fresh
                    out.u32_cols.clear();
                    out.u32_cols.resize(keyVecs.size());
                    out.u32_names = keyNames;
                    out.f32_cols.clear();
                    out.f32_cols.resize(aggFuncs.size());
                    out.f32_names = aggNames;
                    out.rowCount = 0;
                    
                    // Map for AVG: track which extra COUNT slots correspond to which AVG
                    size_t extraCountIdx = aggFuncs.size();  // Extra COUNT slots start after regular aggregates
                    
                    for (uint32_t slot = 0; slot < cap; ++slot) {
                        const uint32_t k0 = keyWords[slot * 8 + 0];
                        if (k0 == 0) continue;  // Empty slot
                        
                        // Extract keys (subtract 1 to remove bias)
                        for (size_t k = 0; k < keyVecs.size(); ++k) {
                            uint32_t keyVal = keyWords[slot * 8 + k];
                            out.u32_cols[k].push_back(keyVal > 0 ? keyVal - 1 : 0);
                        }
                        
                        // Extract aggregates
                        size_t avgCount = 0;
                        for (size_t a = 0; a < aggFuncs.size(); ++a) {
                            float val = 0.0f;
                            uint32_t aggSlotVal = aggWords[slot * 16 + a];
                            
                            if (aggFuncs[a] == AggFunc::Count || aggFuncs[a] == AggFunc::CountStar) {
                                val = static_cast<float>(aggSlotVal);
                            } else if (aggFuncs[a] == AggFunc::Avg) {
                                // Get SUM from slot a
                                float sum = *reinterpret_cast<const float*>(&aggSlotVal);
                                // Get COUNT from extra slot
                                uint32_t countSlotVal = aggWords[slot * 16 + aggFuncs.size() + avgCount];
                                float count = static_cast<float>(countSlotVal);
                                val = count > 0 ? sum / count : 0.0f;
                                avgCount++;
                            } else {
                                // SUM, MIN, MAX are stored as float bits
                                val = *reinterpret_cast<const float*>(&aggSlotVal);
                            }
                            out.f32_cols[a].push_back(val);
                        }
                        out.rowCount++;
                    }
                    
                    // Post-process string columns
                    for (size_t k = 0; k < keyVecs.size(); ++k) {
                        // Check if we have hash->string mapping (for pre-hashed keys)
                        if (k < hashToStringMaps.size() && !hashToStringMaps[k].empty()) {
                            // Use hash lookup
                            std::vector<std::string> strCol;
                            strCol.reserve(out.rowCount);
                            const auto& hashMap = hashToStringMaps[k];
                            
                            if (debug) std::cerr << "[V2] GroupBy: Post-proc string col " << k 
                                                 << " via hash lookup, hashMap.size=" << hashMap.size() << "\n";
                            
                            for (uint32_t hashVal : out.u32_cols[k]) {
                                auto it = hashMap.find(hashVal);
                                if (it != hashMap.end()) {
                                    strCol.push_back(it->second);
                                } else {
                                    strCol.push_back("");
                                }
                            }
                            if (debug) std::cerr << "[V2] GroupBy: Built strCol with " << strCol.size() << " strings via hash lookup\n";
                            out.string_cols.push_back(std::move(strCol));
                            out.string_names.push_back(out.u32_names[k]);
                        } else if (!outputStringMaps[k].empty()) {
                            // Convert IDs back to strings (1-based index)
                            std::vector<std::string> strCol;
                            strCol.reserve(out.rowCount);
                            const auto& map = outputStringMaps[k];
                            
                            if (debug) std::cerr << "[V2] GroupBy: Post-proc string col " << k 
                                                 << " u32_cols[k].size=" << out.u32_cols[k].size() 
                                                 << " map.size=" << map.size() << "\n";
                            
                            for (uint32_t val : out.u32_cols[k]) {
                                if (val > 0 && (val - 1) < map.size()) {
                                    strCol.push_back(map[val - 1]);
                                } else {
                                    strCol.push_back(""); 
                                }
                            }
                            if (debug) std::cerr << "[V2] GroupBy: Built strCol with " << strCol.size() << " strings\n";
                            out.string_cols.push_back(std::move(strCol));
                            out.string_names.push_back(out.u32_names[k]);
                        }
                    }

                    // Build output order - check if any string column was produced
                    out.order.clear();
                    size_t strIdx = 0;
                    for (size_t i = 0; i < out.u32_names.size(); ++i) {
                        bool hasStrings = (!outputStringMaps[i].empty()) || 
                                          (i < hashToStringMaps.size() && !hashToStringMaps[i].empty());
                        if (hasStrings) {
                            out.order.push_back({TableResult::ColRef::Kind::String, strIdx++, out.u32_names[i]});
                        } else {
                            out.order.push_back({TableResult::ColRef::Kind::U32, i, out.u32_names[i]});
                        }
                    }
                    for (size_t i = 0; i < out.f32_names.size(); ++i) {
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
                    OperatorsGPU::release(*htOpt);
                    for (auto* buf : toRelease) buf->release();
                    
                    if (debug) {
                        std::cerr << "[V2] GroupBy: GPU completed with " << out.rowCount << " groups\n";
                        std::cerr << "[V2] GroupBy: GPU output u32_cols.size=" << out.u32_cols.size();
                        for (size_t i = 0; i < out.u32_cols.size(); ++i) {
                            std::cerr << " " << out.u32_names[i] << "(" << out.u32_cols[i].size() << ")";
                        }
                        std::cerr << "\n[V2] GroupBy: GPU output f32_cols.size=" << out.f32_cols.size();
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
                std::cerr << "[V2] GroupBy: GPU path failed, falling back to CPU\n";
            }
        }
    }
    
    throw std::runtime_error("GPU GroupBy failed: conditions not met for any kernel (and CPU fallback is disabled).");
}

} // namespace engine
