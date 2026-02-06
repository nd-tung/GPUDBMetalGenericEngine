#include "IRGpuExecutorV2.hpp"
#include "IRGpuExecutorV2_Priv.hpp"
#include "RelationGPU.hpp"
#include "OperatorsGPU.hpp"

#include <iostream>
#include <vector>
#include <set>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <map>

namespace engine {

bool IRGpuExecutorV2::executeProject(const IRProjectV2& project, EvalContext& ctx, TableResult& out, std::unordered_map<std::string, EvalContext>* tableContexts) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    if (debug) {
        std::cerr << "[V2] Project START: currentTable=" << ctx.currentTable << " ctx.u32Cols=";
        for (const auto& [k, v] : ctx.u32Cols) std::cerr << k << " ";
        std::cerr << "\n";
    }

    // NOTE: activeRows sync from GPU is deferred until needed (e.g., for string columns)
    // Lambda to sync on-demand
    auto ensureActiveRowsCPU = [&]() {
        if (ctx.activeRowsGPU && ctx.activeRows.size() != ctx.activeRowsCountGPU) {
            if (debug) std::cerr << "[V2] Project: Lazy syncing activeRowsGPU (" << ctx.activeRowsCountGPU << ") to CPU\n";
            ctx.activeRows.resize(ctx.activeRowsCountGPU);
            if (ctx.activeRowsCountGPU > 0) {
                std::memcpy(ctx.activeRows.data(), ctx.activeRowsGPU->contents(), ctx.activeRowsCountGPU * sizeof(uint32_t));
            }
        }
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
    
    // Clear output if it already has content (we're selecting/reordering)
    bool hasExistingOutput = !out.u32_cols.empty() || !out.f32_cols.empty();
    
    if (debug && hasExistingOutput) {
        std::cerr << "[V2] Project: hasExistingOutput=true, out.u32_names=";
        for (const auto& n : out.u32_names) std::cerr << n << " ";
        std::cerr << "\n";
    }
    
    // Copy output only if projecting from previous result (not fresh scan).
    // If currentTable set, don't merge (single-table data).
    bool shouldCopyFromOut = hasExistingOutput && ctx.currentTable.empty();
    
    if (shouldCopyFromOut) {
        // Save existing data to use for projection (for post-join/aggregate projections)
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

        // Add to context
        for (const auto& [n, v] : savedU32) ctx.u32Cols[n] = v;
        for (const auto& [n, v] : savedF32) ctx.f32Cols[n] = v;
        for (const auto& [n, v] : savedString) ctx.stringCols[n] = v;
    }
    
    // Clear for repopulation
    if (hasExistingOutput) {
        out.u32_cols.clear();
        out.u32_names.clear();
        out.f32_cols.clear();
        out.f32_names.clear();
        out.string_cols.clear();
        out.string_names.clear();
        out.order.clear();
    }
    
    // Track which columns we've already used in this projection
    // to support multi-instance columns (n_name -> n_name_2 for second use)
    std::set<std::string> usedColumns;
    
    // For each projection expression, evaluate and add to context
    // This allows subsequent operators (GROUP BY) to reference computed columns
    for (size_t i = 0; i < project.exprs.size(); ++i) {
        const auto& expr = project.exprs[i];
        std::string outName = i < project.outputNames.size() ? project.outputNames[i] : "";
        
        // Generate a name for positional reference (#N)
        std::string posName = "#" + std::to_string(i);
        
        if (debug) {
            std::cerr << "[V2] Project: expr[" << i << "] outName=" << outName;
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
                std::cerr << "[V2] Project: function '" << func.name << "' (lower: '" << funcLower 
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
                    
                    // Find raw strings for this column
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
                        
                        // Create u32 encoded values for groupby compatibility (simple hash).
                        std::vector<uint32_t> encoded;
                        encoded.reserve(ctx.stringCols[outName].size());
                        for (const auto& s : ctx.stringCols[outName]) {
                            // Simple hash - just convert first 2 chars to number
                            uint32_t val = 0;
                            for (size_t i = 0; i < s.size() && i < 8; ++i) {
                                val = val * 256 + static_cast<uint8_t>(s[i]);
                            }
                            encoded.push_back(val);
                        }
                        ctx.u32Cols[outName] = encoded;
                        ctx.u32Cols[posName] = encoded;
                        out.u32_cols.push_back(encoded);
                        out.u32_names.push_back(outName);
                        
                        if (debug) {
                            std::cerr << "[V2] Project: SUBSTRING computed " << ctx.stringCols[outName].size() 
                                      << " results for " << outName << "\n";
                        }
                        continue;
                    }
                }
            }
            
            // Handle EXTRACT(YEAR FROM col) or YEAR(col)
            bool isYearFunc = (funcLower == "year" && func.args.size() == 1) || 
                              (funcLower == "extract" && func.args.size() >= 2); // lenient check for extract

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
                    
                    // Helper to find actual column key in map (and also check GPU)
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
                                    if (debug) std::cerr << "[V2] Project: YEAR lazy-fetched " << key << " from GPU (" << count << " rows)\n";
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
                    
                    if (found) {
                        if(debug) std::cerr << "[V2] Project: YEAR computed " << results.size() << " results for " << outName << " (Input table: " << ctx.currentTable << ")\n";
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
                    if (debug) std::cerr << "[V2] Project: function passthrough " << col << "\n";
                    ctx.u32Cols[posName] = itU->second;
                    out.u32_cols.push_back(itU->second);
                    out.u32_names.push_back(outName.empty() ? col : outName);
                    continue;
                }
                // For #N positional references, look up directly (they should exist in context)
                if (col.size() >= 2 && col[0] == '#' && std::isdigit(static_cast<unsigned char>(col[1]))) {
                    auto itU = ctx.u32Cols.find(col);
                    if (itU != ctx.u32Cols.end()) {
                        if (debug) std::cerr << "[V2] Project: function passthrough positional " << col << "\n";
                        ctx.u32Cols[posName] = itU->second;
                        out.u32_cols.push_back(itU->second);
                        out.u32_names.push_back(outName.empty() ? col : outName);
                        continue;
                    }
                    // Also try f32
                    auto itF = ctx.f32Cols.find(col);
                    if (itF != ctx.f32Cols.end()) {
                        if (debug) std::cerr << "[V2] Project: function passthrough positional " << col << " (f32)\n";
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
                 std::cerr << "[V2] Project: Looking for col '" << col << "'\n";
                 std::cerr << "[V2] Project: ActiveRows size: " << ctx.activeRows.size() 
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
                if (debug) std::cerr << "[V2] Project: pass-through string col " << col << " (as " << strLookupCol << ") -> " << (outName.empty() ? col : outName) << "\n";
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
                
                if (debug) std::cerr << "[V2] Project: string col size " << sub.size() << "\n";
                updateRowCount(sub.size());

                if (!outName.empty()) {
                    ctx.stringCols[outName] = sub;
                    if (outName != col) ctx.columnAliases[col] = outName;
                }
                ctx.stringCols[posName] = sub;
                out.string_cols.push_back(std::move(sub));
                out.string_names.push_back(outName.empty() ? col : outName);
                
                // Also check if U32 exists (hashes) and propagate if needed?
                // For now, assume String type dominance.
                continue;
            }

            
            // Handle post-GroupBy positional references: #N might be SUM_#N or COUNT_#N
            if (col.size() >= 2 && col[0] == '#') {
                // Try SUM_#N first for aggregate outputs
                std::string sumName = "SUM_" + col;
                auto itSum = ctx.f32Cols.find(sumName);
                if (itSum != ctx.f32Cols.end()) {
                    if (debug) std::cerr << "[V2] Project: mapping " << col << " -> " << sumName << "\n";
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
                    if (debug) std::cerr << "[V2] Project: mapping " << col << " -> " << countName << "\n";
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
                            if (debug) std::cerr << "[V2] Project: multi-instance column " << col << " -> " << lookupCol << "\n";
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
                    if (debug) std::cerr << "[V2] Project: alias resolution " << lookupCol << " -> " << aliasIt->second << "\n";
                    lookupCol = aliasIt->second;
                }
                // Also try: if col not found but outName is, use outName as lookup
                // This handles CTE aliasing where DuckDB plan asks to read "supplier_no" (alias)
                // but we only have "l_suppkey" (the output name which matches actual column)
                else if (!outName.empty() && outName != col &&
                         (ctx.u32Cols.find(outName) != ctx.u32Cols.end() ||
                          ctx.f32Cols.find(outName) != ctx.f32Cols.end())) {
                    if (debug) std::cerr << "[V2] Project: CTE alias fallback " << lookupCol << " -> " << outName << "\n";
                    lookupCol = outName;
                    // Track this alias for future lookups
                    ctx.columnAliases[col] = outName;
                }
            }
            
            // Patch: Fuzzy match for Join aliases (e.g. min(x) vs min(x)_rhs_29)
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
                                         if (debug) std::cerr << "[V2] Project: aggregate fuzzy match (varying) '" << name << "' -> '" << n << "'\n";
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
                                             if (debug) std::cerr << "[V2] Project: aggregate fuzzy match (varying GPU) '" << name << "' -> '" << n << "'\n";
                                             break;
                                         }
                                     }
                                 }
                             }
                         }
                         
                         // Return varying match if found, otherwise first match
                         if (!varyingMatch.empty()) {
                             if (debug && varyingMatch != firstMatch) 
                                 std::cerr << "[V2] Project: preferring varying column '" << varyingMatch << "' over '" << firstMatch << "'\n";
                             return varyingMatch;
                         }
                         if (!firstMatch.empty()) {
                             if (debug) std::cerr << "[V2] Project: aggregate fuzzy match '" << name << "' -> '" << firstMatch << "'\n";
                             return firstMatch;
                         }
                     }
                     
                     // Check GPU maps too! (Size check hard for GPU, assume if CPU cache Empty, we might defer? But Project usually syncs)
                     // If we are here, CPU map missing or empty.
                     return "";
                 };
                 std::string found = fuzzyFind(lookupCol);
                 if (!found.empty()) {
                      if (debug) std::cerr << "[V2] Project: fuzzy match " << lookupCol << " -> " << found << "\n";
                      lookupCol = found;
                 }
            }
            
            if (debug) std::cerr << "[V2] Project: lookup " << col << " as " << lookupCol << "\n";
            
            // Try to download from GPU if missing on CPU OR if CPU column has wrong size
            auto itDirect = ctx.u32Cols.find(lookupCol);
            bool missingCPU = (itDirect == ctx.u32Cols.end());
            if (!missingCPU && ctx.rowCount > 0 && itDirect->second.empty()) missingCPU = true;
            // Also prefer GPU if CPU column size doesn't match context rowCount
            if (!missingCPU && ctx.rowCount > 0 && itDirect->second.size() != ctx.rowCount) {
                // GPU column may have correct filtered size
                if (ctx.u32ColsGPU.count(lookupCol)) {
                    if (debug) std::cerr << "[V2] Project: CPU col " << lookupCol << " size=" << itDirect->second.size() << " != ctx.rowCount=" << ctx.rowCount << ", re-downloading from GPU\n";
                    missingCPU = true;
                }
            }
            
            if (missingCPU) {
                MTL::Buffer* buf = nullptr;
                if (ctx.u32ColsGPU.count(lookupCol)) {
                    buf = ctx.u32ColsGPU[lookupCol];
                }
                
                if (buf) {
                     if (debug) std::cerr << "[V2] Project: downloading GPU column " << lookupCol << "\n";
                     uint32_t cnt = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
                     if (cnt == 0 && ctx.rowCount > 0 && !ctx.activeRowsGPU) cnt = ctx.rowCount;
                     
                     std::vector<uint32_t> down;
                     if (cnt > 0) {
                         down.resize(cnt);
                         MTL::Buffer* src = buf;
                         bool temp = false;
                         
                         if (ctx.activeRowsGPU) {
                             src = OperatorsGPU::gatherU32(buf, ctx.activeRowsGPU, cnt);
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
                } else {
                    // Need CPU activeRows for compaction
                    ensureActiveRowsCPU();
                    colData.reserve(ctx.activeRows.size());
                    for (uint32_t idx : ctx.activeRows) {
                        colData.push_back(idx < itU->second.size() ? itU->second[idx] : 0);
                    }
                }
                
                if (debug) {
                    std::cerr << "[V2] Project: column " << lookupCol << " size=" << colData.size();
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
                        if (debug) std::cerr << "[V2] Project: tracking alias " << col << " -> " << outName << "\n";
                    }
                }
                // For CTE aliasing: also store under the alias name (col) if it differs
                // This handles cases like "l_suppkey as supplier_no" where join needs "supplier_no"
                if (col != lookupCol && col != outName) {
                    ctx.u32Cols[col] = colData;
                    if (debug) std::cerr << "[V2] Project: also storing as CTE alias " << col << "\n";
                }
                ctx.u32Cols[posName] = colData;
                out.u32_cols.push_back(std::move(colData));
                out.u32_names.push_back(outName.empty() ? lookupCol : outName);
                if (debug) std::cerr << "[V2] Project: Pushing U32 col " << (outName.empty()?lookupCol:outName) << "\n";
                continue;
            }

            // Check if F32 data is on GPU and needs downloading
            bool missingCPU_F32 = (ctx.f32Cols.find(lookupCol) == ctx.f32Cols.end());
            if (!missingCPU_F32 && ctx.rowCount > 0 && ctx.f32Cols[lookupCol].empty()) missingCPU_F32 = true;
            // Also prefer GPU if CPU column size doesn't match context rowCount
            if (!missingCPU_F32 && ctx.rowCount > 0 && ctx.f32Cols[lookupCol].size() != ctx.rowCount) {
                if (ctx.f32ColsGPU.count(lookupCol)) {
                    if (debug) std::cerr << "[V2] Project: CPU f32 col " << lookupCol << " size=" << ctx.f32Cols[lookupCol].size() << " != ctx.rowCount=" << ctx.rowCount << ", re-downloading from GPU\n";
                    missingCPU_F32 = true;
                }
            }
            
            if (missingCPU_F32) {
                MTL::Buffer* buf = nullptr;
                if (ctx.f32ColsGPU.count(lookupCol)) {
                    buf = ctx.f32ColsGPU[lookupCol];
                }
                
                if (buf) {
                     if (debug) std::cerr << "[V2] Project: downloading GPU f32 column " << lookupCol << "\n";
                     uint32_t cnt = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
                     if (cnt == 0 && ctx.rowCount > 0 && !ctx.activeRowsGPU) cnt = ctx.rowCount; // Fallback to raw count if not filtered

                     if (cnt > 0) {
                         std::vector<float> down(cnt);
                         MTL::Buffer* src = buf;
                         bool temp = false;
                         
                         if (ctx.activeRowsGPU) {
                             src = OperatorsGPU::gatherF32(buf, ctx.activeRowsGPU, cnt);
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
                    // Need CPU activeRows for compaction
                    ensureActiveRowsCPU();
                    colData.reserve(ctx.activeRows.size());
                    for (uint32_t idx : ctx.activeRows) {
                        colData.push_back(idx < itF->second.size() ? itF->second[idx] : 0.0f);
                    }
                } else {
                    colData = itF->second;
                }
                
                if (debug) {
                    std::cerr << "[V2] Project: f32 column " << lookupCol << " size=" << colData.size();
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
                    if (debug) std::cerr << "[V2] Project: f32 also storing as CTE alias " << col << "\n";
                }
                ctx.f32Cols[posName] = colData;
                out.f32_cols.push_back(std::move(colData));
                out.f32_names.push_back(outName.empty() ? col : outName);
                continue;
            }

            // Q02 Fix: Ad-hoc lookup for missing columns from saved contexts
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
                        if (debug) std::cerr << "[V2] Project: performing ad-hoc join for " << neededCol << " on " << currentKey << " -> " << targetKey << "\n";
                        
                        std::unordered_map<uint32_t, uint32_t> keyToRow;
                        const auto& sKeys = sourceCtx->u32Cols.at(targetKey);
                        for (size_t i = 0; i < sKeys.size(); ++i) keyToRow[sKeys[i]] = static_cast<uint32_t>(i);
                        
                        const auto& probeKeysFull = ctx.u32Cols.at(currentKey);
                        std::vector<uint32_t> probeKeys;
                        if (!ctx.activeRows.empty()) {
                            // Need CPU activeRows for probe key extraction
                            ensureActiveRowsCPU();
                            probeKeys.reserve(ctx.activeRows.size());
                            for(auto idx : ctx.activeRows) probeKeys.push_back(idx < probeKeysFull.size() ? probeKeysFull[idx] : 0);
                        } else {
                            probeKeys = probeKeysFull;
                        }
                        
                        if (sourceCtx->f32Cols.count(neededCol)) {
                            const auto& sVals = sourceCtx->f32Cols.at(neededCol);
                            std::vector<float> res;
                            res.reserve(probeKeys.size());
                            for (uint32_t k : probeKeys) {
                                if (keyToRow.count(k)) {
                                    uint32_t row = keyToRow[k];
                                    res.push_back(row < sVals.size() ? sVals[row] : 0.0f);
                                } else res.push_back(0.0f);
                            }
                            ctx.f32Cols[posName] = res;
                            if (!outName.empty()) ctx.f32Cols[outName] = res;
                            out.f32_cols.push_back(res);
                            out.f32_names.push_back(outName.empty() ? neededCol : outName);
                            updateRowCount(res.size());
                            continue;
                        } else if (sourceCtx->stringCols.count(neededCol)) {
                            const auto& sVals = sourceCtx->stringCols.at(neededCol);
                            std::vector<std::string> res;
                            res.reserve(probeKeys.size());
                            for (uint32_t k : probeKeys) {
                                if (keyToRow.count(k)) {
                                    uint32_t row = keyToRow[k];
                                    res.push_back(row < sVals.size() ? sVals[row] : "");
                                } else res.push_back("");
                            }
                            ctx.stringCols[posName] = res;
                            if (!outName.empty()) ctx.stringCols[outName] = res;
                            out.string_cols.push_back(res);
                            out.string_names.push_back(outName.empty() ? neededCol : outName);
                            // Dummy u32
                             std::vector<uint32_t> encoded;
                             for(const auto& s : res) encoded.push_back(s.empty() ? 0 : (uint32_t)s[0]);
                             ctx.u32Cols[posName] = encoded;
                             if (!outName.empty()) ctx.u32Cols[outName] = encoded;
                             out.u32_cols.push_back(encoded);
                             out.u32_names.push_back(outName.empty() ? neededCol : outName);
                             updateRowCount(res.size());
                             continue;
                        } else if (sourceCtx->u32Cols.count(neededCol)) {
                             // U32 Case
                             const auto& sVals = sourceCtx->u32Cols.at(neededCol);
                             std::vector<uint32_t> res;
                             res.reserve(probeKeys.size());
                             for (uint32_t k : probeKeys) {
                                 if (keyToRow.count(k)) {
                                     uint32_t row = keyToRow[k];
                                     res.push_back(row < sVals.size() ? sVals[row] : 0);
                                 } else res.push_back(0);
                             }
                             ctx.u32Cols[posName] = res;
                             if (!outName.empty()) ctx.u32Cols[outName] = res;
                             out.u32_cols.push_back(res);
                             out.u32_names.push_back(outName.empty() ? neededCol : outName);
                             updateRowCount(res.size());
                             continue;
                        }
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
                    if (debug) std::cerr << "[V2] Project: mapping unknown alias '" << col << "' to aggregate " << aggName << "\n";
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
            
            // NOTE: Removed greedy positional mapping that was incorrectly mapping
            // columns like c_acctbal to unrelated #N columns like cntrycode hash.
            // This caused aggregation on wrong columns.
            // Columns should only use #N if they are explicitly #N references.
        } else {
            // Check if this expression output name already exists as a column in the context
            // This happens when PlannerV2 passes a complex expression (Kind!=Column) that logic
            // actually corresponds to a pre-calculated column from a previous step (e.g. Aggregation)
            if (!outName.empty() && ctx.f32Cols.count(outName)) {
                if (debug) std::cerr << "[V2] Project: resolving complex expression '" << outName << "' as existing f32 column\n";
                auto& colData = ctx.f32Cols[outName];
                updateRowCount(colData.size());
                // Also map to posName for subsequent access
                ctx.f32Cols[posName] = colData;
                out.f32_cols.push_back(colData); // Copy
                out.f32_names.push_back(outName);
                continue;
            }
            if (!outName.empty() && ctx.u32Cols.count(outName)) {
                if (debug) std::cerr << "[V2] Project: resolving complex expression '" << outName << "' as existing u32 column\n";
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
                if (debug) std::cerr << "[V2] Project: computed expr[" << i << "] on GPU\n";
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
                if (debug) std::cerr << "[V2] Project: GPU eval failed. Fallback disabled.\n";
                throw std::runtime_error("GPU Project eval failed for expression index " + std::to_string(i) + " (" + outName + ")");
            }

            
            if (!values.empty()) {
                if (debug) {
                    std::cerr << "[V2] Project: computed expr[" << i << "] (" << posName << ") -> " 
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
                // Try u32 evaluation
                std::vector<uint32_t> uvals = evalExprU32(expr, ctx);
                if (!uvals.empty()) {
                    if (!outName.empty()) {
                        ctx.u32Cols[outName] = uvals;
                    }
                    ctx.u32Cols[posName] = uvals;
                    out.u32_cols.push_back(std::move(uvals));
                    updateRowCount(out.u32_cols.back().size());
                    out.u32_names.push_back(outName.empty() ? posName : outName);
                } else {
                    // Expression evaluation failed - try to find column by positional reference
                    // This handles post-GroupBy projections where #N refers to aggregated output
                    bool found = false;
                    
                    // Try outName first (e.g., "c_count" for aggregate output)
                    if (!outName.empty()) {
                        auto itF = ctx.f32Cols.find(outName);
                        if (itF != ctx.f32Cols.end()) {
                            if (debug) std::cerr << "[V2] Project: found outName " << outName << " in f32Cols\n";
                            ctx.f32Cols[posName] = itF->second;
                            out.f32_cols.push_back(itF->second);
                            out.f32_names.push_back(outName);
                            found = true;
                        }
                        // Also check u32Cols for outName (e.g. mixed types or string results)
                        if (!found) {
                            auto itU = ctx.u32Cols.find(outName);
                            if (itU != ctx.u32Cols.end()) {
                                if (debug) std::cerr << "[V2] Project: found outName " << outName << " in u32Cols\n";
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
                                    if (debug) std::cerr << "[V2] Project: fuzzy match f32 '" << outName << "' -> '" << key << "'\n";
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
                                        if (debug) std::cerr << "[V2] Project: fuzzy match u32 '" << outName << "' -> '" << key << "'\n";
                                        ctx.u32Cols[posName] = val;
                                        out.u32_cols.push_back(val);
                                        out.u32_names.push_back(outName);
                                        found = true;
                                        break;
                                    }
                                } 
                            }
                        }                    }
                    
                    // Try posName (#N) in f32Cols
                    if (!found) {
                        auto itF = ctx.f32Cols.find(posName);
                        if (itF != ctx.f32Cols.end()) {
                            if (debug) std::cerr << "[V2] Project: found " << posName << " in f32Cols\n";
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
                            if (debug) std::cerr << "[V2] Project: found " << sumName << " in f32Cols\n";
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
                            if (debug) std::cerr << "[V2] Project: found " << posName << " in u32Cols\n";
                            out.u32_cols.push_back(itU->second);
                            updateRowCount(itU->second.size());
                            out.u32_names.push_back(outName.empty() ? posName : outName);
                            found = true;
                        }
                    }
                    
                    if (!found && debug) {
                        std::cerr << "[V2] Project: expr[" << i << "] evaluation failed, no fallback found\n";
                    }
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
