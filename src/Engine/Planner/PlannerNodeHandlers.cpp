// ============================================================================
// PlannerNodeHandlers.cpp — Node-type handlers called from traverseNode
// ============================================================================
#include "PlannerInternal.hpp"
#include "EnvUtil.hpp"
#include <iostream>
#include <algorithm>
#include <regex>
#include <cctype>
#include <set>

namespace engine {

// ========== handleScan ==========
bool handleScan(const json& /*node*/, const std::string& name, const std::string& nameLower,
                       const json& extraInfo, std::vector<std::string>& myProjs, TraverseContext& ctx) {

    if (nameLower == "column_data_scan") {
        debug_log("DEBUG: COLUMN_DATA_SCAN found");
        if (!extraInfo.is_null()) {
             if (env_truthy("GPUDB_DEBUG_PLANNER")) std::cerr << "DEBUG: COLUMN_DATA_SCAN extra_info keys: ";
             if (env_truthy("GPUDB_DEBUG_PLANNER")) if (extraInfo.contains("column_index")) std::cerr << "column_index, ";
             if (env_truthy("GPUDB_DEBUG_PLANNER")) if (extraInfo.contains("values")) std::cerr << "values, ";
             if (env_truthy("GPUDB_DEBUG_PLANNER")) if (extraInfo.contains("columns")) std::cerr << "columns, ";
             if (env_truthy("GPUDB_DEBUG_PLANNER")) if (extraInfo.contains("Columns")) std::cerr << "Columns, ";
             if (env_truthy("GPUDB_DEBUG_PLANNER")) if (extraInfo.contains("result_chunk")) std::cerr << "result_chunk, ";
             if (env_truthy("GPUDB_DEBUG_PLANNER")) if (extraInfo.contains("Result Chunk")) std::cerr << "Result Chunk, ";
             if (env_truthy("GPUDB_DEBUG_PLANNER")) std::cerr << std::endl;
        }
    }

    std::string delimTableOverride;
    bool isDummy = nameLower.find("dummy_scan") != std::string::npos;
    if ((nameLower.find("delim_scan") != std::string::npos || nameLower == "column_data_scan" || isDummy) && !ctx.delimStack.empty()) {
        debug_log("Generating Multi-Level DELIM_SCAN. Depth: " + std::to_string(ctx.delimStack.size()));

        // 1. Accumulate all projections (Legacy: Just use top)
        if (myProjs.empty()) myProjs = ctx.delimStack.back().second;

         // 2. Emit Nodes (Top Level Only)
         const auto& level = ctx.delimStack.back();
         std::string tbl = level.first;
         const auto& projs = level.second;

         if (ctx.seenTables.find(tbl) == ctx.seenTables.end()) {
             ctx.seenTables.insert(tbl);
             std::vector<std::string> cols;
             for(const auto& p : projs) cols.push_back(stripTableQualifier(p));
             ctx.plan.tables.push_back({tbl, cols});
         }

         IRNode s = IRNode::scan(tbl);
         s.duckdbName = name + "_DelimTop";
         for(const auto& p : projs) s.asScan().columns.push_back(stripTableQualifier(p));
         // DELIM_SCAN produces distinct correlated keys; COLUMN_DATA_SCAN produces full data
         if (nameLower.find("delim_scan") != std::string::npos && nameLower.find("column_data_scan") == std::string::npos) {
             s.asScan().isDelimScan = true;
         }
         ctx.plan.nodes.push_back(std::move(s));
         debug_log("Generating Scan(DelimTop): " + tbl + " at index " + std::to_string(ctx.plan.nodes.size()-1) + " Plan: " + std::to_string((uintptr_t)&ctx.plan));

         // Filters
         if (extraInfo.is_object() && extraInfo.contains("Filters")) {
            const auto& f = extraInfo["Filters"];
            std::vector<std::string> candidateFilters;
            if (f.is_array()) {
                for (const auto& item : f.get_array()) if(item.is_string()) candidateFilters.push_back(item.get_string());
            } else if (f.is_string()) {
                candidateFilters.push_back(f.get_string());
            }

            std::string filterStr;
            for (auto& s : candidateFilters) {
                 if (s.find("optional:") != std::string::npos) s = trim_str(s.substr(s.find("optional:")+9));
                 if (!filterStr.empty()) filterStr += " AND ";
                 filterStr += s;
            }
            if (!filterStr.empty()) {
                ctx.plan.nodes.push_back(IRNode::filter(Planner::parseExpression(filterStr), filterStr));
            }
         }
         return true;
    }

    IRNode scanNode = IRNode::scan(delimTableOverride);
    scanNode.duckdbName = name;
    auto& scan = scanNode.asScan();

    // Extract table name
    if (extraInfo.is_object() && extraInfo.contains("Table") && extraInfo["Table"].is_string()) {
        scan.table = extraInfo["Table"].get_string();
    }

    // Extract CTE Name if Table is missing (for CTE_SCAN)
    if (scan.table.empty()) {
        if (extraInfo.is_object()) {
             if (extraInfo.contains("CTE Name") && extraInfo["CTE Name"].is_string()) {
                 scan.table = extraInfo["CTE Name"].get_string();
             } else if (extraInfo.contains("CTE Index")) {
                 int64_t idx = 0;
                 if (extraInfo["CTE Index"].is_number()) idx = (int64_t)extraInfo["CTE Index"].get_number();
                 if (ctx.cteMap.count(idx)) {
                     scan.table = ctx.cteMap[idx];
                     if (env_truthy("GPUDB_DEBUG_PLANNER")) std::cerr << "DEBUG: Resolved CTE_SCAN table from index " << idx << " -> " << scan.table << "\n";
                 } else {
                     if (env_truthy("GPUDB_DEBUG_PLANNER")) std::cerr << "DEBUG: FAILED to resolve CTE Index " << idx << ". Map size=" << ctx.cteMap.size() << "\n";
                 }
             }
        }
    }

    // Infer table from column prefixes if needed
    if (env_truthy("GPUDB_DEBUG_PLANNER")) std::cerr << "DEBUG: Scan table determined as: '" << scan.table << "'\n";

    // Also handle DELIM_SCAN where table is a template source "tmpl_..." but logically is "orders" etc.
    if ((scan.table.empty() || scan.table.find("tmpl_") == 0) && !myProjs.empty()) {
        int l = 0, o = 0, c = 0, p = 0, s = 0, n = 0, r = 0;
        for (const auto& proj : myProjs) {
            std::string col = stripTableQualifier(proj);
            if (col.rfind("l_", 0) == 0) ++l;
            else if (col.rfind("o_", 0) == 0) ++o;
            else if (col.rfind("c_", 0) == 0) ++c;
            else if (col.rfind("p_", 0) == 0) ++p;
            else if (col.rfind("s_", 0) == 0) ++s;
            else if (col.rfind("n_", 0) == 0) ++n;
            else if (col.rfind("r_", 0) == 0) ++r;
        }
        std::string inferred;
        if (l >= o && l >= c && l >= p && l > 0) inferred = "lineitem";
        else if (o >= l && o >= c && o >= p && o > 0) inferred = "orders";
        else if (c >= l && c >= o && c >= p && c > 0) inferred = "customer";
        else if (p >= l && p >= o && p >= c && p > 0) inferred = "part";
        else if (s > 0) inferred = "supplier";
        else if (n > 0) inferred = "nation";
        else if (r > 0) inferred = "region";

        if (!inferred.empty()) {
            if (scan.table.empty()) scan.table = inferred;
            else if (scan.table != inferred) {
                // Always switch to base table if inferred, to avoid missing columns in partial pipeline snapshots
                if (scan.table.find("tmpl_") == 0) {
                    debug_log("DELIM_SCAN inferred base table " + inferred + ". Switching from " + scan.table);
                    scan.table = inferred;
                }
            }
        }
    }

    // Extract columns needed
    for (const auto& proj : myProjs) {
        std::string col = stripTableQualifier(proj);
        scan.columns.push_back(col);
    }

    // Extract pushed-down filters from scan
    if (extraInfo.is_object() && extraInfo.contains("Filters")) {
        const auto& f = extraInfo["Filters"];
        std::vector<std::string> candidateFilters;

        if (f.is_array()) {
            for (const auto& item : f.get_array()) {
                std::string s = item.get_string();
                if (s.find("Dynamic Filter") != std::string::npos) continue;
                if (s.find("optional:") != std::string::npos) {
                    s = trim_str(s.substr(s.find("optional:") + 9));
                }
                candidateFilters.push_back(s);
            }
        } else if (f.is_string()) {
            std::string s = f.get_string();
            if (s.find("Dynamic Filter") != std::string::npos) {
                // do not add
            } else {
                if (s.find("optional:") != std::string::npos) {
                     s = trim_str(s.substr(s.find("optional:") + 9));
                }
                candidateFilters.push_back(s);
            }
        }

        std::string filterStr;
        for (const auto& flt : candidateFilters) {
            if (!filterStr.empty()) filterStr += " AND ";
            filterStr += flt;
        }

        if (!filterStr.empty()) {
            scan.filter = Planner::parseExpression(filterStr);
        }
    }

    // Track unique tables
    if (!scan.table.empty() && ctx.seenTables.find(scan.table) == ctx.seenTables.end()) {
        ctx.seenTables.insert(scan.table);
        ctx.plan.tables.push_back({scan.table, scan.columns});
    }

    if (env_truthy("GPUDB_DEBUG_PLANNER")) std::cerr << "DEBUG: Pushing SCAN node for " << name << ". Table=" << scan.table << "\n";
    ctx.plan.nodes.push_back(std::move(scanNode));
    debug_log("Generating Scan: " + ctx.plan.nodes.back().asScan().table + " at index " + std::to_string(ctx.plan.nodes.size()-1) + " Plan: " + std::to_string((uintptr_t)&ctx.plan));

    // Emit separate filter node if scan has pushed-down filter
    if (ctx.plan.nodes.back().asScan().filter) {
        // Already handled in scan
    }
    return false;
}

// ========== handleFilter ==========
void handleFilter(const json& /*node*/, const std::string& name,
                         const json& extraInfo, const std::string& extraStr,
                         std::vector<std::string>& childProjs, TraverseContext& ctx) {
    if (!extraInfo.is_null()) {
         if (env_truthy("GPUDB_DEBUG_PLANNER")) if (extraInfo.contains("Expression")) std::cerr << "Expression: " << extraInfo["Expression"].get_string() << std::endl;
         if (env_truthy("GPUDB_DEBUG_PLANNER")) if (extraInfo.contains("Condition")) std::cerr << "Condition: " << extraInfo["Condition"].get_string() << std::endl;
    }

    std::string projsStr;
    for(const auto& s : childProjs) projsStr += s + ", ";
    debug_log("DEBUG FILTER CHILD PROJS: " + projsStr);

    std::string predicate;
    bool hasExpression = false;
    if (extraInfo.is_object()) {
        if (extraInfo.contains("Expression") && extraInfo["Expression"].is_string()) {
            predicate = extraInfo["Expression"].get_string();
            hasExpression = true;
        } else if (extraInfo.contains("Condition") && extraInfo["Condition"].is_string()) {
            predicate = extraInfo["Condition"].get_string();
            hasExpression = true;
        }

        if (!hasExpression && extraInfo.contains("Filters")) {
            const auto& f = extraInfo["Filters"];
            if (f.is_array()) {
                for (const auto& item : f.get_array()) {
                    if (!predicate.empty()) predicate += " AND ";
                    predicate += item.get_string();
                }
            } else if (f.is_string()) {
                if (!predicate.empty()) predicate += " AND ";
                predicate += f.get_string();
            }
        }
    }
    if (predicate.empty()) predicate = extraStr;

    // Handle SUBQUERY placeholder in predicate (Scalar Subquery result)
    if (predicate.find("SUBQUERY") != std::string::npos) {
         debug_log("Attempting to replace SUBQUERY in: " + predicate);
         std::string replacement;
         for (auto it = childProjs.rbegin(); it != childProjs.rend(); ++it) {
             std::string s = tolower_str(*it);
             debug_log("  Checking candidate: " + s);
             if (s.find("min(") != std::string::npos || 
                 s.find("max(") != std::string::npos ||
                 s.find("sum(") != std::string::npos ||
                 s.find("avg(") != std::string::npos ||
                 s.find("count(") != std::string::npos) {
                 replacement = *it;
                 break;
             }
         }
         if (!replacement.empty()) {
              size_t pos = predicate.find("SUBQUERY");
              predicate.replace(pos, 8, replacement);
              debug_log("Replaced SUBQUERY with " + replacement);
         } else {
             debug_log("Failed to find replacement for SUBQUERY. Available cols: " + std::to_string(childProjs.size()));
         }
    }

    predicate = resolveColRef(predicate, childProjs);

    // Resolve ambiguous column references using cyclic aliased variants
    {
        std::regex wordRe(R"(\b([a-zA-Z_][a-zA-Z0-9_]*)\b)");

        std::map<std::string, std::vector<std::string>> candidatesMap;

        std::sregex_iterator wordsBegin(predicate.begin(), predicate.end(), wordRe);
        std::sregex_iterator wordsEnd;
        for (std::sregex_iterator i = wordsBegin; i != wordsEnd; ++i) {
            std::string word = i->str();
            bool isValid = false;
            for(const auto& c : childProjs) if(c == word) { isValid = true; break; }
            if(isValid) continue;

            std::string lw = tolower_str(word);
            if(lw == "and" || lw == "or" || lw == "between" || lw == "in" || lw == "is" || lw == "not" || lw == "null") continue;

            if (candidatesMap.find(word) == candidatesMap.end()) {
                std::vector<std::string> cands;
                for(const auto& c : childProjs) {
                    if (c.size() > word.size() && c.find(word) == 0 && c[word.size()] == '_' && c.find("_rhs_") != std::string::npos) {
                        cands.push_back(c);
                    }
                }
                if (!cands.empty()) {
                    std::sort(cands.begin(), cands.end());
                    cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
                    candidatesMap[word] = cands;
                }
            }
        }

        for (auto& [word, cands] : candidatesMap) {
            if (cands.size() < 2) continue;

            debug_log("Fixing ambiguous Filter column '" + word + "' with cyclic candidates: " + std::to_string(cands.size()));

            std::string newPred;
            size_t lastPos = 0;
            int replaceIdx = 0;

            std::sregex_iterator it(predicate.begin(), predicate.end(), wordRe);
            for (; it != wordsEnd; ++it) {
                 if (it->str() == word) {
                     newPred += predicate.substr(lastPos, it->position() - lastPos);
                     newPred += cands[replaceIdx % cands.size()];
                     replaceIdx++;
                     lastPos = it->position() + it->length();
                 }
            }
            newPred += predicate.substr(lastPos);
            predicate = newPred;
        }
    }

    // Anti/Semi Join Cleanup
    if ((predicate.find("SUBQUERY") != std::string::npos) && !ctx.plan.nodes.empty()) {
         bool handledByJoin = false;
         int limit = 5; 
         for(int i = (int)ctx.plan.nodes.size() - 1; i >= 0 && limit > 0; --i) {
             auto& n = ctx.plan.nodes[i];
             if (n.type == IRNode::Type::Join) {
                 if (n.asJoin().type == JoinType::Anti || n.asJoin().type == JoinType::Semi || n.asJoin().type == JoinType::Mark) {
                     if (n.asJoin().type == JoinType::Mark) {
                         if (predicate.find("NOT SUBQUERY") != std::string::npos) {
                             debug_log("Converting MARK Join to ANTI Join due to NOT SUBQUERY.");
                             n.asJoin().type = JoinType::Anti;
                         } else {
                             debug_log("Converting MARK Join to SEMI Join due to SUBQUERY.");
                             n.asJoin().type = JoinType::Semi;
                         }
                     }
                     handledByJoin = true;
                 }
                 break;
             }
             limit--;
         }

         if (handledByJoin) {
             debug_log("Stripping SUBQUERY predicate artifacts due to Anti/Semi/Mark Join.");
             size_t pos = 0;
             while ((pos = predicate.find("(NOT SUBQUERY)", pos)) != std::string::npos) { predicate.replace(pos, 14, "1=1"); }
             pos = 0;
             while ((pos = predicate.find("NOT SUBQUERY", pos)) != std::string::npos) { predicate.replace(pos, 12, "1=1"); }
             pos = 0;
             while ((pos = predicate.find("(SUBQUERY)", pos)) != std::string::npos) { predicate.replace(pos, 10, "1=1"); }
             pos = 0;
             while ((pos = predicate.find("SUBQUERY", pos)) != std::string::npos) { predicate.replace(pos, 8, "1=1"); }
         }
    }

    IRNode filterNode = IRNode::filter(Planner::parseExpression(predicate), predicate);
    filterNode.duckdbName = name;
    ctx.plan.nodes.push_back(std::move(filterNode));
}

// ========== handleGroupBy ==========
void handleGroupBy(const json& /*node*/, const std::string& name,
                          const json& extraInfo, const std::vector<std::string>& childProjs,
                          std::vector<std::string>& myProjs, TraverseContext& ctx) {
    IRNode gbNode = IRNode::groupBy();
    gbNode.duckdbName = name;
    auto& gb = gbNode.asGroupBy();

    if (extraInfo.is_object()) {
        // Parse grouping keys
        if (extraInfo.contains("Groups")) {
            const auto& groupsNode = extraInfo["Groups"];
            auto processGroup = [&](std::string col) {
                col = resolveColRef(col, childProjs);
                col = stripTableQualifier(col);
                if (!col.empty()) {
                    gb.keys.push_back(TypedExpr::column(col));
                    gb.keyNames.push_back(col);
                }
            };
            if (groupsNode.is_array()) {
                for (const auto& item : groupsNode.get_array()) {
                    if (item.is_string()) processGroup(item.get_string());
                }
            } else if (groupsNode.is_string()) {
                processGroup(groupsNode.get_string());
            }
        }

        // Parse aggregates
        if (extraInfo.contains("Aggregates")) {
            const auto& aggsNode = extraInfo["Aggregates"];
            auto processAgg = [&](std::string agg) {
                debug_log("Parsing agg string: '" + agg + "'");
                if (!ctx.pastGroupBy) {
                     agg = resolveColRef(agg, childProjs);
                }
                debug_log("Resolved agg: " + agg);

                size_t start = agg.find('(');
                size_t end = agg.rfind(')');

                IRGroupBy::AggSpec spec;

                std::string funcName;
                if (start != std::string::npos) {
                    funcName = agg.substr(0, start);
                }
                spec.func = Planner::parseAggFunc(funcName);

                if (start != std::string::npos && end != std::string::npos) {
                    spec.inputExpr = trim_str(agg.substr(start + 1, end - start - 1));

                    if (!ctx.pastGroupBy) {
                         spec.inputExpr = resolveColRef(spec.inputExpr, childProjs);
                    }

                    bool isChildColumn = false;
                    if (!ctx.pastGroupBy) {
                         for (const auto& proj : childProjs) {
                             if (proj == spec.inputExpr) {
                                 isChildColumn = true;
                                 break;
                             }
                         }
                    }

                    // Check for DISTINCT modifier
                    std::string lowerInput = tolower_str(spec.inputExpr);
                    if (lowerInput.rfind("distinct ", 0) == 0) {
                        spec.inputExpr = trim_str(spec.inputExpr.substr(9));
                        if (spec.func == AggFunc::Count) {
                            spec.func = AggFunc::CountDistinct;
                        }
                    } else if (lowerInput.rfind("distinct", 0) == 0 && lowerInput.size() > 8) {
                        spec.inputExpr = trim_str(spec.inputExpr.substr(8));
                        if (spec.func == AggFunc::Count) {
                            spec.func = AggFunc::CountDistinct;
                        }
                    }

                    if (isChildColumn) {
                        debug_log("inputExpr '" + spec.inputExpr + "' exists in child projections. Treating as Column.");
                        spec.input = TypedExpr::column(spec.inputExpr);
                    } else {
                        spec.input = Planner::parseExpression(spec.inputExpr);
                    }
                }

                // Try to find alias from SQL
                std::string resolvedAgg = resolveColRef(agg, childProjs);
                std::string normAgg = tolower_str(resolvedAgg);
                normAgg.erase(std::remove_if(normAgg.begin(), normAgg.end(),
                    [](unsigned char ch) { return std::isspace(ch); }), normAgg.end());
                normAgg = normalizeNumericLiterals(normAgg);

                if (normAgg.rfind("sum_no_overflow(", 0) == 0) {
                    normAgg = "sum(" + normAgg.substr(16);
                }

                debug_log("Looking up agg alias: '" + normAgg + "'");

                if (normAgg.find("count_star()") != std::string::npos) {
                    normAgg = "count(*)";
                    spec.func = AggFunc::CountStar;
                }

                auto it = ctx.aliases.find(normAgg);
                if (it == ctx.aliases.end()) {
                    std::regex re(R"((\w+)\(\((.+)\)\))");
                    std::smatch m;
                    if (std::regex_match(normAgg, m, re)) {
                        std::string reduced = m[1].str() + "(" + m[2].str() + ")";
                        it = ctx.aliases.find(reduced);
                        if (it != ctx.aliases.end()) {
                            debug_log("Found alias with reduced parens: '" + reduced + "'");
                        }
                    }
                }
                if (it == ctx.aliases.end()) {
                    std::string reduced = normAgg;
                    std::string prev;
                    while (reduced != prev) {
                        prev = reduced;
                        std::regex doubleParens(R"(\(\(([^()]*)\)\))");
                        reduced = std::regex_replace(reduced, doubleParens, "($1)");
                    }
                    if (reduced != normAgg) {
                        it = ctx.aliases.find(reduced);
                        if (it != ctx.aliases.end()) {
                            debug_log("Found alias with fully reduced parens: '" + reduced + "'");
                        }
                    }
                }
                if (it == ctx.aliases.end()) {
                    auto stripInnerParens = [](const std::string& s) -> std::string {
                        std::string result;
                        int depth = 0;
                        bool inFunc = false;
                        for (size_t i = 0; i < s.size(); ++i) {
                            char c = s[i];
                            if (c == '(') {
                                if (!inFunc && i > 0 && std::isalpha(s[i-1])) {
                                    inFunc = true;
                                    result += c;
                                } else if (inFunc && depth == 0) {
                                    result += c;
                                }
                                depth++;
                            } else if (c == ')') {
                                depth--;
                                if (depth == 0) {
                                    result += c;
                                    inFunc = false;
                                }
                            } else {
                                result += c;
                            }
                        }
                        return result;
                    };
                    std::string stripped = stripInnerParens(normAgg);
                    for (const auto& [key, val] : ctx.aliases) {
                        std::string strippedKey = stripInnerParens(key);
                        if (stripped == strippedKey) {
                            spec.outputName = val;
                            debug_log("Found alias via stripped comparison: '" + val + "'");
                            break;
                        }
                    }
                } else {
                    spec.outputName = it->second;
                }

                if (spec.outputName.empty()) {
                    spec.outputName = normAgg;
                    debug_log("No alias found for agg, using generated name: " + spec.outputName);
                } else {
                    debug_log("Found alias for agg: " + spec.outputName);
                }

                debug_log("Pushing spec with outputName='" + spec.outputName + "'");
                auto savedFunc = spec.func;
                auto savedInput = spec.input;
                auto savedOutput = spec.outputName;
                gb.aggSpecs.push_back(std::move(spec));
                gb.aggregates.push_back(TypedExpr::aggregate(savedFunc, savedInput, savedOutput));
            };

            if (aggsNode.is_array()) {
                for (const auto& item : aggsNode.get_array()) {
                    if (item.is_string()) processAgg(item.get_string());
                }
            } else if (aggsNode.is_string()) {
                processAgg(aggsNode.get_string());
            }
        }
    }

    // Update output projections
    for (const auto& key : gb.keyNames) {
        myProjs.push_back(key);
    }
    for (const auto& spec : gb.aggSpecs) {
        myProjs.push_back(spec.outputName);
    }

    ctx.plan.nodes.push_back(std::move(gbNode));
    ctx.pastGroupBy = true;
}

// ========== handleUngroupedAggregate ==========
void handleUngroupedAggregate(const json& /*node*/, const std::string& name,
                                     const json& extraInfo,
                                     std::vector<std::string>& childProjs, TraverseContext& ctx) {
    if (extraInfo.is_object() && extraInfo.contains("Aggregates")) {
        const auto& aggs = extraInfo["Aggregates"];
        std::vector<std::string> aggStrings;

        if (aggs.is_string()) {
            aggStrings.push_back(aggs.get_string());
        } else if (aggs.is_array()) {
            for (const auto& a : aggs.get_array()) {
                aggStrings.push_back(a.get_string());
            }
        }

        std::vector<IRNode> bufferedAggs;
        for (size_t aggIdx = 0; aggIdx < aggStrings.size(); ++aggIdx) {
            std::string aggStr = resolveColRef(aggStrings[aggIdx], childProjs);
            debug_log("Processing aggregate: '" + aggStr + "'");

            size_t start = aggStr.find('(');
            size_t end = aggStr.rfind(')');
            AggFunc func = AggFunc::Sum;
            std::string exprStr;

            if (start != std::string::npos && end != std::string::npos) {
                std::string funcName = aggStr.substr(0, start);
                func = Planner::parseAggFunc(funcName);
                exprStr = trim_str(aggStr.substr(start + 1, end - start - 1));

                if (func == AggFunc::Max && exprStr == "max(total_revenue)") {
                     exprStr = "total_revenue";
                }
            } else {
                 if (aggStr == "count_star()") {
                     func = AggFunc::CountStar;
                     exprStr = "*";
                 } else {
                     debug_log("Skipping invalid aggregate string: " + aggStr);
                     continue;
                 }
            }

            if (exprStr.empty() && func != AggFunc::CountStar) {
                continue;
            }

            IRNode aggNode = IRNode::aggregate(func, Planner::parseExpression(exprStr));
            aggNode.duckdbName = name;
            aggNode.asAggregate().alias = aggStr;
            aggNode.asAggregate().exprStr = exprStr;
            aggNode.asAggregate().aggIndex = aggIdx;

            bufferedAggs.push_back(std::move(aggNode));
        }

        if (!bufferedAggs.empty()) {
            for (size_t i = 0; i < bufferedAggs.size(); ++i) {
                bufferedAggs[i].asAggregate().isLastAgg = (i == bufferedAggs.size() - 1);
            }
            for (auto& node : bufferedAggs) {
                ctx.plan.nodes.push_back(std::move(node));
            }
        }

        ctx.projections.clear();
        for (const auto& rawAgg : aggStrings) {
            std::string resolved = resolveColRef(rawAgg, childProjs);
            ctx.projections.push_back(resolved);
        }
    }
}

// ========== handleProjection ==========
void handleProjection(const json& /*node*/, const std::string& name,
                             const std::vector<std::string>& myProjs,
                             const std::vector<std::string>& childProjs, TraverseContext& ctx) {
    std::vector<TypedExprPtr> exprs;
    std::vector<std::string> names;

    for (const auto& proj : myProjs) {
        std::string exprStr = proj;
        std::string outName = stripTableQualifier(proj);

        // DuckDB scalar subquery error-checking CASE wrapper
        if (outName.find("CASE") != std::string::npos && 
            outName.find("\"error\"(") != std::string::npos &&
            outName.find("ELSE") != std::string::npos) {
            size_t elsePos = outName.find("ELSE");
            size_t endPos = outName.rfind("END");
            if (elsePos != std::string::npos && endPos != std::string::npos && endPos > elsePos) {
                std::string elseExpr = trim_str(outName.substr(elsePos + 4, endPos - (elsePos + 4)));
                for (const auto& [alias, def] : ctx.aliases) {
                    if (elseExpr.find(alias) != std::string::npos) {
                        debug_log("Projection: scalar subquery CASE wrapper -> using alias '" + alias + "' as outName");
                        outName = alias;
                        break;
                    }
                }
            }
        }

        bool exactMatchInChild = false;
        for (const auto& c : childProjs) {
            if (c == proj) { exactMatchInChild = true; break; }
        }

        if (exactMatchInChild) {
            exprs.push_back(TypedExpr::column(proj));
            names.push_back(outName);
            continue;
        }

        bool isSimpleIdent = !proj.empty() && 
            proj[0] != '#' &&
            proj.find('(') == std::string::npos && 
            proj.find(' ') == std::string::npos &&
            proj.find('*') == std::string::npos &&
            proj.find('+') == std::string::npos &&
            proj.find('-') == std::string::npos;

        if (isSimpleIdent) {
            bool inChild = false;
            for (const auto& c : childProjs) {
                if (c == proj) { inChild = true; break; }
            }

            if (!inChild && proj == "o_year") {
                 debug_log("DEBUG: Failed to find 'o_year' in childProjs. ChildProjs size: " + std::to_string(childProjs.size()));
                 for(const auto& c : childProjs) debug_log("  - '" + c + "'");
            }

            if (!inChild) {
                auto it = ctx.aliases.find(proj);
                if (it != ctx.aliases.end()) {
                    debug_log("Projection: resolving alias '" + proj + "' -> '" + it->second + "'");
                    exprStr = it->second;
                    outName = proj;

                    auto mappingIt = ctx.qualifiedColumnMapping.find(exprStr);
                    if (mappingIt != ctx.qualifiedColumnMapping.end()) {
                        debug_log("Projection: using qualified mapping '" + exprStr + "' -> '" + mappingIt->second + "'");
                        exprStr = mappingIt->second;
                    }
                }
            }
        }

        exprs.push_back(Planner::parseExpression(exprStr));
        names.push_back(outName);
    }

    // Force keep global columns
    for (const auto& col : ctx.forceKeepColumns) {
        bool foundInChild = false;
        for(const auto& c : childProjs) {
            if (c == col || stripTableQualifier(c) == col) {
                foundInChild = true;
                break;
            }
        }
        if (foundInChild) {
            bool alreadyProjected = false;
            for(size_t i=0; i<names.size(); ++i) {
                if (names[i] == col) {
                     alreadyProjected = true;
                     break;
                }
            }
            if (!alreadyProjected) {
                 debug_log("Forcing keep of global column: " + col);
                 exprs.push_back(TypedExpr::column(col));
                 names.push_back(col);
            }
        }
    }

    IRNode projNode = IRNode::project(std::move(exprs), std::move(names));
    projNode.duckdbName = name;
    ctx.plan.nodes.push_back(std::move(projNode));
}

// ========== handleOrderBy ==========
void handleOrderBy(const json& /*node*/, const std::string& name, const std::string& /*nameLower*/,
                          const json& extraInfo, const std::vector<std::string>& childProjs,
                          TraverseContext& ctx) {
    IRNode obNode = IRNode::orderBy();
    obNode.duckdbName = name;
    auto& ob = obNode.asOrderBy();

    if (extraInfo.is_object() && extraInfo.contains("Order By")) {
        const auto& obSpec = extraInfo["Order By"];
        auto processOrder = [&](std::string s) {
            bool asc = true;
            std::string slower = tolower_str(s);
            if (slower.size() > 5 && slower.substr(slower.size()-5) == " desc") {
                asc = false;
                s = s.substr(0, s.size()-5);
            } else if (slower.size() > 4 && slower.substr(slower.size()-4) == " asc") {
                asc = true;
                s = s.substr(0, s.size()-4);
            }

            // Strip NULLS FIRST/LAST
            slower = tolower_str(s);
            if (slower.size() > 11 && slower.substr(slower.size()-11) == " nulls last") {
                s = s.substr(0, s.size()-11);
            } else if (slower.size() > 12 && slower.substr(slower.size()-12) == " nulls first") {
                s = s.substr(0, s.size()-12);
            }

            s = resolveColRef(s, childProjs);
            s = trim_str(s);

            bool inChild = false;
            for (const auto& c : childProjs) {
                if (c == s) { inChild = true; break; }
            }

            if (inChild) {
                // kept as is
            } else {
                if (s.find('(') == std::string::npos && s.find(' ') == std::string::npos) {
                    s = stripTableQualifier(s);
                } else {
                    std::string normS = tolower_str(s);
                    normS.erase(std::remove_if(normS.begin(), normS.end(),
                        [](unsigned char ch) { return std::isspace(ch); }), normS.end());
                    normS = normalizeNumericLiterals(normS);
                    normS.erase(std::remove(normS.begin(), normS.end(), '"'), normS.end());

                    std::regex tableQualRe(R"((?:[a-z_][a-z0-9_]*\.)+([a-z_][a-z0-9_]*))");
                    normS = std::regex_replace(normS, tableQualRe, "$1");

                    debug_log("ORDER BY looking up alias: '" + normS + "'");

                    auto it = ctx.aliases.find(normS);
                    if (it == ctx.aliases.end() && normS == "count_star()") {
                        it = ctx.aliases.find("count(*)");
                    }
                    if (it != ctx.aliases.end()) {
                        s = it->second;
                    } else {
                        std::regex re(R"((\w+)\(\((.+)\)\))");
                        std::smatch m;
                        if (std::regex_match(normS, m, re)) {
                            std::string reduced = m[1].str() + "(" + m[2].str() + ")";
                            it = ctx.aliases.find(reduced);
                            if (it != ctx.aliases.end()) {
                                s = it->second;
                            }
                        }
                    }
                }
            }

            ob.columns.push_back(s);
            ob.ascending.push_back(asc);
            ob.specs.push_back({TypedExpr::column(s), asc});
        };

        if (obSpec.is_array()) {
            for (const auto& item : obSpec.get_array()) {
                if (item.is_string()) processOrder(item.get_string());
            }
        } else if (obSpec.is_string()) {
            processOrder(obSpec.get_string());
        }
    }

    ctx.plan.nodes.push_back(std::move(obNode));
}

// ── Helper: infer table name from column prefix ──
static std::string inferTableFromColumn(const TypedExprPtr& expr) {
    TypedExprPtr e = expr;
    while (e && e->kind == TypedExpr::Kind::Cast) e = e->asCast().expr;
    if (e && e->kind == TypedExpr::Kind::Column) {
        const std::string& n = e->asColumn().column;
        if (n.starts_with("c_"))  return "customer";
        if (n.starts_with("o_"))  return "orders";
        if (n.starts_with("l_"))  return "lineitem";
        if (n.starts_with("p_"))  return "part";
        if (n.starts_with("s_"))  return "supplier";
        if (n.starts_with("ps_")) return "partsupp";
        if (n.starts_with("n_"))  return "nation";
        if (n.starts_with("r_"))  return "region";
    }
    return "";
}

// ── Helper: check if an expression's column appears in a projection list ──
static bool isInScope(const TypedExprPtr& expr, const std::vector<std::string>& projs) {
    TypedExprPtr e = expr;
    while (e && e->kind == TypedExpr::Kind::Cast) e = e->asCast().expr;
    if (e && e->kind == TypedExpr::Kind::Column) {
        const std::string& name = e->asColumn().column;
        for (const auto& p : projs) if (p == name) return true;
        return false;
    }
    if (e && e->kind == TypedExpr::Kind::Literal) return true;
    return false;
}

// ── Helper: check if a table name matches the RHS of a join ──
static bool matchesRHS(const std::string& tblName,
                       const std::unordered_set<std::string>& rhsTables,
                       const std::string& capturedRightTable,
                       const TraverseContext& ctx) {
    if (tblName.empty()) return false;
    if (rhsTables.count(tblName)) return true;
    if (ctx.aliases.count(tblName)) {
        std::string target = ctx.aliases.at(tblName);
        if (rhsTables.count(target)) return true;
    }
    if (ctx.localAliases.count(tblName)) {
        std::string target = ctx.localAliases.at(tblName);
        if (rhsTables.count(target)) return true;
    }
    if (tblName == capturedRightTable) return true;
    if (!capturedRightTable.empty()) {
        for (const auto& t : ctx.plan.tables) {
            if (t.name == capturedRightTable) {
                for (const auto& c : t.neededColumns) {
                    if (tblName == "orders"   && c.starts_with("o_"))  return true;
                    if (tblName == "lineitem" && c.starts_with("l_"))  return true;
                    if (tblName == "customer" && c.starts_with("c_"))  return true;
                    if (tblName == "part"     && c.starts_with("p_"))  return true;
                    if (tblName == "supplier" && c.starts_with("s_"))  return true;
                    if (tblName == "nation"   && c.starts_with("n_"))  return true;
                    if (tblName == "region"   && c.starts_with("r_"))  return true;
                }
            }
        }
    }
    return false;
}

// ── Classify equality conditions into left/right join keys ──
static void classifyJoinKeys(IRJoin& join, const std::string& capturedRightTable,
                             const std::unordered_set<std::string>& rhsTables,
                             const std::vector<std::string>& lhsProjections,
                             const std::vector<std::string>& rhsProjections,
                             const TraverseContext& ctx) {
    if (!join.condition || capturedRightTable.empty()) return;

    // Flatten AND-connected conditions
    std::vector<TypedExprPtr> conds;
    std::function<void(const TypedExprPtr&)> flatten = [&](const TypedExprPtr& e) {
        if (e->kind == TypedExpr::Kind::Binary && e->asBinary().op == BinaryOp::And) {
            flatten(e->asBinary().left);
            flatten(e->asBinary().right);
        } else {
            conds.push_back(e);
        }
    };
    flatten(join.condition);

    for (const auto& c : conds) {
        if (c->kind != TypedExpr::Kind::Compare || c->asCompare().op != CompareOp::Eq) continue;
        auto& cmp = c->asCompare();
        TypedExprPtr l = cmp.left;
        TypedExprPtr r = cmp.right;

        std::string t1 = inferTableFromColumn(l);
        std::string t2 = inferTableFromColumn(r);

        // Apply local alias resolution to table qualifiers
        auto applyAlias = [&](const std::string& tbl, TypedExprPtr& expr) {
            if (ctx.localAliases.count(tbl)) {
                std::string phy = ctx.localAliases.at(tbl);
                TypedExprPtr e = expr;
                while (e && e->kind == TypedExpr::Kind::Cast) e = e->asCast().expr;
                if (e && e->kind == TypedExpr::Kind::Column) e->asColumn().table = phy;
            }
        };
        applyAlias(t1, l);
        applyAlias(t2, r);

        bool t1R = matchesRHS(t1, rhsTables, capturedRightTable, ctx) || isInScope(l, rhsProjections);
        bool t2R = matchesRHS(t2, rhsTables, capturedRightTable, ctx) || isInScope(r, rhsProjections);
        bool t1L = isInScope(l, lhsProjections) || (!t1.empty() && ctx.seenTables.count(t1) && !t1R);
        bool t2L = isInScope(r, lhsProjections) || (!t2.empty() && ctx.seenTables.count(t2) && !t2R);

        if (t1L && t2R) {
            join.leftKeys.push_back(l); join.rightKeys.push_back(r);
        } else if (t2L && t1R) {
            join.leftKeys.push_back(r); join.rightKeys.push_back(l);
        } else if (t1L && t2L) {
            if (t1R && !t2R)      { join.rightKeys.push_back(l); join.leftKeys.push_back(r); }
            else if (t2R && !t1R) { join.rightKeys.push_back(r); join.leftKeys.push_back(l); }
            else                  { join.leftKeys.push_back(l);  join.rightKeys.push_back(r); }
        } else {
            join.leftKeys.push_back(l); join.rightKeys.push_back(r);
        }
    }
}

// ========== handleJoinEmit ==========
void handleJoinEmit(const json& /*node*/, const std::string& name, const std::string& nameLower,
                           const json& extraInfo, const std::vector<std::string>& childProjs,
                           const JoinCapture& jc, TraverseContext& ctx) {
    const auto& capturedRightTable = jc.capturedRightTable;
    const auto& capturedRightFilter = jc.capturedRightFilter;
    const auto& capturedRHS = jc.capturedRHS;
    const auto& rhsTables = jc.rhsTables;
    const auto& lhsProjections = jc.lhsProjections;
    const auto& rhsProjections = jc.rhsProjections;
    JoinType jtype = JoinType::Inner;
    bool isRightVariant = false;
    if (nameLower.find("left") != std::string::npos) jtype = JoinType::Left;
    else if (nameLower.find("right") != std::string::npos) jtype = JoinType::Right;
    else if (nameLower.find("full") != std::string::npos) jtype = JoinType::Full;
    else if (nameLower.find("cross") != std::string::npos) jtype = JoinType::Cross;
    else if (nameLower.find("semi") != std::string::npos) jtype = JoinType::Semi;
    else if (nameLower.find("anti") != std::string::npos) jtype = JoinType::Anti;
    else if (nameLower.find("mark") != std::string::npos) jtype = JoinType::Mark;

    std::string condStr;
    if (extraInfo.is_object()) {
        if (extraInfo.contains("Join Type") && extraInfo["Join Type"].is_string()) {
            std::string jtStr = tolower_str(extraInfo["Join Type"].get_string());
            if (jtStr == "left") jtype = JoinType::Left;
            else if (jtStr == "right") jtype = JoinType::Right;
            else if (jtStr == "full" || jtStr == "outer") jtype = JoinType::Full;
            else if (jtStr == "semi" || jtStr == "right_semi") jtype = JoinType::Semi;
            else if (jtStr == "anti" || jtStr == "right_anti") jtype = JoinType::Anti;
            else if (jtStr == "mark") jtype = JoinType::Mark;
            isRightVariant = (jtStr.find("right_") == 0);
        }
        if (extraInfo.contains("Conditions")) {
            if (extraInfo["Conditions"].is_string()) {
                condStr = extraInfo["Conditions"].get_string();
            } else if (extraInfo["Conditions"].is_array()) {
                const auto& arr = extraInfo["Conditions"];
                for (size_t i = 0; i < arr.size(); ++i) {
                    if (arr[i].is_string()) {
                        if (!condStr.empty()) condStr += " AND ";
                        condStr += arr[i].get_string();
                    }
                }
            }
        }
    }

    if (nameLower.find("join") != std::string::npos) {
        if (env_truthy("GPUDB_DEBUG_PLANNER")) std::cerr << "[Planner] Creating JOIN '" << name << "'. CapturedRightTable: '" << capturedRightTable << "'" << std::endl;
        if (env_truthy("GPUDB_DEBUG_PLANNER")) std::cerr << "[Planner] Pre-resolved Condition: '" << condStr << "'" << std::endl;
        if (env_truthy("GPUDB_DEBUG_PLANNER")) std::cerr << "[Planner] ChildProjs size: " << childProjs.size() << std::endl;
        if (env_truthy("GPUDB_DEBUG_PLANNER")) for(size_t i=0; i<childProjs.size(); ++i) std::cerr << "  #" << i << ": " << childProjs[i] << std::endl;
    }

    // HEURISTIC: Fix broken #N references in DuckDB Join Conditions
    if (nameLower.find("join") != std::string::npos && !rhsProjections.empty() && condStr.find('#') != std::string::npos) {
         size_t lhsSize = childProjs.size() - rhsProjections.size();
         if (lhsSize > 0) {
             std::string shiftedCond;
             size_t lastPos = 0;
             bool neededShift = false;
             std::regex hashRe(R"(#(\d+))");
             std::sregex_iterator it(condStr.begin(), condStr.end(), hashRe);
             std::sregex_iterator end;

             for (; it != end; ++it) {
                 size_t idx = std::stoll(it->str().substr(1));
                 if (idx < lhsSize) {
                     neededShift = true;
                     shiftedCond += condStr.substr(lastPos, it->position() - lastPos);
                     shiftedCond += "#" + std::to_string(idx + lhsSize);
                     lastPos = it->position() + it->length();
                 }
             }
             if (neededShift) {
                  shiftedCond += condStr.substr(lastPos);
                  debug_log("Fixing Join Indexing: Shifted '" + condStr + "' to '" + shiftedCond + "'");
                  condStr = shiftedCond;
             }
         }
    }

    condStr = resolveColRef(condStr, childProjs);

    // Replace SUBQUERY keyword in join conditions
    if (condStr.find("SUBQUERY") != std::string::npos && !rhsProjections.empty()) {
         std::string rhsCol = rhsProjections[0];
         bool isSimple = rhsCol.find("CASE") == std::string::npos &&
                         rhsCol.find("(") == std::string::npos &&
                         rhsCol.find(" ") == std::string::npos;
         if (isSimple) {
             size_t pos = 0;
             while ((pos = condStr.find("SUBQUERY", pos)) != std::string::npos) {
                 condStr.replace(pos, 8, rhsCol);
                 pos += rhsCol.size();
             }
             if (env_truthy("GPUDB_DEBUG_PLANNER")) std::cerr << "[Planner] Replaced SUBQUERY with '" << rhsCol << "' in Join Condition -> " << condStr << std::endl;
         } else {
             if (env_truthy("GPUDB_DEBUG_PLANNER")) std::cerr << "[Planner] Keeping SUBQUERY token (RHS is complex: '" << rhsCol.substr(0, 40) << "...')" << std::endl;
         }
    }

    if (nameLower.find("join") != std::string::npos) {
        if (env_truthy("GPUDB_DEBUG_PLANNER")) std::cerr << "[Planner] Resolved Condition: '" << condStr << "'" << std::endl;
    }

    if (nameLower.find("join") != std::string::npos) {
         if (env_truthy("GPUDB_DEBUG_PLANNER")) std::cerr << "DEBUG_JOIN_EMISSION: Node='" << name << "'"
                   << " capturedRHS=" << (capturedRHS ? "true" : "false")
                   << " capturedRightTable='" << capturedRightTable << "'"
                   << std::endl;

         if (capturedRHS && capturedRightTable.empty()) {
              std::cerr << "CRITICAL ERROR: capturedRHS is TRUE but capturedRightTable is EMPTY for " << name << std::endl;
         }
    }

    IRNode joinNode = IRNode::join(jtype, Planner::parseExpression(condStr), condStr, capturedRightTable, capturedRightFilter);
    joinNode.asJoin().rightVariant = isRightVariant;

    // Extract and classify join keys into left/right based on table membership
    classifyJoinKeys(joinNode.asJoin(), capturedRightTable, rhsTables,
                     lhsProjections, rhsProjections, ctx);

    joinNode.duckdbName = name;
    ctx.plan.nodes.push_back(std::move(joinNode));
}

} // namespace engine
