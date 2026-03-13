// ============================================================================
// PlannerTraversal.cpp — DuckDB JSON tree traversal (traverseNode + helpers)
// ============================================================================
#include "PlannerInternal.hpp"
#include "EnvUtil.hpp"
#include <iostream>
#include <algorithm>
#include <regex>
#include <cctype>
#include "Logger.hpp"

namespace engine {

// Helper to scan JSON for all referenced columns to populate forceKeepColumns
void collectGlobalColumns(const json& j, std::unordered_set<std::string>& cols) {
    if (j.is_array()) {
        const auto& arr = j.get_array();
        for(const auto& item : arr) collectGlobalColumns(item, cols);
        return;
    }
    if (!j.is_object()) return;

    const auto& obj = j.get_object();
    for (const auto& [key, val] : obj) {
        
        // Fields known to contain column references
        if (key == "Projections" || key == "Filters" || key == "Groups" || key == "Aggregates" || key == "Condition" || key == "Expression") {
            auto extractWords = [&](std::string expr) {
                static std::regex colRe(R"([a-zA-Z_][a-zA-Z0-9_]*)");
                std::sregex_iterator begin(expr.begin(), expr.end(), colRe), end;
                for (auto i = begin; i != end; ++i) {
                    std::string match = i->str();
                    std::string up = match; // to upper
                    std::transform(up.begin(), up.end(), up.begin(), ::toupper);
                    
                    static const std::unordered_set<std::string> keywords = {
                        "AND", "OR", "NOT", "IS", "NULL", "LIKE", "IN", "BETWEEN", 
                        "CASE", "WHEN", "THEN", "ELSE", "END", "CAST", "AS", 
                        "SUM", "MIN", "MAX", "AVG", "COUNT", "FIRST", "DISTINCT",
                        "FROM", "WHERE", "GROUP", "BY", "ORDER", "LIMIT", "SUBQUERY",
                        "DATE", "INTERVAL", "YEAR", "MONTH", "DAY"
                    };
                    
                    if (keywords.find(up) == keywords.end()) {
                        cols.insert(match);
                    }
                }
            };

            if (val.is_array()) {
                const auto& arr = val.get_array();
                for(const auto& v : arr) {
                    if (v.is_string()) extractWords(v.get_string());
                }
            } else if (val.is_string()) {
                extractWords(val.get_string());
            }
        }
        
        collectGlobalColumns(val, cols);
    }
}

// --- splitProjectionString: split a comma-separated projection string respecting quotes/parens ---
static void splitProjectionString(const std::string& s, std::vector<std::string>& out) {
    bool inQuote = false;
    int depth = 0;
    std::string current;
    for (size_t j = 0; j < s.size(); ++j) {
        char c = s[j];
        if (c == '\'' && (j == 0 || s[j-1] != '\\')) inQuote = !inQuote;
        if (!inQuote) {
            if (c == '(') depth++;
            else if (c == ')') depth--;
            else if (c == ',' && depth == 0) {
                out.push_back(trim_str(current));
                current.clear();
                continue;
            }
        }
        current += c;
    }
    if (!current.empty()) out.push_back(trim_str(current));
}

// --- parseNodeProjections: extract and resolve projection list from node extra_info ---
static std::vector<std::string> parseNodeProjections(const json& extraInfo,
                                                      const std::string& name,
                                                      const std::vector<std::string>& childProjs) {
    std::vector<std::string> myProjs;
    if (!extraInfo.is_object() || !extraInfo.contains("Projections")) return myProjs;

    debug_log("Parsing Projections for " + name);
    const auto& projNode = extraInfo["Projections"];

    auto processProj = [&](std::string proj) {
        // Cleanup internal DuckDB optimizations
        if (proj.find("__internal_compress") != std::string::npos ||
            proj.find("__internal_decompress") != std::string::npos) {
            size_t start = proj.find('(');
            size_t end = proj.rfind(')');
            if (start != std::string::npos && end != std::string::npos) {
                proj = proj.substr(start + 1, end - start - 1);
                size_t comma = proj.find(',');
                if (comma != std::string::npos) proj = proj.substr(0, comma);
            }
        }
        proj = resolveColRef(proj, childProjs);
        debug_log("Proj: " + proj);
        myProjs.push_back(proj);
    };

    std::vector<std::string> rawProjs;
    if (projNode.is_array()) {
        for (const auto& item : projNode.get_array()) {
            if (item.is_string()) splitProjectionString(item.get_string(), rawProjs);
        }
    } else if (projNode.is_string()) {
        splitProjectionString(projNode.get_string(), rawProjs);
    }

    // Stitch split CASE statements back together
    for (size_t i = 0; i < rawProjs.size(); ++i) {
        std::string current = rawProjs[i];
        while (i + 1 < rawProjs.size()) {
            std::string s_upper = current;
            std::transform(s_upper.begin(), s_upper.end(), s_upper.begin(), ::toupper);
            int caseCount = 0;
            size_t pos = 0;
            while ((pos = s_upper.find("CASE", pos)) != std::string::npos) { caseCount++; pos += 4; }
            int endCount = 0;
            pos = 0;
            while ((pos = s_upper.find("END", pos)) != std::string::npos) { endCount++; pos += 3; }
            if (caseCount > endCount) {
                debug_log("Fixing split CASE projection. Appending next line.");
                current += " " + rawProjs[i + 1];
                i++;
            } else {
                break;
            }
        }
        processProj(current);
    }
    return myProjs;
}

// --- applyRenameMapping: update aliases and qualified column mappings after column renaming ---
static void applyRenameMapping(const std::unordered_map<std::string, std::string>& renameMap,
                                TraverseContext& ctx) {
    for (const auto& [oldName, newName] : renameMap) {
        for (auto& [alias, target] : ctx.localAliases) {
            if (target == oldName) target = newName;
        }
        for (const auto& [alias, target] : ctx.aliases) {
            std::string stripped = stripTableQualifier(target);
            if (stripped == oldName &&
                target.find('.') != std::string::npos &&
                ctx.qualifiedColumnMapping.find(target) == ctx.qualifiedColumnMapping.end()) {
                ctx.qualifiedColumnMapping[target] = newName;
                debug_log("Qualified mapping: " + target + " -> " + newName);
                break;
            }
        }
    }
}

// --- handleCTE: process Common Table Expression nodes ---
static bool handleCTE(const json& node, TraverseContext& ctx) {
    if (!node.contains("children") || !node["children"].is_array()) return false;
    const auto& kids = node["children"].get_array();
    if (kids.size() < 2) return false;

    std::string cteName = "unknown_cte";
    if (node.contains("extra_info") && node["extra_info"].is_object()) {
        auto& ei = node["extra_info"];
        if (env_truthy("GPUDB_DEBUG_PLANNER")) {
            LOG_INFO("PLANNER", "DEBUG: CTE Node extra_info keys:\n");
            for (auto& elt : ei.get_object()) std::cerr << "  CTE Key: " << elt.first << "\n";
        }
        if (ei.contains("CTE Name")) cteName = ei["CTE Name"].get_string();
        if (ei.contains("Table Index")) {
            int64_t idx = 0;
            if (ei["Table Index"].is_number()) idx = (int64_t)ei["Table Index"].get_number();
            ctx.cteMap[idx] = cteName;
        }
    }
    traverseNode(kids[0], ctx);
    ctx.plan.nodes.push_back(IRNode::save(cteName));
    traverseNode(kids[1], ctx);
    return true;
}

// --- tryCaptureJoinRHS: attempt to capture a simple Scan/Filter chain from the RHS subtree ---
static bool tryCaptureJoinRHS(const json& root, JoinCapture& jc, TraverseContext& ctx) {
    auto& capturedRightTable = jc.capturedRightTable;
    auto& capturedRightFilter = jc.capturedRightFilter;
    auto& rhsTables = jc.rhsTables;
    auto& rhsProjections = jc.rhsProjections;

    debug_log("Checking RHS capture for Join");
    json curr = root;
    std::string filterStr;

    while (true) {
        std::string n = curr.contains("name") ? curr["name"].get_string() : "";
        std::string nl = tolower_str(n);
        debug_log("Inspecting RHS node " + n);

        if (nl.find("scan") != std::string::npos || nl == "get" ||
            nl.find("read_csv") != std::string::npos || nl.find("delim_scan") != std::string::npos) {
            if (curr.contains("extra_info") && curr["extra_info"].is_object()) {
                auto& ei = curr["extra_info"];
                std::string tbl;
                if (ei.contains("Table")) {
                    tbl = ei["Table"].get_string();
                } else if (nl.find("delim_scan") != std::string::npos) {
                    return false;
                }
                if (!tbl.empty()) {
                    if (tolower_str(tbl) == "part") {
                        debug_log("Skipping capture for 'part' table to force full Traversal");
                        return false;
                    }
                    capturedRightTable = tbl;
                    rhsTables.insert(capturedRightTable);
                    debug_log("Captured Table " + capturedRightTable);

                    // Extract Scan Filters
                    if (ei.contains("Filters")) {
                        auto& f = ei["Filters"];
                        if (f.is_array()) {
                            for (const auto& x : f.get_array()) {
                                if (x.get_string().find("optional:") == std::string::npos) {
                                    if (!filterStr.empty()) filterStr += " AND ";
                                    filterStr += x.get_string();
                                }
                            }
                        } else if (f.is_string()) {
                            std::string s = f.get_string();
                            if (s.find("optional:") == std::string::npos) {
                                if (!filterStr.empty()) filterStr += " AND ";
                                filterStr += s;
                            }
                        }
                    }
                    if (!filterStr.empty()) capturedRightFilter = Planner::parseExpression(filterStr);

                    // Extract Projections
                    if (ei.contains("Projections")) {
                        const auto& p = ei["Projections"];
                        if (p.is_array()) {
                            for (const auto& item : p.get_array()) {
                                std::string s = item.get_string();
                                if (s.find("__internal_compress") != std::string::npos ||
                                    s.find("__internal_decompress") != std::string::npos) {
                                    size_t start = s.find('(');
                                    size_t end = s.rfind(')');
                                    if (start != std::string::npos && end != std::string::npos) {
                                        s = s.substr(start + 1, end - start - 1);
                                        size_t comma = s.find(',');
                                        if (comma != std::string::npos) s = s.substr(0, comma);
                                    }
                                }
                                s = stripTableQualifier(s);
                                rhsProjections.push_back(s);
                            }
                        }
                    }
                    if (ctx.seenTables.find(capturedRightTable) == ctx.seenTables.end()) {
                        ctx.seenTables.insert(capturedRightTable);
                        ctx.plan.tables.push_back({capturedRightTable, rhsProjections});
                    }
                    return true;
                }
            }
            return false;
        } else if (nl.find("filter") != std::string::npos) {
            debug_log("processing filter node info");
            if (curr.contains("extra_info")) {
                std::string p;
                auto& ei = curr["extra_info"];
                if (ei.is_string()) {
                    p = ei.get_string();
                } else if (ei.is_object()) {
                    if (ei.contains("Expression")) p = ei["Expression"].get_string();
                    else if (ei.contains("Condition")) p = ei["Condition"].get_string();
                    if (ei.contains("Filters")) {
                        auto& f = ei["Filters"];
                        if (f.is_array()) {
                            for (const auto& item : f.get_array()) {
                                if (!p.empty()) p += " AND ";
                                p += item.get_string();
                            }
                        } else if (f.is_string()) {
                            if (!p.empty()) p += " AND ";
                            p += f.get_string();
                        }
                    }
                }
                if (!p.empty()) {
                    if (!filterStr.empty()) filterStr += " AND ";
                    filterStr += p;
                    debug_log("captured filter: " + p);
                }
            }
            if (curr.contains("children") && curr["children"].is_array() && curr["children"].size() == 1) {
                debug_log("descending to child");
                json next = curr["children"][0];
                curr = next;
            } else {
                debug_log("missing children");
                return false;
            }
        } else {
            return false;
        }
    }
}

// --- containsDelimScan: check if a subtree contains DELIM_SCAN/DELIM_GET/COLUMN_DATA_SCAN ---
static bool containsDelimScan(const json& n) {
    std::string nl = tolower_str(n.contains("name") ? n["name"].get_string() : "");
    debug_log("checkDelim visiting: " + nl);
    if (nl.find("delim_scan") != std::string::npos || nl.find("delim_get") != std::string::npos || nl == "column_data_scan") {
        debug_log("checkDelim FOUND: " + nl);
        return true;
    }
    if (n.contains("children")) {
        for (const auto& c : n["children"].get_array()) {
            if (containsDelimScan(c)) return true;
        }
    }
    return false;
}

// --- handleDelimJoinTraversal: process DELIM_JOIN provider/consumer pattern ---
static void handleDelimJoinTraversal(const json& node, const std::string& /*name*/, const std::string& nameLower,
                                      const json::array& kids, bool swapInputs,
                                      JoinCapture& jc, std::vector<std::string>& childProjs, TraverseContext& ctx) {
    auto& capturedRightTable = jc.capturedRightTable;
    auto& capturedRHS = jc.capturedRHS;
    auto& rhsTables = jc.rhsTables;

    debug_log("Processing DELIM_JOIN. Children: " + std::to_string(kids.size()) + " Swap: " + std::to_string(swapInputs));

    const auto& providerNode = swapInputs ? kids[1] : kids[0];
    const auto& consumerNode = swapInputs ? kids[0] : kids[1];

    // Provider First -> Save -> Consumer -> Save -> Scan Provider -> Join
    traverseNode(providerNode, ctx);
    auto lhsProjs = ctx.projections;
    childProjs.insert(childProjs.end(), lhsProjs.begin(), lhsProjs.end());

    std::string lhsSaveID = "tmpl_delim_lhs_" + std::to_string(ctx.plan.nodes.size());
    debug_log("Emitting SAVE for DELIM_JOIN LHS: " + lhsSaveID + " at index " + std::to_string(ctx.plan.nodes.size()) + " Plan: " + std::to_string((uintptr_t)&ctx.plan));
    auto saveNode = IRNode::save(lhsSaveID);
    debug_log("Created Save Node with type: " + std::to_string((int)saveNode.type));
    ctx.plan.nodes.push_back(std::move(saveNode));
    if (ctx.plan.nodes.back().type == IRNode::Type::Save) debug_log("CONFIRMED: Back node is Save.");
    else debug_log("ERROR: Back node is NOT Save! It is " + std::to_string((int)ctx.plan.nodes.back().type));
    debug_log("Post-Emit Size: " + std::to_string(ctx.plan.nodes.size()));

    // Push new DELIM context
    ctx.delimStack.push_back({lhsSaveID, lhsProjs});
    debug_log("Pushed DELIM context: " + lhsSaveID + " (stack size=" + std::to_string(ctx.delimStack.size()) + ")");

    // Traverse Consumer with isolation
    TraverseContext rhsCtx = ctx;
    rhsCtx.seenTables.clear();
    traverseNode(consumerNode, rhsCtx);

    rhsTables = rhsCtx.seenTables;
    for (const auto& t : rhsTables) ctx.seenTables.insert(t);
    for (const auto& [k, v] : rhsCtx.localAliases) ctx.localAliases[k] = v;
    ctx.projections = rhsCtx.projections;
    childProjs.insert(childProjs.end(), ctx.projections.begin(), ctx.projections.end());

    if (!ctx.delimStack.empty()) {
        debug_log("Popping DELIM context: " + ctx.delimStack.back().first);
        ctx.delimStack.pop_back();
    }

    // Determine join type
    JoinType delimJoinType = JoinType::Semi;
    if (node.contains("extra_info") && node["extra_info"].is_object()) {
        auto& ei = node["extra_info"];
        if (ei.contains("Join Type") && ei["Join Type"].is_string()) {
            std::string jtStr = ei["Join Type"].get_string();
            std::string jtLower = jtStr;
            std::transform(jtLower.begin(), jtLower.end(), jtLower.begin(), ::tolower);
            if (jtLower.find("anti") != std::string::npos) delimJoinType = JoinType::Anti;
            else if (jtLower.find("semi") != std::string::npos) delimJoinType = JoinType::Semi;
            else if (jtLower.find("mark") != std::string::npos) delimJoinType = JoinType::Semi;
            else if (jtLower.find("left") != std::string::npos) delimJoinType = JoinType::Left;
            else if (jtLower.find("right") != std::string::npos) delimJoinType = JoinType::Right;
            else if (jtLower.find("inner") != std::string::npos || jtLower == "single") delimJoinType = JoinType::Inner;
            debug_log("DELIM_JOIN read Join Type from plan: '" + jtStr + "'");
        }
    }
    if (delimJoinType == JoinType::Semi) {
        if (nameLower.find("anti") != std::string::npos || nameLower.find("not_exists") != std::string::npos)
            delimJoinType = JoinType::Anti;
    }
    debug_log("DELIM_JOIN emitting join type: " + std::to_string(static_cast<int>(delimJoinType)));

    bool isExplicitDelimJoin = (nameLower.find("delim_join") != std::string::npos);
    bool needsCorrelationJoin = (delimJoinType == JoinType::Semi || delimJoinType == JoinType::Anti) || !isExplicitDelimJoin;

    if (needsCorrelationJoin) {
        std::string rhsSaveID = "tmpl_join_" + std::to_string(ctx.plan.nodes.size());
        ctx.plan.nodes.push_back(IRNode::save(rhsSaveID));
        capturedRightTable = rhsSaveID;
        capturedRHS = true;

        IRNode restoreScan = IRNode::scan(lhsSaveID);
        for (const auto& proj : lhsProjs) restoreScan.asScan().columns.push_back(stripTableQualifier(proj));

        std::string delimCondRaw;
        if (node.contains("extra_info")) {
            auto& ei = node["extra_info"];
            if (ei.contains("Condition") && ei["Condition"].is_string())
                delimCondRaw = ei["Condition"].get_string();
            if (ei.contains("Conditions")) {
                const auto& c = ei["Conditions"];
                if (c.is_string()) delimCondRaw = c.get_string();
                else if (c.is_array()) {
                    for (const auto& item : c.get_array()) {
                        if (item.is_string()) {
                            if (!delimCondRaw.empty()) delimCondRaw += " AND ";
                            delimCondRaw += item.get_string();
                        }
                    }
                }
            }
        }
        debug_log("DELIM_JOIN condition: " + delimCondRaw);

        // Rewrite IS NOT DISTINCT FROM to =
        size_t indfPos = 0;
        while ((indfPos = delimCondRaw.find("IS NOT DISTINCT FROM", indfPos)) != std::string::npos) {
            delimCondRaw.replace(indfPos, 20, "=");
            indfPos += 1;
        }

        ctx.plan.nodes.push_back(restoreScan);

        // Refine Join Condition
        std::string lhsJoinKey, rhsJoinKey;
        bool isEquality = false;
        std::string cond = delimCondRaw;
        if (!cond.empty()) {
            size_t eqPos = cond.find('=');
            if (eqPos != std::string::npos) {
                lhsJoinKey = trim_str(cond.substr(0, eqPos));
                rhsJoinKey = trim_str(cond.substr(eqPos + 1));
                isEquality = true;
            }
        }
        if (isEquality) {
            auto findInLhs = [&](const std::string& key) -> std::string {
                std::string cleanKey = stripTableQualifier(key);
                for (const auto& c : lhsProjs) {
                    if (c == key || stripTableQualifier(c) == cleanKey) return c;
                }
                for (const auto& c : lhsProjs) {
                    if (stripTableQualifier(c).find(cleanKey + "_") == 0) return c;
                }
                return "";
            };
            std::string matchedCol = findInLhs(lhsJoinKey);
            if (!matchedCol.empty()) {
                delimCondRaw = matchedCol + " = " + rhsJoinKey;
                debug_log("Refined DELIM_JOIN condition: " + cond + " -> " + delimCondRaw);
            } else {
                matchedCol = findInLhs(rhsJoinKey);
                if (!matchedCol.empty()) {
                    delimCondRaw = matchedCol + " = " + lhsJoinKey;
                    debug_log("Refined DELIM_JOIN condition: " + cond + " -> " + delimCondRaw);
                } else {
                    debug_log("Warning: DELIM_JOIN condition keys not found in LHS projections. Keeping original: " + cond);
                }
            }
        }

        if (delimCondRaw.empty()) delimCondRaw = "1=1";

        JoinType correlationJoinType = (delimJoinType == JoinType::Anti || delimJoinType == JoinType::Semi)
                                          ? JoinType::Semi : JoinType::Inner;
        ctx.plan.nodes.push_back(IRNode::join(correlationJoinType, Planner::parseExpression(delimCondRaw), delimCondRaw, rhsSaveID, nullptr));
    } else {
        debug_log("DELIM_JOIN: Skipping outer correlation join (type=" +
                  std::to_string(static_cast<int>(delimJoinType)) +
                  "). Consumer output used directly.");
    }
}

// --- handleBushyJoinTraversal: process right-deep / bushy join patterns ---
static void handleBushyJoinTraversal(const json::array& kids, JoinCapture& jc,
                                      std::vector<std::string>& childProjs, TraverseContext& ctx) {
    auto& capturedRightTable = jc.capturedRightTable;
    auto& rhsTables = jc.rhsTables;
    auto& lhsProjections = jc.lhsProjections;
    auto& rhsProjections = jc.rhsProjections;

    // Traverse RHS (Build Side) first using isolated context
    TraverseContext rhsCtx = ctx;
    rhsCtx.seenTables.clear();
    traverseNode(kids[1], rhsCtx);

    rhsTables = rhsCtx.seenTables;
    for (const auto& t : rhsTables) ctx.seenTables.insert(t);

    // Rename RHS columns with unique suffix
    std::string uniqueSuffix = "_rhs_" + std::to_string(ctx.plan.nodes.size());
    std::vector<std::string> renamedRhsProjs;
    std::vector<TypedExprPtr> projectExprs;
    std::vector<std::string> projectNames;

    int complexCounter = 0;
    for (const auto& col : rhsCtx.projections) {
        projectExprs.push_back(TypedExpr::column(col));
        std::string baseName = col;
        if (baseName.size() > 64 || baseName.find("CASE") != std::string::npos ||
            baseName.find("SUBQUERY") != std::string::npos || baseName.find('"') != std::string::npos) {
            baseName = "complex_expr_" + std::to_string(complexCounter++);
        }
        std::string newName = baseName + uniqueSuffix;
        projectNames.push_back(newName);
        renamedRhsProjs.push_back(newName);
    }

    if (!renamedRhsProjs.empty()) {
        rhsCtx.plan.nodes.push_back(IRNode::project(projectExprs, projectNames));
        rhsCtx.projections = renamedRhsProjs;
    }

    rhsProjections = rhsCtx.projections;
    ctx.projections = rhsCtx.projections;

    for (const auto& [k, v] : rhsCtx.localAliases) ctx.localAliases[k] = v;

    std::string saveID = "tmpl_join_" + std::to_string(ctx.plan.nodes.size());
    ctx.plan.nodes.push_back(IRNode::save(saveID));
    capturedRightTable = saveID;
    rhsTables.insert(saveID);

    // Traverse LHS (Probe Side)
    traverseNode(kids[0], ctx);
    lhsProjections = ctx.projections;

    childProjs.insert(childProjs.end(), lhsProjections.begin(), lhsProjections.end());
    childProjs.insert(childProjs.end(), rhsProjections.begin(), rhsProjections.end());
}


void traverseNode(const json& node, TraverseContext& ctx) {
    if (!node.is_object()) return;
    
    std::string name = node.contains("name") && node["name"].is_string() 
        ? node["name"].get_string() : "";
    
    debug_log("Traversing node: " + name);
    
    std::string nameLower = tolower_str(name);

    // Handle CTE (Common Table Expressions)
    if (name == "CTE") {
        if (handleCTE(node, ctx)) return;
    }

    std::vector<std::string> childProjs;
    
    // Capture Join RHS logic (bundled for passing to handleJoinEmit)
    JoinCapture jc;
    auto& capturedRightTable = jc.capturedRightTable;
    auto& capturedRHS = jc.capturedRHS;
    auto& rhsTables = jc.rhsTables;
    auto& rhsProjections = jc.rhsProjections;

    if (nameLower.find("join") != std::string::npos && 
        node.contains("children") && node["children"].is_array()) {
            
        debug_log("[TraverseNode] Logic for JOIN: " + name);
            
        const auto& kids = node["children"].get_array();
        if (kids.size() == 2) {
            
            // Handle RHS capture (simple & complex)
            // Skip DELIM_JOIN here, it has special handling below
            if (nameLower.find("delim_join") == std::string::npos) {
                if (tryCaptureJoinRHS(kids[1], jc, ctx)) {
                    // Simple Capture (Scan/Filter chain)
                    capturedRHS = true;
                    
                    // Need to extract RHS projections to resolve join conditions (e.g. #0) correctly
                    // Use a temporary context to traverse the RHS and get projections
                    {
                        TraverseContext fhCtx = ctx;
                        fhCtx.seenTables.clear();
                        fhCtx.projections.clear();
                        Plan dummyPlan;
                        TraverseContext dummyCtx {
                            dummyPlan,
                            fhCtx.aliases,
                            fhCtx.localAliases,
                            fhCtx.projections,
                            fhCtx.seenTables,
                            fhCtx.pastGroupBy,
                            fhCtx.delimStack,
                            fhCtx.cteMap,
                            fhCtx.forceKeepColumns,
                            {}
                        };
                        traverseNode(kids[1], dummyCtx);
                        rhsProjections = dummyCtx.projections;
                        
                        for(const auto& t : dummyCtx.seenTables) {
                            ctx.seenTables.insert(t);
                        }
                    }

                    // Traverse LHS only
                    traverseNode(kids[0], ctx);
                    
                    // Rename duplicate columns from RHS to avoid ambiguity in filters
                    std::unordered_map<std::string, std::string> renameMap;
                    auto renamedRhsProjs = renameDuplicateColumns(ctx.projections, rhsProjections, renameMap);
                    applyRenameMapping(renameMap, ctx);
                    
                    childProjs.insert(childProjs.end(), ctx.projections.begin(), ctx.projections.end()); // LHS projs
                    childProjs.insert(childProjs.end(), renamedRhsProjs.begin(), renamedRhsProjs.end());   // RHS projs (renamed)
                } else {
                    // Complex Capture (Subtree, Joins, etc.)
                    // Traverse RHS first, Save it, then LHS.
                    debug_log("RHS is complex (tryCapture failed). Using Traversal-First Strategy.");
                    
                    // 1. Traverse RHS (Appends nodes to Plan)
                    traverseNode(kids[1], ctx);
                    
                    // 2. Save RHS Result
                    std::string saveID = "tmpl_join_" + std::to_string(ctx.plan.nodes.size());
                    ctx.plan.nodes.push_back(IRNode::save(saveID));
                    capturedRightTable = saveID;
                    rhsTables.insert(saveID);
                    rhsProjections = ctx.projections;
                    capturedRHS = true;
                    
                    // 3. Traverse LHS (Appends nodes to Plan, result in currentCtx)
                    traverseNode(kids[0], ctx);
                    
                    // 4. Combine Projections with duplicate column renaming
                    std::unordered_map<std::string, std::string> renameMap;
                    auto renamedRhsProjs = renameDuplicateColumns(ctx.projections, rhsProjections, renameMap);
                    applyRenameMapping(renameMap, ctx);
                    
                    childProjs.insert(childProjs.end(), ctx.projections.begin(), ctx.projections.end()); // LHS
                    childProjs.insert(childProjs.end(), renamedRhsProjs.begin(), renamedRhsProjs.end());   // RHS (renamed)
                }
            }
        }
    }

    // Visit children first (post-order) - ONLY if not manual logic
    bool handled = false;
    if (!capturedRHS && node.contains("children") && node["children"].is_array()) {
        const auto& kids = node["children"].get_array();
        debug_log("Entering children traversal block. Name: " + nameLower + " Children: " + std::to_string(kids.size()));
        
        // Handle Complex Join Re-ordering (Right-Deep / Bushy)
        if (nameLower.find("join") != std::string::npos && kids.size() >= 2) {
            bool hasDelimScanInLHS = containsDelimScan(kids[0]);
            bool hasDelimScanInRHS = containsDelimScan(kids[1]);
            bool isDelimJoin = (nameLower.find("delim_join") != std::string::npos) || hasDelimScanInRHS || hasDelimScanInLHS;
            if (isDelimJoin) debug_log("Processing DELIM_JOIN at node " + std::to_string(ctx.plan.nodes.size()) + " Plan: " + std::to_string((uintptr_t)&ctx.plan));

            if (isDelimJoin) {
                bool swapInputs = hasDelimScanInLHS && !hasDelimScanInRHS;
                handleDelimJoinTraversal(node, name, nameLower, kids, swapInputs, jc, childProjs, ctx);
                handled = true;
            } else {
                handleBushyJoinTraversal(kids, jc, childProjs, ctx);
                handled = true;
            }
        }

        if (!handled) {
            for (const auto& child : node["children"].get_array()) {
                traverseNode(child, ctx);
                // Merge child projections
                childProjs.insert(childProjs.end(), ctx.projections.begin(), ctx.projections.end());
            }
        }
    }
    ctx.projections = childProjs;

    if (handled) return;
    
    // Extract extra_info
    json extraInfo;
    std::string extraStr;
    if (node.contains("extra_info")) {
        if (node["extra_info"].is_object()) {
            extraInfo = node["extra_info"];
        } else if (node["extra_info"].is_string()) {
            extraStr = node["extra_info"].get_string();
        }
    }
    
    // Parse projections from this node
    std::vector<std::string> myProjs = parseNodeProjections(extraInfo, name, childProjs);
    
    // ========== SCAN ==========
    if (nameLower.find("scan") != std::string::npos || 
        nameLower == "get" || 
        nameLower.find("read_csv") != std::string::npos) {
        if (handleScan(node, name, nameLower, extraInfo, myProjs, ctx)) return;
    }
    // ========== FILTER ==========
    else if (nameLower.find("filter") != std::string::npos) {
        handleFilter(node, name, extraInfo, extraStr, childProjs, ctx);
    }
    // ========== GROUP_BY ==========
    else if (nameLower.find("group_by") != std::string::npos) {
        handleGroupBy(node, name, extraInfo, childProjs, myProjs, ctx);
    }
    // ========== UNGROUPED_AGGREGATE / AGGREGATE ==========
    else if (name == "UNGROUPED_AGGREGATE" || name == "AGGREGATE") {
        handleUngroupedAggregate(node, name, extraInfo, childProjs, ctx);
    }
    // ========== PROJECTION ==========
    else if (name.find("PROJECTION") != std::string::npos) {
        handleProjection(node, name, myProjs, childProjs, ctx);
    }
    // ========== ORDER_BY / TOP_N ==========
    else if (name == "ORDER_BY" || name == "ORDER" || nameLower.find("top_n") != std::string::npos) {
        handleOrderBy(node, name, nameLower, extraInfo, childProjs, ctx);
    }
    // ========== LIMIT ==========
    else if (name == "LIMIT") {
        int64_t count = 10;
        std::regex re("(\\d+)");
        std::smatch m;
        if (std::regex_search(extraStr, m, re)) {
            count = std::stoll(m[1].str());
        }
        
        IRNode limNode = IRNode::limit(count);
        limNode.duckdbName = name;
        ctx.plan.nodes.push_back(std::move(limNode));
    }
    // ========== DISTINCT ==========
    else if (nameLower.find("distinct") != std::string::npos &&
             nameLower.find("scan") == std::string::npos &&
             nameLower.find("join") == std::string::npos) {
        // DuckDB may output HASH_DISTINCT or STREAMING_DISTINCT nodes
        IRNode distNode = IRNode::distinct();
        distNode.duckdbName = name;
        ctx.plan.nodes.push_back(std::move(distNode));
    }
    // ========== JOIN ==========
    else if (name.find("JOIN") != std::string::npos) {
        handleJoinEmit(node, name, nameLower, extraInfo, childProjs, jc, ctx);
    }
    // Update projections for parent
    if (!myProjs.empty()) {
        ctx.projections = myProjs;
    }
    
    std::string proj_list;
    for(auto& s : ctx.projections) proj_list += s + ", ";
    debug_log("Node " + name + " output projections: " + proj_list);
}

} // namespace engine
