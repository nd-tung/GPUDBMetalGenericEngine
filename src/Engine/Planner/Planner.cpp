#include "Planner.hpp"
#include "DuckDBAdapter.hpp"
#include <string>
#include <algorithm>
#include <regex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cctype>
#include <unordered_map>

using nlohmann::json;

namespace engine {

static std::string tolower_copy(std::string s){ std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){return std::tolower(c);}); return s; }

static std::string trim_copy(std::string s) {
    auto first = s.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    auto last = s.find_last_not_of(" \t\n\r");
    return s.substr(first, last - first + 1);
}

static std::string strip_outer_parens(std::string s) {
    s = trim_copy(std::move(s));
    while (!s.empty() && s.front() == '(' && s.back() == ')') {
        int depth = 0;
        bool ok = true;
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '(') depth++;
            else if (s[i] == ')') {
                depth--;
                if (depth == 0 && i + 1 != s.size()) {
                    ok = false;
                    break;
                }
            }
            if (depth < 0) { ok = false; break; }
        }
        if (!ok || depth != 0) break;
        s = trim_copy(s.substr(1, s.size() - 2));
    }
    return s;
}

static std::string strip_order_nulls_clause(std::string s) {
    // Strip optional trailing "NULLS FIRST/LAST" which DuckDB may include in ORDER BY.
    // We don't currently model NULL ordering; for TPC-H data this is effectively a no-op
    // but avoids spurious ORDER BY mismatches downstream.
    s = trim_copy(std::move(s));
    std::string low = tolower_copy(s);

    auto rstrip = [&](const std::string& suffix) -> bool {
        if (low.size() < suffix.size()) return false;
        if (low.compare(low.size() - suffix.size(), suffix.size(), suffix) != 0) return false;
        s = trim_copy(s.substr(0, s.size() - suffix.size()));
        low = tolower_copy(s);
        return true;
    };

    // Try a couple of common variants.
    if (rstrip(" nulls last")) return s;
    if (rstrip(" nulls first")) return s;
    return s;
}

static std::string normalize_expr_for_alias(std::string s) {
    s = strip_outer_parens(std::move(s));
    s = tolower_copy(std::move(s));
    // Drop any explicit alias portion.
    auto posAs = s.rfind(" as ");
    if (posAs != std::string::npos) s = s.substr(0, posAs);

    // Remove whitespace.
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }), s.end());

    // Remove table qualifiers like "t.col" -> "col".
    static const std::regex qualRe(R"(\b[a-z_][a-z0-9_]*\.)");
    s = std::regex_replace(s, qualRe, "");

    // Normalize DuckDB count_star() to count(*).
    if (s.find("count_star()") != std::string::npos) return "count(*)";
    if (s.find("count(*)") != std::string::npos) return "count(*)";
    return s;
}

static std::unordered_map<std::string, std::string> parse_select_agg_aliases(const std::string& sql) {
    std::unordered_map<std::string, std::string> out;

    std::string s = sql;
    // Find SELECT ... FROM at top-level.
    std::string sl = tolower_copy(s);
    size_t sel = sl.find("select");
    if (sel == std::string::npos) return out;

    int depth = 0;
    size_t fromPos = std::string::npos;
    for (size_t i = sel + 6; i + 4 <= sl.size(); ++i) {
        char c = sl[i];
        if (c == '(') depth++;
        else if (c == ')') depth--;
        if (depth == 0 && sl.compare(i, 4, "from") == 0) { fromPos = i; break; }
    }
    if (fromPos == std::string::npos) return out;

    std::string list = s.substr(sel + 6, fromPos - (sel + 6));
    // Split select list by top-level commas.
    depth = 0;
    size_t start = 0;
    for (size_t i = 0; i <= list.size(); ++i) {
        if (i == list.size() || (list[i] == ',' && depth == 0)) {
            std::string item = trim_copy(list.substr(start, i - start));
            start = i + 1;
            if (item.empty()) continue;

            std::string itemLower = tolower_copy(item);
            // Only consider explicit AS aliases.
            size_t asPos = itemLower.rfind(" as ");
            if (asPos == std::string::npos) continue;
            std::string expr = trim_copy(item.substr(0, asPos));
            std::string alias = trim_copy(item.substr(asPos + 4));
            alias = strip_outer_parens(std::move(alias));
            if (!alias.empty() && (alias.front() == '"' || alias.front() == '\'')) {
                if (alias.size() >= 2 && alias.back() == alias.front()) {
                    alias = alias.substr(1, alias.size() - 2);
                }
            }

            const std::string norm = normalize_expr_for_alias(expr);
            if (!norm.empty() && !alias.empty()) out[norm] = alias;
            continue;
        }
        if (list[i] == '(') depth++;
        else if (list[i] == ')') depth--;
    }

    return out;
}

// Map explicit SELECT aliases (alias -> expression). Used to recover computed projection
// expressions when DuckDB's JSON plan only lists the output name (e.g. "rev").
static std::unordered_map<std::string, std::string> parse_select_proj_aliases(const std::string& sql) {
    std::unordered_map<std::string, std::string> out;

    std::string s = sql;
    std::string sl = tolower_copy(s);
    size_t sel = sl.find("select");
    if (sel == std::string::npos) return out;

    int depth = 0;
    size_t fromPos = std::string::npos;
    for (size_t i = sel + 6; i + 4 <= sl.size(); ++i) {
        char c = sl[i];
        if (c == '(') depth++;
        else if (c == ')') depth--;
        if (depth == 0 && sl.compare(i, 4, "from") == 0) { fromPos = i; break; }
    }
    if (fromPos == std::string::npos) return out;

    std::string list = s.substr(sel + 6, fromPos - (sel + 6));
    depth = 0;
    size_t start = 0;
    for (size_t i = 0; i <= list.size(); ++i) {
        if (i == list.size() || (list[i] == ',' && depth == 0)) {
            std::string item = trim_copy(list.substr(start, i - start));
            start = i + 1;
            if (item.empty()) continue;

            std::string itemLower = tolower_copy(item);
            size_t asPos = itemLower.rfind(" as ");
            if (asPos == std::string::npos) continue;

            std::string expr = trim_copy(item.substr(0, asPos));
            std::string alias = trim_copy(item.substr(asPos + 4));
            alias = strip_outer_parens(std::move(alias));
            if (!alias.empty() && (alias.front() == '"' || alias.front() == '\'')) {
                if (alias.size() >= 2 && alias.back() == alias.front()) {
                    alias = alias.substr(1, alias.size() - 2);
                }
            }
            if (alias.empty() || expr.empty()) continue;

            // Strip a simple table qualifier prefix in the expression to match how our
            // downstream base_ident/ExprEval resolves identifiers.
            expr = std::regex_replace(expr, std::regex(R"(\b[a-z_][a-z0-9_]*\.)", std::regex::icase), "");
            out[alias] = expr;
            continue;
        }
        if (list[i] == '(') depth++;
        else if (list[i] == ')') depth--;
    }
    return out;
}

// Helper to resolve column references like "#0" to actual names
static std::string resolveColumnRef(const std::string& ref, const std::vector<std::string>& projections) {
    std::string expr = ref;
    size_t pos = 0;
    while ((pos = expr.find('#', pos)) != std::string::npos) {
        if (pos + 1 < expr.size() && std::isdigit(static_cast<unsigned char>(expr[pos+1]))) {
            size_t end = pos + 1;
            while (end < expr.size() && std::isdigit(static_cast<unsigned char>(expr[end]))) end++;
            std::string numStr = expr.substr(pos + 1, end - (pos + 1));
            try {
                size_t idx = std::stoull(numStr);
                if (idx < projections.size()) {
                    std::string replacement = projections[idx];
                    expr.replace(pos, end - pos, replacement);
                    pos += replacement.length();
                    continue;
                }
            } catch (...) {}
        }
        pos++;
    }
    return expr;
}

static std::vector<std::string> traverse(const json& node,
                                         Plan& p,
                                         const std::unordered_map<std::string, std::string>& aggAliases,
                                         const std::unordered_map<std::string, std::string>& projAliases) {
    std::vector<std::string> childProjections;

    // 1. Visit children first (Post-Order)
    if (node.is_object() && node.contains("children") && node["children"].is_array()) {
        const auto& children = node["children"];
        for (const auto& child : children.get_array()) {
            auto childProjs = traverse(child, p, aggAliases, projAliases);
            childProjections.insert(childProjections.end(), childProjs.begin(), childProjs.end());
        }
    } else if (node.is_array()) {
        for (const auto& child : node.get_array()) {
            auto childProjs = traverse(child, p, aggAliases, projAliases);
            childProjections.insert(childProjections.end(), childProjs.begin(), childProjs.end());
        }
        return childProjections;
    }

    if (!node.is_object()) return {};

    std::string name = node.contains("name") && node["name"].is_string() ? node["name"].get_string() : "";
    std::string nameLower = tolower_copy(name);
    
    std::vector<std::string> myProjections;

    // Parse projections for this node
    if (node.contains("extra_info") && node["extra_info"].is_object()) {
        const auto& ei = node["extra_info"];
        json projNode;
        if (ei.contains("Projections")) projNode = ei["Projections"];
        
        auto processProj = [&](std::string proj) {
            if (proj.find("__internal_") != std::string::npos) {
                size_t start = proj.find('(');
                size_t end = proj.rfind(')');
                if (start != std::string::npos && end != std::string::npos && end > start) {
                    proj = proj.substr(start + 1, end - start - 1);
                }
            }
            proj = resolveColumnRef(proj, childProjections);

            // DuckDB sometimes reports computed expressions only by their output name (alias).
            // If we can recover the original expression from the SQL, inline it as "expr AS alias"
            // so downstream TableResult postprocess can evaluate it.
            {
                std::string trimmed = trim_copy(proj);
                const bool looksLikeBareIdent = (trimmed.find('(') == std::string::npos &&
                                                trimmed.find(' ') == std::string::npos);
                if (looksLikeBareIdent) {
                    auto it = projAliases.find(trimmed);
                    if (it != projAliases.end() && !it->second.empty()) {
                        proj = it->second + " AS " + trimmed;
                    }
                }
            }
            myProjections.push_back(proj);
        };

        if (projNode.is_array()) {
            for (const auto& item : projNode.get_array()) {
                if (item.is_string()) processProj(item.get_string());
            }
        } else if (projNode.is_string()) {
            processProj(projNode.get_string());
        }
    }

    // extra_info string fallback
    std::string extra;
    if (node.contains("extra_info") && node["extra_info"].is_string()) {
        extra = node["extra_info"].get_string();
    }

    if (nameLower.find("scan") != std::string::npos || nameLower == "get" || nameLower.find("read_csv") != std::string::npos) {
        IRNode s; s.type = IRNode::Type::Scan;
        std::string t;
        std::string scanFilters;
        if (node.contains("extra_info") && node["extra_info"].is_object()) {
            if (node["extra_info"].contains("Table") && node["extra_info"]["Table"].is_string()) {
                t = node["extra_info"]["Table"].get_string();
            }
            // Extract filters from Scan
            if (node["extra_info"].contains("Filters")) {
                const auto& f = node["extra_info"]["Filters"];
                if (f.is_array()) {
                    for (const auto& item : f.get_array()) {
                        if (!scanFilters.empty()) scanFilters += " AND ";
                        scanFilters += item.get_string();
                    }
                } else if (f.is_string()) {
                    scanFilters = f.get_string();
                }
            }
        }
        if (t.empty()) {
            // DuckDB READ_CSV scans for views often omit a "Table" field.
            // Infer the logical table by looking at projected column name prefixes.
            auto inferTable = [&](const std::vector<std::string>& projs) -> std::string {
                int l = 0, o = 0, c = 0;
                for (auto p : projs) {
                    p = resolveColumnRef(p, childProjections);
                    auto dot = p.rfind('.');
                    if (dot != std::string::npos && dot + 1 < p.size()) p = p.substr(dot + 1);
                    if (p.rfind("l_", 0) == 0) ++l;
                    else if (p.rfind("o_", 0) == 0) ++o;
                    else if (p.rfind("c_", 0) == 0) ++c;
                }
                if (c > o && c > l) return "customer";
                if (o > c && o > l) return "orders";
                if (l > 0) return "lineitem";
                return "";
            };

            t = inferTable(myProjections);
        }
        if (t.empty()) {
            t = extra;
            auto pos = t.find('['); if (pos != std::string::npos) t = t.substr(0,pos);
            auto sp = t.find('\n'); if (sp != std::string::npos) t = t.substr(0,sp);
        }
        if (t.empty()) t = "lineitem";
        s.scan.table = t;
        p.nodes.push_back(s);

        if (!scanFilters.empty()) {
            IRNode f; f.type = IRNode::Type::Filter;
            f.filter.predicate = scanFilters;
            p.nodes.push_back(f);
        }
    } else if (nameLower.find("filter") != std::string::npos) {
        IRNode f; f.type = IRNode::Type::Filter; 
        std::string predicate;
        if (node.contains("extra_info") && node["extra_info"].is_object()) {
            const auto& ei = node["extra_info"];
            // DuckDB often stores the predicate here for FILTER nodes.
            if (ei.contains("Expression") && ei["Expression"].is_string()) {
                predicate = ei["Expression"].get_string();
            } else if (ei.contains("Condition") && ei["Condition"].is_string()) {
                predicate = ei["Condition"].get_string();
            }
            if (ei.contains("Filters")) {
                if (ei["Filters"].is_array()) {
                    for (const auto& item : ei["Filters"].get_array()) {
                        if (!predicate.empty()) predicate += " AND ";
                        predicate += item.get_string();
                    }
                } else if (ei["Filters"].is_string()) {
                    predicate = ei["Filters"].get_string();
                }
            }
        }
        if (predicate.empty()) predicate = extra;
        f.filter.predicate = predicate; 
        p.nodes.push_back(f);
    } else if (nameLower.find("group_by") != std::string::npos) {
        IRNode g; g.type = IRNode::Type::GroupBy;
        
        if (node.contains("extra_info") && node["extra_info"].is_object()) {
            const auto& ei = node["extra_info"];
            
            if (ei.contains("Groups")) {
                const auto& groupsNode = ei["Groups"];
                auto processGroup = [&](std::string col) {
                    col = resolveColumnRef(col, childProjections);
                    if (!col.empty()) g.groupBy.keys.push_back(col);
                };
                if (groupsNode.is_array()) {
                    for (const auto& item : groupsNode.get_array()) if (item.is_string()) processGroup(item.get_string());
                } else if (groupsNode.is_string()) {
                    processGroup(groupsNode.get_string());
                }
            }
            
            if (ei.contains("Aggregates")) {
                const auto& aggsNode = ei["Aggregates"];
                auto processAgg = [&](std::string agg) {
                    size_t start = agg.find('(');
                    size_t end = agg.rfind(')');
                    if (start != std::string::npos && end != std::string::npos && end > start) {
                        std::string colRef = agg.substr(start + 1, end - start - 1);
                        if (!colRef.empty()) {
                            std::string resolved = resolveColumnRef(colRef, childProjections);
                            agg = agg.substr(0, start + 1) + resolved + agg.substr(end);
                        }
                    }
                    g.groupBy.aggs.push_back(agg);

                    // Attach SQL SELECT alias if we can match this aggregate expression.
                    const std::string norm = normalize_expr_for_alias(agg);
                    auto itAlias = aggAliases.find(norm);
                    if (itAlias != aggAliases.end() && !itAlias->second.empty()) {
                        g.groupBy.aggs.back() = agg + " AS " + itAlias->second;
                    }

                    std::string aggLower = tolower_copy(agg);
                    if (aggLower.find("sum(") != std::string::npos) g.groupBy.aggFuncs.push_back("sum");
                    else if (aggLower.find("avg(") != std::string::npos) g.groupBy.aggFuncs.push_back("avg");
                    else if (aggLower.find("min(") != std::string::npos) g.groupBy.aggFuncs.push_back("min");
                    else if (aggLower.find("max(") != std::string::npos) g.groupBy.aggFuncs.push_back("max");
                    else if (aggLower.find("count(") != std::string::npos || aggLower.find("count_star") != std::string::npos) g.groupBy.aggFuncs.push_back("count");
                    else g.groupBy.aggFuncs.push_back("sum");
                };
                if (aggsNode.is_array()) {
                    for (const auto& item : aggsNode.get_array()) if (item.is_string()) processAgg(item.get_string());
                } else if (aggsNode.is_string()) {
                    processAgg(aggsNode.get_string());
                }
            }
        }
        p.nodes.push_back(g);
    } else if (name.find("PROJECTION") != std::string::npos) {
        IRNode pr; pr.type = IRNode::Type::Project;
        pr.project.columns = myProjections;
        p.nodes.push_back(pr);
    } else if (name == "UNGROUPED_AGGREGATE" || name == "AGGREGATE") {
        IRNode a; a.type = IRNode::Type::Aggregate; a.aggregate.func = "sum"; 
        if (node.contains("extra_info") && node["extra_info"].is_object()) {
             const auto& ei = node["extra_info"];
             if (ei.contains("Aggregates")) {
                 // Simplified handling for ungrouped
                 json aggs = ei["Aggregates"];
                 std::string exprStr;
                 if (aggs.is_string()) exprStr = aggs.get_string();
                 else if (aggs.is_array() && aggs.size()>0) exprStr = aggs[0].get_string();
                 
                 // Resolve #0, #1 etc.
                 size_t start = exprStr.find('(');
                 size_t end = exprStr.rfind(')');
                 if (start != std::string::npos && end != std::string::npos && end > start) {
                     std::string colRef = exprStr.substr(start + 1, end - start - 1);
                     std::string resolved = resolveColumnRef(colRef, childProjections);
                     
                     // Strip function wrapper if it matches func
                     std::string funcName = exprStr.substr(0, start);
                     std::string lowerFunc = tolower_copy(funcName);
                     auto canonicalAgg = [&](const std::string& f) -> std::string {
                         // DuckDB sometimes uses internal names like sum_no_overflow(...)
                         // Normalize any prefix match to the canonical SQL aggregate.
                         if (f.rfind("sum", 0) == 0) return "sum";
                         if (f.rfind("avg", 0) == 0) return "avg";
                         if (f.rfind("min", 0) == 0) return "min";
                         if (f.rfind("max", 0) == 0) return "max";
                         if (f.rfind("count", 0) == 0) return "count";
                         return "";
                     };

                     if (auto canon = canonicalAgg(lowerFunc); !canon.empty()) {
                         a.aggregate.expr = resolved;
                         a.aggregate.func = canon;
                     } else {
                         a.aggregate.expr = exprStr.substr(0, start + 1) + resolved + exprStr.substr(end);
                     }
                 } else {
                     a.aggregate.expr = resolveColumnRef(exprStr, childProjections);
                 }
             }
        }
        if (a.aggregate.expr.empty()) a.aggregate.expr = extra;

        // Check for arithmetic ops
        if (a.aggregate.expr.find('*') != std::string::npos || 
            a.aggregate.expr.find('/') != std::string::npos ||
            a.aggregate.expr.find('+') != std::string::npos ||
            a.aggregate.expr.find('-') != std::string::npos) {
            a.aggregate.hasExpression = true;
        }

        p.nodes.push_back(a);
    } else if (name == "ORDER_BY" || name == "ORDER" || nameLower.find("top_n") != std::string::npos) {
        IRNode o; o.type = IRNode::Type::OrderBy;
        if (node.contains("extra_info") && node["extra_info"].is_object()) {
            const auto& ei = node["extra_info"];
            if (ei.contains("Order By")) {
                const auto& ob = ei["Order By"];
                auto processOrder = [&](std::string s) {
                    // Format: "expr DESC" or "expr ASC"
                    bool asc = true;
                    if (s.size() > 5 && s.substr(s.size()-5) == " DESC") {
                        asc = false;
                        s = s.substr(0, s.size()-5);
                    } else if (s.size() > 4 && s.substr(s.size()-4) == " ASC") {
                        asc = true;
                        s = s.substr(0, s.size()-4);
                    }

                    // Optional: "... NULLS FIRST/LAST".
                    s = strip_order_nulls_clause(std::move(s));

                    // Resolve #0, #1 etc and normalize qualified identifiers like
                    // memory.main.lineitem.l_quantity -> l_quantity
                    s = resolveColumnRef(s, childProjections);
                    // Trim spaces
                    if (!s.empty()) {
                        size_t first = s.find_first_not_of(" \t\n\r");
                        if (first != std::string::npos) s.erase(0, first);
                        size_t last = s.find_last_not_of(" \t\n\r");
                        if (last != std::string::npos) s.erase(last + 1);
                    }
                    // If it's a bare qualified column (no spaces/paren), keep the last segment.
                    if (s.find('(') == std::string::npos && s.find(' ') == std::string::npos) {
                        auto dot = s.rfind('.');
                        if (dot != std::string::npos && dot + 1 < s.size()) {
                            s = s.substr(dot + 1);
                        }
                    } else {
                        // If this ORDER BY is an aggregate expression and the SELECT list provided
                        // an alias, rewrite to the alias so execution can order by the output column.
                        const std::string norm = normalize_expr_for_alias(s);
                        auto itAlias = aggAliases.find(norm);
                        if (itAlias != aggAliases.end() && !itAlias->second.empty()) {
                            s = itAlias->second;
                        }
                    }
                    o.orderBy.columns.push_back(s);
                    o.orderBy.ascending.push_back(asc);
                };
                if (ob.is_array()) {
                    for (const auto& item : ob.get_array()) if (item.is_string()) processOrder(item.get_string());
                } else if (ob.is_string()) {
                    processOrder(ob.get_string());
                }
            }
        }
        if (o.orderBy.columns.empty()) {
            o.orderBy.ascending.push_back(true); // Fallback
        }
        p.nodes.push_back(o);
    } else if (name == "LIMIT") {
        IRNode l; l.type = IRNode::Type::Limit;
        std::regex re("(\\d+)");
        std::smatch m;
        if (std::regex_search(extra, m, re)) l.limit.count = std::stoll(m[1].str());
        else l.limit.count = 10;
        p.nodes.push_back(l);
    } else if (name.find("JOIN") != std::string::npos) {
        IRNode j; j.type = IRNode::Type::Join;
        j.join.joinType = "inner";
        if (node.contains("extra_info") && node["extra_info"].is_object()) {
            const auto& ei = node["extra_info"];
            if (ei.contains("Join Type") && ei["Join Type"].is_string()) j.join.joinType = tolower_copy(ei["Join Type"].get_string());
            if (ei.contains("Conditions") && ei["Conditions"].is_string()) j.join.condition = ei["Conditions"].get_string();
        }
        p.nodes.push_back(j);
    }

    return myProjections;
}

Plan Planner::fromSQL(const std::string& sql) {
    Plan p;
    const auto aggAliases = parse_select_agg_aliases(sql);
    const auto projAliases = parse_select_proj_aliases(sql);
    std::string raw = DuckDBAdapter::explainJSON(sql);
    
    // Extract JSON array from DuckDB output (skip header, find balanced brackets)
    std::string jsonStr;
    auto start = raw.find('[');
    if (start != std::string::npos) {
        // Find matching closing bracket
        int depth = 0;
        size_t end = start;
        for (size_t i = start; i < raw.size(); i++) {
            if (raw[i] == '[') depth++;
            else if (raw[i] == ']') {
                depth--;
                if (depth == 0) {
                    end = i + 1;
                    break;
                }
            }
        }
        jsonStr = raw.substr(start, end - start);
        // Remove any trailing junk (e.g., shell prompt '%')
        while (!jsonStr.empty() && (jsonStr.back() == '%' || jsonStr.back() == '\n' || jsonStr.back() == '\r')) {
            jsonStr.pop_back();
        }
    } else {
        jsonStr = raw;
    }
    
    // Try to parse DuckDB JSON (bug in nlohmann::json has been fixed)
    bool ok = false;
    try {
        json j = json::parse(jsonStr);
        if (j.is_array() && j.size() > 0) {
            if (const char* dbg = std::getenv("GPUDB_DEBUG_PLAN_JSON")) {
                (void)dbg;
                std::cerr << "[Planner] DuckDB raw bytes: " << raw.size() << "\n";
                std::cerr << "[Planner] Extracted JSON bytes: " << jsonStr.size() << "\n";
                if (j[0].contains("name") && j[0]["name"].is_string()) {
                    std::cerr << "[Planner] Root op: " << j[0]["name"].get_string() << "\n";
                }
                std::string snippet = jsonStr.substr(0, std::min<size_t>(jsonStr.size(), 400));
                std::cerr << "[Planner] JSON snippet: " << snippet << (jsonStr.size() > 400 ? "..." : "") << "\n";
            }
            traverse(j[0], p, aggAliases, projAliases);
            ok = !p.nodes.empty(); // Success if we got any nodes
        }
    } catch (const std::exception& e) {
        // Fall back to regex parser if JSON parsing fails
        ok = false;
    } catch (...) {
        // Fall back to regex parser if JSON parsing fails
        ok = false;
    }

    // DuckDB can represent ORDER BY ... LIMIT as a TopN/other node and omit a
    // standalone LIMIT node from the JSON plan. Our host dispatch expects a
    // Limit IR node (e.g., to print only top-K sorted indices), so if JSON
    // parsing succeeded but no Limit node exists, recover it from the SQL text.
    if (ok) {
        bool hasLimit = false;
        for (const auto& n : p.nodes) {
            if (n.type == IRNode::Type::Limit) {
                hasLimit = true;
                break;
            }
        }
        if (!hasLimit) {
            std::regex re_limit(R"(limit\s+(\d+))", std::regex::icase);
            std::smatch m;
            if (std::regex_search(sql, m, re_limit) && m.size() > 1) {
                IRNode l;
                l.type = IRNode::Type::Limit;
                l.limit.count = std::stoll(m[1].str());
                l.limit.offset = 0;
                p.nodes.push_back(l);
            }
        }
    }
    
    if (!ok) {
        std::string low = tolower_copy(sql);
        std::string table; std::string predicate; std::string aggexpr;
        
        // Check for JOIN
        std::regex re_join(R"(from\s+([A-Za-z_][A-Za-z0-9_]*)\s+join\s+([A-Za-z_][A-Za-z0-9_]*)\s+on\s+([^\s]+)\s*=\s*([^\s]+))", std::regex::icase);
        std::smatch m_join;
        if (std::regex_search(sql, m_join, re_join) && m_join.size()>4) {
            // Found JOIN
            table = m_join[1].str();
            std::string rightTable = m_join[2].str();
            
            IRNode s; s.type = IRNode::Type::Scan; s.scan.table = table; p.nodes.push_back(s);
            IRNode s2; s2.type = IRNode::Type::Scan; s2.scan.table = rightTable; p.nodes.push_back(s2);
            IRNode j; j.type = IRNode::Type::Join;
            j.join.rightTable = rightTable;
            j.join.condition = m_join[3].str() + "=" + m_join[4].str();
            j.join.joinType = "inner";
            p.nodes.push_back(j);
        } else {
            // Simple query
            std::regex re_from(R"(from\s+([A-Za-z_][A-Za-z0-9_\.]*)\b)", std::regex::icase);
            std::smatch m;
            if (std::regex_search(sql, m, re_from) && m.size()>1) table = m[1].str();
            if (table.empty()) table = "lineitem";
            IRNode s; s.type = IRNode::Type::Scan; s.scan.table = table; p.nodes.push_back(s);
        }
        
        // Parse aggregation function: SUM, COUNT, AVG, MIN, MAX
        std::string aggFunc;
        std::regex re_agg(R"(select\s+(sum|count|avg|min|max)\s*\()", std::regex::icase);
        std::smatch m;
        if (std::regex_search(sql, m, re_agg)) {
            aggFunc = m[1].str();
            std::transform(aggFunc.begin(), aggFunc.end(), aggFunc.begin(), ::tolower);
            
            // Find matching closing parenthesis
            size_t start = m.position() + m.length();
            int depth = 1;
            size_t end = start;
            for (size_t i = start; i < sql.size() && depth > 0; ++i) {
                if (sql[i] == '(') depth++;
                else if (sql[i] == ')') {
                    depth--;
                    if (depth == 0) {
                        end = i;
                        break;
                    }
                }
            }
            aggexpr = sql.substr(start, end - start);
        }
        std::regex re_where(R"(where\s+(.+?)(?:\s+group\s+by|\s+order\s+by|\s+limit|$))", std::regex::icase);
        if (std::regex_search(sql, m, re_where) && m.size()>1) predicate = m[1].str();
        
        // Parse ORDER BY
        std::regex re_orderby(R"(order\s+by\s+([A-Za-z_][A-Za-z0-9_]*)\s*(asc|desc)?)", std::regex::icase);
        if (std::regex_search(sql, m, re_orderby) && m.size()>1) {
            IRNode o; o.type = IRNode::Type::OrderBy;
            o.orderBy.columns.push_back(m[1].str());
            bool isAsc = true;
            if (m.size() > 2 && m[2].matched) {
                std::string dir = m[2].str();
                std::transform(dir.begin(), dir.end(), dir.begin(), ::tolower);
                isAsc = (dir != "desc");
            }
            o.orderBy.ascending.push_back(isAsc);
            p.nodes.push_back(o);
        }
        
        // Parse GROUP BY
        std::regex re_groupby(R"(group\s+by\s+([A-Za-z_][A-Za-z0-9_,\s]*?)(?:\s+order\s+by|\s+limit|$))", std::regex::icase);
        if (std::regex_search(sql, m, re_groupby) && m.size()>1) {
            IRNode g; g.type = IRNode::Type::GroupBy;
            std::string groupCols = m[1].str();
            // Split by comma for multiple columns
            size_t start = 0;
            while (start < groupCols.size()) {
                size_t comma = groupCols.find(',', start);
                if (comma == std::string::npos) comma = groupCols.size();
                std::string col = groupCols.substr(start, comma - start);
                // Trim spaces
                col.erase(0, col.find_first_not_of(" \t\n\r"));
                col.erase(col.find_last_not_of(" \t\n\r") + 1);
                if (!col.empty()) g.groupBy.keys.push_back(col);
                start = comma + 1;
            }
            if (!aggexpr.empty()) {
                g.groupBy.aggs.push_back(aggexpr);
                // Also extract and store the function name
                if (!aggFunc.empty()) {
                    g.groupBy.aggFuncs.push_back(aggFunc);
                } else {
                    g.groupBy.aggFuncs.push_back("sum");  // default
                }
            }
            p.nodes.push_back(g);
        }
        
        // Parse LIMIT
        std::regex re_limit(R"(limit\s+(\d+))", std::regex::icase);
        if (std::regex_search(sql, m, re_limit) && m.size()>1) {
            IRNode l; l.type = IRNode::Type::Limit;
            l.limit.count = std::stoll(m[1].str());
            l.limit.offset = 0;
            p.nodes.push_back(l);
        }
        
        // Only add aggregate if aggregation function is present AND no GROUP BY
        if (!aggexpr.empty() && !aggFunc.empty()) {
            bool hasGroupBy = false;
            for (auto& n : p.nodes) if (n.type == IRNode::Type::GroupBy) hasGroupBy = true;
            
            if (!hasGroupBy) {
                if (!predicate.empty()) { IRNode f; f.type = IRNode::Type::Filter; f.filter.predicate = predicate; p.nodes.push_back(f); }
                IRNode a; a.type = IRNode::Type::Aggregate; a.aggregate.func = aggFunc; a.aggregate.expr = aggexpr;
                // Check if expression contains arithmetic operators
                a.aggregate.hasExpression = (aggexpr.find('*') != std::string::npos || 
                                             aggexpr.find('/') != std::string::npos ||
                                             aggexpr.find('+') != std::string::npos ||
                                             aggexpr.find('-') != std::string::npos);
                p.nodes.push_back(a);
            }
        }
    }
    /*
    // Ensure order Scan -> Join/Filter -> GroupBy -> OrderBy -> Limit -> Aggregate
    std::vector<IRNode> ordered;
    for (auto& n : p.nodes) if (n.type==IRNode::Type::Scan) ordered.push_back(n);
    for (auto& n : p.nodes) if (n.type==IRNode::Type::Join) ordered.push_back(n);
    for (auto& n : p.nodes) if (n.type==IRNode::Type::Filter) ordered.push_back(n);
    for (auto& n : p.nodes) if (n.type==IRNode::Type::GroupBy) ordered.push_back(n);
    for (auto& n : p.nodes) if (n.type==IRNode::Type::OrderBy) ordered.push_back(n);
    for (auto& n : p.nodes) if (n.type==IRNode::Type::Limit) ordered.push_back(n);
    for (auto& n : p.nodes) if (n.type==IRNode::Type::Aggregate) ordered.push_back(n);
    p.nodes.swap(ordered);
    */
    return p;
}

} // namespace engine
