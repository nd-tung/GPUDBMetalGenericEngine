// ============================================================================
// PlannerSQLParser.cpp — SQL-level parsing (column refs, aliases, SELECT lists)
// ============================================================================
#include "PlannerInternal.hpp"
#include "EnvUtil.hpp"
#include <iostream>
#include <algorithm>
#include <cctype>
#include "Logger.hpp"

namespace engine {

// Resolve #0, #1, etc. references to actual column names
std::string resolveColRef(const std::string& ref, const std::vector<std::string>& projections) {
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
            } catch (...) {
                if (env_truthy("GPUDB_DEBUG_PLANNER"))
                    LOG_ERROR("Planner", "resolveProjectionRefs: failed to parse index '" << numStr << "'\n");
            }
        }
        pos++;
    }
    return expr;
}

// Parse the outermost SELECT column names (alias if present, else expression)
std::vector<std::string> parseSelectColumnNames(const std::string& sql) {
    std::vector<std::string> cols;
    std::string sl = tolower_str(sql);

    // Find the first SELECT that is NOT preceded by '(' at the same depth
    // i.e. the outermost SELECT
    size_t selPos = std::string::npos;
    for (size_t i = 0; i + 6 <= sl.size(); ++i) {
        if (sl.compare(i, 6, "select") == 0) {
            if (i == 0 || !std::isalnum(static_cast<unsigned char>(sl[i-1]))) {
                // Check we're not inside parentheses
                int d = 0;
                for (size_t k = 0; k < i; ++k) {
                    if (sl[k] == '(') d++;
                    else if (sl[k] == ')') d--;
                }
                if (d == 0) { selPos = i; break; }
            }
        }
    }
    if (selPos == std::string::npos) return cols;

    // Skip optional DISTINCT
    size_t afterSelect = selPos + 6;
    {
        std::string rest = trim_str(sl.substr(afterSelect));
        if (rest.compare(0, 8, "distinct") == 0 &&
            (rest.size() == 8 || !std::isalnum(static_cast<unsigned char>(rest[8])))) {
            afterSelect = sl.find("distinct", afterSelect) + 8;
        }
    }

    // Find FROM at depth 0
    int depth = 0;
    size_t fromPos = std::string::npos;
    for (size_t i = afterSelect; i + 4 <= sl.size(); ++i) {
        char c = sl[i];
        if (c == '(') depth++;
        else if (c == ')') { depth--; if (depth < 0) break; }
        if (depth == 0 && sl.compare(i, 4, "from") == 0 &&
            (i == 0 || !std::isalnum(static_cast<unsigned char>(sl[i-1]))) &&
            (i + 4 >= sl.size() || !std::isalnum(static_cast<unsigned char>(sl[i+4])))) {
            fromPos = i;
            break;
        }
    }
    if (fromPos == std::string::npos) return cols;

    // Extract column list using original SQL (preserve case)
    std::string list = sql.substr(afterSelect, fromPos - afterSelect);
    std::string listLower = tolower_str(list);

    depth = 0;
    size_t start = 0;
    for (size_t i = 0; i <= list.size(); ++i) {
        if (i == list.size() || (list[i] == ',' && depth == 0)) {
            std::string item = trim_str(list.substr(start, i - start));
            start = i + 1;
            if (item.empty()) continue;

            std::string itemLower = tolower_str(item);
            // Check for " as " alias
            size_t asPos = itemLower.rfind(" as ");
            if (asPos != std::string::npos) {
                std::string alias = trim_str(item.substr(asPos + 4));
                alias = strip_parens(std::move(alias));
                // Remove quotes
                if (alias.size() >= 2 && (alias.front() == '"' || alias.front() == '\'') &&
                    alias.back() == alias.front()) {
                    alias = alias.substr(1, alias.size() - 2);
                }
                cols.push_back(tolower_str(alias));
            } else {
                // No alias — use the expression as-is (lowercased, stripped)
                std::string name = trim_str(item);
                // Remove whitespace for normalization
                std::string norm;
                for (char c : name) {
                    if (!std::isspace(static_cast<unsigned char>(c))) norm += std::tolower(static_cast<unsigned char>(c));
                }
                cols.push_back(norm);
            }
            continue;
        }
        if (list[i] == '(') depth++;
        else if (list[i] == ')') depth--;
    }

    // If the only column is "*", return empty to mean "all columns"
    if (cols.size() == 1 && cols[0] == "*") return {};

    return cols;
}

// Parse aggregate aliases from SQL SELECT clause
std::unordered_map<std::string, std::string> parseSelectAliases(const std::string& sql) {
    std::unordered_map<std::string, std::string> out;
    
    // Helper to parse aliases from one SELECT...FROM block
    auto parseOneSelectBlock = [&out](const std::string& sql, size_t selPos) {
        std::string sl = tolower_str(sql);
        
        // Find FROM at depth 0 (relative to this SELECT)
        int depth = 0;
        size_t fromPos = std::string::npos;
        for (size_t i = selPos + 6; i + 4 <= sl.size(); ++i) {
            char c = sl[i];
            if (c == '(') depth++;
            else if (c == ')') {
                depth--;
                if (depth < 0) break;  // Gone beyond our scope
            }
            if (depth == 0 && sl.compare(i, 4, "from") == 0) { fromPos = i; break; }
        }
        if (fromPos == std::string::npos) return;
        
        std::string list = sql.substr(selPos + 6, fromPos - (selPos + 6));
        depth = 0;
        size_t start = 0;
        for (size_t i = 0; i <= list.size(); ++i) {
            if (i == list.size() || (list[i] == ',' && depth == 0)) {
                std::string item = trim_str(list.substr(start, i - start));
                start = i + 1;
                if (item.empty()) continue;
                
                std::string itemLower = tolower_str(item);
                size_t asPos = itemLower.rfind(" as ");
                if (asPos == std::string::npos) continue;
                
                std::string expr = trim_str(item.substr(0, asPos));
                std::string alias = trim_str(item.substr(asPos + 4));
                alias = strip_parens(std::move(alias));
                if (!alias.empty() && (alias.front() == '"' || alias.front() == '\'')) {
                    if (alias.size() >= 2 && alias.back() == alias.front()) {
                        alias = alias.substr(1, alias.size() - 2);
                    }
                }
                if (!alias.empty() && !expr.empty()) {
                    // Normalize expression for matching
                    std::string normExpr = tolower_str(expr);
                    normExpr.erase(std::remove_if(normExpr.begin(), normExpr.end(), 
                        [](unsigned char ch) { return std::isspace(ch); }), normExpr.end());
                    // Normalize operators: <> -> !=
                    normExpr = normalizeOperators(normExpr);
                    out[normExpr] = alias;
                    out[alias] = expr;  // reverse mapping too
                }
                continue;
            }
            if (list[i] == '(') depth++;
            else if (list[i] == ')') depth--;
        }
    };
    
    // Find ALL SELECT keywords (including in subqueries) and parse each
    std::string sl = tolower_str(sql);
    size_t pos = 0;
    while ((pos = sl.find("select", pos)) != std::string::npos) {
        // Make sure it's a word boundary (not in a comment or string)
        if (pos == 0 || !std::isalnum(sl[pos-1])) {
            parseOneSelectBlock(sql, pos);
        }
        pos += 6;
    }
    
    return out;
}

} // namespace engine
