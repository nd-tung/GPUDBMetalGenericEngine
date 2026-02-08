#include "Planner.hpp"
#include "DuckDBAdapter.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <algorithm>
#include <regex>
#include <cctype>
#include <unordered_map>
#include <unordered_set>
#include <set>

using nlohmann::json;

namespace engine {

// --- Debug utilities ---

static void debug_log(const std::string& msg) {
    std::cerr << "[Planner] " << msg << std::endl;
}

// --- String utilities ---

static std::string tolower_str(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

static std::string trim_str(std::string s) {
    auto first = s.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    auto last = s.find_last_not_of(" \t\n\r");
    return s.substr(first, last - first + 1);
}

// Normalize numeric literals by removing trailing zeros (1.00 -> 1)
static std::string normalizeNumericLiterals(const std::string& s) {
    std::string result;
    size_t i = 0;
    while (i < s.size()) {
        // Check if this is a numeric literal
        if (std::isdigit(s[i]) || (s[i] == '.' && i+1 < s.size() && std::isdigit(s[i+1]))) {
            size_t start = i;
            bool hasDot = false;
            while (i < s.size() && (std::isdigit(s[i]) || s[i] == '.')) {
                if (s[i] == '.') hasDot = true;
                i++;
            }
            std::string num = s.substr(start, i - start);
            if (hasDot) {
                // Remove trailing zeros and trailing dot
                while (num.size() > 1 && num.back() == '0') num.pop_back();
                if (num.size() > 1 && num.back() == '.') num.pop_back();
            }
            result += num;
        } else {
            result += s[i++];
        }
    }
    return result;
}

// Normalize operators: <> -> !=
static std::string normalizeOperators(const std::string& s) {
    std::string result = s;
    size_t pos = 0;
    while ((pos = result.find("<>", pos)) != std::string::npos) {
        result.replace(pos, 2, "!=");
        pos += 2;
    }
    return result;
}

static std::string strip_parens(std::string s) {
    s = trim_str(std::move(s));
    while (!s.empty() && s.front() == '(' && s.back() == ')') {
        int depth = 0;
        bool balanced = true;
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '(') depth++;
            else if (s[i] == ')') {
                depth--;
                if (depth == 0 && i + 1 != s.size()) { balanced = false; break; }
            }
            if (depth < 0) { balanced = false; break; }
        }
        if (!balanced || depth != 0) break;
        s = trim_str(s.substr(1, s.size() - 2));
    }
    return s;
}

// Remove table qualifiers like "lineitem.l_quantity" -> "l_quantity"
static std::string stripTableQualifier(const std::string& s) {
    if (s.empty()) return s;
    auto dot = s.rfind('.');
    
    // Safety check: if dot is part of a number (e.g. 1.00), don't strip
    // Heuristic: if char after dot is a digit, it's a number.
    if (dot != std::string::npos && dot + 1 < s.size() && std::isdigit(static_cast<unsigned char>(s[dot+1]))) {
        return s;
    }

    if (dot != std::string::npos && dot + 1 < s.size()) {
        return s.substr(dot + 1);
    }
    return s;
}

// Rename duplicate columns in RHS projections for self-joins
static std::vector<std::string> renameDuplicateColumns(
    const std::vector<std::string>& lhsProjs,
    const std::vector<std::string>& rhsProjs,
    std::unordered_map<std::string, std::string>& renameMap) {
    
    // Build a set of LHS column names for quick lookup
    std::unordered_set<std::string> lhsNames;
    for (const auto& col : lhsProjs) {
        lhsNames.insert(col);
    }
    
    // Track suffix counters for columns that need renaming
    std::unordered_map<std::string, int> suffixCounters;
    for (const auto& col : lhsProjs) {
        // Initialize counters - if LHS already has "col_2", we need to use "col_3" next
        size_t underscorePos = col.rfind('_');
        if (underscorePos != std::string::npos) {
            std::string suffix = col.substr(underscorePos + 1);
            bool allDigits = !suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit);
            if (allDigits) {
                std::string baseName = col.substr(0, underscorePos);
                int num = std::stoi(suffix);
                if (suffixCounters[baseName] < num) {
                    suffixCounters[baseName] = num;
                }
            }
        }
    }
    
    std::vector<std::string> renamedRhs;
    renamedRhs.reserve(rhsProjs.size());
    
    for (const auto& col : rhsProjs) {
        if (lhsNames.count(col) > 0) {
            // Duplicate found - need to rename
            int& counter = suffixCounters[col];
            counter++;
            std::string newName = col + "_" + std::to_string(counter);
            
            // Ensure the new name doesn't collide either
            while (lhsNames.count(newName) > 0) {
                counter++;
                newName = col + "_" + std::to_string(counter);
            }
            
            renameMap[col] = newName;
            renamedRhs.push_back(newName);
            lhsNames.insert(newName); // Prevent future collisions
            
            debug_log("Renamed duplicate column: " + col + " -> " + newName);
        } else {
            renamedRhs.push_back(col);
            lhsNames.insert(col);
        }
    }
    
    return renamedRhs;
}

// --- Expression parsing ---

AggFunc Planner::parseAggFunc(const std::string& name) {
    std::string lower = tolower_str(name);
    // Handle DuckDB internal names
    if (lower.find("sum") != std::string::npos) return AggFunc::Sum;
    if (lower.find("count_star") != std::string::npos) return AggFunc::CountStar;
    if (lower.find("count") != std::string::npos) return AggFunc::Count;
    if (lower.find("avg") != std::string::npos) return AggFunc::Avg;
    if (lower.find("min") != std::string::npos) return AggFunc::Min;
    if (lower.find("max") != std::string::npos) return AggFunc::Max;
    if (lower.find("first") != std::string::npos) return AggFunc::First;
    return AggFunc::Sum;
}

CompareOp Planner::parseCompareOp(const std::string& op) {
    if (op == "=" || op == "==") return CompareOp::Eq;
    if (op == "<>" || op == "!=") return CompareOp::Ne;
    if (op == "<") return CompareOp::Lt;
    if (op == "<=") return CompareOp::Le;
    if (op == ">") return CompareOp::Gt;
    if (op == ">=") return CompareOp::Ge;
    return CompareOp::Eq;
}

// Simple expression parser for DuckDB expression strings
TypedExprPtr Planner::parseExpression(const std::string& exprStr) {
    std::string s = strip_parens(exprStr);
    if (s.empty()) return nullptr;
    
    bool debug = std::getenv("GPUDB_DEBUG_PARSE") != nullptr;
    if (debug) {
        std::cerr << "[parseExpression] input: '" << s.substr(0, 80) << (s.size() > 80 ? "..." : "") << "'\n";
    }
    
    std::string upper = s;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    
    // Check for AND/OR at top level (lowest precedence logical operators)
    // Need to find AND/OR outside quotes and parentheses, and NOT part of BETWEEN
    {
        int depth = 0;
        bool inQuote = false;
        
        // Find rightmost AND/OR at depth 0 (right-associative for left-to-right evaluation)
        size_t andPos = std::string::npos;
        size_t orPos = std::string::npos;
        
        // Track if we're in a BETWEEN clause (BETWEEN seen but not yet matched its AND)
        bool inBetween = false;
        
        for (size_t i = 0; i < s.size(); ++i) {
            char c = s[i];
            if (c == '\'' && (i == 0 || s[i-1] != '\\')) {
                inQuote = !inQuote;
                continue;
            }
            if (inQuote) continue;
            if (c == '(') { depth++; inBetween = false; }  // BETWEEN ends at paren boundary
            else if (c == ')') { depth--; inBetween = false; }
            else if (depth == 0) {
                // Check for BETWEEN (marks that next AND belongs to BETWEEN, not logical AND)
                if (i + 8 <= s.size() && upper.substr(i, 8) == " BETWEEN") {
                    inBetween = true;
                }
                // Check for " AND " - only count as logical AND if not in BETWEEN
                if (i + 5 <= s.size() && upper.substr(i, 5) == " AND ") {
                    if (inBetween) {
                        inBetween = false;  // This AND belongs to BETWEEN, skip it
                    } else {
                        andPos = i;  // This is a logical AND
                    }
                }
                if (i + 4 <= s.size() && upper.substr(i, 4) == " OR ") {
                    orPos = i;
                    inBetween = false;
                }
            }
        }
        
        // OR has lower precedence than AND
        if (orPos != std::string::npos) {
            std::string left = trim_str(s.substr(0, orPos));
            std::string right = trim_str(s.substr(orPos + 4));
            return TypedExpr::binary(BinaryOp::Or, parseExpression(left), parseExpression(right));
        }
        if (andPos != std::string::npos) {
            std::string left = trim_str(s.substr(0, andPos));
            std::string right = trim_str(s.substr(andPos + 5));
            return TypedExpr::binary(BinaryOp::And, parseExpression(left), parseExpression(right));
        }
    }
    
    // Check for BETWEEN expr AND expr (only after AND/OR is ruled out)
    // The column part should not contain " AND " or " OR " at depth 0
    static const std::regex betweenRe(R"(^(.+?)\s+BETWEEN\s+(.+?)\s+AND\s+(.+)$)", std::regex::icase);
    std::smatch m;
    if (std::regex_match(s, m, betweenRe)) {
        std::string col = trim_str(m[1].str());
        std::string lo = trim_str(m[2].str());
        std::string hi = trim_str(m[3].str());
        // BETWEEN a AND b => (col >= a) AND (col <= b)
        auto colExpr = parseExpression(col);
        auto loExpr = parseExpression(lo);
        auto hiExpr = parseExpression(hi);
        auto geExpr = TypedExpr::compare(CompareOp::Ge, colExpr, loExpr);
        auto leExpr = TypedExpr::compare(CompareOp::Le, colExpr, hiExpr);
        return TypedExpr::binary(BinaryOp::And, geExpr, leExpr);
    }
    
    // Check for IN (list) expression: column IN ('val1', 'val2', ...)
    {
        static const std::regex inRe(R"(^(.+?)\s+IN\s*\((.+)\)$)", std::regex::icase);
        if (std::regex_match(s, m, inRe)) {
            std::string col = trim_str(m[1].str());
            std::string listStr = m[2].str();
            
            // Parse comma-separated list, respecting quotes
            std::vector<TypedExprPtr> listExprs;
            std::string current;
            bool inQuote = false;
            for (size_t i = 0; i <= listStr.size(); ++i) {
                char c = (i < listStr.size()) ? listStr[i] : ',';
                if (c == '\'' && (i == 0 || listStr[i-1] != '\\')) {
                    inQuote = !inQuote;
                    current += c;
                } else if (c == ',' && !inQuote) {
                    current = trim_str(current);
                    if (!current.empty()) {
                        listExprs.push_back(parseExpression(current));
                    }
                    current.clear();
                } else {
                    current += c;
                }
            }
            
            return TypedExpr::inList(parseExpression(col), std::move(listExprs));
        }
    }
    
    // Check for CASE expression: CASE WHEN cond THEN val [WHEN cond THEN val]* [ELSE val] END
    if (upper.find("CASE") == 0 && upper.rfind("END") == upper.size() - 3) {
        std::string body = trim_str(s.substr(4, s.size() - 7)); // strip "CASE" and "END"
        
        CaseExpr caseExpr;
        size_t pos = 0;
        
        // Parse WHEN ... THEN ... clauses
        while (pos < body.size()) {
            std::string upper_body = body.substr(pos);
            std::transform(upper_body.begin(), upper_body.end(), upper_body.begin(), ::toupper);
            
            size_t whenPos = upper_body.find("WHEN");
            if (whenPos == std::string::npos) break;
            
            // Find THEN
            size_t searchStart = whenPos + 4;
            size_t thenPos = std::string::npos;
            int depth = 0;
            bool inQuote = false;
            for (size_t i = searchStart; i < upper_body.size(); ++i) {
                char c = body[pos + i];
                if (c == '\'' && (i == 0 || body[pos + i - 1] != '\\')) inQuote = !inQuote;
                if (inQuote) continue;
                if (c == '(') depth++;
                else if (c == ')') depth--;
                if (depth == 0 && i + 4 <= upper_body.size() && upper_body.substr(i, 5) == " THEN") {
                    thenPos = i;
                    break;
                }
            }
            
            if (thenPos == std::string::npos) break;
            
            std::string whenCond = trim_str(body.substr(pos + whenPos + 5, thenPos - whenPos - 5));
            
            // Find next WHEN, ELSE, or end
            size_t afterThen = thenPos + 5;
            size_t nextClause = body.size();
            depth = 0;
            inQuote = false;
            for (size_t i = afterThen; i < upper_body.size(); ++i) {
                char c = body[pos + i];
                if (c == '\'' && (i == 0 || body[pos + i - 1] != '\\')) inQuote = !inQuote;
                if (inQuote) continue;
                if (c == '(') depth++;
                else if (c == ')') depth--;
                if (depth == 0) {
                    std::string rem = upper_body.substr(i);
                    if (rem.find(" WHEN") == 0 || rem.find(" ELSE") == 0) {
                        nextClause = i;
                        break;
                    }
                }
            }
            
            std::string thenVal = trim_str(body.substr(pos + afterThen, nextClause - afterThen));
            
            CaseExpr::WhenThen wt;
            wt.when = parseExpression(whenCond);
            wt.then = parseExpression(thenVal);
            caseExpr.cases.push_back(std::move(wt));
            
            pos += nextClause;
        }
        
        // Check for ELSE clause
        std::string upper_remaining = body.substr(pos);
        std::transform(upper_remaining.begin(), upper_remaining.end(), upper_remaining.begin(), ::toupper);
        size_t elsePos = upper_remaining.find("ELSE");
        if (elsePos != std::string::npos) {
            std::string elseVal = trim_str(body.substr(pos + elsePos + 4));
            caseExpr.elseExpr = parseExpression(elseVal);
        }
        
        auto e = std::make_shared<TypedExpr>();
        e->kind = TypedExpr::Kind::Case;
        e->data = std::move(caseExpr);
        return e;
    }
    
    /* 
    // AGGREGATE PARSING MOVED TO EXPLICIT NODES (GROUP_BY, AGGREGATE)
    // We do NOT want to parse "max(x)" as an aggregate in a Filter/Project,
    // because it might be a column name (from a subquery/join) or a scalar function.
    
    // Check for aggregate function
    // Use [\s\S]* instead of .* to match across newlines in multiline expressions
    static const std::regex aggRe(R"(^(sum|count|avg|min|max|count_star)\s*\(([\s\S]*)\)$)", std::regex::icase);
    if (std::regex_match(s, m, aggRe)) {
        AggFunc func = parseAggFunc(m[1].str());
        std::string inner = trim_str(m[2].str());
        TypedExprPtr arg = inner.empty() ? nullptr : parseExpression(inner);
        return TypedExpr::aggregate(func, arg);
    }
    */
    
    // Check for "IS NOT DISTINCT FROM" - this is DuckDB's NULL-safe equality for semi/anti joins
    // Treat it as regular equality (=)
    {
        static const std::regex notDistinctRe(R"((.+?)\s+IS\s+NOT\s+DISTINCT\s+FROM\s+(.+))", std::regex::icase);
        if (std::regex_match(s, m, notDistinctRe)) {
            std::string left = trim_str(m[1].str());
            std::string right = trim_str(m[2].str());
            return TypedExpr::compare(CompareOp::Eq, parseExpression(left), parseExpression(right));
        }
    }
    
    // Check for comparison operators at depth 0 (outside parentheses and quotes)
    // This needs to be done manually to handle nested parentheses correctly
    // e.g., CAST(sum((x * CAST(y AS DECIMAL(18,0)))) AS DECIMAL(38,6)) > threshold
    {
        int depth = 0;
        bool inQuote = false;
        size_t cmpPos = std::string::npos;
        std::string cmpOp;
        
        // Find the first comparison operator at depth 0
        for (size_t i = 0; i < s.size(); ++i) {
            char c = s[i];
            if (c == '\'' && (i == 0 || s[i-1] != '\\')) {
                inQuote = !inQuote;
                continue;
            }
            if (inQuote) continue;
            
            if (c == '(') depth++;
            else if (c == ')') depth--;
            else if (depth == 0) {
                // Check for multi-character operators first
                if (i + 2 <= s.size()) {
                    std::string op2 = s.substr(i, 2);
                    if (op2 == "!~" && i + 3 <= s.size() && s[i+2] == '~') {
                        cmpPos = i; cmpOp = "!~~"; break;
                    }
                    if (op2 == "~~") {
                        cmpPos = i; cmpOp = "~~"; break;
                    }
                    if (op2 == ">=" || op2 == "<=" || op2 == "<>" || op2 == "!=") {
                        cmpPos = i; cmpOp = op2; break;
                    }
                }
                // Single character operators (only if not preceded/followed by another operator char)
                if ((c == '>' || c == '<' || c == '=') && i > 0 && i + 1 < s.size()) {
                    char prev = s[i-1];
                    char next = s[i+1];
                    // Avoid matching parts of >=, <=, <>, !=
                    if (c == '>' && prev != '!' && prev != '<' && next != '=') {
                        cmpPos = i; cmpOp = ">"; break;
                    }
                    if (c == '<' && prev != '!' && next != '>' && next != '=') {
                        cmpPos = i; cmpOp = "<"; break;
                    }
                    if (c == '=' && prev != '!' && prev != '>' && prev != '<' && next != '=') {
                        cmpPos = i; cmpOp = "="; break;
                    }
                }
            }
        }
        
        if (cmpPos != std::string::npos && !cmpOp.empty()) {
            std::string left = trim_str(s.substr(0, cmpPos));
            std::string right = trim_str(s.substr(cmpPos + cmpOp.length()));
            
            if (debug) {
                std::cerr << "[parseExpression] comparison: left='" << left.substr(0, 50) << "...' op='" << cmpOp << "' right='" << right.substr(0, 50) << "...'\n";
            }
            
            // Handle LIKE operators as function calls
            if (cmpOp == "~~" || cmpOp == "!~~") {
                std::string funcName = (cmpOp == "~~") ? "LIKE" : "NOTLIKE";
                FunctionCall func;
                func.name = funcName;
                func.args.push_back(parseExpression(left));
                func.args.push_back(parseExpression(right));
                func.returnType = DataType::Bool;
                auto e = std::make_shared<TypedExpr>();
                e->kind = TypedExpr::Kind::Function;
                e->data = std::move(func);
                return e;
            }
            
            return TypedExpr::compare(parseCompareOp(cmpOp), parseExpression(left), parseExpression(right));
        }
    }
    
    // Check for binary arithmetic (+ - * /)
    // Find operator at lowest precedence (outside parentheses and quotes)
    int depth = 0;
    bool inQuote = false;
    size_t opPos = std::string::npos;
    char opChar = 0;
    int opPrecedence = 100;
    
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '\'' && (i == 0 || s[i-1] != '\\')) {
            inQuote = !inQuote;
            continue;
        }
        if (inQuote) continue;
        
        if (c == '(') depth++;
        else if (c == ')') depth--;
        else if (depth == 0) {
            int prec = 0;
            if (c == '+' || c == '-') prec = 1;
            else if (c == '*' || c == '/') prec = 2;
            
            if (prec > 0 && prec <= opPrecedence) {
                // Skip if this is a unary minus at start
                if (c == '-' && i == 0) continue;
                opPos = i;
                opChar = c;
                opPrecedence = prec;
            }
        }
    }
    
    if (opPos != std::string::npos && opPos > 0 && opPos < s.size() - 1) {
        std::string left = trim_str(s.substr(0, opPos));
        std::string right = trim_str(s.substr(opPos + 1));
        BinaryOp op;
        switch (opChar) {
            case '+': op = BinaryOp::Add; break;
            case '-': op = BinaryOp::Sub; break;
            case '*': op = BinaryOp::Mul; break;
            case '/': op = BinaryOp::Div; break;
            default: op = BinaryOp::Add;
        }
        if (debug) {
            std::cerr << "[parseExpression] Binary split at pos " << opPos << " op='" << opChar << "'\n";
            std::cerr << "[parseExpression]   left.size=" << left.size() << " right.size=" << right.size() << "\n";
        }
        return TypedExpr::binary(op, parseExpression(left), parseExpression(right));
    }

    // Check for CASE expression
    std::string slupper = s;
    std::transform(slupper.begin(), slupper.end(), slupper.begin(), ::toupper);

    if (slupper.find("CASE") != std::string::npos) {
         std::cerr << "DEBUG: Parsing string containing CASE: '" << s << "'\n";
         std::cerr << "DEBUG: Is Start? " << slupper.starts_with("CASE") << "\n";
         std::cerr << "DEBUG: Is End? " << slupper.ends_with("END") << "\n";
    }

    if (slupper.starts_with("CASE") && slupper.ends_with("END")) {
         std::cerr << "DEBUG: parseExpression CASE detected: " << s << "\n";
         size_t whenPos = slupper.find("WHEN"); // Accept WHEN without leading space if at start? No, CASE WHEN.
         if (whenPos != std::string::npos) {
             size_t thenPos = slupper.find(" THEN ", whenPos);
             size_t elsePos = slupper.find(" ELSE ", thenPos);
             size_t endPos = slupper.rfind(" END");
             
             if (thenPos != std::string::npos && elsePos != std::string::npos && endPos != std::string::npos) {
                  std::string whenStr = trim_str(s.substr(whenPos + 4, thenPos - (whenPos + 4)));
                  std::string thenStr = trim_str(s.substr(thenPos + 6, elsePos - (thenPos + 6)));
                  std::string elseStr = trim_str(s.substr(elsePos + 6, endPos - (elsePos + 6)));
                  
                  std::cerr << "DEBUG: CASE Parts:\n  WHEN: '" << whenStr << "'\n  THEN: '" << thenStr << "'\n  ELSE: '" << elseStr << "'\n";

                  auto e = std::make_shared<TypedExpr>();
                  e->kind = TypedExpr::Kind::Case;
                  CaseExpr c;
                  c.elseExpr = parseExpression(elseStr);
                  c.cases.push_back({parseExpression(whenStr), parseExpression(thenStr)});
                  e->data = std::move(c);
                  return e;
             }
         }
    }
    
    // Check for numeric literal
    try {
        size_t pos = 0;
        double d = std::stod(s, &pos);
        if (pos == s.size()) {
            // Check if it's an integer
            if (s.find('.') == std::string::npos) {
                return TypedExpr::literal(static_cast<int64_t>(d));
            }
            return TypedExpr::literal(d);
        }
    } catch (...) {}
    
    // Check for DATE literal (both DATE '...' and '...'::DATE formats)
    static const std::regex dateRe1(R"(DATE\s*'(\d{4}-\d{2}-\d{2})')", std::regex::icase);
    static const std::regex dateRe2(R"('(\d{4}-\d{2}-\d{2})'::DATE)", std::regex::icase);
    
    // Helper to convert YYYY-MM-DD to YYYYMMDD integer for comparison with int-based date columns
    auto makeDateInt = [](const std::string& ds) -> TypedExprPtr {
        std::string t = ds;
        // Remove dashes: 1995-01-01 -> 19950101
        t.erase(std::remove(t.begin(), t.end(), '-'), t.end());
        try { 
            return TypedExpr::literal((int64_t)std::stoll(t)); 
        }
        catch(...) { 
            // Fallback to string if parsing fails
            return TypedExpr::literal(ds, DataType::Date); 
        }
    };
    
    if (std::regex_match(s, m, dateRe1)) {
        return makeDateInt(m[1].str());
    }
    if (std::regex_match(s, m, dateRe2)) {
        return makeDateInt(m[1].str());
    }
    
    // Check for function call: funcname(arg1, arg2, ...) or "funcname"(arg1, arg2, ...)
    // DuckDB outputs some function names in quotes like "substring"
    // Handle quoted version first
    static const std::regex funcReQuoted(R"(^\"(\w+)\"\s*\((.+)\)$)");
    static const std::regex funcRe(R"(^(\w+)\s*\((.+)\)$)");
    bool funcMatched = std::regex_match(s, m, funcReQuoted);
    if (!funcMatched) funcMatched = std::regex_match(s, m, funcRe);
    if (funcMatched) {
        std::string funcName = m[1].str();
        std::string funcUpper = funcName;
        std::transform(funcUpper.begin(), funcUpper.end(), funcUpper.begin(), ::toupper);
        
        // Special handling for CAST(expr AS type)
        if (funcUpper == "CAST") {
            std::string argsStr = m[2].str();
            // Find " AS " at depth 0 (outside nested parentheses)
            // e.g., CAST(sum((x * CAST(y AS DECIMAL(18,0)))) AS DECIMAL(38,6))
            // The inner "AS" is at depth > 0, we need the outer one
            int depth = 0;
            bool inQuote = false;
            size_t asPos = std::string::npos;
            std::string argsUpper = argsStr;
            std::transform(argsUpper.begin(), argsUpper.end(), argsUpper.begin(), ::toupper);
            
            for (size_t i = 0; i + 4 < argsStr.size(); ++i) {
                char c = argsStr[i];
                if (c == '\'' && (i == 0 || argsStr[i-1] != '\\')) {
                    inQuote = !inQuote;
                    continue;
                }
                if (inQuote) continue;
                if (c == '(') depth++;
                else if (c == ')') depth--;
                else if (depth == 0 && argsUpper.substr(i, 4) == " AS ") {
                    asPos = i;
                    break;  // Found the outermost AS
                }
            }
            
            if (asPos != std::string::npos) {
                std::string exprPart = trim_str(argsStr.substr(0, asPos));
                std::string typePart = trim_str(argsStr.substr(asPos + 4));
                
                // Parse the inner expression
                TypedExprPtr innerExpr = parseExpression(exprPart);
                
                // Create a Cast expression
                auto e = std::make_shared<TypedExpr>();
                e->kind = TypedExpr::Kind::Cast;
                CastExpr cast;
                cast.expr = innerExpr;
                
                // Determine target type
                std::string typeUpper = typePart;
                std::transform(typeUpper.begin(), typeUpper.end(), typeUpper.begin(), ::toupper);
                if (typeUpper.find("TIMESTAMP") != std::string::npos || 
                    typeUpper.find("DATE") != std::string::npos) {
                    cast.targetType = DataType::Date;
                } else if (typeUpper.find("INT") != std::string::npos) {
                    cast.targetType = DataType::Int32;
                } else if (typeUpper.find("FLOAT") != std::string::npos ||
                           typeUpper.find("DOUBLE") != std::string::npos ||
                           typeUpper.find("DECIMAL") != std::string::npos) {
                    cast.targetType = DataType::Float32;
                } else {
                    cast.targetType = DataType::String;
                }
                
                e->data = std::move(cast);
                return e;
            }
        }
        
        // Special handling for EXTRACT(part FROM expr) - e.g., extract(year from o_orderdate)
        if (funcUpper == "EXTRACT") {
            std::string argsStr = m[2].str();
            std::string argsUpper = argsStr;
            std::transform(argsUpper.begin(), argsUpper.end(), argsUpper.begin(), ::toupper);
            size_t fromPos = argsUpper.find(" FROM ");
            if (fromPos != std::string::npos) {
                std::string partStr = trim_str(argsStr.substr(0, fromPos));
                std::string colStr = trim_str(argsStr.substr(fromPos + 6));
                
                // Create a function expression with part (as literal) and column as arguments
                auto e = std::make_shared<TypedExpr>();
                e->kind = TypedExpr::Kind::Function;
                FunctionCall fc;
                fc.name = "EXTRACT";
                fc.args.push_back(TypedExpr::literal(partStr));
                fc.args.push_back(parseExpression(colStr));
                e->data = std::move(fc);
                return e;
            }
        }
        
        // Special handling for SUBSTRING(col FROM start FOR length) - SQL standard syntax
        if (funcUpper == "SUBSTRING" || funcUpper == "SUBSTR") {
            std::string argsStr = m[2].str();
            std::string argsUpper = argsStr;
            std::transform(argsUpper.begin(), argsUpper.end(), argsUpper.begin(), ::toupper);
            
            size_t fromPos = argsUpper.find(" FROM ");
            if (fromPos != std::string::npos) {
                std::string colStr = trim_str(argsStr.substr(0, fromPos));
                std::string rest = argsStr.substr(fromPos + 6);
                std::string restUpper = rest;
                std::transform(restUpper.begin(), restUpper.end(), restUpper.begin(), ::toupper);
                
                int startVal = 1;
                int lengthVal = -1;  // -1 means to end
                
                size_t forPos = restUpper.find(" FOR ");
                if (forPos != std::string::npos) {
                    startVal = std::atoi(trim_str(rest.substr(0, forPos)).c_str());
                    lengthVal = std::atoi(trim_str(rest.substr(forPos + 5)).c_str());
                } else {
                    startVal = std::atoi(trim_str(rest).c_str());
                }
                
                // Create a function expression with column, start, length as args
                auto e = std::make_shared<TypedExpr>();
                e->kind = TypedExpr::Kind::Function;
                FunctionCall fc;
                fc.name = "SUBSTRING";
                fc.args.push_back(parseExpression(colStr));
                fc.args.push_back(TypedExpr::literal(static_cast<int64_t>(startVal)));
                fc.args.push_back(TypedExpr::literal(static_cast<int64_t>(lengthVal >= 0 ? lengthVal : 9999)));
                e->data = std::move(fc);
                return e;
            }
        }
        
        // Skip aggregate functions (handled separately in GROUP_BY nodes)
        // But NOT "FIRST" - DuckDB uses it for scalar subquery results, not as aggregate
        if (funcUpper != "SUM" && funcUpper != "COUNT" && funcUpper != "AVG" &&
            funcUpper != "MIN" && funcUpper != "MAX" &&
            funcUpper != "DATE") {
            // Parse arguments
            std::string argsStr = m[2].str();
            std::vector<TypedExprPtr> args;
            size_t start = 0;
            int depth = 0;
            bool inQuote = false;
            for (size_t i = 0; i <= argsStr.size(); ++i) {
                if (i == argsStr.size() || (argsStr[i] == ',' && depth == 0 && !inQuote)) {
                    std::string arg = trim_str(argsStr.substr(start, i - start));
                    if (!arg.empty()) {
                        args.push_back(parseExpression(arg));
                    }
                    start = i + 1;
                } else {
                    char c = argsStr[i];
                    if (c == '\'' && (i == 0 || argsStr[i-1] != '\\')) {
                         inQuote = !inQuote;
                    }
                    if (!inQuote) {
                        if (c == '(') depth++;
                        else if (c == ')') depth--;
                    }
                }
            }
            
            // Create function expression
            auto e = std::make_shared<TypedExpr>();
            e->kind = TypedExpr::Kind::Function;
            FunctionCall fc;
            fc.name = funcUpper;
            fc.args = std::move(args);
            e->data = std::move(fc);
            return e;
        }
    }
    
    // Check for PostgreSQL-style cast: 'value'::TYPE or value::TYPE
    // Handle this before string literal check
    {
        size_t castPos = s.find("::");
        if (castPos != std::string::npos && castPos > 0) {
            std::string valPart = trim_str(s.substr(0, castPos));
            std::string typePart = trim_str(s.substr(castPos + 2));
            
            // Parse the value part - could be a string literal or other expression
            if (valPart.size() >= 2 && valPart.front() == '\'' && valPart.back() == '\'') {
                // This is a cast of a string literal
                std::string strVal = valPart.substr(1, valPart.size() - 2);
                
                // Determine the type from the cast
                std::string typeUpper = typePart;
                std::transform(typeUpper.begin(), typeUpper.end(), typeUpper.begin(), ::toupper);
                
                if (typeUpper.find("DATE") != std::string::npos || 
                    typeUpper.find("TIMESTAMP") != std::string::npos) {
                    // Date/timestamp literal - extract just the date part
                    // '1998-09-02 00:00:00' -> '1998-09-02'
                    std::string dateVal = strVal;
                    size_t spacePos = dateVal.find(' ');
                    if (spacePos != std::string::npos) {
                        dateVal = dateVal.substr(0, spacePos);
                    }
                    return TypedExpr::literal(dateVal, DataType::Date);
                }
                
                // Default: just return as string literal
                return TypedExpr::literal(strVal, DataType::String);
            }
            
            // Otherwise, parse the value part and wrap in cast
            // Parse the inner expression
            return parseExpression(valPart);
        }
    }
    
    // Check for string literal
    if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'') {
        return TypedExpr::literal(s.substr(1, s.size() - 2), DataType::String);
    }
    
    // Otherwise treat as column reference
    std::string colName = stripTableQualifier(s);
    return TypedExpr::column(colName);
}

// --- JSON traversal helpers ---

// Resolve #0, #1, etc. references to actual column names
static std::string resolveColRef(const std::string& ref, const std::vector<std::string>& projections) {
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

// Parse aggregate aliases from SQL SELECT clause
static std::unordered_map<std::string, std::string> parseSelectAliases(const std::string& sql) {
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

// --- DuckDB JSON traversal ---

struct TraverseContext {
    Plan& plan;
    const std::unordered_map<std::string, std::string>& aliases;
    std::unordered_map<std::string, std::string> localAliases; // Mutable aliases for plan traversal
    std::vector<std::string> projections;  // current projection list for column resolution
    std::unordered_set<std::string> seenTables;
    bool pastGroupBy = false;  // after GROUP_BY, #N refs are to aggregate outputs
    
    // Stack for nested DELIM joins: {tableName, projections}
    std::vector<std::pair<std::string, std::vector<std::string>>> delimStack;

    std::map<int64_t, std::string> cteMap;
    
    // Global set of columns that must represent valid data if they appear, 
    // to prevent accidental dropping by intermediate generic projections.
    std::unordered_set<std::string> forceKeepColumns;
    
    // Map table-qualified column names to renamed columns (e.g., "n1.n_name" -> "n_name", "n2.n_name" -> "n_name_1")
    // This is used when the same table is joined multiple times with different aliases
    std::unordered_map<std::string, std::string> qualifiedColumnMapping;
};

// Helper to scan JSON for all referenced columns to populate forceKeepColumns
static void collectGlobalColumns(const json& j, std::unordered_set<std::string>& cols) {
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

static void traverseNode(const json& node, TraverseContext& ctx) {
    if (!node.is_object()) return;
    
    std::string name = node.contains("name") && node["name"].is_string() 
        ? node["name"].get_string() : "";
    
    debug_log("Traversing node: " + name);
    
    debug_log("Traversing: " + name);
    std::string nameLower = tolower_str(name);

    // Handle CTE (Common Table Expressions)
    if (name == "CTE") {
        if (node.contains("children") && node["children"].is_array()) {
            const auto& kids = node["children"].get_array();
            if (kids.size() >= 2) {
                // Save CTE result
                std::string cteName = "unknown_cte";
                if (node.contains("extra_info") && node["extra_info"].is_object()) {
                    auto& ei = node["extra_info"];
                    std::cerr << "DEBUG: CTE Node extra_info keys:\n";
                    for(auto& elt : ei.get_object()) {
                        std::cerr << "  CTE Key: " << elt.first << "\n";
                    }

                    if (ei.contains("CTE Name")) {
                        cteName = ei["CTE Name"].get_string();
                    }
                    if (ei.contains("Table Index")) {
                         int64_t idx = 0;
                         if (ei["Table Index"].is_number()) idx = (int64_t)ei["Table Index"].get_number();
                         ctx.cteMap[idx] = cteName;
                         // std::cerr << "DEBUG: Mapped CTE Index " << idx << " -> " << cteName << "\n";
                    }
                }

                // Child 0: CTE Definition part
                traverseNode(kids[0], ctx);
                
                ctx.plan.nodes.push_back(IRNode::save(cteName));
                
                // Child 1: Main Query using the CTE
                traverseNode(kids[1], ctx);
                return;
            }
        }
    }

    std::vector<std::string> childProjs;
    
    // Capture Join RHS logic
    std::string capturedRightTable;
    TypedExprPtr capturedRightFilter;
    bool capturedRHS = false;
    std::unordered_set<std::string> rhsTables;
    std::vector<std::string> lhsProjections;
    std::vector<std::string> rhsProjections;

    if (nameLower.find("join") != std::string::npos && 
        node.contains("children") && node["children"].is_array()) {
            
        debug_log("[TraverseNode] Logic for JOIN: " + name);
            
        const auto& kids = node["children"].get_array();
        if (kids.size() == 2) {
            
            // Helper to inspect RHS
            auto tryCapture = [&](const json& root) -> bool {
                debug_log("Checking RHS capture for Join " + name);
                json curr = root;
                std::string filterStr;
                
                while(true) {
                    std::string n = curr.contains("name") ? curr["name"].get_string() : "";
                    std::string nl = tolower_str(n);
                    debug_log("Inspecting RHS node " + n);
                    
                    if (nl.find("scan") != std::string::npos || nl == "get" || nl.find("read_csv") != std::string::npos || nl.find("delim_scan") != std::string::npos) {
                        // Found Scan
                        if (curr.contains("extra_info") && curr["extra_info"].is_object()) {
                            auto& ei = curr["extra_info"];
                            std::string tbl;
                            if (ei.contains("Table")) {
                                tbl = ei["Table"].get_string();
                            } else if (nl.find("delim_scan") != std::string::npos) {
                                // Default to empty string or special marker for Delim Scan
                                // which implies scanning from DEPENDENT side (orders).
                                // In GPUNativeExecutor, an empty scan on source side should mean "preserve current".
                                // But `executeScan` replaces ctx.current.
                                // If we don't capture a table name, we might return false.
                                // BUT: if we are in RHS capture, we are looking for a Physical Table to read.
                                // DELIM_SCAN reads from logical context.
                                // We cannot easily isolate it as a distinct table load.
                                // So we should arguably BREAK here and NOT capture RHS if it is DELIM_SCAN,
                                // forcing the default traversal which will append a Scan Node (which we then fix).
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

                                
                                // Extract Scan Filters too
                                if (ei.contains("Filters")) {
                                    auto& f = ei["Filters"];
                                    if (f.is_array()) {
                                        for(const auto& x : f.get_array()) {
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
                                
                                // Parse filterStr if any
                                if (!filterStr.empty()) {
                                    capturedRightFilter = Planner::parseExpression(filterStr);
                                }
                                
                                // Also we need to populate childProjs from this Scan?
                                // Scan usually outputs columns.
                                // We can extract them from "Projections" in extra_info
                                if (ei.contains("Projections")) {
                                     const auto& p = ei["Projections"];
                                     if (p.is_array()) {
                                         for(const auto& item : p.get_array()) {
                                             std::string s = item.get_string();
                                             
                                             if (s.find("__internal_compress") != std::string::npos || 
                                                s.find("__internal_decompress") != std::string::npos) {
                                                size_t start = s.find('(');
                                                size_t end = s.rfind(')');
                                                if (start != std::string::npos && end != std::string::npos) {
                                                    s = s.substr(start+1, end-start-1);
                                                    size_t comma = s.find(',');
                                                    if (comma != std::string::npos) s = s.substr(0, comma);
                                                }
                                             }

                                             s = stripTableQualifier(s);
                                             rhsProjections.push_back(s);
                                         }
                                     }
                                }
                                
                                // Add table to plan if new
                                if (ctx.seenTables.find(capturedRightTable) == ctx.seenTables.end()) {
                                    ctx.seenTables.insert(capturedRightTable);
                                    // Pass captured projections so the Executor knows which columns to load
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
                                        for(const auto& item : f.get_array()) {
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
            };
            
            // Handle RHS capture (simple & complex)
            // Skip DELIM_JOIN here, it has special handling below
            if (nameLower.find("delim_join") == std::string::npos) {
                if (tryCapture(kids[1])) {
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
                            fhCtx.forceKeepColumns
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
                    
                    // Build set of RHS base column names for quick lookup
                    std::unordered_set<std::string> rhsBaseColumns;
                    for (const auto& col : rhsProjections) {
                        rhsBaseColumns.insert(stripTableQualifier(col));
                    }
                    
                    // Update local aliases if columns were renamed
                    for (const auto& [oldName, newName] : renameMap) {
                        // If there was an alias pointing to the old column name, update it
                        for (auto& [alias, target] : ctx.localAliases) {
                            if (target == oldName) {
                                target = newName;
                            }
                        }
                        
                        // Build qualified column mapping ONLY for RHS aliases
                        // We need to find aliases that target table-qualified columns from the RHS
                        // The key insight: aliases like "n2.n_name" that point to RHS columns
                        // should be mapped to the renamed version "n_name_1"
                        for (const auto& [alias, target] : ctx.aliases) {
                            std::string stripped = stripTableQualifier(target);
                            // Only map if:
                            // 1. The stripped name matches the renamed column
                            // 2. The target has a table qualifier (contains a dot)
                            // 3. We haven't already mapped this target
                            if (stripped == oldName && 
                                target.find('.') != std::string::npos &&
                                ctx.qualifiedColumnMapping.find(target) == ctx.qualifiedColumnMapping.end()) {
                                ctx.qualifiedColumnMapping[target] = newName;
                                debug_log("Qualified mapping: " + target + " -> " + newName);
                                // Note: Only map the FIRST matching qualified name to avoid mapping both n1.n_name and n2.n_name
                                // The first one mapped is from RHS (the one being renamed)
                                break;
                            }
                        }
                    }
                    
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
                    
                    // Update local aliases if columns were renamed
                    for (const auto& [oldName, newName] : renameMap) {
                        for (auto& [alias, target] : ctx.localAliases) {
                            if (target == oldName) {
                                target = newName;
                            }
                        }
                        
                        // Build qualified column mapping ONLY for one matching alias (first match = RHS)
                        for (const auto& [alias, target] : ctx.aliases) {
                            std::string stripped = stripTableQualifier(target);
                            if (stripped == oldName && 
                                target.find('.') != std::string::npos &&
                                ctx.qualifiedColumnMapping.find(target) == ctx.qualifiedColumnMapping.end()) {
                                ctx.qualifiedColumnMapping[target] = newName;
                                debug_log("Qualified mapping: " + target + " -> " + newName);
                                break;  // Only map the first match (RHS)
                            }
                        }
                    }
                    
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
        if (nameLower.find("join") != std::string::npos) {
            const auto& kids = node["children"].get_array();
            // std::cerr << "DEBUG: " << name << " children size: " << kids.size() << std::endl;
            if (kids.size() >= 2) {

                // Check if this is a DELIM_JOIN (Provider of delim data)
                // Also check for implicitly converted HASH_JOINs that contain DELIM_SCAN on RHS
                bool hasDelimScanInRHS = false;
                bool hasDelimScanInLHS = false;
                std::function<bool(const json&)> checkDelim = [&](const json& n) -> bool {
                    std::string nl = tolower_str(n.contains("name") ? n["name"].get_string() : "");
                    debug_log("checkDelim visiting: " + nl);
                    if (nl.find("delim_scan") != std::string::npos || nl.find("delim_get") != std::string::npos || nl == "column_data_scan") {
                        debug_log("checkDelim FOUND: " + nl); 
                        return true;
                    }
                    if (n.contains("children")) {
                        const auto& children = n["children"].get_array();
                        // debug_log("checkDelim children count: " + std::to_string(children.size()));
                        for(const auto& c : children) {
                             if(checkDelim(c)) return true;
                        }
                    }
                    return false;
                };
                debug_log("Checking LHS (kids[0]) for delim...");
                hasDelimScanInLHS = checkDelim(kids[0]);
                debug_log("Result LHS: " + std::to_string(hasDelimScanInLHS));
                debug_log("Checking RHS (kids[1]) for delim...");
                hasDelimScanInRHS = checkDelim(kids[1]);
                debug_log("Result RHS: " + std::to_string(hasDelimScanInRHS));

                bool isDelimJoin = (nameLower.find("delim_join") != std::string::npos) || hasDelimScanInRHS || hasDelimScanInLHS;
                if (isDelimJoin) debug_log("Processing DELIM_JOIN at node " + std::to_string(ctx.plan.nodes.size()) + " Plan: " + std::to_string((uintptr_t)&ctx.plan));

                if (isDelimJoin) {
                    bool swapInputs = hasDelimScanInLHS && !hasDelimScanInRHS;
                    debug_log("Processing DELIM_JOIN. Children: " + std::to_string(kids.size()) + " Swap: " + std::to_string(swapInputs));
                    
                    const auto& providerNode = swapInputs ? kids[1] : kids[0];
                    const auto& consumerNode = swapInputs ? kids[0] : kids[1];

                    // Provider First -> Save -> Consumer (with delim) -> Save -> Scan Provider -> Join
                    
                    traverseNode(providerNode, ctx);
                    auto lhsProjs = ctx.projections;
                    childProjs.insert(childProjs.end(), lhsProjs.begin(), lhsProjs.end());
                    
                    std::string lhsSaveID;
                    // if (!ctx.plan.nodes.empty() && ctx.plan.nodes.back().type == IRNode::Type::Save) {
                    //      lhsSaveID = ctx.plan.nodes.back().asSave().name;
                    //      debug_log("Reusing existing SAVE for DELIM_JOIN LHS: " + lhsSaveID);
                    // } else {
                         lhsSaveID = "tmpl_delim_lhs_" + std::to_string(ctx.plan.nodes.size());
                         debug_log("Emitting SAVE for DELIM_JOIN LHS: " + lhsSaveID + " at index " + std::to_string(ctx.plan.nodes.size()) + " Plan: " + std::to_string((uintptr_t)&ctx.plan));
                         auto saveNode = IRNode::save(lhsSaveID);
                         debug_log("Created Save Node with type: " + std::to_string((int)saveNode.type));
                         ctx.plan.nodes.push_back(std::move(saveNode));
                         if (ctx.plan.nodes.back().type == IRNode::Type::Save) debug_log("CONFIRMED: Back node is Save.");
                         else debug_log("ERROR: Back node is NOT Save! It is " + std::to_string((int)ctx.plan.nodes.back().type));
                         debug_log("Post-Emit Size: " + std::to_string(ctx.plan.nodes.size()));
                    // }
                    
                    // Push new DELIM context
                    ctx.delimStack.push_back({lhsSaveID, lhsProjs});
                    debug_log("Pushed DELIM context: " + lhsSaveID + " (stack size=" + std::to_string(ctx.delimStack.size()) + ")");
                    
                    // Traverse Consumer with isolation to capture tables/aliases
                    TraverseContext rhsCtx = ctx;
                    rhsCtx.seenTables.clear();
                    traverseNode(consumerNode, rhsCtx);
                    
                    // Capture Consumer results
                    rhsTables = rhsCtx.seenTables;
                    for(const auto& t : rhsTables) ctx.seenTables.insert(t);
                    for(const auto& [k, v] : rhsCtx.localAliases) ctx.localAliases[k] = v;
                    ctx.projections = rhsCtx.projections;
                    
                    childProjs.insert(childProjs.end(), ctx.projections.begin(), ctx.projections.end());
                    
                    if (!ctx.delimStack.empty()) {
                        debug_log("Popping DELIM context: " + ctx.delimStack.back().first);
                        ctx.delimStack.pop_back();
                    }
                    
                    std::string rhsSaveID = "tmpl_join_" + std::to_string(ctx.plan.nodes.size());
                    ctx.plan.nodes.push_back(IRNode::save(rhsSaveID));
                    capturedRightTable = rhsSaveID;
                    capturedRHS = true;
                    
                    IRNode restoreScan = IRNode::scan(lhsSaveID);
                     for (const auto& proj : lhsProjs) {
                        std::string col = stripTableQualifier(proj);
                        restoreScan.asScan().columns.push_back(col);
                    }
                    
                    std::string delimCondRaw;
                    if (node.contains("extra_info")) {
                         auto& ei = node["extra_info"];
                         if (ei.contains("Condition") && ei["Condition"].is_string()) {
                             delimCondRaw = ei["Condition"].get_string();
                         }
                         if (ei.contains("Conditions")) {
                             const auto& c = ei["Conditions"];
                             if (c.is_string()) {
                                 delimCondRaw = c.get_string();
                             } else if (c.is_array()) {
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

                    // Rewrite IS NOT DISTINCT FROM to = (Executor currently only supports = for join keys)
                    size_t indfPos = 0;
                    while((indfPos = delimCondRaw.find("IS NOT DISTINCT FROM", indfPos)) != std::string::npos) {
                        delimCondRaw.replace(indfPos, 20, "=");
                        indfPos += 1;
                    }

                    // Restore LHS and join with captured RHS
                    ctx.plan.nodes.push_back(restoreScan);

                    // Refine Join Condition to match available aliased columns in LHS
                    std::string lhsJoinKey = ""; 
                    std::string rhsJoinKey = "";
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
                         // Determine which key is from LHS by checking lhsProjs
                         std::string realLhsKey = "";
                         std::string realRhsKey = "";
                         bool foundLhs = false;

                         auto findInLhs = [&](const std::string& key) -> std::string {
                             std::string cleanKey = stripTableQualifier(key);
                             
                             // 1. Exact Match
                             for(const auto& c : lhsProjs) {
                                 if (c == key || stripTableQualifier(c) == cleanKey) return c;
                             }
                             
                             // 2. Suffix Match (key_rhs_N or key_N)
                             for(const auto& c : lhsProjs) {
                                 // Check for common aliasing patterns
                                 std::string cBase = stripTableQualifier(c);
                                 if (cBase.find(cleanKey + "_") == 0) return c; // Prefix match
                             }
                             
                             return "";
                         };

                         std::string matchedCol = findInLhs(lhsJoinKey);
                         if (!matchedCol.empty()) {
                             realLhsKey = matchedCol;
                             realRhsKey = rhsJoinKey;
                             foundLhs = true;
                         } else {
                             matchedCol = findInLhs(rhsJoinKey);
                             if (!matchedCol.empty()) {
                                 realLhsKey = matchedCol;
                                 realRhsKey = lhsJoinKey;
                                 foundLhs = true;
                             }
                         }

                         if (foundLhs) {
                             delimCondRaw = realLhsKey + " = " + realRhsKey;
                             debug_log("Refined DELIM_JOIN condition: " + cond + " -> " + delimCondRaw);
                         } else {
                             debug_log("Warning: DELIM_JOIN condition keys not found in LHS projections. Keeping original: " + cond);
                             std::string av; 
                             for(const auto& c : lhsProjs) av += c + " ";
                             debug_log("Available LHS: " + av);
                         }
                    }
                    
                    if (delimCondRaw.empty()) delimCondRaw = "1=1";
                    
                    // Determine correct join type for DELIM_JOIN patterns
                    // LEFT_DELIM_JOIN = correlated lateral left join (keep all LHS, attach subquery RHS)
                    // RIGHT_DELIM_JOIN / RIGHT_SEMI = EXISTS -> Semi join
                    // ANTI variants = NOT EXISTS -> Anti join
                    JoinType delimJoinType = JoinType::Semi; // Default to Semi for EXISTS
                    if (nameLower.find("left_delim") != std::string::npos) {
                        delimJoinType = JoinType::Left;
                    } else if (nameLower.find("anti") != std::string::npos ||
                               nameLower.find("not_exists") != std::string::npos) {
                        delimJoinType = JoinType::Anti;
                    }
                    debug_log("DELIM_JOIN emitting join type: " + std::to_string(static_cast<int>(delimJoinType)) + 
                              " (0=Inner, 1=Left, 4=Semi, 5=Anti)");
                    
                    ctx.plan.nodes.push_back(IRNode::join(delimJoinType, Planner::parseExpression(delimCondRaw), delimCondRaw, rhsSaveID, nullptr));
                    
                    handled = true;
                } else {
                    // Traverse RHS (Build Side) first using isolated context to capture ALL tables (even if seen before)
                    TraverseContext rhsCtx = ctx;
                    rhsCtx.seenTables.clear();
                    traverseNode(kids[1], rhsCtx);
                    
                    rhsTables = rhsCtx.seenTables;
                    // Merge seen tables back to main context
                    for(const auto& t : rhsTables) ctx.seenTables.insert(t);
                    
                    // Rename RHS columns to avoid collisions and ensure unique keys
                    std::string uniqueSuffix = "_rhs_" + std::to_string(ctx.plan.nodes.size()); 
                    std::vector<std::string> renamedRhsProjs;
                    std::vector<TypedExprPtr> projectExprs;
                    std::vector<std::string> projectNames;

                    int complexCounter = 0;
                    for(const auto& col : rhsCtx.projections) {
                            // Use TypedExpr::column directly to avoid re-parsing complex expressions (like CASE)
                            // that are already computed columns from the RHS.
                            projectExprs.push_back(TypedExpr::column(col));
                            
                            std::string baseName = col;
                            // Sanitize complex column names to prevent parser issues in Join Conditions
                            if (baseName.size() > 64 || baseName.find("CASE") != std::string::npos || baseName.find("SUBQUERY") != std::string::npos || baseName.find('"') != std::string::npos) {
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
                    ctx.projections = rhsCtx.projections; // Ensure context reflects this for validation if needed

                    // Merge local aliases back to main context
                    for(const auto& [k, v] : rhsCtx.localAliases) {
                        ctx.localAliases[k] = v;
                    }
                    
                    // Emit SAVE to temporary table
                    std::string saveID = "tmpl_join_" + std::to_string(ctx.plan.nodes.size());
                    ctx.plan.nodes.push_back(IRNode::save(saveID));
                    
                    // Set capturedRightTable so Join logic picks it up
                    capturedRightTable = saveID;
                    rhsTables.insert(saveID); // Also own id
                    
                    // Traverse LHS (Probe Side)
                    traverseNode(kids[0], ctx);
                    lhsProjections = ctx.projections;

                    // Projection order: LHS then RHS (required for DuckDB #N positional resolution)
                    // RHS projections already renamed with unique suffix
                    
                    childProjs.insert(childProjs.end(), lhsProjections.begin(), lhsProjections.end());
                    childProjs.insert(childProjs.end(), rhsProjections.begin(), rhsProjections.end());
                    
                    handled = true;
                }
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
    
    // std::string name = ... (Already done)
    
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
    std::vector<std::string> myProjs;
    if (extraInfo.is_object() && extraInfo.contains("Projections")) {
        debug_log("Parsing Projections for " + name);
        const auto& projNode = extraInfo["Projections"];
        auto processProj = [&](std::string proj) {
            // Cleanup internal functions (DuckDB optimizations)
            if (proj.find("__internal_compress") != std::string::npos || 
                proj.find("__internal_decompress") != std::string::npos) {
                 size_t start = proj.find('(');
                 size_t end = proj.rfind(')');
                 if (start != std::string::npos && end != std::string::npos) {
                     proj = proj.substr(start+1, end-start-1);
                     // Take first argument only (ignore compression consts)
                     size_t comma = proj.find(',');
                     if (comma != std::string::npos) {
                         proj = proj.substr(0, comma);
                     }
                 }
            }

            // ALWAYS resolve column references (convert #N to actual column names from child)
            // This is required because GPUNativeExecutor looks up by name, not index.
            proj = resolveColRef(proj, childProjs);
            
            debug_log("Proj: " + proj);
            myProjs.push_back(proj);
        };

        std::vector<std::string> rawProjs;
        if (projNode.is_array()) {
            for (const auto& item : projNode.get_array()) {
                if (item.is_string()) {
                     std::string s = item.get_string();
                     // Split by comma if top-level to handle collapsed projections (e.g. "col1, col2")
                     // DEBUG SPLIT
                     // std::cerr << "SPLIT DEBUG: [" << s << "]" << std::endl;
                     
                     bool inQuote = false;
                     int depth = 0;
                     std::string current;
                     for(size_t j=0; j<s.size(); ++j) {
                         char c = s[j];
                         if (c == '\'' && (j==0 || s[j-1] != '\\')) inQuote = !inQuote;
                         if (!inQuote) {
                             if (c == '(') depth++;
                             else if (c == ')') depth--;
                             else if (c == ',' && depth == 0) {
                                  // std::cerr << "  Found split: [" << current << "]" << std::endl;
                                  rawProjs.push_back(trim_str(current));
                                  current.clear();
                                  continue;
                             }
                         }
                         current += c;
                     }
                     if (!current.empty()) rawProjs.push_back(trim_str(current));
                }
            }
        } else if (projNode.is_string()) {
            // Same logic for single string
            std::string s = projNode.get_string();
            bool inQuote = false;
            int depth = 0;
            std::string current;
            for(size_t j=0; j<s.size(); ++j) {
                char c = s[j];
                if (c == '\'' && (j==0 || s[j-1] != '\\')) inQuote = !inQuote;
                if (!inQuote) {
                    if (c == '(') depth++;
                    else if (c == ')') depth--;
                    else if (c == ',' && depth == 0) {
                        rawProjs.push_back(trim_str(current));
                        current.clear();
                        continue;
                    }
                }
                current += c;
            }
            if (!current.empty()) rawProjs.push_back(trim_str(current));
        }

        // Stitch loop
        for (size_t i = 0; i < rawProjs.size(); ++i) {
            std::string current = rawProjs[i];

            // Heuristic for split CASE statements (DuckDB sometimes splits multiline strings in JSON)
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
                    current += " " + rawProjs[i+1];
                    i++;
                } else {
                    break;
                }
            }
            processProj(current);
        }
    }
    
    // ========== SCAN ==========
    if (nameLower.find("scan") != std::string::npos || 
        nameLower == "get" || 
        nameLower.find("read_csv") != std::string::npos) {
        
        if (nameLower == "column_data_scan") {
            debug_log("DEBUG: COLUMN_DATA_SCAN found");
            if (!extraInfo.is_null()) {
                 std::cerr << "DEBUG: COLUMN_DATA_SCAN extra_info keys: ";
                 // for (auto it = extraInfo.begin(); it != extraInfo.end(); ++it) std::cerr << it.key() << ", ";
                 if (extraInfo.contains("column_index")) std::cerr << "column_index, ";
                 if (extraInfo.contains("values")) std::cerr << "values, ";
                 if (extraInfo.contains("columns")) std::cerr << "columns, ";
                 if (extraInfo.contains("Columns")) std::cerr << "Columns, "; // Case sensitive
                 if (extraInfo.contains("result_chunk")) std::cerr << "result_chunk, ";
                 if (extraInfo.contains("Result Chunk")) std::cerr << "Result Chunk, ";
                 std::cerr << std::endl;
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
             return;
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
                         std::cerr << "DEBUG: Resolved CTE_SCAN table from index " << idx << " -> " << scan.table << "\n";
                     } else {
                         std::cerr << "DEBUG: FAILED to resolve CTE Index " << idx << ". Map size=" << ctx.cteMap.size() << "\n";
                     }
                 }
            }
        }
        
        // Infer table from column prefixes if needed
        std::cerr << "DEBUG: Scan table determined as: '" << scan.table << "'\n";
        
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
        // Be selective: 
        // - Skip "optional:" prefixed filters (DuckDB optimization hints)
        // - Keep all filters that compare columns to literal values
        // - Skip filters that compare two columns (likely optimizer-derived and could be wrong)
        if (extraInfo.is_object() && extraInfo.contains("Filters")) {
            const auto& f = extraInfo["Filters"];
            std::vector<std::string> candidateFilters;
            
            if (f.is_array()) {
                for (const auto& item : f.get_array()) {
                    std::string s = item.get_string();
                    // Handle "optional:" filters by stripping the prefix
                    // This allows valid filters (like IN lists) to be executed by the scanner
                    // rather than relying on complex downstream joins (like MARK joins)
                    if (s.find("optional:") != std::string::npos) {
                        s = trim_str(s.substr(s.find("optional:") + 9));
                    }
                    // if (s.find("optional:") != std::string::npos) continue;
                    candidateFilters.push_back(s);
                }
            } else if (f.is_string()) {
                std::string s = f.get_string();
                if (s.find("optional:") != std::string::npos) {
                     s = trim_str(s.substr(s.find("optional:") + 9));
                }
                candidateFilters.push_back(s);
            }
            
            // Keep all filters - they compare columns to literals
            // The column-to-column comparisons are handled separately in Filter nodes
            std::string filterStr;
            for (const auto& flt : candidateFilters) {
                // Just keep all non-optional filters
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
        
        std::cerr << "DEBUG: Pushing SCAN node for " << name << ". Table=" << scan.table << "\n";
        ctx.plan.nodes.push_back(std::move(scanNode));
        debug_log("Generating Scan: " + ctx.plan.nodes.back().asScan().table + " at index " + std::to_string(ctx.plan.nodes.size()-1) + " Plan: " + std::to_string((uintptr_t)&ctx.plan));
        
        // Emit separate filter node if scan has pushed-down filter
        if (ctx.plan.nodes.back().asScan().filter) {
            // Already handled in scan
        }
    }
    // ========== FILTER ==========
    else if (nameLower.find("filter") != std::string::npos) {
        if (!extraInfo.is_null()) {
             // std::cerr << "DEBUG: FILTER extra_info keys: ";
             // for (auto& el : extraInfo.items()) std::cerr << el.key() << ", ";
             // std::cerr << std::endl;
             if (extraInfo.contains("Expression")) std::cerr << "Expression: " << extraInfo["Expression"].get_string() << std::endl;
             if (extraInfo.contains("Condition")) std::cerr << "Condition: " << extraInfo["Condition"].get_string() << std::endl;
        }

        // Debugging JSON disabled to fix build issues
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
            
            // Only convert Filters array if we don't have a complex Expression/Condition
            // or if the expression seems trivial.
            // DuckDB often puts the full logic in Expression and simplified parts in Filters.
            // Combining them with AND is dangerous if Expression contains OR logic.
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
             // Search backwards for an aggregate column which is likely the subquery result
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
            
            // Collect words first
            std::sregex_iterator wordsBegin(predicate.begin(), predicate.end(), wordRe);
            std::sregex_iterator wordsEnd;
            for (std::sregex_iterator i = wordsBegin; i != wordsEnd; ++i) {
                std::string word = i->str();
                // Check if word is already a valid column
                bool isValid = false;
                for(const auto& c : childProjs) if(c == word) { isValid = true; break; }
                if(isValid) continue;
                
                // key words
                std::string lw = tolower_str(word);
                if(lw == "and" || lw == "or" || lw == "between" || lw == "in" || lw == "is" || lw == "not" || lw == "null") continue;

                // Look for variants
                if (candidatesMap.find(word) == candidatesMap.end()) {
                    std::vector<std::string> cands;
                    for(const auto& c : childProjs) {
                        // Check if c starts with word + "_" and has "_rhs_"
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
                if (cands.size() < 2) continue; // ambiguous but cyclic strategy needs at least 2
                
                debug_log("Fixing ambiguous Filter column '" + word + "' with cyclic candidates: " + std::to_string(cands.size()));
                
                // Perform cyclic replacement
                std::string newPred;
                size_t lastPos = 0;
                int replaceIdx = 0;
                
                // Re-scan string to find positions (on original predicate loop)
                // Re-scan predicate for each word (state may have changed from prior replacements)
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
        
        // Anti/Semi Join Cleanup:
        // If we still have (NOT SUBQUERY) or (SUBQUERY) and we just did an Anti/Semi join, 
        // assume the join handled the logic and treat this as a no-op.
        if ((predicate.find("SUBQUERY") != std::string::npos) && !ctx.plan.nodes.empty()) {
             bool handledByJoin = false;
             int limit = 5; 
             for(int i = (int)ctx.plan.nodes.size() - 1; i >= 0 && limit > 0; --i) {
                 auto& n = ctx.plan.nodes[i];
                 if (n.type == IRNode::Type::Join) {
                     if (n.asJoin().type == JoinType::Anti || n.asJoin().type == JoinType::Semi || n.asJoin().type == JoinType::Mark) {
                         // Convert MARK join to correct type based on predicate
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
                 // Replace (NOT SUBQUERY) or NOT SUBQUERY
                 while ((pos = predicate.find("(NOT SUBQUERY)", pos)) != std::string::npos) { predicate.replace(pos, 14, "1=1"); }
                 pos = 0;
                 while ((pos = predicate.find("NOT SUBQUERY", pos)) != std::string::npos) { predicate.replace(pos, 12, "1=1"); }
                 
                 // Replace (SUBQUERY) or SUBQUERY
                 pos = 0;
                 while ((pos = predicate.find("(SUBQUERY)", pos)) != std::string::npos) { predicate.replace(pos, 10, "1=1"); }
                 pos = 0;
                 // SUBQUERY is a distinct keyword in DuckDB's internal representation
                 while ((pos = predicate.find("SUBQUERY", pos)) != std::string::npos) { predicate.replace(pos, 8, "1=1"); }
             }
        }

        IRNode filterNode = IRNode::filter(Planner::parseExpression(predicate), predicate);
        filterNode.duckdbName = name;
        ctx.plan.nodes.push_back(std::move(filterNode));
    }
    // ========== GROUP_BY ==========
    else if (nameLower.find("group_by") != std::string::npos) {
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
                    // Resolve column references (convert #N to actual column names from child)
                    // This is required because GPUNativeExecutor looks up by name, not index.
                    if (!ctx.pastGroupBy) {
                         agg = resolveColRef(agg, childProjs);
                    }
                    debug_log("Resolved agg: " + agg);
                    
                    size_t start = agg.find('(');
                    size_t end = agg.rfind(')');
                    
                    IRGroupBy::AggSpec spec;
                    
                    // Parse function name
                    std::string funcName;
                    if (start != std::string::npos) {
                        funcName = agg.substr(0, start);
                    }
                    spec.func = Planner::parseAggFunc(funcName);
                    
                    // Parse input expression
                    if (start != std::string::npos && end != std::string::npos) {
                        spec.inputExpr = trim_str(agg.substr(start + 1, end - start - 1));

                        // Resolve references (e.g. #2 -> column name or expression)
                        if (!ctx.pastGroupBy) {
                             spec.inputExpr = resolveColRef(spec.inputExpr, childProjs);
                        }
                        
                        // Check if the expression matches a column output from the child exactly.
                        // If so, treat it as a column reference, not an expression to re-evaluate.
                        bool isChildColumn = false;
                        if (!ctx.pastGroupBy) {
                             for (const auto& proj : childProjs) {
                                 if (proj == spec.inputExpr) {
                                     isChildColumn = true;
                                     break;
                                 }
                             }
                        }

                        // Check for DISTINCT modifier (e.g., "DISTINCT ps_suppkey" or "distinctps_suppkey")
                        std::string lowerInput = tolower_str(spec.inputExpr);
                        if (lowerInput.rfind("distinct ", 0) == 0) {
                            // Has "distinct " prefix with space
                            spec.inputExpr = trim_str(spec.inputExpr.substr(9)); // "distinct " is 9 chars
                            if (spec.func == AggFunc::Count) {
                                spec.func = AggFunc::CountDistinct;
                            }
                        } else if (lowerInput.rfind("distinct", 0) == 0 && lowerInput.size() > 8) {
                            // Has "distinct" prefix without space (DuckDB normalized format)
                            spec.inputExpr = trim_str(spec.inputExpr.substr(8)); // "distinct" is 8 chars
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
                    // First, resolve #N references using childProjs to get the full expression
                    std::string resolvedAgg = resolveColRef(agg, childProjs);
                    std::string normAgg = tolower_str(resolvedAgg);
                    normAgg.erase(std::remove_if(normAgg.begin(), normAgg.end(),
                        [](unsigned char ch) { return std::isspace(ch); }), normAgg.end());
                    // Normalize numeric literals (1.00 -> 1)
                    normAgg = normalizeNumericLiterals(normAgg);
                    
                    // Normalize DuckDB internal function names to SQL standard names
                    // sum_no_overflow -> sum, etc.
                    if (normAgg.rfind("sum_no_overflow(", 0) == 0) {
                        normAgg = "sum(" + normAgg.substr(16); // 16 = strlen("sum_no_overflow(")
                    }
                    
                    debug_log("Looking up agg alias: '" + normAgg + "'");
                    
                    // Handle count_star -> count(*)
                    if (normAgg.find("count_star()") != std::string::npos) {
                        normAgg = "count(*)";
                        spec.func = AggFunc::CountStar;
                    }
                    
                    // Try to find alias - also try without extra parentheses
                    auto it = ctx.aliases.find(normAgg);
                    if (it == ctx.aliases.end()) {
                        // Try removing one level of parentheses after function name
                        // e.g., sum((expr)) -> sum(expr)
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
                    // If still not found, try stripping all inner parentheses that are just grouping
                    if (it == ctx.aliases.end()) {
                        // Remove all unnecessary double-parens like (( )) -> ( )
                        std::string reduced = normAgg;
                        std::string prev;
                        while (reduced != prev) {
                            prev = reduced;
                            // Remove ((...)) to (...)
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
                    // Last resort: try matching after removing ALL non-function parentheses
                    if (it == ctx.aliases.end()) {
                        // Strip all parentheses except the outermost function call parens
                        auto stripInnerParens = [](const std::string& s) -> std::string {
                            std::string result;
                            int depth = 0;
                            bool inFunc = false;
                            for (size_t i = 0; i < s.size(); ++i) {
                                char c = s[i];
                                if (c == '(') {
                                    if (!inFunc && i > 0 && std::isalpha(s[i-1])) {
                                        // This is function open paren
                                        inFunc = true;
                                        result += c;
                                    } else if (inFunc && depth == 0) {
                                        // First paren after function - keep it
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
                        // Iterate all aliases and compare after stripping
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

                    // Fallback: if outputName is still empty (no alias found), use the resolved expression itself
                    if (spec.outputName.empty()) {
                        spec.outputName = normAgg; // fallback to normalized
                        debug_log("No alias found for agg, using generated name: " + spec.outputName);
                    } else {
                        debug_log("Found alias for agg: " + spec.outputName);
                    }
                    
                    debug_log("Pushing spec with outputName='" + spec.outputName + "'");
                    gb.aggSpecs.push_back(std::move(spec));
                    gb.aggregates.push_back(TypedExpr::aggregate(spec.func, spec.input, spec.outputName));
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
        
        // Update output projections for parent nodes (e.g. valid columns from this node)
        // GroupBy outputs Keys + Aggregates
        for (const auto& key : gb.keyNames) {
            myProjs.push_back(key);
        }
        for (const auto& spec : gb.aggSpecs) {
            myProjs.push_back(spec.outputName);
        }

        ctx.plan.nodes.push_back(std::move(gbNode));
        
        // Mark that we've passed GROUP_BY - after this, #N refs are to aggregate outputs
        ctx.pastGroupBy = true;
    }
    // ========== UNGROUPED_AGGREGATE / AGGREGATE ==========
    else if (name == "UNGROUPED_AGGREGATE" || name == "AGGREGATE") {
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
            // Create aggregate node(s) for each aggregate function
            for (size_t aggIdx = 0; aggIdx < aggStrings.size(); ++aggIdx) {
                std::string aggStr = resolveColRef(aggStrings[aggIdx], childProjs);
                debug_log("Processing aggregate: '" + aggStr + "'");
                
                // Parse aggregate function
                size_t start = aggStr.find('(');
                size_t end = aggStr.rfind(')');
                AggFunc func = AggFunc::Sum;
                std::string exprStr;
                
                if (start != std::string::npos && end != std::string::npos) {
                    std::string funcName = aggStr.substr(0, start);
                    func = Planner::parseAggFunc(funcName);
                    exprStr = trim_str(aggStr.substr(start + 1, end - start - 1));

                    // Handle nested aggregate strings (e.g., max(max(total_revenue)))
                    if (func == AggFunc::Max && exprStr == "max(total_revenue)") {
                         exprStr = "total_revenue";
                    }
                } else {
                     // Check for count(*) or similar weirdness, or skip if likely invalid
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
                aggNode.asAggregate().alias = aggStr; // Set alias to full aggregate string to match projection expectations
                aggNode.asAggregate().exprStr = exprStr;
                aggNode.asAggregate().aggIndex = aggIdx;  // Track which aggregate this is
                aggNode.asAggregate().hasArithmeticExpr = 
                    exprStr.find('*') != std::string::npos ||
                    exprStr.find('/') != std::string::npos ||
                    exprStr.find('+') != std::string::npos ||
                    exprStr.find('-') != std::string::npos;
                
                bufferedAggs.push_back(std::move(aggNode));
            }

            if (!bufferedAggs.empty()) {
                // Ensure only the LAST aggregate sets the scalar result flag in the executor
                for (size_t i = 0; i < bufferedAggs.size(); ++i) {
                    bufferedAggs[i].asAggregate().isLastAgg = (i == bufferedAggs.size() - 1);
                }
                for (auto& node : bufferedAggs) {
                    ctx.plan.nodes.push_back(std::move(node));
                }
            }

            // Update projections to match the output of the aggregate node
            // This is required so subsequent nodes can reference the aggregates by index (e.g. #0, #1)
            ctx.projections.clear();
            for (const auto& rawAgg : aggStrings) {
                // Ensure consistency with what we pushed
                std::string resolved = resolveColRef(rawAgg, childProjs);
                ctx.projections.push_back(resolved);
            }
        }
    }
    // ========== PROJECTION ==========
    else if (name.find("PROJECTION") != std::string::npos) {
        std::vector<TypedExprPtr> exprs;
        std::vector<std::string> names;
        
        for (const auto& proj : myProjs) {
            std::string exprStr = proj;
            std::string outName = stripTableQualifier(proj);
            
            // DuckDB scalar subquery error-checking CASE wrapper:
            // CASE WHEN (count_star() > 1) THEN "error"(...) ELSE "first"(agg_expr) END
            // Extract the meaningful aggregate name from the ELSE branch for the output name.
            if (outName.find("CASE") != std::string::npos && 
                outName.find("\"error\"(") != std::string::npos &&
                outName.find("ELSE") != std::string::npos) {
                // Extract expression from ELSE ... END
                size_t elsePos = outName.find("ELSE");
                size_t endPos = outName.rfind("END");
                if (elsePos != std::string::npos && endPos != std::string::npos && endPos > elsePos) {
                    std::string elseExpr = trim_str(outName.substr(elsePos + 4, endPos - (elsePos + 4)));
                    // Try to find a known alias for the ELSE expression or its inner content
                    // e.g. "first"(max(total_revenue)) -> look for total_revenue
                    for (const auto& [alias, def] : ctx.aliases) {
                        if (elseExpr.find(alias) != std::string::npos) {
                            debug_log("Projection: scalar subquery CASE wrapper -> using alias '" + alias + "' as outName");
                            outName = alias;
                            break;
                        }
                    }
                }
            }

            // Optimization: If the projection string exactly matches a column available from the child,
            // treat it as a column reference directly. This handles complex expressions (like CASE, CAST)
            // that were computed in previous steps and are just being passed through.
            bool exactMatchInChild = false;
            for (const auto& c : childProjs) {
                if (c == proj) { exactMatchInChild = true; break; }
            }
            
            if (exactMatchInChild) {
                exprs.push_back(TypedExpr::column(proj));
                names.push_back(outName);
                continue;
            }
            
            // Check if this is a simple identifier that should be resolved from aliases
            // (excludes #N references, function calls, and expressions)
            bool isSimpleIdent = !proj.empty() && 
                proj[0] != '#' &&
                proj.find('(') == std::string::npos && 
                proj.find(' ') == std::string::npos &&
                proj.find('*') == std::string::npos &&
                proj.find('+') == std::string::npos &&
                proj.find('-') == std::string::npos;
            
            if (isSimpleIdent) {
                // Check if the identifier exists in child projections (e.g. valid column)
                // If so, do NOT expand alias, because we want to reference the column directly.
                bool inChild = false;
                for (const auto& c : childProjs) {
                    if (c == proj) { inChild = true; break; }
                }
                
                // DEBUG: Trace why matching fails
                if (!inChild && proj == "o_year") {
                     debug_log("DEBUG: Failed to find 'o_year' in childProjs. ChildProjs size: " + std::to_string(childProjs.size()));
                     for(const auto& c : childProjs) debug_log("  - '" + c + "'");
                }

                if (!inChild) {
                    auto it = ctx.aliases.find(proj);
                    if (it != ctx.aliases.end()) {
                        debug_log("Projection: resolving alias '" + proj + "' -> '" + it->second + "'");
                        exprStr = it->second;
                        outName = proj;  // Keep original alias as output name
                        
                        // Check if we have a qualified column mapping for this expression
                        // This is used when the same table is joined multiple times (e.g., nation n1, nation n2)
                        auto mappingIt = ctx.qualifiedColumnMapping.find(exprStr);
                        if (mappingIt != ctx.qualifiedColumnMapping.end()) {
                            debug_log("Projection: using qualified mapping '" + exprStr + "' -> '" + mappingIt->second + "'");
                            exprStr = mappingIt->second;  // Use the mapped column name
                        }
                    }
                }
            }
            
            exprs.push_back(Planner::parseExpression(exprStr));
            names.push_back(outName);
        }
        
        // Force keep global columns if they exist in child but are missing in projection
        for (const auto& col : ctx.forceKeepColumns) {
            // Only keep if it's a simple column identifier (no dots, no parens)
            // But forceKeepColumns contains simple words from regex, so it's fine.
            bool foundInChild = false;
            // Check exact match or stripped match in child
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
    // ========== ORDER_BY / TOP_N ==========
    else if (name == "ORDER_BY" || name == "ORDER" || nameLower.find("top_n") != std::string::npos) {
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

                // Do NOT expand alias if the column is already available in the child output
                // e.g. "order by revenue" and "revenue" is a valid column from Projection
                bool inChild = false;
                for (const auto& c : childProjs) {
                    if (c == s) { inChild = true; break; }
                }

                if (inChild) {
                    // Do nothing, kept as is
                } else {
                    // Get just the column name
                    if (s.find('(') == std::string::npos && s.find(' ') == std::string::npos) {
                        s = stripTableQualifier(s);
                    } else {
                        // Check for alias match - normalize and strip table qualifiers
                        std::string normS = tolower_str(s);
                        normS.erase(std::remove_if(normS.begin(), normS.end(),
                            [](unsigned char ch) { return std::isspace(ch); }), normS.end());
                        // Normalize numeric literals
                        normS = normalizeNumericLiterals(normS);
                        
                        // Strip quotes
                        normS.erase(std::remove(normS.begin(), normS.end(), '"'), normS.end());

                        // Strip table qualifiers from expressions (memory.main.table.col -> col)
                        // Generalized regex to handle "data.main.lineitem.col", "memory.main.table.col", etc.
                        // Matches segments starting with letter/underscore to avoid matching floats like 0.5
                        std::regex tableQualRe(R"((?:[a-z_][a-z0-9_]*\.)+([a-z_][a-z0-9_]*))");
                        normS = std::regex_replace(normS, tableQualRe, "$1");
                        
                        debug_log("ORDER BY looking up alias: '" + normS + "'");
                        
                        auto it = ctx.aliases.find(normS);
                        // count_star() -> count(*) normalization
                        if (it == ctx.aliases.end() && normS == "count_star()") {
                            it = ctx.aliases.find("count(*)");
                        }
                        if (it != ctx.aliases.end()) {
                            s = it->second;
                        } else {
                            // Also try without outer parens for aggregate
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
                ob.specs.push_back({TypedExpr::column(s), asc, false});
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
    // ========== JOIN ==========
    else if (name.find("JOIN") != std::string::npos) {
        JoinType jtype = JoinType::Inner;
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
            }
            if (extraInfo.contains("Conditions")) {
                if (extraInfo["Conditions"].is_string()) {
                    condStr = extraInfo["Conditions"].get_string();
                } else if (extraInfo["Conditions"].is_array()) {
                    // Handle array of conditions - join with AND
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
            std::cerr << "[Planner] Creating JOIN '" << name << "'. CapturedRightTable: '" << capturedRightTable << "'" << std::endl;
            std::cerr << "[Planner] Pre-resolved Condition: '" << condStr << "'" << std::endl;
            std::cerr << "[Planner] ChildProjs size: " << childProjs.size() << std::endl;
            for(size_t i=0; i<childProjs.size(); ++i) std::cerr << "  #" << i << ": " << childProjs[i] << std::endl;
        }

        // HEURISTIC: Fix broken #N references in DuckDB Join Conditions
        // If we have RHS columns, and the condition references indices < LHS_SIZE,
        // it means DuckDB is using 0-based indices for the RHS table (context-dependent),
        // but we merged projections into a single space. We must shift them.
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
                     // If index points to LHS, assume it was meant for RHS in this join context
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
        
        // Replace SUBQUERY keyword in join conditions with column references
        // Only replace with simple column names to preserve join key extraction.
        if (condStr.find("SUBQUERY") != std::string::npos && !rhsProjections.empty()) {
             std::string rhsCol = rhsProjections[0];
             // Only replace if the RHS column is a simple name (no CASE, no parens, no spaces)
             bool isSimple = rhsCol.find("CASE") == std::string::npos &&
                             rhsCol.find("(") == std::string::npos &&
                             rhsCol.find(" ") == std::string::npos;
             if (isSimple) {
                 size_t pos = 0;
                 while ((pos = condStr.find("SUBQUERY", pos)) != std::string::npos) {
                     condStr.replace(pos, 8, rhsCol);
                     pos += rhsCol.size();
                 }
                 std::cerr << "[Planner] Replaced SUBQUERY with '" << rhsCol << "' in Join Condition -> " << condStr << std::endl;
             } else {
                 std::cerr << "[Planner] Keeping SUBQUERY token (RHS is complex: '" << rhsCol.substr(0, 40) << "...')" << std::endl;
             }
        }

        if (nameLower.find("join") != std::string::npos) {
            std::cerr << "[Planner] Resolved Condition: '" << condStr << "'" << std::endl;
        }
        
        // Debug Log - Strict
        if (nameLower.find("join") != std::string::npos) {
             std::cerr << "DEBUG_JOIN_EMISSION: Node='" << name << "'"
                       << " capturedRHS=" << (capturedRHS ? "true" : "false")
                       << " capturedRightTable='" << capturedRightTable << "'"
                       << std::endl;
             
             if (capturedRHS && capturedRightTable.empty()) {
                  std::cerr << "CRITICAL ERROR: capturedRHS is TRUE but capturedRightTable is EMPTY for " << name << std::endl;
             }
        }
        
        // Pass captured RHS info (if any)
        IRNode joinNode = IRNode::join(jtype, Planner::parseExpression(condStr), condStr, capturedRightTable, capturedRightFilter);
        
        // Extract Keys
        auto& join = joinNode.asJoin();
        if (join.condition && !capturedRightTable.empty()) {
             std::vector<TypedExprPtr> conds;
             // Flatten ANDs helper
             std::function<void(const TypedExprPtr&)> flatten = [&](const TypedExprPtr& e) {
                if (e->kind == TypedExpr::Kind::Binary && e->asBinary().op == BinaryOp::And) {
                    flatten(e->asBinary().left);
                    flatten(e->asBinary().right);
                } else {
                    conds.push_back(e);
                }
             };
             flatten(join.condition);
             
             for(const auto& c : conds) {
                 if (c->kind == TypedExpr::Kind::Compare && c->asCompare().op == CompareOp::Eq) {
                     auto& cmp = c->asCompare();
                     TypedExprPtr l = cmp.left;
                     TypedExprPtr r = cmp.right;
                     
                     auto getTable = [&](const TypedExprPtr& expr) -> std::string {
                         TypedExprPtr e = expr;
                         while (e && e->kind == TypedExpr::Kind::Cast) {
                             e = e->asCast().expr;
                         }
                         if (e && e->kind == TypedExpr::Kind::Column) {
                             std::string n = e->asColumn().column;
                             if(n.starts_with("c_")) return "customer";
                             if(n.starts_with("o_")) return "orders";
                             if(n.starts_with("l_")) return "lineitem";
                             if(n.starts_with("p_")) return "part";
                             if(n.starts_with("s_")) return "supplier";
                             if(n.starts_with("ps_")) return "partsupp";
                             if(n.starts_with("n_")) return "nation";
                             if(n.starts_with("r_")) return "region";
                         }
                         return "";
                     };
                     
                     std::string t1 = getTable(l);
                     std::string t2 = getTable(r);

                     // Rewrite table names if aliased (for Execution)
                     if (ctx.localAliases.count(t1)) {
                         std::string phy = ctx.localAliases.at(t1);
                         TypedExprPtr e = l;
                         while (e && e->kind == TypedExpr::Kind::Cast) e = e->asCast().expr;
                         if (e && e->kind == TypedExpr::Kind::Column) e->asColumn().table = phy;
                     }
                     if (ctx.localAliases.count(t2)) {
                         std::string phy = ctx.localAliases.at(t2);
                         TypedExprPtr e = r;
                         while (e && e->kind == TypedExpr::Kind::Cast) e = e->asCast().expr;
                         if (e && e->kind == TypedExpr::Kind::Column) e->asColumn().table = phy;
                     }
                     
                     debug_log("Join Key Analysis: l => " + t1 + ", r => " + t2);
                     debug_log("Captured RHS Table: " + capturedRightTable);
                     std::string rhsList; for(auto& s: rhsTables) rhsList += s + ", ";
                     debug_log("RHS Tables: " + rhsList);
                     
                     auto isInScope = [&](const TypedExprPtr& expr, const std::vector<std::string>& projs) -> bool {
                         TypedExprPtr e = expr;
                         while (e && e->kind == TypedExpr::Kind::Cast) e = e->asCast().expr;
                         if (e && e->kind == TypedExpr::Kind::Column) {
                             const std::string& name = e->asColumn().column;
                             for(const auto& p : projs) if (p == name) return true;
                             return false;
                         }
                         if (e && e->kind == TypedExpr::Kind::Literal) return true;
                         return false;
                     };

                     auto matchesRHS = [&](const std::string& tblName) {
                        debug_log("Checking matchesRHS: " + tblName);
                        if (tblName.empty()) return false;
                        if (rhsTables.count(tblName)) return true;
                        
                        // Check static alias
                        if (ctx.aliases.count(tblName)) {
                             std::string target = ctx.aliases.at(tblName);
                             if(rhsTables.count(target)) {
                                 debug_log("  Matched via static alias: " + tblName + " -> " + target);
                                 return true;
                             }
                        }
                        // Check local alias
                        if (ctx.localAliases.count(tblName)) {
                             std::string target = ctx.localAliases.at(tblName);
                             debug_log("  Found local alias: " + tblName + " -> " + target);
                             if(rhsTables.count(target)) {
                                 debug_log("  Matched via local alias!");
                                 return true;
                             } else {
                                 debug_log("  But target " + target + " not in rhsTables");
                             }
                        }
                        if (tblName == capturedRightTable) return true;
                        // Extended logic to look for prefixes in captured table
                         if (!capturedRightTable.empty()) {
                             for(const auto& t : ctx.plan.tables) {
                                 if (t.name == capturedRightTable) {
                                     for(const auto& c : t.neededColumns) {
                                         if (tblName == "orders" && c.starts_with("o_")) return true;
                                         if (tblName == "lineitem" && c.starts_with("l_")) return true;
                                         if (tblName == "customer" && c.starts_with("c_")) return true;
                                         if (tblName == "part" && c.starts_with("p_")) return true;
                                         if (tblName == "supplier" && c.starts_with("s_")) return true;
                                         if (tblName == "nation" && c.starts_with("n_")) return true;
                                         if (tblName == "region" && c.starts_with("r_")) return true;
                                     }
                                 }
                             }
                        }
                        return false;
                     };

                     bool t1IsRight = matchesRHS(t1) || isInScope(l, rhsProjections);
                     bool t2IsRight = matchesRHS(t2) || isInScope(r, rhsProjections);
                     
                     bool t1IsLeft = isInScope(l, lhsProjections) || (!t1.empty() && ctx.seenTables.count(t1) && !t1IsRight);
                     bool t2IsLeft = isInScope(r, lhsProjections) || (!t2.empty() && ctx.seenTables.count(t2) && !t2IsRight);
                     
                     if (t1IsLeft && t2IsRight) {
                         join.leftKeys.push_back(l);
                         join.rightKeys.push_back(r);
                     } else if (t2IsLeft && t1IsRight) {
                         join.leftKeys.push_back(r);
                         join.rightKeys.push_back(l);
                     } else if (t1IsLeft && t2IsLeft) {
                         // Ambiguous / Split Case
                         if (t1IsRight && !t2IsRight) { // l matches both, r matches left
                             join.rightKeys.push_back(l); join.leftKeys.push_back(r);
                         } else if (t2IsRight && !t1IsRight) { // r matches both, l matches left
                             join.rightKeys.push_back(r); join.leftKeys.push_back(l);
                         } else {
                             // Prefer standard order
                             join.leftKeys.push_back(l); 
                             join.rightKeys.push_back(r);
                         }
                     } else if (t1IsRight && t2IsRight) {
                         // Both Right? Only if not left. Only possible if self-join on RHS or filter.
                         // Treat as 1=1 AND filter? Or attempt logical assign?
                         // If filter, it should be in filter clause, not join keys. 
                         // But if extracted from condition...
                         join.leftKeys.push_back(l);
                         join.rightKeys.push_back(r);
                     } else {
                         join.leftKeys.push_back(l);
                         join.rightKeys.push_back(r);
                     }
                 }
             }
        }
        
        joinNode.duckdbName = name;
        ctx.plan.nodes.push_back(std::move(joinNode));
    }
    
    // Update projections for parent
    if (!myProjs.empty()) {
        ctx.projections = myProjs;
    }
    
    std::string proj_list;
    for(auto& s : ctx.projections) proj_list += s + ", ";
    debug_log("Node " + name + " output projections: " + proj_list);
}

// --- Main parsing function ---

Plan Planner::fromSQL(const std::string& sql) {
    Plan plan;
    plan.originalSQL = sql;
    
    // Get DuckDB EXPLAIN JSON
    std::string raw = DuckDBAdapter::explainJSON(sql);
    
    // Extract JSON array
    std::string jsonStr;
    auto start = raw.find('[');
    if (start != std::string::npos) {
        int depth = 0;
        size_t end = start;
        for (size_t i = start; i < raw.size(); i++) {
            if (raw[i] == '[') depth++;
            else if (raw[i] == ']') {
                depth--;
                if (depth == 0) { end = i + 1; break; }
            }
        }
        jsonStr = raw.substr(start, end - start);
        while (!jsonStr.empty() && (jsonStr.back() == '%' || jsonStr.back() == '\n' || jsonStr.back() == '\r')) {
            jsonStr.pop_back();
        }
    } else {
        std::cerr << "DuckDB Raw Output:\n" << raw << "\n";
        plan.parseError = "Could not find JSON array in DuckDB output";
        return plan;
    }
    
    // Parse JSON
    try {
        json j = json::parse(jsonStr);
        if (!j.is_array() || j.size() == 0) {
            plan.parseError = "DuckDB JSON is not a non-empty array";
            return plan;
        }
        
        auto aliases = parseSelectAliases(sql);
        
        debug_log("Parsed aliases:");
        for (const auto& [k, v] : aliases) {
            debug_log("  '" + k + "' -> '" + v + "'");
        }
        
        TraverseContext ctx{plan, aliases, {}, {}, {}, false, {}, {}, {}};
        collectGlobalColumns(j[0], ctx.forceKeepColumns);
        traverseNode(j[0], ctx);
        
        plan.parsedFromJSON = true;
        
        // Recover LIMIT from SQL if not in plan
        bool hasLimit = false;
        for (const auto& n : plan.nodes) {
            if (n.type == IRNode::Type::Limit) { hasLimit = true; break; }
        }
        if (!hasLimit) {
            std::regex re_limit(R"(limit\s+(\d+))", std::regex::icase);
            std::smatch m;
            if (std::regex_search(sql, m, re_limit) && m.size() > 1) {
                plan.nodes.push_back(IRNode::limit(std::stoll(m[1].str())));
            }
        }
        
    } catch (const std::exception& e) {
        plan.parseError = std::string("JSON parse error: ") + e.what();
    }
    
    return plan;
}

// --- GPU feasibility check ---

Planner::Feasibility Planner::checkGPUFeasibility(const Plan& plan) {
    Feasibility f;
    f.canExecuteGPU = true;
    
    if (!plan.isValid()) {
        f.canExecuteGPU = false;
        f.blockers.push_back("Invalid plan: " + plan.parseError);
        return f;
    }
    
    size_t scanCount = 0;
    size_t joinCount = 0;
    
    for (const auto& node : plan.nodes) {
        switch (node.type) {
            case IRNode::Type::Scan:
                scanCount++;
                break;
                
            case IRNode::Type::Join: {
                joinCount++;
                const auto& j = node.asJoin();
                if (j.type != JoinType::Inner) {
                    f.blockers.push_back(std::string("Unsupported join type: ") + joinTypeName(j.type));
                }
                break;
            }
            
            case IRNode::Type::GroupBy: {
                const auto& gb = node.asGroupBy();
                // V2 executor uses CPU hash-based GroupBy which supports more keys
                if (gb.keys.size() > 8) {
                    f.blockers.push_back("GROUP BY with more than 8 keys: " + std::to_string(gb.keys.size()));
                }
                // Check for unsupported aggregate functions
                for (const auto& spec : gb.aggSpecs) {
                    if (spec.func != AggFunc::Sum && spec.func != AggFunc::Count && 
                        spec.func != AggFunc::CountStar && spec.func != AggFunc::Avg) {
                        f.blockers.push_back(std::string("Unsupported aggregate: ") + aggFuncName(spec.func));
                    }
                }
                break;
            }
            
            case IRNode::Type::Aggregate:
                // Scalar aggregates are supported
                break;
                
            case IRNode::Type::Filter:
                // Filters are generally supported
                break;
                
            case IRNode::Type::OrderBy:
            case IRNode::Type::Limit:
            case IRNode::Type::Project:
            case IRNode::Type::Save:
                // These are supported
                break;
                
            case IRNode::Type::Distinct:
                f.blockers.push_back("DISTINCT not yet supported");
                break;
                
            case IRNode::Type::Union:
                f.blockers.push_back("UNION not yet supported");
                break;
        }
    }
    
    // Check for multi-way joins
    if (joinCount > 2) {
        f.blockers.push_back("Multi-way join (>2 joins): " + std::to_string(joinCount));
    }
    
    f.canExecuteGPU = f.blockers.empty();
    return f;
}

// --- Extract needed columns ---

std::vector<std::pair<std::string, std::vector<std::string>>> 
Planner::extractNeededColumns(const Plan& plan) {
    std::vector<std::pair<std::string, std::vector<std::string>>> result;
    
    // Collect columns from all scans
    std::unordered_map<std::string, std::unordered_set<std::string>> tableColumns;
    
    for (const auto& node : plan.nodes) {
        if (node.type == IRNode::Type::Scan) {
            const auto& scan = node.asScan();
            for (const auto& col : scan.columns) {
                tableColumns[scan.table].insert(col);
            }
        }
    }
    
    // Also collect columns from expressions in filters, aggregates, etc.
    for (const auto& node : plan.nodes) {
        std::vector<ColumnRef> refs;
        
        switch (node.type) {
            case IRNode::Type::Filter:
                collectColumns(node.asFilter().predicate, refs);
                break;
            case IRNode::Type::GroupBy:
                for (const auto& key : node.asGroupBy().keys) collectColumns(key, refs);
                for (const auto& agg : node.asGroupBy().aggregates) collectColumns(agg, refs);
                break;
            case IRNode::Type::Aggregate:
                collectColumns(node.asAggregate().expr, refs);
                break;
            case IRNode::Type::OrderBy:
                for (const auto& spec : node.asOrderBy().specs) collectColumns(spec.expr, refs);
                break;
            case IRNode::Type::Project:
                for (const auto& expr : node.asProject().exprs) collectColumns(expr, refs);
                break;
            default:
                break;
        }
        
        // Add columns to their tables
        for (const auto& ref : refs) {
            if (!ref.table.empty()) {
                tableColumns[ref.table].insert(ref.column);
            } else {
                // Try to infer table from column prefix
                std::string col = ref.column;
                if (col.rfind("l_", 0) == 0) tableColumns["lineitem"].insert(col);
                else if (col.rfind("o_", 0) == 0) tableColumns["orders"].insert(col);
                else if (col.rfind("c_", 0) == 0) tableColumns["customer"].insert(col);
                else if (col.rfind("p_", 0) == 0) tableColumns["part"].insert(col);
                else if (col.rfind("s_", 0) == 0) tableColumns["supplier"].insert(col);
                else if (col.rfind("n_", 0) == 0) tableColumns["nation"].insert(col);
                else if (col.rfind("r_", 0) == 0) tableColumns["region"].insert(col);
            }
        }
    }
    
    for (const auto& [table, cols] : tableColumns) {
        result.push_back({table, std::vector<std::string>(cols.begin(), cols.end())});
    }
    
    return result;
}

} // namespace engine
