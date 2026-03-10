// ============================================================================
// PlannerExprParser.cpp — Expression parsing (CAST, CASE, comparisons, etc.)
// ============================================================================
#include "PlannerInternal.hpp"
#include "EnvUtil.hpp"
#include <iostream>
#include <algorithm>
#include <regex>
#include <cctype>

namespace engine {

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

// Helper: parse CAST(expr AS type) — find outermost " AS " at depth 0,
// determine target DataType from the type string, and build a CastExpr node.
static TypedExprPtr parseCastExpression(const std::string& argsStr) {
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
            break;
        }
    }

    if (asPos == std::string::npos) return nullptr;

    std::string exprPart = trim_str(argsStr.substr(0, asPos));
    std::string typePart = trim_str(argsStr.substr(asPos + 4));

    TypedExprPtr innerExpr = Planner::parseExpression(exprPart);

    auto e = std::make_shared<TypedExpr>();
    e->kind = TypedExpr::Kind::Cast;
    CastExpr cast;
    cast.expr = innerExpr;

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

// Helper: parse CASE WHEN cond THEN val [WHEN...] [ELSE val] END expression
// with depth-aware/quote-aware scanning for clause boundaries.
static TypedExprPtr parseCaseExpression(const std::string& s) {
    std::string upper = s;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    if (!(upper.find("CASE") == 0 && upper.rfind("END") == upper.size() - 3))
        return nullptr;

    std::string body = trim_str(s.substr(4, s.size() - 7)); // strip "CASE" and "END"
    CaseExpr caseExpr;
    size_t pos = 0;

    while (pos < body.size()) {
        std::string upper_body = body.substr(pos);
        std::transform(upper_body.begin(), upper_body.end(), upper_body.begin(), ::toupper);

        size_t whenPos = upper_body.find("WHEN");
        if (whenPos == std::string::npos) break;

        // Find THEN at depth 0
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
        wt.when = Planner::parseExpression(whenCond);
        wt.then = Planner::parseExpression(thenVal);
        caseExpr.cases.push_back(std::move(wt));
        pos += nextClause;
    }

    // Check for ELSE clause
    std::string upper_remaining = body.substr(pos);
    std::transform(upper_remaining.begin(), upper_remaining.end(), upper_remaining.begin(), ::toupper);
    size_t elsePos = upper_remaining.find("ELSE");
    if (elsePos != std::string::npos) {
        std::string elseVal = trim_str(body.substr(pos + elsePos + 4));
        caseExpr.elseExpr = Planner::parseExpression(elseVal);
    }

    auto e = std::make_shared<TypedExpr>();
    e->kind = TypedExpr::Kind::Case;
    e->data = std::move(caseExpr);
    return e;
}

// Helper: parse comparison expressions at depth 0 (handles >=, <=, <>, !=, ~~, !~~, >, <, =)
static TypedExprPtr parseComparisonExpression(const std::string& s) {
    int depth = 0;
    bool inQuote = false;
    size_t cmpPos = std::string::npos;
    std::string cmpOp;

    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '\'' && (i == 0 || s[i-1] != '\\')) { inQuote = !inQuote; continue; }
        if (inQuote) continue;

        if (c == '(') depth++;
        else if (c == ')') depth--;
        else if (depth == 0) {
            if (i + 2 <= s.size()) {
                std::string op2 = s.substr(i, 2);
                if (op2 == "!~" && i + 3 <= s.size() && s[i+2] == '~') { cmpPos = i; cmpOp = "!~~"; break; }
                if (op2 == "~~") { cmpPos = i; cmpOp = "~~"; break; }
                if (op2 == ">=" || op2 == "<=" || op2 == "<>" || op2 == "!=") { cmpPos = i; cmpOp = op2; break; }
            }
            if ((c == '>' || c == '<' || c == '=') && i > 0 && i + 1 < s.size()) {
                char prev = s[i-1], next = s[i+1];
                if (c == '>' && prev != '!' && prev != '<' && next != '=') { cmpPos = i; cmpOp = ">"; break; }
                if (c == '<' && prev != '!' && next != '>' && next != '=') { cmpPos = i; cmpOp = "<"; break; }
                if (c == '=' && prev != '!' && prev != '>' && prev != '<' && next != '=') { cmpPos = i; cmpOp = "="; break; }
            }
        }
    }

    if (cmpPos == std::string::npos || cmpOp.empty()) return nullptr;

    std::string left = trim_str(s.substr(0, cmpPos));
    std::string right = trim_str(s.substr(cmpPos + cmpOp.length()));

    if (cmpOp == "~~" || cmpOp == "!~~") {
        std::string funcName = (cmpOp == "~~") ? "LIKE" : "NOTLIKE";
        FunctionCall func;
        func.name = funcName;
        func.args.push_back(Planner::parseExpression(left));
        func.args.push_back(Planner::parseExpression(right));
        func.returnType = DataType::Bool;
        auto e = std::make_shared<TypedExpr>();
        e->kind = TypedExpr::Kind::Function;
        e->data = std::move(func);
        return e;
    }

    return TypedExpr::compare(Planner::parseCompareOp(cmpOp), Planner::parseExpression(left), Planner::parseExpression(right));
}

// Helper: parse function calls — EXTRACT(part FROM expr), SUBSTRING(col FROM n FOR m),
// and generic function(arg1, arg2, ...) with comma-splitting. Skips aggregates.
static TypedExprPtr parseFunctionCallExpression(const std::string& funcUpper, const std::string& argsStr) {
    // Already handled: CAST (via parseCastExpression)

    // EXTRACT(part FROM expr)
    if (funcUpper == "EXTRACT") {
        std::string argsUpper = argsStr;
        std::transform(argsUpper.begin(), argsUpper.end(), argsUpper.begin(), ::toupper);
        size_t fromPos = argsUpper.find(" FROM ");
        if (fromPos != std::string::npos) {
            auto e = std::make_shared<TypedExpr>();
            e->kind = TypedExpr::Kind::Function;
            FunctionCall fc;
            fc.name = "EXTRACT";
            fc.args.push_back(TypedExpr::literal(trim_str(argsStr.substr(0, fromPos))));
            fc.args.push_back(Planner::parseExpression(trim_str(argsStr.substr(fromPos + 6))));
            e->data = std::move(fc);
            return e;
        }
    }

    // SUBSTRING(col FROM start FOR length) — SQL standard syntax
    if (funcUpper == "SUBSTRING" || funcUpper == "SUBSTR") {
        std::string argsUpper = argsStr;
        std::transform(argsUpper.begin(), argsUpper.end(), argsUpper.begin(), ::toupper);
        size_t fromPos = argsUpper.find(" FROM ");
        if (fromPos != std::string::npos) {
            std::string colStr = trim_str(argsStr.substr(0, fromPos));
            std::string rest = argsStr.substr(fromPos + 6);
            std::string restUpper = rest;
            std::transform(restUpper.begin(), restUpper.end(), restUpper.begin(), ::toupper);

            int startVal = 1, lengthVal = -1;
            size_t forPos = restUpper.find(" FOR ");
            if (forPos != std::string::npos) {
                startVal = std::atoi(trim_str(rest.substr(0, forPos)).c_str());
                lengthVal = std::atoi(trim_str(rest.substr(forPos + 5)).c_str());
            } else {
                startVal = std::atoi(trim_str(rest).c_str());
            }

            auto e = std::make_shared<TypedExpr>();
            e->kind = TypedExpr::Kind::Function;
            FunctionCall fc;
            fc.name = "SUBSTRING";
            fc.args.push_back(Planner::parseExpression(colStr));
            fc.args.push_back(TypedExpr::literal(static_cast<int64_t>(startVal)));
            fc.args.push_back(TypedExpr::literal(static_cast<int64_t>(lengthVal >= 0 ? lengthVal : 9999)));
            e->data = std::move(fc);
            return e;
        }
    }

    // Skip aggregate functions (handled separately in GROUP_BY nodes)
    if (funcUpper == "SUM" || funcUpper == "COUNT" || funcUpper == "AVG" ||
        funcUpper == "MIN" || funcUpper == "MAX" || funcUpper == "DATE")
        return nullptr;

    // Generic function: split args by comma at depth 0
    std::vector<TypedExprPtr> args;
    size_t start = 0;
    int depth = 0;
    bool inQuote = false;
    for (size_t i = 0; i <= argsStr.size(); ++i) {
        if (i == argsStr.size() || (argsStr[i] == ',' && depth == 0 && !inQuote)) {
            std::string arg = trim_str(argsStr.substr(start, i - start));
            if (!arg.empty()) args.push_back(Planner::parseExpression(arg));
            start = i + 1;
        } else {
            char c = argsStr[i];
            if (c == '\'' && (i == 0 || argsStr[i-1] != '\\')) inQuote = !inQuote;
            if (!inQuote) {
                if (c == '(') depth++;
                else if (c == ')') depth--;
            }
        }
    }

    auto e = std::make_shared<TypedExpr>();
    e->kind = TypedExpr::Kind::Function;
    FunctionCall fc;
    fc.name = funcUpper;
    fc.args = std::move(args);
    e->data = std::move(fc);
    return e;
}

// -- Extracted: parseLogicalAndOr --
// Depth/quote-aware scan for rightmost top-level AND/OR, splitting into binary node.
static TypedExprPtr parseLogicalAndOr(const std::string& s, const std::string& upper) {
    int depth = 0;
    bool inQuote = false;
    size_t andPos = std::string::npos;
    size_t orPos = std::string::npos;
    bool inBetween = false;

    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '\'' && (i == 0 || s[i-1] != '\\')) { inQuote = !inQuote; continue; }
        if (inQuote) continue;
        if (c == '(') { depth++; inBetween = false; }
        else if (c == ')') { depth--; inBetween = false; }
        else if (depth == 0) {
            if (i + 8 <= s.size() && upper.substr(i, 8) == " BETWEEN") inBetween = true;
            if (i + 5 <= s.size() && upper.substr(i, 5) == " AND ") {
                if (inBetween) inBetween = false;
                else andPos = i;
            }
            if (i + 4 <= s.size() && upper.substr(i, 4) == " OR ") {
                orPos = i;
                inBetween = false;
            }
        }
    }

    if (orPos != std::string::npos) {
        std::string left = trim_str(s.substr(0, orPos));
        std::string right = trim_str(s.substr(orPos + 4));
        return TypedExpr::binary(BinaryOp::Or, Planner::parseExpression(left), Planner::parseExpression(right));
    }
    if (andPos != std::string::npos) {
        std::string left = trim_str(s.substr(0, andPos));
        std::string right = trim_str(s.substr(andPos + 5));
        return TypedExpr::binary(BinaryOp::And, Planner::parseExpression(left), Planner::parseExpression(right));
    }
    return nullptr;
}

// -- Extracted: parseInListExpression --
// Matches "column IN (val1, val2, ...)" and returns an inList node.
static TypedExprPtr parseInListExpression(const std::string& s) {
    static const std::regex inRe(R"(^(.+?)\s+IN\s*\((.+)\)$)", std::regex::icase);
    std::smatch m;
    if (!std::regex_match(s, m, inRe)) return nullptr;

    std::string col = trim_str(m[1].str());
    std::string listStr = m[2].str();

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
            if (!current.empty()) listExprs.push_back(Planner::parseExpression(current));
            current.clear();
        } else {
            current += c;
        }
    }
    return TypedExpr::inList(Planner::parseExpression(col), std::move(listExprs));
}

// -- Extracted: parseBinaryArithmetic --
// Depth/quote-aware scan for binary arithmetic operators (+, -, *, /).
static TypedExprPtr parseBinaryArithmetic(const std::string& s, bool debug) {
    int depth = 0;
    bool inQuote = false;
    size_t opPos = std::string::npos;
    char opChar = 0;
    int opPrecedence = 100;

    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '\'' && (i == 0 || s[i-1] != '\\')) { inQuote = !inQuote; continue; }
        if (inQuote) continue;
        if (c == '(') depth++;
        else if (c == ')') depth--;
        else if (depth == 0) {
            int prec = 0;
            if (c == '+' || c == '-') prec = 1;
            else if (c == '*' || c == '/') prec = 2;
            if (prec > 0 && prec <= opPrecedence) {
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
        return TypedExpr::binary(op, Planner::parseExpression(left), Planner::parseExpression(right));
    }
    return nullptr;
}

// -- Extracted: parseLiteralOrDate --
// Tries numeric literal (int/float) and DATE 'YYYY-MM-DD' patterns.
static TypedExprPtr parseLiteralOrDate(const std::string& s) {
    // Numeric literal
    try {
        size_t pos = 0;
        double d = std::stod(s, &pos);
        if (pos == s.size()) {
            if (s.find('.') == std::string::npos)
                return TypedExpr::literal(static_cast<int64_t>(d));
            return TypedExpr::literal(d);
        }
    } catch (...) {
        if (env_truthy("GPUDB_DEBUG_PLANNER"))
            std::cerr << "[Planner] parseExpression: numeric parse failed for '" << s.substr(0, 60) << "'\n";
    }

    static const std::regex dateRe1(R"(DATE\s*'(\d{4}-\d{2}-\d{2})')", std::regex::icase);
    static const std::regex dateRe2(R"('(\d{4}-\d{2}-\d{2})'::DATE)", std::regex::icase);

    auto makeDateInt = [](const std::string& ds) -> TypedExprPtr {
        std::string t = ds;
        t.erase(std::remove(t.begin(), t.end(), '-'), t.end());
        try { return TypedExpr::literal((int64_t)std::stoll(t)); }
        catch(...) { return TypedExpr::literal(ds, DataType::Date); }
    };

    std::smatch m;
    if (std::regex_match(s, m, dateRe1)) return makeDateInt(m[1].str());
    if (std::regex_match(s, m, dateRe2)) return makeDateInt(m[1].str());
    return nullptr;
}

// -- Extracted: parsePostgresCast --
// Handles 'value'::TYPE and value::TYPE PostgreSQL-style cast expressions.
static TypedExprPtr parsePostgresCast(const std::string& s) {
    size_t castPos = s.find("::");
    if (castPos == std::string::npos || castPos == 0) return nullptr;

    std::string valPart = trim_str(s.substr(0, castPos));
    std::string typePart = trim_str(s.substr(castPos + 2));

    if (valPart.size() >= 2 && valPart.front() == '\'' && valPart.back() == '\'') {
        std::string strVal = valPart.substr(1, valPart.size() - 2);
        std::string typeUpper = typePart;
        std::transform(typeUpper.begin(), typeUpper.end(), typeUpper.begin(), ::toupper);
        if (typeUpper.find("DATE") != std::string::npos ||
            typeUpper.find("TIMESTAMP") != std::string::npos) {
            std::string dateVal = strVal;
            size_t spacePos = dateVal.find(' ');
            if (spacePos != std::string::npos) dateVal = dateVal.substr(0, spacePos);
            return TypedExpr::literal(dateVal, DataType::Date);
        }
        return TypedExpr::literal(strVal, DataType::String);
    }
    return Planner::parseExpression(valPart);
}

// Simple expression parser for DuckDB expression strings
TypedExprPtr Planner::parseExpression(const std::string& exprStr) {
    std::string s = strip_parens(exprStr);
    if (s.empty()) return nullptr;
    
    bool debug = env_truthy("GPUDB_DEBUG_PARSE");
    if (debug) {
        std::cerr << "[parseExpression] input: '" << s.substr(0, 80) << (s.size() > 80 ? "..." : "") << "'\n";
    }
    
    std::string upper = s;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    
    // Check for AND/OR at top level (lowest precedence logical operators)
    {
        auto logicResult = parseLogicalAndOr(s, upper);
        if (logicResult) return logicResult;
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
    
    // Check for IN (list) expression
    {
        auto inResult = parseInListExpression(s);
        if (inResult) return inResult;
    }
    
    // Check for CASE expression: CASE WHEN cond THEN val [WHEN cond THEN val]* [ELSE val] END
    {
        auto caseResult = parseCaseExpression(s);
        if (caseResult) return caseResult;
    }
    
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
    
    // Check for comparison operators at depth 0
    {
        auto cmpResult = parseComparisonExpression(s);
        if (cmpResult) {
            if (debug) {
                std::cerr << "[parseExpression] comparison matched\n";
            }
            return cmpResult;
        }
    }
    
    // Check for binary arithmetic (+ - * /)
    {
        auto arithResult = parseBinaryArithmetic(s, debug);
        if (arithResult) return arithResult;
    }

    // Check for numeric/date literals
    {
        auto litResult = parseLiteralOrDate(s);
        if (litResult) return litResult;
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
            auto castExpr = parseCastExpression(m[2].str());
            if (castExpr) return castExpr;
        }
        
        // Handle EXTRACT, SUBSTRING, and generic function calls
        {
            auto funcResult = parseFunctionCallExpression(funcUpper, m[2].str());
            if (funcResult) return funcResult;
        }
    }
    
    // Check for PostgreSQL-style cast: 'value'::TYPE or value::TYPE
    {
        auto castResult = parsePostgresCast(s);
        if (castResult) return castResult;
    }
    
    // Check for string literal
    if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'') {
        return TypedExpr::literal(s.substr(1, s.size() - 2), DataType::String);
    }
    
    // Otherwise treat as column reference
    std::string colName = stripTableQualifier(s);
    return TypedExpr::column(colName);
}

} // namespace engine
