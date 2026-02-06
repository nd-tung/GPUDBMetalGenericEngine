#include "TypedExprEval.hpp"
#include <regex>
#include <algorithm>
#include <cctype>

namespace engine {

static std::string trim_str(std::string s) {
    auto first = s.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    auto last = s.find_last_not_of(" \t\n\r");
    return s.substr(first, last - first + 1);
}

static std::string tolower_str(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

static std::string stripTableQualifier(const std::string& s) {
    if (s.empty()) return s;
    auto dot = s.rfind('.');
    
    // Safety check: if dot is part of a number (e.g. 1.00), don't strip
    if (dot != std::string::npos && dot + 1 < s.size() && std::isdigit(static_cast<unsigned char>(s[dot+1]))) {
        return s;
    }

    if (dot != std::string::npos && dot + 1 < s.size()) {
        return s.substr(dot + 1);
    }
    return s;
}

// ============================================================================
// Parse predicate string
// ============================================================================

TypedExprPtr parsePredicateString(const std::string& pred) {
    std::string s = trim_str(pred);
    if (s.empty()) return nullptr;
    
    // Handle AND conjunction
    std::string slower = tolower_str(s);
    
    // Split by AND at top level (outside parentheses)
    std::vector<std::string> clauses;
    int depth = 0;
    size_t start = 0;
    
    for (size_t i = 0; i + 3 <= slower.size(); ++i) {
        char c = s[i];
        if (c == '(') depth++;
        else if (c == ')') depth--;
        
        if (depth == 0 && slower.compare(i, 4, " and") == 0) {
            // Check it's not part of a larger word
            if ((i == 0 || !std::isalnum(static_cast<unsigned char>(slower[i-1]))) &&
                (i + 4 >= slower.size() || !std::isalnum(static_cast<unsigned char>(slower[i+4])))) {
                clauses.push_back(trim_str(s.substr(start, i - start)));
                start = i + 4;
            }
        }
    }
    clauses.push_back(trim_str(s.substr(start)));
    
    // If we have multiple clauses, create AND chain
    if (clauses.size() > 1) {
        TypedExprPtr result = parsePredicateString(clauses[0]);
        for (size_t i = 1; i < clauses.size(); ++i) {
            result = TypedExpr::binary(BinaryOp::And, result, parsePredicateString(clauses[i]));
        }
        return result;
    }
    
    // Single clause: parse comparison
    static const std::regex cmpRe(R"((.+?)\s*(>=|<=|<>|!=|>|<|=)\s*(.+))");
    std::smatch m;
    if (std::regex_match(s, m, cmpRe)) {
        std::string left = trim_str(m[1].str());
        std::string opStr = m[2].str();
        std::string right = trim_str(m[3].str());
        
        CompareOp op;
        if (opStr == "=" || opStr == "==") op = CompareOp::Eq;
        else if (opStr == "<>" || opStr == "!=") op = CompareOp::Ne;
        else if (opStr == "<") op = CompareOp::Lt;
        else if (opStr == "<=") op = CompareOp::Le;
        else if (opStr == ">") op = CompareOp::Gt;
        else if (opStr == ">=") op = CompareOp::Ge;
        else op = CompareOp::Eq;
        
        TypedExprPtr leftExpr = parseArithmeticString(left);
        TypedExprPtr rightExpr = parseArithmeticString(right);
        
        return TypedExpr::compare(op, leftExpr, rightExpr);
    }
    
    // Fallback: treat as boolean column
    return TypedExpr::column(stripTableQualifier(s));
}

// ============================================================================
// Parse arithmetic expression string
// ============================================================================

TypedExprPtr parseArithmeticString(const std::string& expr) {
    std::string s = trim_str(expr);
    if (s.empty()) return nullptr;
    
    // Strip outer parentheses
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
    
    if (s.empty()) return nullptr;
    
    // Check for DATE literal
    static const std::regex dateRe(R"(DATE\s*'(\d{4}-\d{2}-\d{2})')", std::regex::icase);
    std::smatch m;
    if (std::regex_match(s, m, dateRe)) {
        std::string dateStr = m[1].str();
        // Convert YYYY-MM-DD to YYYYMMDD integer
        dateStr.erase(std::remove(dateStr.begin(), dateStr.end(), '-'), dateStr.end());
        return TypedExpr::literal(std::stoll(dateStr));
    }
    
    // Check for string literal
    if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'') {
        return TypedExpr::literal(s.substr(1, s.size() - 2), DataType::String);
    }
    
    // Check for numeric literal
    bool isNumeric = true;
    bool hasDot = false;
    size_t start = 0;
    if (!s.empty() && (s[0] == '-' || s[0] == '+')) start = 1;
    for (size_t i = start; i < s.size() && isNumeric; ++i) {
        if (s[i] == '.') {
            if (hasDot) isNumeric = false;
            hasDot = true;
        } else if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
            isNumeric = false;
        }
    }
    if (isNumeric && !s.empty() && (start < s.size())) {
        try {
            if (hasDot) {
                return TypedExpr::literal(std::stod(s));
            } else {
                return TypedExpr::literal(std::stoll(s));
            }
        } catch (...) {}
    }
    
    // Find binary operator at lowest precedence (outside parentheses)
    // Precedence: + - (level 1), * / (level 2)
    int depth = 0;
    size_t opPos = std::string::npos;
    char opChar = 0;
    int opPrecedence = 100;
    
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '(') depth++;
        else if (c == ')') depth--;
        else if (depth == 0) {
            int prec = 0;
            if (c == '+' || c == '-') prec = 1;
            else if (c == '*' || c == '/') prec = 2;
            
            if (prec > 0 && prec <= opPrecedence) {
                // Skip if this is a unary minus at start or after an operator
                if ((c == '-' || c == '+') && (i == 0 || s[i-1] == '(' || s[i-1] == '+' || 
                    s[i-1] == '-' || s[i-1] == '*' || s[i-1] == '/')) {
                    continue;
                }
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
        
        return TypedExpr::binary(op, parseArithmeticString(left), parseArithmeticString(right));
    }
    
    // Check for aggregate function
    static const std::regex aggRe(R"(^(sum|count|avg|min|max|count_star)\s*\((.*)\)$)", std::regex::icase);
    if (std::regex_match(s, m, aggRe)) {
        std::string funcName = tolower_str(m[1].str());
        std::string inner = trim_str(m[2].str());
        
        AggFunc func;
        if (funcName.find("sum") != std::string::npos) func = AggFunc::Sum;
        else if (funcName.find("count_star") != std::string::npos) func = AggFunc::CountStar;
        else if (funcName.find("count") != std::string::npos) func = AggFunc::Count;
        else if (funcName.find("avg") != std::string::npos) func = AggFunc::Avg;
        else if (funcName.find("min") != std::string::npos) func = AggFunc::Min;
        else if (funcName.find("max") != std::string::npos) func = AggFunc::Max;
        else func = AggFunc::Sum;
        
        TypedExprPtr arg = inner.empty() ? nullptr : parseArithmeticString(inner);
        return TypedExpr::aggregate(func, arg);
    }
    
    // Column reference
    return TypedExpr::column(stripTableQualifier(s));
}

} // namespace engine
