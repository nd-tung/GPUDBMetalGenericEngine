// ============================================================================
// PlannerUtils.cpp — String utilities and helpers for the Planner subsystem
// ============================================================================
#include "PlannerInternal.hpp"
#include "EnvUtil.hpp"
#include <iostream>
#include <algorithm>
#include <cctype>
#include "Logger.hpp"

namespace engine {

// --- Debug utilities ---

void debug_log(const std::string& msg) {
    static const bool enabled = env_truthy("GPUDB_DEBUG_PLANNER");
    if (!enabled) return;
    LOG_INFO("Planner", msg);
}

// --- String utilities ---

std::string tolower_str(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

std::string trim_str(std::string s) {
    auto first = s.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    auto last = s.find_last_not_of(" \t\n\r");
    return s.substr(first, last - first + 1);
}

// Normalize numeric literals by removing trailing zeros (1.00 -> 1)
std::string normalizeNumericLiterals(const std::string& s) {
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
std::string normalizeOperators(const std::string& s) {
    std::string result = s;
    size_t pos = 0;
    while ((pos = result.find("<>", pos)) != std::string::npos) {
        result.replace(pos, 2, "!=");
        pos += 2;
    }
    return result;
}

std::string strip_parens(std::string s) {
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
std::string stripTableQualifier(const std::string& s) {
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
std::vector<std::string> renameDuplicateColumns(
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

} // namespace engine
