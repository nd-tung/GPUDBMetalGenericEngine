#pragma once
// Inline utility helpers shared across executor modules.
// Split from GpuExecutorDetail.hpp for focused compilation.

#include "EvalContext.hpp"
#include "Operators.hpp"
#include "Schema.hpp"
#include "IR.hpp"

#include <string>
#include <vector>
#include <set>
#include <cctype>
#include <cstring>
#include <iostream>
#include "Logger.hpp"

namespace engine {

// ========== GPU gather → CPU vector helpers ==========
// Gather GPU buffer by index buffer into a CPU vector, releasing the gathered GPU buffer.
// T = uint32_t calls gatherU32, T = float calls gatherF32.
template<typename T>
inline std::vector<T> gatherToVector(MTL::Buffer* buf, MTL::Buffer* indices, uint32_t count) {
    GpuBuffer gathered;
    if constexpr (std::is_same_v<T, float>)
        gathered = GpuOps::gatherF32(buf, indices, count);
    else
        gathered = GpuOps::gatherU32(buf, indices, count);
    std::vector<T> result(count);
    std::memcpy(result.data(), gathered->contents(), count * sizeof(T));
    return result;
}

// ========== GPU buffer access helpers ==========
// Typed getters that return a non-owning pointer (or null) to reduce
// repeated if-count-then-at boilerplate across executor files.

// Return the raw GPU buffer for a u32 column, or nullptr if absent.
inline MTL::Buffer* getU32GPU(const EvalContext& ctx, const std::string& col) {
    auto it = ctx.u32ColsGPU.find(col);
    return (it != ctx.u32ColsGPU.end() && it->second) ? it->second.get() : nullptr;
}

// Return the raw GPU buffer for a f32 column, or nullptr if absent.
inline MTL::Buffer* getF32GPU(const EvalContext& ctx, const std::string& col) {
    auto it = ctx.f32ColsGPU.find(col);
    return (it != ctx.f32ColsGPU.end() && it->second) ? it->second.get() : nullptr;
}

// Check if a flat string column exists and has data.
inline bool hasFlatString(const EvalContext& ctx, const std::string& col) {
    auto it = ctx.flatStringCols.find(col);
    return it != ctx.flatStringCols.end() && it->second.chars && it->second.rowCount > 0;
}

// Return pointer to a FlatStringCol, or nullptr if absent/empty.
inline const FlatStringCol* getFlatString(const EvalContext& ctx, const std::string& col) {
    auto it = ctx.flatStringCols.find(col);
    if (it != ctx.flatStringCols.end() && it->second.chars && it->second.rowCount > 0)
        return &it->second;
    return nullptr;
}

// Return pointer to a DictEncoded column, or nullptr if absent/invalid.
inline const DictEncoded* getDictCol(const EvalContext& ctx, const std::string& col) {
    auto it = ctx.dictCols.find(col);
    if (it != ctx.dictCols.end() && it->second.valid())
        return &it->second;
    return nullptr;
}

// ========== String utility helpers ==========

// Split a condition string by " AND " into parts.
inline std::vector<std::string> splitConditionByAnd(const std::string& s) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start < s.size()) {
        size_t andPos = s.find(" AND ", start);
        if (andPos == std::string::npos) {
            parts.push_back(s.substr(start));
            break;
        }
        parts.push_back(s.substr(start, andPos - start));
        start = andPos + 5;
    }
    return parts;
}

// Parse a self-comparison condition ("col IS NOT DISTINCT FROM col" or "col = col").
// Returns the column name if it is a self-comparison, empty string otherwise.
// If isINDF is non-null, sets it to true when the pattern is IS NOT DISTINCT FROM.
inline std::string parseSelfComparison(const std::string& part, bool* isINDF = nullptr) {
    if (isINDF) *isINDF = false;
    // Check "col IS NOT DISTINCT FROM col"
    size_t indfPos = part.find("IS NOT DISTINCT FROM");
    if (indfPos != std::string::npos) {
        std::string lhs = part.substr(0, indfPos);
        std::string rhs = part.substr(indfPos + 20);
        while (!lhs.empty() && std::isspace(static_cast<unsigned char>(lhs.back()))) lhs.pop_back();
        while (!lhs.empty() && std::isspace(static_cast<unsigned char>(lhs.front()))) lhs.erase(0, 1);
        while (!rhs.empty() && std::isspace(static_cast<unsigned char>(rhs.back()))) rhs.pop_back();
        while (!rhs.empty() && std::isspace(static_cast<unsigned char>(rhs.front()))) rhs.erase(0, 1);
        // Strip trailing garbage (e.g., closing parentheses)
        auto endPos = rhs.find_first_of(" )");
        if (endPos != std::string::npos) rhs = rhs.substr(0, endPos);
        if (lhs == rhs || (!lhs.empty() && !rhs.empty() && lhs.find(rhs) != std::string::npos)) {
            if (isINDF) *isINDF = true;
            return lhs;
        }
        return "";
    }
    // Check "col = col"
    size_t eqPos = part.find(" = ");
    if (eqPos != std::string::npos) {
        std::string lhs = part.substr(0, eqPos);
        std::string rhs = part.substr(eqPos + 3);
        while (!lhs.empty() && std::isspace(static_cast<unsigned char>(lhs.back()))) lhs.pop_back();
        while (!rhs.empty() && std::isspace(static_cast<unsigned char>(rhs.front()))) rhs.erase(0, 1);
        if (lhs == rhs) return lhs;
    }
    return "";
}

inline std::string trim_copy(std::string s) {
    auto first = s.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    auto last = s.find_last_not_of(" \t\n\r");
    return s.substr(first, last - first + 1);
}

inline std::string base_ident(std::string s) {
    s = trim_copy(std::move(s));
    while (!s.empty() && s.front() == '(' && s.back() == ')') {
        s = s.substr(1, s.size() - 2);
        s = trim_copy(std::move(s));
    }
    auto dot = s.rfind('.');
    if (dot != std::string::npos && dot + 1 < s.size()) s = s.substr(dot + 1);
    return trim_copy(std::move(s));
}

inline std::string tableForColumn(const std::string& col) {
    return SchemaRegistry::instance().tableForColumn(base_ident(col));
}

inline std::string cleanupColumnName(const std::string& name) {
    std::string n = name;
    static const std::vector<std::string> prefixes = {
        "__internal_decompress_string(",
        "__internal_compress_string_utinyint(",
        "__internal_compress_string_uhugeint(",
        "__internal_decompress_integral_integer(",
        "__internal_decompress_integral_bigint(",
        "__internal_compress_integral_utinyint(",
        "__internal_compress_integral_usmallint(",
        "__internal_compress_integral_uinteger(",
        "__internal_compress_integral_ubigint(",
        "__internal_decompress_integral_usmallint(",
        "__internal_decompress_integral_uinteger(",
        "__internal_decompress_integral_ubigint(",
    };
    for (const auto& prefix : prefixes) {
        if (n.rfind(prefix, 0) == 0 && !n.empty() && n.back() == ')') {
            n = n.substr(prefix.size(), n.size() - prefix.size() - 1);
            auto comma = n.find(',');
            if (comma != std::string::npos) {
                n = n.substr(0, comma);
            }
            n = trim_copy(n);
        }
    }
    return n;
}

inline void collectColumnsFromExpr(const TypedExprPtr& expr, std::set<std::string>& cols) {
    if (!expr) return;
    std::vector<ColumnRef> refs;
    collectColumns(expr, refs);
    for (const auto& ref : refs) {
        cols.insert(ref.column);
    }
}

inline bool isColumnEqualsLiteral(const TypedExprPtr& expr, std::string& colName, std::string& literalVal) {
    if (!expr || expr->kind != TypedExpr::Kind::Compare) return false;
    const auto& cmp = expr->asCompare();
    if (cmp.op != CompareOp::Eq) return false;
    if (!cmp.left || !cmp.right) return false;
    
    const TypedExprPtr* colExpr = nullptr;
    const TypedExprPtr* litExpr = nullptr;
    
    if (cmp.left->kind == TypedExpr::Kind::Column && cmp.right->kind == TypedExpr::Kind::Literal) {
        colExpr = &cmp.left;
        litExpr = &cmp.right;
    } else if (cmp.right->kind == TypedExpr::Kind::Column && cmp.left->kind == TypedExpr::Kind::Literal) {
        colExpr = &cmp.right;
        litExpr = &cmp.left;
    } else {
        return false;
    }
    
    colName = (*colExpr)->asColumn().column;
    const auto& lit = (*litExpr)->asLiteral();
    if (std::holds_alternative<std::string>(lit.value)) {
        literalVal = std::get<std::string>(lit.value);
        return true;
    }
    return false;
}

inline TypedExprPtr makeCompareWithColumn(const TypedExprPtr& original, const std::string& newColName) {
    if (!original || original->kind != TypedExpr::Kind::Compare) return original;
    const auto& cmp = original->asCompare();
    
    // Create new column expression
    auto newCol = TypedExpr::column(newColName);
    
    // Build new Compare with the new column
    if (cmp.left && cmp.left->kind == TypedExpr::Kind::Column) {
        return TypedExpr::compare(cmp.op, newCol, cmp.right);
    } else if (cmp.right && cmp.right->kind == TypedExpr::Kind::Column) {
        return TypedExpr::compare(cmp.op, cmp.left, newCol);
    }
    
    return original;
}

// Transform predicates to use suffixed column names for multi-instance columns
// availableCols is the set of all column names currently available in the context
inline TypedExprPtr transformMultiInstancePredicate(const TypedExprPtr& pred, 
                                                     const std::set<std::string>& availableCols, 
                                                     bool debug) {
    if (!pred) return pred;
    
    if (debug) {
        LOG_INFO("Exec", "transformMultiInstancePredicate: pred kind=" << static_cast<int>(pred->kind));
        LOG_INFO("Exec", "availableCols: ");
        for (const auto& c : availableCols) std::cerr << c << ", ";
        LOG_INFO("FILTER", "\n");
    }
    
    // Only handle Binary predicates (AND/OR)
    if (pred->kind != TypedExpr::Kind::Binary) {
        return pred;
    }
    
    const auto& bin = pred->asBinary();
    
    if (bin.op == BinaryOp::Or) {
        // Recurse into both sides of OR
        auto newLeft = transformMultiInstancePredicate(bin.left, availableCols, debug);
        auto newRight = transformMultiInstancePredicate(bin.right, availableCols, debug);
        if (newLeft != bin.left || newRight != bin.right) {
            return TypedExpr::binary(BinaryOp::Or, newLeft, newRight);
        }
        return pred;
    }
    
    if (bin.op != BinaryOp::And) return pred;
    
    // Check if both sides are col = literal with same column but different literals
    std::string leftCol, leftLit;
    std::string rightCol, rightLit;
    
    if (!isColumnEqualsLiteral(bin.left, leftCol, leftLit)) {
        if (debug) {
            LOG_INFO("Exec", "transformMultiInstancePredicate: left side is not col=literal, recursing\n");
        }
        // Maybe left side is another AND - recurse
        auto newLeft = transformMultiInstancePredicate(bin.left, availableCols, debug);
        auto newRight = transformMultiInstancePredicate(bin.right, availableCols, debug);
        if (newLeft != bin.left || newRight != bin.right) {
            return TypedExpr::binary(BinaryOp::And, newLeft, newRight);
        }
        return pred;
    }
    
    if (!isColumnEqualsLiteral(bin.right, rightCol, rightLit)) {
        return pred;
    }
    
    // Both sides are col = literal
    if (leftCol != rightCol) {
        return pred;  // Different columns, no transformation needed
    }
    
    if (leftLit == rightLit) {
        return pred;  // Same value, no transformation needed (probably a no-op anyway)
    }
    
    // Same column, different values! 
    // Look for a suffixed version of the column (col_1, col_2, etc.)
    // When the unsuffixed column exists (e.g., n_name from nation_1), 
    // prefer the higher suffix (n_name_2 for nation_2) to get a different instance
    std::string altCol;
    bool hasUnsuffixed = (availableCols.count(leftCol) > 0);
    
    if (hasUnsuffixed) {
        // Prefer higher suffix (start from 2) to get different instance
        for (int suffix = 2; suffix >= 1; --suffix) {
            std::string candidate = leftCol + "_" + std::to_string(suffix);
            if (availableCols.count(candidate) > 0) {
                altCol = candidate;
                break;
            }
        }
    } else {
        // No unsuffixed column, try suffixes in order
        for (int suffix = 1; suffix <= 2; ++suffix) {
            std::string candidate = leftCol + "_" + std::to_string(suffix);
            if (availableCols.count(candidate) > 0) {
                altCol = candidate;
                break;
            }
        }
    }
    
    if (altCol.empty()) {
        return pred;  // No alternative column found
    }
    
    if (debug) {
        LOG_INFO("Exec", "transformMultiInstancePredicate: " << leftCol << "='" << leftLit  << "' AND " << rightCol << "='" << rightLit << "' -> using " << altCol  << " for second\n");
    }
    
    // Create transformed predicate: (leftCol = leftLit AND altCol = rightLit)
    auto newRightCompare = makeCompareWithColumn(bin.right, altCol);
    
    return TypedExpr::binary(BinaryOp::And, bin.left, newRightCompare);
}

} // namespace engine
