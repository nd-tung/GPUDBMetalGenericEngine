#pragma once
#include "IRGpuExecutorV2.hpp"
// TypedExprEval.hpp removed

#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <set>
#include <map>
#include <unordered_map>
#include <variant>

namespace MTL { class Buffer; }

namespace engine {

// Move EvalContext definition here so it can be shared across translation units
struct EvalContext {
    // Column data keyed by column name
    std::unordered_map<std::string, std::vector<uint32_t>> u32Cols;
    std::unordered_map<std::string, std::vector<float>> f32Cols;
    
    // GPU storage - Metal buffers
    std::unordered_map<std::string, MTL::Buffer*> u32ColsGPU;
    std::unordered_map<std::string, MTL::Buffer*> f32ColsGPU;
    
    // Raw string columns for pattern matching (LIKE, CONTAINS)
    std::unordered_map<std::string, std::vector<std::string>> stringCols;
    
    // Column aliases: maps alias -> canonical name
    // e.g., "supplier_no" -> "l_suppkey" for CTE aliasing
    std::unordered_map<std::string, std::string> columnAliases;
    
    // Active row indices (selection vector)
    std::vector<uint32_t> activeRows;
    
    // GPU selection vector
    MTL::Buffer* activeRowsGPU = nullptr;
    uint32_t activeRowsCountGPU = 0;
    
    // Join mapping (for join results)
    std::vector<uint32_t> leftIndices;
    std::vector<uint32_t> rightIndices;
    
    // Row count
    size_t rowCount = 0;

    // Flag to indicate if this context represents a scalar result (even if broadcasted)
    bool isScalarResult = false;
    
    // Which table is "current" for column lookups
    std::string currentTable;
};

struct ScanInstance {
    std::string baseTable;     // Original table name (e.g., "nation")
    std::string instanceKey;   // Instance-qualified key (e.g., "nation_1", "nation_2")
    int instanceNum;           // 1-based instance number
    size_t nodeIndex;          // Index in plan.nodes
};

// Function declarations for shared helpers (implemented in respective .cpp files)
std::map<size_t, ScanInstance> buildScanInstanceMap(const PlanV2& plan);
std::unordered_map<std::string, std::set<std::string>> collectNeededColumnsV2(const PlanV2& plan);
std::unordered_map<std::string, std::set<std::string>> collectPatternMatchColumnsV2(const PlanV2& plan);

// Helper for table loading (Scan logic)
struct IRGpuLoader {
    static void loadTables(
        const std::unordered_map<std::string, std::set<std::string>>& tableColsMap,
        const std::unordered_map<std::string, std::set<std::string>>& patternMatchCols,
        const std::map<size_t, ScanInstance>& scanInstanceMap,
        const std::string& datasetPath,
        std::unordered_map<std::string, EvalContext>& tableContexts,
        IRGpuExecutorV2::ExecutionResult& result,
        bool debug
    );
};

// Inline helpers

inline bool env_truthy(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return (s == "1" || s == "true" || s == "on" || s == "yes");
}

inline std::string trim_copy(std::string s) {
    auto first = s.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    auto last = s.find_last_not_of(" \t\n\r");
    return s.substr(first, last - first + 1);
}

inline std::string lower_compact(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }), s.end());
    return s;
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

inline std::string resolveColumnAlias(const std::string& col) {
    if (col == "supplier_no") return "l_suppkey";
    return col;
}

inline std::string tableForColumn(const std::string& col) {
    const std::string c = base_ident(col);
    if (c.rfind("l_", 0) == 0) return "lineitem";
    if (c.rfind("o_", 0) == 0) return "orders";
    if (c.rfind("c_", 0) == 0) return "customer";
    if (c.rfind("p_", 0) == 0) return "part";
    if (c.rfind("s_", 0) == 0) return "supplier";
    if (c.rfind("ps_", 0) == 0) return "partsupp";
    if (c.rfind("n_", 0) == 0) return "nation";
    if (c == "nation") return "nation";
    if (c.rfind("r_", 0) == 0) return "region";
    return "";
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

extern thread_local size_t g_aggregateCounter;

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
        std::cerr << "[V2] transformMultiInstancePredicate: pred kind=" << static_cast<int>(pred->kind) << "\n";
        std::cerr << "[V2] availableCols: ";
        for (const auto& c : availableCols) std::cerr << c << ", ";
        std::cerr << "\n";
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
            std::cerr << "[V2] transformMultiInstancePredicate: left side is not col=literal, recursing\n";
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
        std::cerr << "[V2] transformMultiInstancePredicate: " << leftCol << "='" << leftLit 
                  << "' AND " << rightCol << "='" << rightLit << "' -> using " << altCol 
                  << " for second\n";
    }
    
    // Create transformed predicate: (leftCol = leftLit AND altCol = rightLit)
    auto newRightCompare = makeCompareWithColumn(bin.right, altCol);
    
    return TypedExpr::binary(BinaryOp::And, bin.left, newRightCompare);
}

// Orchestrate the complex logic of setting up a join (finding tables, handling scalar subqueries, etc.)
// Returns true if join was successful or skipped legitimately, false on error
bool orchestrateJoin(
    const IRJoinV2& join,
    const std::string& datasetPath,
    EvalContext& currentCtx,
    std::unordered_map<std::string, EvalContext>& tableContexts,
    std::vector<EvalContext>& savedPipelines,
    std::vector<std::set<std::string>>& savedPipelineTables,
    std::set<std::string>& joinedTables,
    bool& hasPipeline,
    IRGpuExecutorV2::ExecutionResult& result
);

} // namespace engine
