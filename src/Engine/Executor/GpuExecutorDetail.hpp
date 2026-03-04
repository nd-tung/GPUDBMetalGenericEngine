#pragma once
#include "GpuExecutor.hpp"
#include "Operators.hpp"
#include "EnvUtil.hpp"

#include <Metal/Metal.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <iostream>

// ============================================================================
// Configuration constants (previously magic numbers scattered across the codebase)
// ============================================================================
namespace engine::config {
    // Maximum number of suffix variations to try when resolving column names
    // (e.g., col_1, col_2, ... col_N for multi-instance table columns)
    constexpr int kMaxColumnSuffixSearch = 9;

    // Threshold row count below which a date value is treated as days-since-epoch
    // rather than YYYYMMDD format
    constexpr uint32_t kDateFormatThreshold = 100000;

    // Sample size for detecting whether a column has varying values
    constexpr size_t kColumnSampleSize = 100;

    // Thread spawn threshold for parallel CPU string gather in joins
    constexpr uint32_t kParallelStringGatherThreshold = 10000;

    // Maximum keys and aggregates per group in GPU hash table layout
    constexpr uint32_t kMaxGroupByKeys = 8;
    constexpr uint32_t kMaxGroupByAggs = 16;

    // GPU block sort threshold — elements <= this use shared-memory bitonic sort
    constexpr uint32_t kBlockSortThreshold = 1024;
} // namespace engine::config
#include <set>
#include <map>
#include <unordered_map>

namespace engine {

// Pre-flattened Arrow-style GPU string column buffers.
// Standalone struct so it can be forward-declared in public headers.
struct FlatStringCol {
    MTL::Buffer* chars   = nullptr;  // raw character bytes
    MTL::Buffer* offsets = nullptr;  // uint32_t[rowCount] start offset per string
    MTL::Buffer* lengths = nullptr;  // uint32_t[rowCount] length per string
    uint32_t rowCount   = 0;
    uint32_t totalBytes = 0;

    void release() {
        if (chars)   { chars->release();   chars   = nullptr; }
        if (offsets) { offsets->release();  offsets = nullptr; }
        if (lengths) { lengths->release();  lengths = nullptr; }
        rowCount = 0;
        totalBytes = 0;
    }
};

// Dictionary-encoded string column: sorted unique strings with per-row IDs.
// Provides collision-free integer encoding (unlike FNV1a hashes) and O(1) reverse mapping.
// GPU-native: dictionary IDs are the primary representation on GPU; strings are only
// materialized at output time. Dict IDs compact/gather like any u32 column.
struct DictEncoded {
    std::vector<std::string> dictionary;  // sorted unique strings
    std::vector<uint32_t> ids;            // per-row dictionary ID (CPU mirror, may be lazy)
    MTL::Buffer* idsGPU = nullptr;        // per-row dictionary ID (GPU) — primary representation
    uint32_t rowCount = 0;

    // Lookup: given a string value, return its dictionary ID (or UINT32_MAX if not found)
    uint32_t lookupId(const std::string& value) const {
        // Binary search since dictionary is sorted
        auto it = std::lower_bound(dictionary.begin(), dictionary.end(), value);
        if (it != dictionary.end() && *it == value)
            return static_cast<uint32_t>(it - dictionary.begin());
        return UINT32_MAX;
    }

    // Reverse lookup: given a dictionary ID, return the string (or "" if out of range)
    const std::string& lookupString(uint32_t id) const {
        static const std::string empty;
        return (id < dictionary.size()) ? dictionary[id] : empty;
    }

    // Sync CPU mirror from GPU buffer (lazy — call when CPU ids needed)
    void ensureIdsCPU() {
        if (idsGPU && ids.size() != rowCount) {
            ids.resize(rowCount);
            if (rowCount > 0) {
                std::memcpy(ids.data(), idsGPU->contents(), rowCount * sizeof(uint32_t));
            }
        }
    }

    // Materialize full string column from dict IDs (for output or legacy consumers)
    std::vector<std::string> materialize() const {
        std::vector<std::string> result(rowCount);
        const uint32_t* idPtr = nullptr;
        if (idsGPU) idPtr = static_cast<const uint32_t*>(idsGPU->contents());
        else if (!ids.empty()) idPtr = ids.data();
        if (!idPtr) return result;
        for (uint32_t i = 0; i < rowCount; ++i) {
            uint32_t id = idPtr[i];
            if (id < dictionary.size()) result[i] = dictionary[id];
        }
        return result;
    }

    // Check if this dict encoding is valid and has data
    bool valid() const { return !dictionary.empty() && (idsGPU || !ids.empty()) && rowCount > 0; }

    // Release GPU resources
    void release() {
        if (idsGPU) { idsGPU->release(); idsGPU = nullptr; }
        ids.clear();
        dictionary.clear();
        rowCount = 0;
    }
};

// Move EvalContext definition here so it can be shared across translation units
struct EvalContext {
    // Column data keyed by column name
    std::unordered_map<std::string, std::vector<uint32_t>> u32Cols;
    std::unordered_map<std::string, std::vector<float>> f32Cols;
    
    // GPU storage - Metal buffers
    std::unordered_map<std::string, MTL::Buffer*> u32ColsGPU;
    std::unordered_map<std::string, MTL::Buffer*> f32ColsGPU;
    
    // Raw string columns for pattern matching (LIKE, CONTAINS)
    // NOTE: With GPU-native dictionary encoding, stringCols is now a LAZY CACHE.
    // Primary string data lives in dictCols. stringCols is populated on-demand
    // when pattern matching needs raw strings (LIKE, CONTAINS) or at final output.
    std::unordered_map<std::string, std::vector<std::string>> stringCols;

    // Pre-flattened string columns (Arrow-style GPU buffers, created at load time)
    // Uses standalone FlatStringCol struct above.
    // NOTE: Built lazily from dictCols when GPU string pattern matching is needed.
    std::unordered_map<std::string, FlatStringCol> flatStringCols;
    
    // Dictionary-encoded string columns — PRIMARY string representation.
    // Dict IDs are GPU-resident u32 values that propagate through the pipeline
    // like normal u32 columns (compact, gather, join, groupby all work on IDs).
    // Strings are only materialized from dict at output time.
    std::unordered_map<std::string, DictEncoded> dictCols;
    
    // Column aliases: maps alias -> canonical name
    // e.g., "supplier_no" -> "l_suppkey" for CTE aliasing
    std::unordered_map<std::string, std::string> columnAliases;
    
    // Active row indices (selection vector)
    std::vector<uint32_t> activeRows;
    
    // GPU selection vector
    MTL::Buffer* activeRowsGPU = nullptr;
    uint32_t activeRowsCountGPU = 0;
    
    // Row count
    size_t rowCount = 0;

    // Lazy sync: download activeRowsGPU to CPU activeRows on demand.
    // Call this before any code path that reads activeRows (the CPU vector).
    void ensureActiveRowsCPU() {
        if (activeRowsGPU && activeRows.size() != activeRowsCountGPU) {
            activeRows.resize(activeRowsCountGPU);
            if (activeRowsCountGPU > 0) {
                std::memcpy(activeRows.data(), activeRowsGPU->contents(),
                            activeRowsCountGPU * sizeof(uint32_t));
            }
        }
    }

    // Flag to indicate if this context represents a scalar result (even if broadcasted)
    bool isScalarResult = false;

    // Ensure stringCols[colName] is populated from dictCols (lazy materialization).
    // Call before any code path that needs raw string data (LIKE, CONTAINS).
    void ensureStringCol(const std::string& colName) {
        if (stringCols.count(colName) && !stringCols[colName].empty()) return;
        auto dit = dictCols.find(colName);
        if (dit != dictCols.end() && dit->second.valid()) {
            stringCols[colName] = dit->second.materialize();
        }
    }

    // Ensure flatStringCols[colName] is built from stringCols (lazy).
    // Needs forward-declared flattenStringCol — implemented externally.
    // Callers should check flatStringCols.count(colName) first if possible.

    // Check if a column has dictionary encoding available
    bool hasDictCol(const std::string& colName) const {
        auto it = dictCols.find(colName);
        return it != dictCols.end() && it->second.valid();
    }

    // Compact dictCols using activeRowsGPU (GPU gather of dict IDs)
    void compactDictCols(uint32_t compactCount) {
        for (auto& [name, dict] : dictCols) {
            if (dict.idsGPU) {
                uint32_t bufRows = (uint32_t)(dict.idsGPU->length() / sizeof(uint32_t));
                if (bufRows > compactCount) {
                    MTL::Buffer* compacted = GpuOps::gatherU32(dict.idsGPU, activeRowsGPU, compactCount, true);
                    if (compacted) {
                        // NOTE: do NOT release old idsGPU — may be shared with tableContexts
                        dict.idsGPU = compacted;
                        dict.rowCount = compactCount;
                        dict.ids.clear();  // Invalidate CPU mirror (lazy sync)
                    }
                }
            }
        }
    }

    // Compact dictCols using an explicit index buffer (GPU gather of dict IDs)
    void compactDictCols(MTL::Buffer* indexBuf, uint32_t newCount) {
        for (auto& [name, dict] : dictCols) {
            if (dict.idsGPU) {
                MTL::Buffer* gathered = GpuOps::gatherU32(dict.idsGPU, indexBuf, newCount, false);
                if (gathered) {
                    dict.idsGPU->release();
                    dict.idsGPU = gathered;
                    dict.rowCount = newCount;
                    dict.ids.clear();
                }
            }
        }
    }

    // Compact flatStringCols using activeRowsGPU (GPU gather of chars/offsets/lengths)
    void compactFlatStringCols(uint32_t compactCount) {
        for (auto& [name, flat] : flatStringCols) {
            if (flat.chars && flat.offsets && flat.lengths && flat.rowCount > compactCount) {
                auto r = GpuOps::gatherFlatString(
                    flat.chars, flat.offsets, flat.lengths,
                    activeRowsGPU, compactCount, true);
                if (r.chars) {
                    // NOTE: do NOT release old buffers — may be shared with tableContexts
                    flat.chars = r.chars;
                    flat.offsets = r.offsets;
                    flat.lengths = r.lengths;
                    flat.rowCount = r.rowCount;
                    flat.totalBytes = r.totalBytes;
                }
            }
        }
    }

    // Compact flatStringCols using an explicit index buffer
    void compactFlatStringCols(MTL::Buffer* indexBuf, uint32_t newCount) {
        for (auto& [name, flat] : flatStringCols) {
            if (flat.chars && flat.offsets && flat.lengths) {
                auto r = GpuOps::gatherFlatString(
                    flat.chars, flat.offsets, flat.lengths,
                    indexBuf, newCount, true);
                if (r.chars) {
                    // NOTE: do NOT release old buffers — may be shared with tableContexts
                    flat.chars = r.chars;
                    flat.offsets = r.offsets;
                    flat.lengths = r.lengths;
                    flat.rowCount = r.rowCount;
                    flat.totalBytes = r.totalBytes;
                }
            }
        }
    }

    // Ensure flatStringCols[colName] is built from dictCols or stringCols (lazy).
    // Implementation uses flattenStringCol() free function (declared below struct).
    void ensureFlatStringCol(const std::string& colName);

    // Safely erase a flat string column, releasing its GPU buffers first
    void eraseFlatStringCol(const std::string& colName) {
        auto it = flatStringCols.find(colName);
        if (it != flatStringCols.end()) {
            it->second.release();
            flatStringCols.erase(it);
        }
    }

    // Gather all GPU-side columns (u32, f32, dict, flat string) by index array.
    // Releases old GPU buffers and replaces with gathered versions.
    void gatherAllGPU(MTL::Buffer* indices, uint32_t count) {
        for (auto& [name, buf] : u32ColsGPU) {
            if (!buf) continue;
            MTL::Buffer* gathered = GpuOps::gatherU32(buf, indices, count);
            buf->release();
            buf = gathered;
        }
        for (auto& [name, buf] : f32ColsGPU) {
            if (!buf) continue;
            MTL::Buffer* gathered = GpuOps::gatherF32(buf, indices, count);
            buf->release();
            buf = gathered;
        }
        compactDictCols(indices, count);
        compactFlatStringCols(indices, count);
    }

    // Invalidate stringCols entries that have dict or flat-string equivalents.
    void invalidateStringColsForDictFlat() {
        for (const auto& [name, dc] : dictCols)
            stringCols.erase(name);
        for (const auto& [name, fc] : flatStringCols)
            stringCols.erase(name);
    }

    // Reset active rows tracking (releases activeRowsGPU if set).
    void clearActiveRows() {
        activeRows.clear();
        if (activeRowsGPU) {
            activeRowsGPU->release();
            activeRowsGPU = nullptr;
        }
        activeRowsCountGPU = 0;
    }

    // Which table is "current" for column lookups
    std::string currentTable;
    
    // Columns originating from DELIM_SCAN (correlation keys)
    // These should be prioritized during join column name collision resolution
    std::unordered_set<std::string> isDelimCorrelation;

    // Sequential counter for positional aggregate output columns (#0, #1, ...)
    size_t aggregateCounter = 0;
};

struct ScanInstance {
    std::string baseTable;     // Original table name (e.g., "nation")
    std::string instanceKey;   // Instance-qualified key (e.g., "nation_1", "nation_2")
    int instanceNum;           // 1-based instance number
    size_t nodeIndex;          // Index in plan.nodes
};

// Function declarations for shared helpers (implemented in respective .cpp files)
std::map<size_t, ScanInstance> buildScanInstanceMap(const Plan& plan);
std::unordered_map<std::string, std::set<std::string>> collectNeededColumns(const Plan& plan);
// GPU dedup helper: deduplicate an EvalContext by u32 key columns (GpuExecutor.cpp)
uint32_t deduplicateContext(EvalContext& ctx, const std::vector<std::string>& dedupCols, bool debug);
// Flatten/dict helpers (Scan.cpp) — callable from Join.cpp and other modules
void flattenStringCol(EvalContext& ctx, const std::string& colName);
void buildDictCol(EvalContext& ctx, const std::string& colName);

// Helper for table loading (Scan logic)
struct IRGpuLoader {
    static void loadTables(
        const std::unordered_map<std::string, std::set<std::string>>& tableColsMap,
        const std::map<size_t, ScanInstance>& scanInstanceMap,
        const std::string& datasetPath,
        std::unordered_map<std::string, EvalContext>& tableContexts,
        GpuExecutor::ExecutionResult& result,
        bool debug
    );
};

// Inline helpers  (env_truthy is in EnvUtil.hpp)

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
    const std::string c = base_ident(col);
    if (c.rfind("ps_", 0) == 0) return "partsupp"; // Must be before p_ and s_
    if (c.rfind("l_", 0) == 0) return "lineitem";
    if (c.rfind("o_", 0) == 0) return "orders";
    if (c.rfind("c_", 0) == 0) return "customer";
    if (c.rfind("p_", 0) == 0) return "part";
    if (c.rfind("s_", 0) == 0) return "supplier";
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
        std::cerr << "[Exec] transformMultiInstancePredicate: pred kind=" << static_cast<int>(pred->kind) << "\n";
        std::cerr << "[Exec] availableCols: ";
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
            std::cerr << "[Exec] transformMultiInstancePredicate: left side is not col=literal, recursing\n";
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
        std::cerr << "[Exec] transformMultiInstancePredicate: " << leftCol << "='" << leftLit 
                  << "' AND " << rightCol << "='" << rightLit << "' -> using " << altCol 
                  << " for second\n";
    }
    
    // Create transformed predicate: (leftCol = leftLit AND altCol = rightLit)
    auto newRightCompare = makeCompareWithColumn(bin.right, altCol);
    
    return TypedExpr::binary(BinaryOp::And, bin.left, newRightCompare);
}

} // namespace engine
