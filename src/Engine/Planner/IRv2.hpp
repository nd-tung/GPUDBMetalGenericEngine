#pragma once
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <cstdint>
#include "TypedExpr.hpp"

namespace engine {

// ============================================================================
// IRv2: Expanded IR with full DuckDB operator metadata.
// This replaces the regex-based fallback with proper typed nodes.
// ============================================================================

// Join types
enum class JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
    Semi,
    Anti,
    Mark    // Used by DuckDB for NOT IN subquery decorrelation
};

inline const char* joinTypeName(JoinType t) {
    switch (t) {
        case JoinType::Inner: return "INNER";
        case JoinType::Left: return "LEFT";
        case JoinType::Right: return "RIGHT";
        case JoinType::Full: return "FULL";
        case JoinType::Cross: return "CROSS";
        case JoinType::Semi: return "SEMI";
        case JoinType::Anti: return "ANTI";
        case JoinType::Mark: return "MARK";
    }
    return "?";
}

// Order direction
struct OrderSpec {
    TypedExprPtr expr;
    bool ascending = true;
    bool nullsFirst = false;
};

// ============================================================================
// IR Node types with full metadata
// ============================================================================

struct IRScanV2 {
    std::string table;
    std::string alias;  // table alias if any
    
    // Columns to project from this scan (when known)
    std::vector<std::string> columns;
    
    // Pushed-down filter (from DuckDB optimization)
    TypedExprPtr filter;
    
    // Estimated row count (from DuckDB stats)
    std::optional<uint64_t> estimatedRows;

    // For constant value lists (e.g. from IN clause optimization)
    std::vector<int64_t> literalValues; // Using int64 to accommodate larger integers
};

struct IRFilterV2 {
    TypedExprPtr predicate;
    
    // Original string form for debugging
    std::string predicateStr;
};

struct IRProjectV2 {
    // Each projection is an expression (column ref, arithmetic, function call, etc.)
    std::vector<TypedExprPtr> exprs;
    
    // Output column names (may differ from expression names due to aliases)
    std::vector<std::string> outputNames;
};

struct IRJoinV2 {
    JoinType type = JoinType::Inner;
    
    // Join condition as typed expression (e.g., left.key = right.key)
    TypedExprPtr condition;
    
    // For equi-joins: left and right key expressions
    std::vector<TypedExprPtr> leftKeys;
    std::vector<TypedExprPtr> rightKeys;
    
    // Right table name (for simple cases)
    std::string rightTable;
    
    // Filter to apply to right table before joining
    TypedExprPtr rightFilter;

    // Original condition string for debugging
    std::string conditionStr;
};

struct IRGroupByV2 {
    // Grouping key expressions
    std::vector<TypedExprPtr> keys;
    std::vector<std::string> keyNames;  // output names for keys
    
    // Aggregate expressions with full metadata
    std::vector<TypedExprPtr> aggregates;
    
    // For simpler dispatch: parsed aggregate info
    struct AggSpec {
        AggFunc func;
        std::string inputExpr;      // expression string (e.g., "l_extendedprice * (1 - l_discount)")
        TypedExprPtr input;         // parsed expression
        std::string outputName;     // alias
        DataType resultType = DataType::Float64;
    };
    std::vector<AggSpec> aggSpecs;
};

struct IROrderByV2 {
    std::vector<OrderSpec> specs;
    
    // Simple column-name interface for backward compatibility
    std::vector<std::string> columns;
    std::vector<bool> ascending;
};

struct IRLimitV2 {
    int64_t count = -1;      // -1 means no limit
    int64_t offset = 0;
};

struct IRAggregateV2 {
    // Scalar (ungrouped) aggregate
    AggFunc func;
    TypedExprPtr expr;
    std::string alias;
    
    // Original expression string
    std::string exprStr;
    bool hasArithmeticExpr = false;
    
    // Index of this aggregate within a multi-aggregate operation (for UNGROUPED_AGGREGATE with multiple aggs)
    size_t aggIndex = 0;
    bool isLastAgg = true; // Whether this is the last aggregate in the block
};

struct IRDistinctV2 {
    std::vector<TypedExprPtr> exprs;
};

struct IRUnionV2 {
    bool all = false;  // UNION ALL vs UNION
};

struct IRSaveV2 {
    std::string name;
};

// ============================================================================
// IRNodeV2: variant-based IR node
// ============================================================================

struct IRNodeV2 {
    enum class Type {
        Scan,
        Filter,
        Project,
        Join,
        GroupBy,
        OrderBy,
        Limit,
        Aggregate,
        Distinct,
        Union,
        Save
    } type;

    // Node data (use std::variant for type safety)
    std::variant<
        IRScanV2,
        IRFilterV2,
        IRProjectV2,
        IRJoinV2,
        IRGroupByV2,
        IROrderByV2,
        IRLimitV2,
        IRAggregateV2,
        IRDistinctV2,
        IRUnionV2,
        IRSaveV2
    > data;

    // Metadata from DuckDB
    std::string duckdbName;           // Original operator name from DuckDB
    std::optional<uint64_t> estRows;  // Estimated output rows
    std::optional<double> estCost;    // Estimated cost
    
    // Child node indices (for tree representation, not used in linear pipeline)
    std::vector<size_t> children;

    // Constructors
    static IRNodeV2 scan(const std::string& table) {
        IRNodeV2 n;
        n.type = Type::Scan;
        n.data = IRScanV2{table, "", {}, nullptr, std::nullopt, {}};
        return n;
    }

    static IRNodeV2 filter(TypedExprPtr pred, const std::string& predStr = "") {
        IRNodeV2 n;
        n.type = Type::Filter;
        n.data = IRFilterV2{pred, predStr};
        return n;
    }

    static IRNodeV2 project(std::vector<TypedExprPtr> exprs, std::vector<std::string> names = {}) {
        IRNodeV2 n;
        n.type = Type::Project;
        n.data = IRProjectV2{std::move(exprs), std::move(names)};
        return n;
    }

    static IRNodeV2 join(JoinType jt, TypedExprPtr cond, const std::string& condStr = "", const std::string& rTable = "", TypedExprPtr rFilter = nullptr) {
        IRNodeV2 n;
        n.type = Type::Join;
        n.data = IRJoinV2{jt, cond, {}, {}, rTable, rFilter, condStr};
        return n;
    }

    static IRNodeV2 groupBy() {
        IRNodeV2 n;
        n.type = Type::GroupBy;
        n.data = IRGroupByV2{};
        return n;
    }

    static IRNodeV2 orderBy() {
        IRNodeV2 n;
        n.type = Type::OrderBy;
        n.data = IROrderByV2{};
        return n;
    }

    static IRNodeV2 limit(int64_t count, int64_t offset = 0) {
        IRNodeV2 n;
        n.type = Type::Limit;
        n.data = IRLimitV2{count, offset};
        return n;
    }

    static IRNodeV2 aggregate(AggFunc func, TypedExprPtr expr, const std::string& alias = "") {
        IRNodeV2 n;
        n.type = Type::Aggregate;
        n.data = IRAggregateV2{func, expr, alias, "", false};
        return n;
    }

    // Accessors
    IRScanV2& asScan() { return std::get<IRScanV2>(data); }
    const IRScanV2& asScan() const { return std::get<IRScanV2>(data); }
    
    IRFilterV2& asFilter() { return std::get<IRFilterV2>(data); }
    const IRFilterV2& asFilter() const { return std::get<IRFilterV2>(data); }
    
    IRProjectV2& asProject() { return std::get<IRProjectV2>(data); }
    const IRProjectV2& asProject() const { return std::get<IRProjectV2>(data); }
    
    IRJoinV2& asJoin() { return std::get<IRJoinV2>(data); }
    const IRJoinV2& asJoin() const { return std::get<IRJoinV2>(data); }
    
    IRGroupByV2& asGroupBy() { return std::get<IRGroupByV2>(data); }
    const IRGroupByV2& asGroupBy() const { return std::get<IRGroupByV2>(data); }
    
    IROrderByV2& asOrderBy() { return std::get<IROrderByV2>(data); }
    const IROrderByV2& asOrderBy() const { return std::get<IROrderByV2>(data); }
    
    IRLimitV2& asLimit() { return std::get<IRLimitV2>(data); }
    const IRLimitV2& asLimit() const { return std::get<IRLimitV2>(data); }
    
    IRAggregateV2& asAggregate() { return std::get<IRAggregateV2>(data); }
    const IRAggregateV2& asAggregate() const { return std::get<IRAggregateV2>(data); }

    static IRNodeV2 save(const std::string& name) {
        IRNodeV2 n;
        n.type = Type::Save;
        n.data = IRSaveV2{name};
        return n;
    }

    IRSaveV2& asSave() { return std::get<IRSaveV2>(data); }
    const IRSaveV2& asSave() const { return std::get<IRSaveV2>(data); }
};

// ============================================================================
// PlanV2: expanded plan structure
// ============================================================================

struct PlanV2 {
    std::vector<IRNodeV2> nodes;  // Linear pipeline order
    
    // Metadata
    std::string originalSQL;
    bool parsedFromJSON = false;
    std::string parseError;
    
    // Schema information inferred from query
    struct TableInfo {
        std::string name;
        std::vector<std::string> neededColumns;
    };
    std::vector<TableInfo> tables;
    
    // Output schema
    std::vector<std::string> outputColumns;
    std::vector<DataType> outputTypes;
    
    // Check if plan is valid
    bool isValid() const { return !nodes.empty() && parseError.empty(); }
    
    // Find first node of a type
    const IRNodeV2* findFirst(IRNodeV2::Type t) const {
        for (const auto& n : nodes) if (n.type == t) return &n;
        return nullptr;
    }
    
    // Count nodes of a type
    size_t countType(IRNodeV2::Type t) const {
        size_t c = 0;
        for (const auto& n : nodes) if (n.type == t) ++c;
        return c;
    }
};

// ============================================================================
// IR conversion utilities
// ============================================================================

// Convert legacy IR nodes to V2 format
inline IRNodeV2 upgradeNode(const struct IRNode& legacy);  // Forward declaration

} // namespace engine
