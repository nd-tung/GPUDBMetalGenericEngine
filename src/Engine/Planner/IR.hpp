#pragma once
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include "TypedExpr.hpp"

namespace engine {

// --- IRv2: Expanded IR with full DuckDB operator metadata ---

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
};

// --- IR Node types ---

struct IRScan {
    std::string table;
    std::string alias;  // table alias if any
    
    // Columns to project from this scan (when known)
    std::vector<std::string> columns;
    
    // Pushed-down filter (from DuckDB optimization)
    TypedExprPtr filter;

    // True for DELIM_SCAN nodes (need deduplication on key columns).
    // False for COLUMN_DATA_SCAN nodes (need full data from saved pipeline).
    bool isDelimScan = false;
};

struct IRFilter {
    TypedExprPtr predicate;
    
    // String form for debugging
    std::string predicateStr;
};

struct IRProject {
    // Each projection is an expression (column ref, arithmetic, function call, etc.)
    std::vector<TypedExprPtr> exprs;
    
    // Output column names (may differ from expression names due to aliases)
    std::vector<std::string> outputNames;
};

struct IRJoin {
    JoinType type = JoinType::Inner;
    
    // If true, this is a RIGHT variant (RIGHT_SEMI, RIGHT_ANTI)
    // For RIGHT_ANTI: keep RIGHT rows not matching LEFT
    // For RIGHT_SEMI: keep RIGHT rows matching LEFT
    bool rightVariant = false;
    
    // Join condition as typed expression (e.g., left.key = right.key)
    TypedExprPtr condition;
    
    // For equi-joins: left and right key expressions
    std::vector<TypedExprPtr> leftKeys;
    std::vector<TypedExprPtr> rightKeys;
    
    // Right table name (for simple cases)
    std::string rightTable;
    
    // Filter to apply to right table before joining
    TypedExprPtr rightFilter;

    // Condition string for debugging
    std::string conditionStr;
};

struct IRGroupBy {
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

struct IROrderBy {
    std::vector<OrderSpec> specs;
    
    // Simple column-name interface for backward compatibility
    std::vector<std::string> columns;
    std::vector<bool> ascending;
};

struct IRLimit {
    int64_t count = -1;      // -1 means no limit
    int64_t offset = 0;
};

struct IRAggregate {
    // Scalar (ungrouped) aggregate
    AggFunc func;
    TypedExprPtr expr;
    std::string alias;
    
    // Expression string
    std::string exprStr;
    
    // Index of this aggregate within a multi-aggregate operation (for UNGROUPED_AGGREGATE with multiple aggs)
    size_t aggIndex = 0;
    bool isLastAgg = true; // Whether this is the last aggregate in the block
};

struct IRDistinct {
    std::vector<TypedExprPtr> exprs;
};

struct IRSave {
    std::string name;
};

// --- IRNode: variant-based IR node ---

struct IRNode {
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
        Save
    } type;

    // Node data (use std::variant for type safety)
    std::variant<
        IRScan,
        IRFilter,
        IRProject,
        IRJoin,
        IRGroupBy,
        IROrderBy,
        IRLimit,
        IRAggregate,
        IRDistinct,
        IRSave
    > data;

    // Metadata from DuckDB
    std::string duckdbName;           // Original operator name from DuckDB

    // Constructors
    static IRNode scan(const std::string& table) {
        IRNode n;
        n.type = Type::Scan;
        n.data = IRScan{table, "", {}, nullptr};
        return n;
    }

    static IRNode filter(TypedExprPtr pred, const std::string& predStr = "") {
        IRNode n;
        n.type = Type::Filter;
        n.data = IRFilter{pred, predStr};
        return n;
    }

    static IRNode project(std::vector<TypedExprPtr> exprs, std::vector<std::string> names = {}) {
        IRNode n;
        n.type = Type::Project;
        n.data = IRProject{std::move(exprs), std::move(names)};
        return n;
    }

    static IRNode join(JoinType jt, TypedExprPtr cond, const std::string& condStr = "", const std::string& rTable = "", TypedExprPtr rFilter = nullptr) {
        IRNode n;
        n.type = Type::Join;
        n.data = IRJoin{jt, false, cond, {}, {}, rTable, rFilter, condStr};
        return n;
    }

    static IRNode groupBy() {
        IRNode n;
        n.type = Type::GroupBy;
        n.data = IRGroupBy{};
        return n;
    }

    static IRNode orderBy() {
        IRNode n;
        n.type = Type::OrderBy;
        n.data = IROrderBy{};
        return n;
    }

    static IRNode limit(int64_t count, int64_t offset = 0) {
        IRNode n;
        n.type = Type::Limit;
        n.data = IRLimit{count, offset};
        return n;
    }

    static IRNode aggregate(AggFunc func, TypedExprPtr expr, const std::string& alias = "") {
        IRNode n;
        n.type = Type::Aggregate;
        n.data = IRAggregate{func, expr, alias, ""};
        return n;
    }

    // Accessors
    IRScan& asScan() { return std::get<IRScan>(data); }
    const IRScan& asScan() const { return std::get<IRScan>(data); }
    
    IRFilter& asFilter() { return std::get<IRFilter>(data); }
    const IRFilter& asFilter() const { return std::get<IRFilter>(data); }
    
    IRProject& asProject() { return std::get<IRProject>(data); }
    const IRProject& asProject() const { return std::get<IRProject>(data); }
    
    IRJoin& asJoin() { return std::get<IRJoin>(data); }
    const IRJoin& asJoin() const { return std::get<IRJoin>(data); }
    
    IRGroupBy& asGroupBy() { return std::get<IRGroupBy>(data); }
    const IRGroupBy& asGroupBy() const { return std::get<IRGroupBy>(data); }
    
    IROrderBy& asOrderBy() { return std::get<IROrderBy>(data); }
    const IROrderBy& asOrderBy() const { return std::get<IROrderBy>(data); }
    
    IRLimit& asLimit() { return std::get<IRLimit>(data); }
    const IRLimit& asLimit() const { return std::get<IRLimit>(data); }
    
    IRAggregate& asAggregate() { return std::get<IRAggregate>(data); }
    const IRAggregate& asAggregate() const { return std::get<IRAggregate>(data); }

    static IRNode save(const std::string& name) {
        IRNode n;
        n.type = Type::Save;
        n.data = IRSave{name};
        return n;
    }

    IRSave& asSave() { return std::get<IRSave>(data); }
    const IRSave& asSave() const { return std::get<IRSave>(data); }
};

// --- Plan structure ---

struct Plan {
    std::vector<IRNode> nodes;  // Linear pipeline order
    
    // Metadata
    std::string parseError;
    
    // Schema information inferred from query
    struct TableInfo {
        std::string name;
        std::vector<std::string> neededColumns;
    };
    std::vector<TableInfo> tables;
    
    // Output schema
    std::vector<std::string> outputColumns;
    
    // Check if plan is valid
    bool isValid() const { return !nodes.empty() && parseError.empty(); }
};

} // namespace engine
