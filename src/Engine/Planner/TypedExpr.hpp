#pragma once
#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <optional>
#include <cstdint>

namespace engine {

// ============================================================================
// TypedExpr: A fully-typed expression tree for generic SQL expression handling.
// Supports: columns, literals, binary/unary ops, function calls, aggregates.
// ============================================================================

enum class DataType {
    Unknown,
    Int32,
    Int64,
    Float32,
    Float64,
    Date,       // stored as YYYYMMDD integer
    String,     // stored as hash or raw
    Bool
};

inline const char* dataTypeName(DataType t) {
    switch (t) {
        case DataType::Unknown: return "unknown";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        case DataType::Float32: return "float32";
        case DataType::Float64: return "float64";
        case DataType::Date: return "date";
        case DataType::String: return "string";
        case DataType::Bool: return "bool";
    }
    return "?";
}

// Forward declarations
struct TypedExpr;
using TypedExprPtr = std::shared_ptr<TypedExpr>;

// Literal value variant
using LiteralValue = std::variant<
    std::monostate,    // null
    int64_t,           // integers
    double,            // floats
    std::string        // strings/dates
>;

// Comparison operators
enum class CompareOp {
    Eq,     // =
    Ne,     // <> or !=
    Lt,     // <
    Le,     // <=
    Gt,     // >
    Ge,     // >=
    Like,   // LIKE
    In,     // IN (list)
    IsNull, // IS NULL
    IsNotNull // IS NOT NULL
};

// Binary arithmetic/logical operators
enum class BinaryOp {
    Add,    // +
    Sub,    // -
    Mul,    // *
    Div,    // /
    Mod,    // %
    And,    // AND
    Or,     // OR
};

// Unary operators
enum class UnaryOp {
    Neg,    // -x
    Not,    // NOT x
};

// Aggregate function types
enum class AggFunc {
    Sum,
    Count,
    CountStar,
    CountDistinct,  // COUNT(DISTINCT ...)
    Avg,
    Min,
    Max,
    First,
    Last
};

inline const char* aggFuncName(AggFunc f) {
    switch (f) {
        case AggFunc::Sum: return "SUM";
        case AggFunc::Count: return "COUNT";
        case AggFunc::CountStar: return "COUNT(*)";
        case AggFunc::CountDistinct: return "COUNT_DISTINCT";
        case AggFunc::Avg: return "AVG";
        case AggFunc::Min: return "MIN";
        case AggFunc::Max: return "MAX";
        case AggFunc::First: return "FIRST";
        case AggFunc::Last: return "LAST";
    }
    return "?";
}

// ============================================================================
// TypedExpr node types
// ============================================================================

struct ColumnRef {
    std::string table;      // optional table qualifier
    std::string column;     // column name
    DataType inferredType = DataType::Unknown;
};

struct Literal {
    LiteralValue value;
    DataType type = DataType::Unknown;
};

struct BinaryExpr {
    BinaryOp op;
    TypedExprPtr left;
    TypedExprPtr right;
};

struct UnaryExpr {
    UnaryOp op;
    TypedExprPtr operand;
};

struct CompareExpr {
    CompareOp op;
    TypedExprPtr left;
    TypedExprPtr right;  // null for IS NULL, or list for IN
    std::vector<TypedExprPtr> inList;  // for IN operator
};

struct FunctionCall {
    std::string name;   // function name (upper case)
    std::vector<TypedExprPtr> args;
    DataType returnType = DataType::Unknown;
};

struct AggregateExpr {
    AggFunc func;
    TypedExprPtr arg;   // null for COUNT(*)
    bool distinct = false;
    std::string alias;  // output column name
};

struct CaseExpr {
    struct WhenThen {
        TypedExprPtr when;
        TypedExprPtr then;
    };
    std::vector<WhenThen> cases;
    TypedExprPtr elseExpr;  // optional
};

struct CastExpr {
    TypedExprPtr expr;
    DataType targetType;
};

// Alias wrapper
struct AliasExpr {
    TypedExprPtr expr;
    std::string alias;
};

// ============================================================================
// TypedExpr: variant wrapper for all expression types
// ============================================================================

struct TypedExpr {
    enum class Kind {
        Column,
        Literal,
        Binary,
        Unary,
        Compare,
        Function,
        Aggregate,
        Case,
        Cast,
        Alias
    } kind;

    // Data stored in variant
    std::variant<
        ColumnRef,
        Literal,
        BinaryExpr,
        UnaryExpr,
        CompareExpr,
        FunctionCall,
        AggregateExpr,
        CaseExpr,
        CastExpr,
        AliasExpr
    > data;

    // Inferred result type
    DataType resultType = DataType::Unknown;

    // Constructors
    static TypedExprPtr column(const std::string& name, const std::string& table = "") {
        auto e = std::make_shared<TypedExpr>();
        e->kind = Kind::Column;
        e->data = ColumnRef{table, name, DataType::Unknown};
        return e;
    }

    static TypedExprPtr literal(int64_t v) {
        auto e = std::make_shared<TypedExpr>();
        e->kind = Kind::Literal;
        e->data = Literal{v, DataType::Int64};
        e->resultType = DataType::Int64;
        return e;
    }

    static TypedExprPtr literal(double v) {
        auto e = std::make_shared<TypedExpr>();
        e->kind = Kind::Literal;
        e->data = Literal{v, DataType::Float64};
        e->resultType = DataType::Float64;
        return e;
    }

    static TypedExprPtr literal(const std::string& v, DataType type = DataType::String) {
        auto e = std::make_shared<TypedExpr>();
        e->kind = Kind::Literal;
        e->data = Literal{v, type};
        e->resultType = type;
        return e;
    }

    static TypedExprPtr binary(BinaryOp op, TypedExprPtr left, TypedExprPtr right) {
        auto e = std::make_shared<TypedExpr>();
        e->kind = Kind::Binary;
        e->data = BinaryExpr{op, left, right};
        return e;
    }

    static TypedExprPtr unary(UnaryOp op, TypedExprPtr operand) {
        auto e = std::make_shared<TypedExpr>();
        e->kind = Kind::Unary;
        e->data = UnaryExpr{op, operand};
        return e;
    }

    static TypedExprPtr compare(CompareOp op, TypedExprPtr left, TypedExprPtr right = nullptr) {
        auto e = std::make_shared<TypedExpr>();
        e->kind = Kind::Compare;
        e->data = CompareExpr{op, left, right, {}};
        e->resultType = DataType::Bool;
        return e;
    }

    static TypedExprPtr inList(TypedExprPtr left, std::vector<TypedExprPtr> list) {
        auto e = std::make_shared<TypedExpr>();
        e->kind = Kind::Compare;
        e->data = CompareExpr{CompareOp::In, left, nullptr, std::move(list)};
        e->resultType = DataType::Bool;
        return e;
    }

    static TypedExprPtr aggregate(AggFunc func, TypedExprPtr arg = nullptr, const std::string& alias = "") {
        auto e = std::make_shared<TypedExpr>();
        e->kind = Kind::Aggregate;
        e->data = AggregateExpr{func, arg, false, alias};
        return e;
    }

    static TypedExprPtr alias(TypedExprPtr expr, const std::string& name) {
        auto e = std::make_shared<TypedExpr>();
        e->kind = Kind::Alias;
        e->data = AliasExpr{expr, name};
        e->resultType = expr ? expr->resultType : DataType::Unknown;
        return e;
    }

    // Accessors
    const ColumnRef& asColumn() const { return std::get<ColumnRef>(data); }
    const Literal& asLiteral() const { return std::get<Literal>(data); }
    const BinaryExpr& asBinary() const { return std::get<BinaryExpr>(data); }
    const UnaryExpr& asUnary() const { return std::get<UnaryExpr>(data); }
    const CompareExpr& asCompare() const { return std::get<CompareExpr>(data); }
    const FunctionCall& asFunction() const { return std::get<FunctionCall>(data); }
    const AggregateExpr& asAggregate() const { return std::get<AggregateExpr>(data); }
    const CaseExpr& asCase() const { return std::get<CaseExpr>(data); }
    const CastExpr& asCast() const { return std::get<CastExpr>(data); }
    const AliasExpr& asAlias() const { return std::get<AliasExpr>(data); }

    ColumnRef& asColumn() { return std::get<ColumnRef>(data); }
    AggregateExpr& asAggregate() { return std::get<AggregateExpr>(data); }
};

// ============================================================================
// Expression utilities
// ============================================================================

// Extract all column references from an expression tree
inline void collectColumns(const TypedExprPtr& expr, std::vector<ColumnRef>& out) {
    if (!expr) return;
    switch (expr->kind) {
        case TypedExpr::Kind::Column:
            out.push_back(expr->asColumn());
            break;
        case TypedExpr::Kind::Binary: {
            const auto& b = expr->asBinary();
            collectColumns(b.left, out);
            collectColumns(b.right, out);
            break;
        }
        case TypedExpr::Kind::Unary:
            collectColumns(expr->asUnary().operand, out);
            break;
        case TypedExpr::Kind::Compare: {
            const auto& c = expr->asCompare();
            collectColumns(c.left, out);
            collectColumns(c.right, out);
            for (const auto& e : c.inList) collectColumns(e, out);
            break;
        }
        case TypedExpr::Kind::Function:
            for (const auto& a : expr->asFunction().args) collectColumns(a, out);
            break;
        case TypedExpr::Kind::Aggregate:
            collectColumns(expr->asAggregate().arg, out);
            break;
        case TypedExpr::Kind::Case: {
            const auto& cs = expr->asCase();
            for (const auto& wt : cs.cases) {
                collectColumns(wt.when, out);
                collectColumns(wt.then, out);
            }
            collectColumns(cs.elseExpr, out);
            break;
        }
        case TypedExpr::Kind::Cast:
            collectColumns(expr->asCast().expr, out);
            break;
        case TypedExpr::Kind::Alias:
            collectColumns(expr->asAlias().expr, out);
            break;
        default:
            break;
    }
}

// Get all unique column names from an expression
inline std::vector<std::string> getColumnNames(const TypedExprPtr& expr) {
    std::vector<ColumnRef> refs;
    collectColumns(expr, refs);
    std::vector<std::string> names;
    for (const auto& r : refs) {
        bool found = false;
        for (const auto& n : names) if (n == r.column) { found = true; break; }
        if (!found) names.push_back(r.column);
    }
    return names;
}

// Check if expression contains any aggregates
inline bool hasAggregate(const TypedExprPtr& expr) {
    if (!expr) return false;
    if (expr->kind == TypedExpr::Kind::Aggregate) return true;
    switch (expr->kind) {
        case TypedExpr::Kind::Binary: {
            const auto& b = expr->asBinary();
            return hasAggregate(b.left) || hasAggregate(b.right);
        }
        case TypedExpr::Kind::Unary:
            return hasAggregate(expr->asUnary().operand);
        case TypedExpr::Kind::Compare: {
            const auto& c = expr->asCompare();
            if (hasAggregate(c.left) || hasAggregate(c.right)) return true;
            for (const auto& e : c.inList) if (hasAggregate(e)) return true;
            return false;
        }
        case TypedExpr::Kind::Function:
            for (const auto& a : expr->asFunction().args) if (hasAggregate(a)) return true;
            return false;
        case TypedExpr::Kind::Case: {
            const auto& cs = expr->asCase();
            for (const auto& wt : cs.cases) {
                if (hasAggregate(wt.when) || hasAggregate(wt.then)) return true;
            }
            return hasAggregate(cs.elseExpr);
        }
        case TypedExpr::Kind::Cast:
            return hasAggregate(expr->asCast().expr);
        case TypedExpr::Kind::Alias:
            return hasAggregate(expr->asAlias().expr);
        default:
            return false;
    }
}

// Unwrap alias to get the underlying expression
inline TypedExprPtr unwrapAlias(const TypedExprPtr& expr) {
    if (!expr) return nullptr;
    if (expr->kind == TypedExpr::Kind::Alias) {
        return unwrapAlias(expr->asAlias().expr);
    }
    return expr;
}

// Get alias name if present
inline std::string getAliasName(const TypedExprPtr& expr) {
    if (!expr) return "";
    if (expr->kind == TypedExpr::Kind::Alias) {
        return expr->asAlias().alias;
    }
    if (expr->kind == TypedExpr::Kind::Aggregate) {
        return expr->asAggregate().alias;
    }
    return "";
}

} // namespace engine
