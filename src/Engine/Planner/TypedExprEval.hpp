#pragma once
#include "TypedExpr.hpp"
#include "Schema.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <cmath>

namespace engine {

// ============================================================================
// Generic expression evaluator for CPU-side postprocessing
// Works with TypedExpr trees from the new IR
// ============================================================================

// Row accessor function type: (rowIndex, columnName) -> value
using RowGetterF64 = std::function<double(size_t, const std::string&)>;
using RowGetterU32 = std::function<uint32_t(size_t, const std::string&)>;

// Column existence checker
using ColumnExistsFn = std::function<bool(const std::string&)>;

// Expression evaluation result
struct EvalResult {
    enum class Kind { Float, Int, Bool, String, Null } kind;
    double floatVal = 0.0;
    int64_t intVal = 0;
    bool boolVal = false;
    std::string strVal;
    
    static EvalResult fromFloat(double v) { return {Kind::Float, v, 0, false, ""}; }
    static EvalResult fromInt(int64_t v) { return {Kind::Int, 0.0, v, false, ""}; }
    static EvalResult fromBool(bool v) { return {Kind::Bool, 0.0, 0, v, ""}; }
    static EvalResult fromString(const std::string& s) { return {Kind::String, 0.0, 0, false, s}; }
    static EvalResult null() { return {Kind::Null, 0.0, 0, false, ""}; }
    
    double asFloat() const {
        switch (kind) {
            case Kind::Float: return floatVal;
            case Kind::Int: return static_cast<double>(intVal);
            case Kind::Bool: return boolVal ? 1.0 : 0.0;
            default: return 0.0;
        }
    }
    
    int64_t asInt() const {
        switch (kind) {
            case Kind::Float: return static_cast<int64_t>(floatVal);
            case Kind::Int: return intVal;
            case Kind::Bool: return boolVal ? 1 : 0;
            default: return 0;
        }
    }
    
    bool asBool() const {
        switch (kind) {
            case Kind::Float: return floatVal != 0.0;
            case Kind::Int: return intVal != 0;
            case Kind::Bool: return boolVal;
            case Kind::String: return !strVal.empty();
            default: return false;
        }
    }
};

// ============================================================================
// TypedExprEvaluator: Evaluates TypedExpr trees row-by-row
// ============================================================================

class TypedExprEvaluator {
public:
    TypedExprEvaluator(const RowGetterF64& getFloat, const RowGetterU32& getU32)
        : getFloat_(getFloat), getU32_(getU32) {}
    
    EvalResult eval(const TypedExprPtr& expr, size_t row) const {
        if (!expr) return EvalResult::null();
        
        switch (expr->kind) {
            case TypedExpr::Kind::Column: {
                const auto& col = expr->asColumn();
                // Try to get as float first, fall back to u32
                double v = getFloat_(row, col.column);
                return EvalResult::fromFloat(v);
            }
            
            case TypedExpr::Kind::Literal: {
                const auto& lit = expr->asLiteral();
                if (auto* i = std::get_if<int64_t>(&lit.value)) {
                    return EvalResult::fromInt(*i);
                }
                if (auto* d = std::get_if<double>(&lit.value)) {
                    return EvalResult::fromFloat(*d);
                }
                if (auto* s = std::get_if<std::string>(&lit.value)) {
                    // Handle date literals
                    if (lit.type == DataType::Date) {
                        // Parse YYYY-MM-DD to YYYYMMDD integer
                        std::string ds = *s;
                        ds.erase(std::remove(ds.begin(), ds.end(), '-'), ds.end());
                        try {
                            return EvalResult::fromInt(std::stoll(ds));
                        } catch (...) {
                            return EvalResult::fromString(*s);
                        }
                    }
                    return EvalResult::fromString(*s);
                }
                return EvalResult::null();
            }
            
            case TypedExpr::Kind::Binary: {
                const auto& bin = expr->asBinary();
                EvalResult left = eval(bin.left, row);
                EvalResult right = eval(bin.right, row);
                
                switch (bin.op) {
                    case BinaryOp::Add:
                        return EvalResult::fromFloat(left.asFloat() + right.asFloat());
                    case BinaryOp::Sub:
                        return EvalResult::fromFloat(left.asFloat() - right.asFloat());
                    case BinaryOp::Mul:
                        return EvalResult::fromFloat(left.asFloat() * right.asFloat());
                    case BinaryOp::Div:
                        if (right.asFloat() == 0.0) return EvalResult::null();
                        return EvalResult::fromFloat(left.asFloat() / right.asFloat());
                    case BinaryOp::Mod:
                        return EvalResult::fromInt(left.asInt() % right.asInt());
                    case BinaryOp::And:
                        return EvalResult::fromBool(left.asBool() && right.asBool());
                    case BinaryOp::Or:
                        return EvalResult::fromBool(left.asBool() || right.asBool());
                }
                return EvalResult::null();
            }
            
            case TypedExpr::Kind::Unary: {
                const auto& un = expr->asUnary();
                EvalResult operand = eval(un.operand, row);
                
                switch (un.op) {
                    case UnaryOp::Neg:
                        return EvalResult::fromFloat(-operand.asFloat());
                    case UnaryOp::Not:
                        return EvalResult::fromBool(!operand.asBool());
                }
                return EvalResult::null();
            }
            
            case TypedExpr::Kind::Compare: {
                const auto& cmp = expr->asCompare();
                EvalResult left = eval(cmp.left, row);
                EvalResult right = cmp.right ? eval(cmp.right, row) : EvalResult::null();
                
                switch (cmp.op) {
                    case CompareOp::Eq:
                        return EvalResult::fromBool(left.asFloat() == right.asFloat());
                    case CompareOp::Ne:
                        return EvalResult::fromBool(left.asFloat() != right.asFloat());
                    case CompareOp::Lt:
                        return EvalResult::fromBool(left.asFloat() < right.asFloat());
                    case CompareOp::Le:
                        return EvalResult::fromBool(left.asFloat() <= right.asFloat());
                    case CompareOp::Gt:
                        return EvalResult::fromBool(left.asFloat() > right.asFloat());
                    case CompareOp::Ge:
                        return EvalResult::fromBool(left.asFloat() >= right.asFloat());
                    case CompareOp::IsNull:
                        return EvalResult::fromBool(left.kind == EvalResult::Kind::Null);
                    case CompareOp::IsNotNull:
                        return EvalResult::fromBool(left.kind != EvalResult::Kind::Null);
                    default:
                        return EvalResult::fromBool(false);
                }
            }
            
            case TypedExpr::Kind::Alias: {
                return eval(expr->asAlias().expr, row);
            }
            
            case TypedExpr::Kind::Cast: {
                const auto& cast = expr->asCast();
                EvalResult inner = eval(cast.expr, row);
                // Simple type coercion
                switch (cast.targetType) {
                    case DataType::Float32:
                    case DataType::Float64:
                        return EvalResult::fromFloat(inner.asFloat());
                    case DataType::Int32:
                    case DataType::Int64:
                        return EvalResult::fromInt(inner.asInt());
                    default:
                        return inner;
                }
            }
            
            default:
                return EvalResult::null();
        }
    }
    
    // Evaluate a comparison predicate and return boolean
    bool evalPredicate(const TypedExprPtr& pred, size_t row) const {
        return eval(pred, row).asBool();
    }
    
    // Evaluate expression and return as float
    float evalFloat(const TypedExprPtr& expr, size_t row) const {
        return static_cast<float>(eval(expr, row).asFloat());
    }
    
private:
    RowGetterF64 getFloat_;
    RowGetterU32 getU32_;
};

// ============================================================================
// Aggregate accumulator for typed aggregates
// ============================================================================

struct AggAccumulator {
    AggFunc func;
    double sum = 0.0;
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    uint64_t count = 0;
    
    void reset() {
        sum = 0.0;
        min_val = std::numeric_limits<double>::max();
        max_val = std::numeric_limits<double>::lowest();
        count = 0;
    }
    
    void add(double value) {
        switch (func) {
            case AggFunc::Sum:
            case AggFunc::Avg:
                sum += value;
                count++;
                break;
            case AggFunc::Count:
            case AggFunc::CountStar:
                count++;
                break;
            case AggFunc::Min:
                if (value < min_val) min_val = value;
                count++;
                break;
            case AggFunc::Max:
                if (value > max_val) max_val = value;
                count++;
                break;
            default:
                break;
        }
    }
    
    void addCount() {
        count++;
    }
    
    double result() const {
        switch (func) {
            case AggFunc::Sum:
                return sum;
            case AggFunc::Avg:
                return count > 0 ? sum / static_cast<double>(count) : 0.0;
            case AggFunc::Count:
            case AggFunc::CountStar:
                return static_cast<double>(count);
            case AggFunc::Min:
                return count > 0 ? min_val : 0.0;
            case AggFunc::Max:
                return count > 0 ? max_val : 0.0;
            default:
                return 0.0;
        }
    }
    
    uint64_t countResult() const {
        return count;
    }
};

// ============================================================================
// Helper to extract column names needed for expression evaluation
// ============================================================================

inline std::vector<std::string> getNeededColumnsForExpr(const TypedExprPtr& expr) {
    if (!expr) return {};
    return getColumnNames(expr);
}

// ============================================================================
// Predicate evaluator using TypedExpr
// ============================================================================

inline bool evaluatePredicate(const TypedExprPtr& pred, 
                              size_t row,
                              const RowGetterF64& getFloat,
                              const RowGetterU32& getU32) {
    TypedExprEvaluator evaluator(getFloat, getU32);
    return evaluator.evalPredicate(pred, row);
}

// ============================================================================
// Expression string to TypedExpr conversion helpers
// ============================================================================

// Parse a simple predicate string into a TypedExpr
// Supports: col op literal, col op DATE 'YYYY-MM-DD', AND conjunction
TypedExprPtr parsePredicateString(const std::string& pred);

// Parse an arithmetic expression string into a TypedExpr
TypedExprPtr parseArithmeticString(const std::string& expr);

} // namespace engine
