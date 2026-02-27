#pragma once
#include <string>
#include "IR.hpp"

namespace engine {

// ============================================================================
// Planner: Generic SQL planner using DuckDB EXPLAIN JSON
// No regex fallback - fully parses DuckDB operator tree
// ============================================================================

class Planner {
public:
    // Parse SQL to a V2 plan using DuckDB EXPLAIN (FORMAT JSON)
    static Plan fromSQL(const std::string& sql);

    // Parsing helpers (public for use by traversal)
    static TypedExprPtr parseExpression(const std::string& exprStr);
    static AggFunc parseAggFunc(const std::string& name);
    static CompareOp parseCompareOp(const std::string& op);
};

} // namespace engine
