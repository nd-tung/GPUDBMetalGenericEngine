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
    
    // Check if we can execute a plan on GPU (returns blockers if not)
    struct Feasibility {
        bool canExecuteGPU = false;
        std::vector<std::string> blockers;
    };
    static Feasibility checkGPUFeasibility(const Plan& plan);
    
    // Extract all needed columns from all tables in the plan
    static std::vector<std::pair<std::string, std::vector<std::string>>> 
        extractNeededColumns(const Plan& plan);

    // Parsing helpers (public for use by traversal)
    static TypedExprPtr parseExpression(const std::string& exprStr);
    static AggFunc parseAggFunc(const std::string& name);
    static CompareOp parseCompareOp(const std::string& op);
};

} // namespace engine
