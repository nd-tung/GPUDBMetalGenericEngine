#pragma once
#include <string>
#include "IRv2.hpp"

namespace engine {

// ============================================================================
// PlannerV2: Generic SQL planner using DuckDB EXPLAIN JSON
// No regex fallback - fully parses DuckDB operator tree
// ============================================================================

class PlannerV2 {
public:
    // Parse SQL to a V2 plan using DuckDB EXPLAIN (FORMAT JSON)
    static PlanV2 fromSQL(const std::string& sql);
    
    // Check if we can execute a plan on GPU (returns blockers if not)
    struct Feasibility {
        bool canExecuteGPU = false;
        std::vector<std::string> blockers;
    };
    static Feasibility checkGPUFeasibility(const PlanV2& plan);
    
    // Extract all needed columns from all tables in the plan
    static std::vector<std::pair<std::string, std::vector<std::string>>> 
        extractNeededColumns(const PlanV2& plan);

    // Parsing helpers (public for use by traversal)
    static TypedExprPtr parseExpression(const std::string& exprStr);
    static AggFunc parseAggFunc(const std::string& name);
    static CompareOp parseCompareOp(const std::string& op);
};

} // namespace engine
