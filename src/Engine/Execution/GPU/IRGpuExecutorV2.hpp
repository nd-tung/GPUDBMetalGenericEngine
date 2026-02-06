#pragma once

#include <optional>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>

#include "IRv2.hpp"
#include "TypedExpr.hpp"
#include "Schema.hpp"
#include "../ResultTable.hpp"

namespace MTL { class Buffer; }

namespace engine {

struct EvalContext;

// ============================================================================
// IRGpuExecutorV2: Generic GPU executor using V2 IR (no regex fallbacks)
// Uses TypedExpr trees for expression evaluation and SchemaRegistry for types
// ============================================================================

class IRGpuExecutorV2 {
public:
    struct ExecutionResult {
        bool success = false;
        std::string error;
        TableResult table;
        
        // Scalar aggregate result (for queries like Q6)
        bool isScalarAggregate = false;
        double scalarValue = 0.0;
        std::string scalarName;
    };

    // Main entry point: execute a V2 plan and return results
    static ExecutionResult execute(const PlanV2& plan, const std::string& datasetPath);

    // Check if a plan can be executed on GPU
    static bool canExecuteGPU(const PlanV2& plan);
    

    // Get list of blockers preventing GPU execution
    static std::vector<std::string> getGPUBlockers(const PlanV2& plan);

private:
    // Execute individual operators
    static bool executeScan(const IRScanV2& scan, const std::string& datasetPath, EvalContext& ctx);
    static bool executeFilter(const IRFilterV2& filter, EvalContext& ctx);
    static bool executeJoin(const IRJoinV2& join, const std::string& datasetPath, 
                            EvalContext& leftCtx, EvalContext& rightCtx, EvalContext& outCtx);
    static bool executeGroupBy(const IRGroupByV2& groupBy, EvalContext& ctx, TableResult& out);
    static bool executeAggregate(const IRAggregateV2& agg, EvalContext& ctx, 
                                  double& outValue, std::string& outName);
    static bool executeOrderBy(const IROrderByV2& order, TableResult& table);
    static bool executeLimit(const IRLimitV2& limit, TableResult& table);
    static bool executeProject(const IRProjectV2& project, EvalContext& ctx, TableResult& out, std::unordered_map<std::string, EvalContext>* tableContexts = nullptr);

    // TypedExpr evaluation on GPU buffers
    static std::vector<float> evalExprFloat(const TypedExprPtr& expr, const EvalContext& ctx);
    static MTL::Buffer* evalExprFloatGPU(const TypedExprPtr& expr, EvalContext& ctx);
    static std::vector<uint32_t> evalExprU32(const TypedExprPtr& expr, const EvalContext& ctx);
    static MTL::Buffer* evalExprU32GPU(const TypedExprPtr& expr, EvalContext& ctx);
    static std::vector<bool> evalPredicate(const TypedExprPtr& pred, const EvalContext& ctx);
    
    // Recursive GPU filter helper
    static bool executeGPUFilterRecursive(const TypedExprPtr& expr, EvalContext& ctx);
    
    // Helper to get column data
    static std::pair<bool, bool> getColumnData(const std::string& colName, const EvalContext& ctx,
                                                std::vector<uint32_t>*& u32Out, 
                                                std::vector<float>*& f32Out);

    // Orchestrate the complex logic of setting up a join (finding tables, handling scalar subqueries, etc.)
    static bool orchestrateJoin(
        const IRJoinV2& join,
        const std::string& datasetPath,
        EvalContext& currentCtx,
        std::unordered_map<std::string, EvalContext>& tableContexts,
        std::vector<EvalContext>& savedPipelines,
        std::vector<std::set<std::string>>& savedPipelineTables,
        std::set<std::string>& joinedTables,
        bool& hasPipeline,
        ExecutionResult& result
    );
};

} // namespace engine
