#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "QueryExecutionContext.hpp"
#include "Operators.hpp"
#include "GpuColumnStore.hpp"
#include <Metal/Metal.hpp>

#include "Planner.hpp"
#include "KernelTimer.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <iostream>
#include <map>
#include <set>
#include <unordered_set>
#include "Logger.hpp"

namespace engine {

// Forward declarations (implemented in ExecNodes.cpp)
void handleExecScanNode(const IRScan& scan, size_t nodeIdx, QueryExecutionContext& qctx);
bool handleExecFilterNode(const IRFilter& filter, QueryExecutionContext& qctx);
bool handleExecGroupByNode(const IRGroupBy& groupBy, QueryExecutionContext& qctx);
bool handleExecAggregateNode(const IRAggregate& agg, QueryExecutionContext& qctx);
bool handleExecOrderByNode(const IROrderBy& orderBy, QueryExecutionContext& qctx);
bool handleExecProjectNode(const IRProject& project, QueryExecutionContext& qctx);

// Forward declarations (implemented in GpuExecutorPostProcess.cpp)
void materializeContextToResult(EvalContext& currentCtx, TableResult& tableResult,
    bool isScalarAggregate, bool debug);
void resolveOutputColumnNames(EvalContext& currentCtx, TableResult& tableResult, bool debug);
void recoverStringColumns(EvalContext& currentCtx, TableResult& tableResult,
    const std::string& datasetPath, bool debug);
void filterOutputColumns(const Plan& plan, TableResult& tableResult,
    bool isScalarAggregate, bool debug);
std::unordered_map<std::string, std::vector<std::string>>
extractDelimCorrelationCols(const Plan& plan, bool debug);

// --- Main Execution Entry Point ---

GpuExecutor::ExecutionResult GpuExecutor::execute(const Plan& plan, const std::string& datasetPath) {
    ExecutionResult result;
    result.success = false;
    
    // Reset kernel timer for this query
    KernelTimer::instance().reset();

    if (!plan.isValid()) {
        result.error = "Invalid plan: " + plan.parseError;
        return result;
    }

    auto blockers = getUnsupportedFeatures(plan);
    if (!blockers.empty()) {
        result.error = "GPU execution blocked: " + blockers[0];
        for (size_t i = 1; i < blockers.size(); ++i) {
            result.error += ", " + blockers[i];
        }
        return result;
    }

    const bool debug = env_truthy("GPUDB_DEBUG_OPS");

    if (debug) {
        LOG_INFO("Exec", "Plan Nodes (" << plan.nodes.size() << "):\n");
        for (size_t i = 0; i < plan.nodes.size(); ++i) {
            const auto& n = plan.nodes[i];
            std::string name = n.duckdbName;
            if (n.type == IRNode::Type::Save) name = "Save(" + n.asSave().name + ")";
            else if (n.type == IRNode::Type::Scan) name = "Scan(" + n.asScan().table + ")";
            else if (n.type == IRNode::Type::Join) name = "Join(" + n.asJoin().conditionStr + ")";
            else if (name.empty()) name = "[Empty/Unknown Type=" + std::to_string((int)n.type) + "]";
            LOG_DEBUG("ENGINE", "  #" << i << ": " << name);
        }
    }

    // Build scan instance map for tables that appear multiple times
    auto scanInstanceMap = buildScanInstanceMap(plan);
    
    if (debug && !scanInstanceMap.empty()) {
        LOG_INFO("Exec", "Table instances for self-joins:\n");
        for (const auto& [nodeIdx, inst] : scanInstanceMap) {
            LOG_INFO("ENGINE", "  Node " << nodeIdx << ": " << inst.baseTable  << " -> " << inst.instanceKey);
        }
    }

    // Collect all tables and columns needed
    auto tableColsMap = collectNeededColumns(plan);

    if (debug) {
        LOG_INFO("Exec", "Columns needed per table:\n");
        for (const auto& [t, cs] : tableColsMap) {
            LOG_INFO("ENGINE", "  " << t << ": ");
            if (debug) for (const auto& c : cs) std::cerr << c << " ";
            LOG_DEBUG("ENGINE", "\n");
        }
    }

    // Build execution contexts for each table
    std::unordered_map<std::string, EvalContext> tableContexts;

    // Execute operators in pipeline order
    EvalContext currentCtx;
    TableResult tableResult;

    // Save previous pipeline contexts for multi-pipeline query merges
    std::vector<EvalContext> savedPipelines;

    IRGpuLoader::loadTables(tableColsMap, scanInstanceMap, datasetPath, tableContexts, result, debug);
    if (!result.error.empty()) { return result; }
    
    auto loadEnd = std::chrono::high_resolution_clock::now();

    if (debug) {
        LOG_INFO("Exec", "Loaded " << tableContexts.size() << " tables in "  << result.table.uploadMs << "ms\n");
    }

    // Track which tables have been joined into the current context
    std::set<std::string> joinedTables;
    
    bool hasPipeline = false;

    std::vector<std::set<std::string>> savedPipelineTables;

    // Bundle join pipeline state for cleaner parameter passing
    JoinPipelineState joinState{tableContexts, savedPipelines, savedPipelineTables, joinedTables, hasPipeline};

    // Pre-scan: extract DELIM correlation columns from self-comparison join conditions
    auto delimCorrelationCols = extractDelimCorrelationCols(plan, debug);

    // Bundle all mutable execution state into a single context for clean handler signatures
    QueryExecutionContext qctx{
        currentCtx, tableContexts, tableResult, joinState, result,
        plan, scanInstanceMap, delimCorrelationCols, debug
    };

    // Per-node timing: track wall-clock time per operator type
    static const char* nodeTypeNames[] = {
        "Scan", "Filter", "Project", "Join", "GroupBy",
        "OrderBy", "Limit", "Aggregate", "Distinct", "Save"
    };
    struct NodeTiming { const char* name; double wallMs; double gpuMs; };
    std::vector<NodeTiming> nodeTimings;

    for (size_t nodeIdx = 0; nodeIdx < plan.nodes.size(); ++nodeIdx) {
        const auto& node = plan.nodes[nodeIdx];
        if (debug) {
            LOG_INFO("Exec", "Executing Node " << nodeIdx << " Type=" << (int)node.type);
            if (node.type == IRNode::Type::Save) {
                 LOG_INFO("Exec", "... Save Name: " << node.asSave().name);
            }
        }
        double gpuBefore = KernelTimer::instance().totalGpuMs();
        auto nodeStart = std::chrono::high_resolution_clock::now();
        switch (node.type) {
            case IRNode::Type::Scan: {
                const auto& scan = node.asScan();
                handleExecScanNode(scan, nodeIdx, qctx);
                break;
            }

            case IRNode::Type::Filter: {
                if (!handleExecFilterNode(node.asFilter(), qctx)) {
                    return result;
                }
                break;
            }

            case IRNode::Type::Join: {
                if (!executeJoinPipeline(node.asJoin(), currentCtx, joinState, result)) {
                    return result;
                }
                break;
            }

            case IRNode::Type::GroupBy: {
                if (!handleExecGroupByNode(node.asGroupBy(), qctx)) {
                    return result;
                }
                break;
            }

            case IRNode::Type::Aggregate: {
                if (!handleExecAggregateNode(node.asAggregate(), qctx)) {
                    return result;
                }
                break;
            }

            case IRNode::Type::OrderBy: {
                if (!handleExecOrderByNode(node.asOrderBy(), qctx)) {
                    return result;
                }
                break;
            }

            case IRNode::Type::Limit: {
                if (!executeLimit(node.asLimit(), tableResult)) {
                    result.error = "Limit execution failed";
                    return result;
                }
                if (debug) {
                    LOG_INFO("Exec", "Limit: " << tableResult.rowCount << " rows\n");
                }
                break;
            }

            case IRNode::Type::Distinct: {
                if (!executeDistinct(node.asDistinct(), currentCtx)) {
                    result.error = "Distinct execution failed";
                    return result;
                }
                tableResult.rowCount = currentCtx.rowCount;
                if (debug) {
                    LOG_INFO("Exec", "Distinct: " << currentCtx.rowCount << " rows\n");
                }
                break;
            }

            case IRNode::Type::Project: {
                if (!handleExecProjectNode(node.asProject(), qctx)) {
                    return result;
                }
                break;
            }

            case IRNode::Type::Save: {
                if (debug) {
                    LOG_INFO("Exec", "Save: storing " << currentCtx.rowCount << " rows into " << node.asSave().name);
                }
                tableContexts[node.asSave().name] = currentCtx;
                break;
            }

            default:
                break;
        }
        auto nodeEnd = std::chrono::high_resolution_clock::now();
        int typeIdx = static_cast<int>(node.type);
        const char* name = (typeIdx >= 0 && typeIdx < 10) ? nodeTypeNames[typeIdx] : "Unknown";
        double nodeWallMs = std::chrono::duration<double, std::milli>(nodeEnd - nodeStart).count();
        double nodeGpuMs = KernelTimer::instance().totalGpuMs() - gpuBefore;
        nodeTimings.push_back({name, nodeWallMs, nodeGpuMs});
    }

    materializeContextToResult(currentCtx, tableResult, result.isScalarAggregate, debug);


    auto endTime = std::chrono::high_resolution_clock::now();
    double pipelineWallMs = std::chrono::duration<double, std::milli>(endTime - loadEnd).count();

    resolveOutputColumnNames(currentCtx, tableResult, debug);


    recoverStringColumns(currentCtx, tableResult, datasetPath, debug);
    tableResult.uploadMs = result.table.uploadMs;

    // Materialize any GPU-only columns to CPU for output printing
    for (size_t i = 0; i < tableResult.u32Cols.size(); ++i) {
        if (tableResult.u32Cols[i].empty() &&
            i < tableResult.u32ColsGPU.size() && tableResult.u32ColsGPU[i]) {
            uint32_t rc = tableResult.rowCount;
            tableResult.u32Cols[i].resize(rc);
            std::memcpy(tableResult.u32Cols[i].data(),
                        tableResult.u32ColsGPU[i]->contents(), rc * sizeof(uint32_t));
        }
    }
    for (size_t i = 0; i < tableResult.f32Cols.size(); ++i) {
        if (tableResult.f32Cols[i].empty() &&
            i < tableResult.f32ColsGPU.size() && tableResult.f32ColsGPU[i]) {
            uint32_t rc = tableResult.rowCount;
            tableResult.f32Cols[i].resize(rc);
            std::memcpy(tableResult.f32Cols[i].data(),
                        tableResult.f32ColsGPU[i]->contents(), rc * sizeof(float));
        }
    }

    filterOutputColumns(plan, tableResult, result.isScalarAggregate, debug);


    // CPU post-processing = pipeline wall-clock minus GPU kernel time + column cleanup time
    auto postEnd = std::chrono::high_resolution_clock::now();
    double postProcessMs = std::chrono::duration<double, std::milli>(postEnd - endTime).count();
    double cpuPipelineMs = pipelineWallMs - tableResult.gpuMs;
    tableResult.cpuPostMs = cpuPipelineMs + postProcessMs;

    // Per-node wall-clock breakdown
    tableResult.nodeTimings.reserve(nodeTimings.size());
    for (auto& nt : nodeTimings)
        tableResult.nodeTimings.push_back({nt.name, nt.wallMs, nt.gpuMs});

    result.success = true;
    result.table = std::move(tableResult);
    return result;
}

} // namespace engine
