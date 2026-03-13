#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "EngineError.hpp"

#include <iostream>
#include <set>
#include "Logger.hpp"

namespace engine {

bool GpuExecutor::executeFilter(const IRFilter& filter, EvalContext& ctx) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    if (!filter.predicate) {
        LOG_DEBUG("Exec", "Filter: no predicate\n");
        return true;  // No-op filter
    }
    
    if (debug) {
        LOG_INFO("Exec", "Filter predicate kind=" << static_cast<int>(filter.predicate->kind));
        LOG_INFO("Exec", "Filter predicateStr=" << filter.predicateStr);
    }
    
    // Optimization: Empty input -> Empty output
    if (ctx.rowCount == 0) {
        LOG_DEBUG("Exec", "Filter input empty, returning early.\n");
        return true;
    }
    
    // Build set of available column names for multi-instance transformation
    std::set<std::string> availableCols;
    for (const auto& [name, _] : ctx.u32Cols) availableCols.insert(name);
    for (const auto& [name, _] : ctx.f32Cols) availableCols.insert(name);
    for (const auto& [name, _] : ctx.stringCols) availableCols.insert(name);
    for (const auto& [name, _] : ctx.dictCols) availableCols.insert(name);
    // Also include GPU columns
    for (const auto& [name, _] : ctx.u32ColsGPU) availableCols.insert(name);
    for (const auto& [name, _] : ctx.f32ColsGPU) availableCols.insert(name);
    
    // Transform predicate for multi-instance columns (like n_name -> n_name_2)
    auto pred = transformMultiInstancePredicate(filter.predicate, availableCols, debug);

    // Try GPU Filter first
    if (!ctx.u32ColsGPU.empty() || !ctx.f32ColsGPU.empty()) {
        if (executeFilterRecursive(pred, ctx)) {
            LOG_DEBUG("Exec", "GPU Filter success, count=" << ctx.activeRowsCountGPU);
            // Lazy sync: only update rowCount here.
            // CPU activeRows vector is populated on demand via ctx.ensureActiveRowsCPU().
            if (ctx.activeRowsGPU) {
                ctx.rowCount = ctx.activeRowsCountGPU;
            } else if (ctx.activeRowsCountGPU == 0) {
                ctx.rowCount = 0;
                ctx.activeRows.clear();
            }
            return true;
        } else {
            LOG_ERROR("Exec", "GPU Filter failed/unsupported: " << filter.predicateStr);
            ENGINE_THROW("GPU Filter failed, and CPU fallback is disabled.");
        }
    }
    
    ENGINE_THROW("GPU Filter path not applicable, and CPU fallback is disabled.");
}

} // namespace engine
