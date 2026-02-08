#include "GpuExecutor.hpp"
#include "GpuExecutorPriv.hpp"
#include "Relation.hpp"

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cstring>

namespace engine {

bool GpuExecutor::executeFilter(const IRFilter& filter, EvalContext& ctx) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    if (!filter.predicate) {
        if (debug) std::cerr << "[Exec] Filter: no predicate\n";
        return true;  // No-op filter
    }
    
    if (debug) {
        std::cerr << "[Exec] Filter predicate kind=" << static_cast<int>(filter.predicate->kind) << "\n";
        std::cerr << "[Exec] Filter predicateStr=" << filter.predicateStr << "\n";
    }
    
    // Optimization: Empty input -> Empty output
    if (ctx.rowCount == 0) {
        if (debug) std::cerr << "[Exec] Filter input empty, returning early.\n";
        return true;
    }
    
    // Build set of available column names for multi-instance transformation
    std::set<std::string> availableCols;
    for (const auto& [name, _] : ctx.u32Cols) availableCols.insert(name);
    for (const auto& [name, _] : ctx.f32Cols) availableCols.insert(name);
    for (const auto& [name, _] : ctx.stringCols) availableCols.insert(name);
    // Also include GPU columns
    for (const auto& [name, _] : ctx.u32ColsGPU) availableCols.insert(name);
    for (const auto& [name, _] : ctx.f32ColsGPU) availableCols.insert(name);
    
    // Transform predicate for multi-instance columns (like n_name -> n_name_2)
    auto pred = transformMultiInstancePredicate(filter.predicate, availableCols, debug);

    // Try GPU Filter first
    if (!ctx.u32ColsGPU.empty() || !ctx.f32ColsGPU.empty()) {
        if (executeGPUFilterRecursive(pred, ctx)) {
            if (debug) std::cerr << "[Exec] GPU Filter success, count=" << ctx.activeRowsCountGPU << "\n";
            // Sync activeRows to CPU for operations that require it
            if (ctx.activeRowsGPU) {
                ctx.rowCount = ctx.activeRowsCountGPU;
                ctx.activeRows.resize(ctx.rowCount);
                if (ctx.rowCount > 0) {
                   const uint32_t* ptr = static_cast<const uint32_t*>(ctx.activeRowsGPU->contents());
                   std::memcpy(ctx.activeRows.data(), ptr, ctx.rowCount * sizeof(uint32_t));
                }
            } else {
                 if (ctx.activeRowsCountGPU == 0) {
                     ctx.rowCount = 0;
                     ctx.activeRows.clear();
                 }
            }
            return true;
        } else {
            std::cerr << "[Exec] GPU Filter failed/unsupported: " << filter.predicateStr << "\n";
            throw std::runtime_error("GPU Filter failed, and CPU fallback is disabled.");
        }
    }
    
    throw std::runtime_error("GPU Filter path not applicable, and CPU fallback is disabled.");
}

} // namespace engine
