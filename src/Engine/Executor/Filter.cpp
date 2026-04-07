#include "GpuExecutor.hpp"
#include "GpuExecutorDetail.hpp"
#include "GpuColumnStore.hpp"
#include "EngineError.hpp"

#include <functional>
#include <iostream>
#include <set>
#include <unordered_set>
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

    // Collect column names referenced in the filter predicate.
    // We only need to GPU-upload columns that are actually used by the filter —
    // uploading ALL context columns is unnecessarily expensive after large joins.
    std::unordered_set<std::string> predCols;
    {
        // Walk the predicate expression tree to collect all column references.
        std::function<void(const TypedExprPtr&)> collectCols = [&](const TypedExprPtr& e) {
            if (!e) return;
            if (e->kind == TypedExpr::Kind::Column) {
                const std::string& c = e->asColumn().column;
                predCols.insert(c);
                // Also insert suffixed variants (multi-instance tables like n_name_2)
                for (int sfx = 1; sfx <= 9; ++sfx)
                    predCols.insert(c + "_" + std::to_string(sfx));
            } else if (e->kind == TypedExpr::Kind::Binary) {
                const auto& b = e->asBinary();
                collectCols(b.left); collectCols(b.right);
            } else if (e->kind == TypedExpr::Kind::Compare) {
                const auto& cmp = e->asCompare();
                collectCols(cmp.left); collectCols(cmp.right);
            } else if (e->kind == TypedExpr::Kind::Function) {
                for (const auto& arg : e->asFunction().args) collectCols(arg);
            } else if (e->kind == TypedExpr::Kind::Unary) {
                collectCols(e->asUnary().operand);
            } else if (e->kind == TypedExpr::Kind::Case) {
                const auto& c2 = e->asCase();
                for (const auto& w : c2.cases) { collectCols(w.when); collectCols(w.then); }
                collectCols(c2.elseExpr);
            } else if (e->kind == TypedExpr::Kind::Cast) {
                collectCols(e->asCast().expr);
            } else if (e->kind == TypedExpr::Kind::Alias) {
                collectCols(e->asAlias().expr);
            }
        };
        collectCols(pred);
    }

    // Targeted lazy-upload: only upload CPU-only columns referenced by the predicate.
    // This avoids uploading entire post-join contexts (20+ large columns) to GPU
    // on every filter call, which caused 10-30× regressions for q10/q02.
    {
        auto& store = GpuColumnStore::instance();
        if (store.device()) {
            for (const auto& colName : predCols) {
                auto fit = ctx.f32Cols.find(colName);
                if (fit != ctx.f32Cols.end() && !fit->second.empty() && !ctx.f32ColsGPU.count(colName)) {
                    const auto& vec = fit->second;
                    MTL::Buffer* buf = store.device()->newBuffer(
                        vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                    if (buf) {
                        ctx.f32ColsGPU[colName].reset(buf);
                        LOG_DEBUG("Exec", "Filter: lazy-uploaded f32 col '" << colName
                                  << "' (" << vec.size() << " rows) to GPU");
                    }
                }
                auto uit = ctx.u32Cols.find(colName);
                if (uit != ctx.u32Cols.end() && !uit->second.empty() && !ctx.u32ColsGPU.count(colName)) {
                    const auto& vec = uit->second;
                    MTL::Buffer* buf = store.device()->newBuffer(
                        vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    if (buf) {
                        ctx.u32ColsGPU[colName].reset(buf);
                        LOG_DEBUG("Exec", "Filter: lazy-uploaded u32 col '" << colName
                                  << "' (" << vec.size() << " rows) to GPU");
                    }
                }
            }
        }
    }


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
