#pragma once
// Umbrella header — includes all focused sub-headers for backward compatibility.
// New code should prefer including only the specific headers it needs:
//   EngineConfig.hpp  — configuration constants
//   FlatStringCol.hpp — GPU Arrow-style flat string columns
//   DictEncoded.hpp   — dictionary-encoded string columns
//   EvalContext.hpp    — per-operator execution context
//   DetailHelpers.hpp  — inline utility helpers
//
// This file also defines ScanInstance and forward-declares shared helpers.

#include "EngineConfig.hpp"
#include "FlatStringCol.hpp"
#include "DictEncoded.hpp"
#include "EvalContext.hpp"
#include "DetailHelpers.hpp"
#include "GpuExecutor.hpp"
#include "EnvUtil.hpp"

#include <map>
#include <set>

namespace engine {

struct ScanInstance {
    std::string baseTable;     // Original table name (e.g., "nation")
    std::string instanceKey;   // Instance-qualified key (e.g., "nation_1", "nation_2")
    int instanceNum;           // 1-based instance number
    size_t nodeIndex;          // Index in plan.nodes
};

// GroupBy key-building output (shared between GroupBy.cpp and GroupByKeys.cpp)
struct GroupByKeyData {
    std::vector<std::vector<uint32_t>> keyVecs;
    std::vector<std::string> keyNames;
    std::vector<std::vector<std::string>> outputStringMaps;
    std::vector<std::unordered_map<uint32_t, std::string>> hashToStringMaps;
    std::vector<bool> keyFromF32;
    std::vector<MTL::Buffer*> keyBufsGPU;
};

// Unwrap Cast/Alias wrappers to get the underlying expression.
inline const TypedExpr* unwrapExpr(const TypedExpr* e) {
    while (e) {
        if (e->kind == TypedExpr::Kind::Cast) e = e->asCast().expr.get();
        else if (e->kind == TypedExpr::Kind::Alias) e = e->asAlias().expr.get();
        else break;
    }
    return e;
}

// Function declarations for shared helpers (implemented in respective .cpp files)
std::map<size_t, ScanInstance> buildScanInstanceMap(const Plan& plan);
std::unordered_map<std::string, std::set<std::string>> collectNeededColumns(const Plan& plan);
// GPU dedup helper: deduplicate an EvalContext by u32 key columns (GpuExecutor.cpp)
uint32_t deduplicateContext(EvalContext& ctx, const std::vector<std::string>& dedupCols, bool debug);
// Flatten/dict helpers (Scan.cpp) — callable from Join.cpp and other modules
void flattenStringCol(EvalContext& ctx, const std::string& colName);
void buildDictCol(EvalContext& ctx, const std::string& colName);

// Helper for table loading (Scan logic)
struct IRGpuLoader {
    static void loadTables(
        const std::unordered_map<std::string, std::set<std::string>>& tableColsMap,
        const std::map<size_t, ScanInstance>& scanInstanceMap,
        const std::string& datasetPath,
        std::unordered_map<std::string, EvalContext>& tableContexts,
        GpuExecutor::ExecutionResult& result,
        bool debug
    );
};

} // namespace engine
