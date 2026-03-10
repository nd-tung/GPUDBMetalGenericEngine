// ============================================================================
// JoinUtils.cpp — Small utility functions shared across join files
// ============================================================================
#include "JoinInternal.hpp"

namespace engine {

// -- ensureColumnOnGPU --
// Uploads a u32 column to GPU, compacting via activeRows gather if needed.
MTL::Buffer* ensureColumnOnGPU(EvalContext& ctx, const std::string& col, bool debug) {
    auto& store = GpuColumnStore::instance();
    uint32_t expectedSize = ctx.activeRowsGPU ? ctx.activeRowsCountGPU : (uint32_t)ctx.rowCount;
    if (ctx.u32ColsGPU.count(col)) {
        MTL::Buffer* existing = ctx.u32ColsGPU.at(col);
        if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0 &&
            existing->length() / sizeof(uint32_t) > expectedSize) {
            if (debug) std::cerr << "[Exec] ensureGPU: compacting GPU buf " << col << " from " << (existing->length()/sizeof(uint32_t)) << " to " << expectedSize << "\n";
            auto compactedBuf = GpuOps::gatherU32(existing, ctx.activeRowsGPU, ctx.activeRowsCountGPU, true);
            if (compactedBuf) {
                if (debug) {
                    uint32_t* p = (uint32_t*)compactedBuf->contents();
                    std::cerr << "[Exec] ensureGPU: compacted " << col << " first 5:";
                    for (uint32_t i = 0; i < std::min(expectedSize, 5u); i++) std::cerr << " " << p[i];
                    if (debug) std::cerr << "\n";
                }
                ctx.u32ColsGPU[col].reset(compactedBuf);
                return compactedBuf;
            }
        }
        return existing;
    }
    if (ctx.u32Cols.count(col)) {
         const auto& vec = ctx.u32Cols.at(col);
         if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0 && vec.size() > expectedSize) {
             auto fullBuf = store.device()->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
             if (fullBuf) {
                 auto compactedBuf = GpuOps::gatherU32(fullBuf, ctx.activeRowsGPU, ctx.activeRowsCountGPU, true);
                 fullBuf->release();
                 if (compactedBuf) {
                     ctx.u32ColsGPU[col].reset(compactedBuf);
                     return compactedBuf;
                 }
             }
         }
         auto buf = store.device()->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
         ctx.u32ColsGPU[col].reset(buf);
         return buf;
    }
    return nullptr;
}

// -- findColWithSuffix --
// Find a column in EvalContext (u32 or f32→u32 bitwise conversion),
// trying the base name and then _1 … _9 suffixes.
std::string findColWithSuffix(EvalContext& ctx, const std::string& col) {
    if (ctx.u32Cols.find(col) != ctx.u32Cols.end()) return col;
    if (ctx.f32Cols.find(col) != ctx.f32Cols.end()) {
        const auto& fVec = ctx.f32Cols.at(col);
        std::vector<uint32_t> uVec(fVec.size());
        if (!fVec.empty()) std::memcpy(uVec.data(), fVec.data(), fVec.size() * sizeof(uint32_t));
        ctx.u32Cols[col] = std::move(uVec);
        return col;
    }
    for (int suffix = 1; suffix <= 9; ++suffix) {
        std::string suffixedCol = col + "_" + std::to_string(suffix);
        if (ctx.u32Cols.find(suffixedCol) != ctx.u32Cols.end()) return suffixedCol;
        if (ctx.f32Cols.find(suffixedCol) != ctx.f32Cols.end()) {
            const auto& fVec = ctx.f32Cols.at(suffixedCol);
            std::vector<uint32_t> uVec(fVec.size());
            if (!fVec.empty()) std::memcpy(uVec.data(), fVec.data(), fVec.size() * sizeof(uint32_t));
            ctx.u32Cols[suffixedCol] = std::move(uVec);
            return suffixedCol;
        }
    }
    return "";
}

// -- fuzzyResolveColumn --
// Fuzzy-resolve a column name that wasn't found directly.
// Tries prefix aliases, positional refs, suffix match, _rhs_ stripping.
std::string fuzzyResolveColumn(EvalContext& ctx, const std::string& colName,
                               const std::unordered_set<std::string>& excludeCols) {
    // 1. Suffixed versions
    std::string s = findColWithSuffix(ctx, colName);
    if (!s.empty() && excludeCols.find(s) == excludeCols.end()) return s;

    // 2. Prefix aliases (l_ -> o_, c_, etc.)
    if (colName.size() > 2 && colName[1] == '_') {
        std::string suffix = colName.substr(2);
        static const std::vector<std::string> prefixes = {"l_", "o_", "c_", "p_", "s_", "ps_", "n_", "r_"};
        for (const auto& p : prefixes) {
            std::string alt = p + suffix;
            std::string res = findColWithSuffix(ctx, alt);
            if (!res.empty() && excludeCols.find(res) == excludeCols.end()) return res;
        }
    }

    // 3. Positional refs (#0..#9)
    for (int i = 0; i < 10; ++i) {
        std::string posRef = "#" + std::to_string(i);
        if (excludeCols.find(posRef) != excludeCols.end()) continue;
        if (ctx.u32Cols.count(posRef)) return posRef;
        if (ctx.f32Cols.count(posRef)) {
            const auto& fVec = ctx.f32Cols.at(posRef);
            std::vector<uint32_t> uVec(fVec.size());
            if (!fVec.empty()) std::memcpy(uVec.data(), fVec.data(), fVec.size() * sizeof(uint32_t));
            ctx.u32Cols[posRef] = std::move(uVec);
            return posRef;
        }
    }

    // 4. Suffix match on u32/f32
    auto underscorePos = colName.find('_');
    if (underscorePos != std::string::npos) {
        std::string sfx = colName.substr(underscorePos);
        for (const auto& [n, _] : ctx.u32Cols) {
            if (n.size() >= sfx.size() && n.rfind(sfx) == n.size() - sfx.size()) return n;
        }
        for (const auto& [n, _] : ctx.f32Cols) {
            if (n.size() >= sfx.size() && n.rfind(sfx) == n.size() - sfx.size()) {
                findColWithSuffix(ctx, n);
                return n;
            }
        }
    }

    // 5. Strip _rhs_N suffix
    size_t rhsPos = colName.find("_rhs_");
    if (rhsPos != std::string::npos) {
        std::string base = colName.substr(0, rhsPos);
        if (ctx.u32Cols.count(base) || ctx.f32Cols.count(base)) return base;
        if (base.size() > 2 && base[1] == '_') {
            std::string sfx = base.substr(2);
            static const std::vector<std::string> prefixes = {"l_", "o_", "c_", "p_", "s_", "ps_", "n_", "r_"};
            for (const auto& p : prefixes) {
                std::string alt = p + sfx;
                if (ctx.u32Cols.count(alt) || ctx.f32Cols.count(alt)) return alt;
            }
        }
    }

    return "";
}

// -- hasColumnOrSuffixed --
// Check if column (or a suffixed version _1…_9) exists in a context.
bool hasColumnOrSuffixed(const EvalContext& ctx, const std::string& colName) {
    if (ctx.u32Cols.find(colName) != ctx.u32Cols.end()) return true;
    if (ctx.f32Cols.find(colName) != ctx.f32Cols.end()) return true;
    for (int suffix = 1; suffix <= 9; ++suffix) {
        std::string suffixedCol = colName + "_" + std::to_string(suffix);
        if (ctx.u32Cols.find(suffixedCol) != ctx.u32Cols.end()) return true;
        if (ctx.f32Cols.find(suffixedCol) != ctx.f32Cols.end()) return true;
    }
    // Try rhs suffixes (e.g. col_rhs_10)
    std::string rhsPattern = colName + "_rhs_";
    for (const auto& [name, _] : ctx.u32Cols) {
        if (name.find(rhsPattern) == 0) return true;
    }
    for (const auto& [name, _] : ctx.f32Cols) {
        if (name.find(rhsPattern) == 0) return true;
    }
    return false;
}

} // namespace engine
