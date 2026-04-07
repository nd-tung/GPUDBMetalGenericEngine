// ============================================================================
// JoinCore.cpp — Core join execution: key resolution, hash join, output scatter
// ============================================================================
#include "JoinInternal.hpp"
#include <future>
#include <thread>
#include "Logger.hpp"

namespace engine {

// Helper: extract all equi-join key pairs from a condition expression,
// and collect non-equality conditions as residual post-join filters.
static void extractJoinKeyPairs(const TypedExprPtr& expr, 
                                std::vector<std::pair<std::string, std::string>>& keyPairs,
                                std::vector<TypedExprPtr>* residuals = nullptr) {
    if (!expr) return;
    
    if (expr->kind == TypedExpr::Kind::Compare) {
        const auto& cmp = expr->asCompare();
        if (cmp.op == CompareOp::Eq && cmp.left && cmp.right) {
            if (cmp.left->kind == TypedExpr::Kind::Column && 
                cmp.right->kind == TypedExpr::Kind::Column) {
                keyPairs.emplace_back(cmp.left->asColumn().column, 
                                      cmp.right->asColumn().column);
                return;
            }
        }
        // Non-equality compare or non-column operands → residual
        if (residuals) residuals->push_back(expr);
    } else if (expr->kind == TypedExpr::Kind::Binary) {
        const auto& bin = expr->asBinary();
        if (bin.op == BinaryOp::And) {
            extractJoinKeyPairs(bin.left, keyPairs, residuals);
            extractJoinKeyPairs(bin.right, keyPairs, residuals);
        } else {
            // Other binary ops → residual
            if (residuals) residuals->push_back(expr);
        }
    } else {
        if (residuals) residuals->push_back(expr);
    }
}


// ============================================================================
// Extracted helpers for executeJoin()
// ============================================================================

static void materializeJoinContext(EvalContext& ctx, const char* label, bool /*debug*/) {
    if (!ctx.activeRowsGPU) return;  // No filter applied — nothing to materialize
    uint32_t count = ctx.activeRowsCountGPU;
    LOG_DEBUG("Exec", "Join: materializing " << label  << " ctx (" << count << " active rows from " << ctx.rowCount << ")\n");
    // If the filter matched 0 rows, clear everything and set rowCount=0
    if (count == 0) {
        for (auto& [name, vec] : ctx.u32Cols) vec.clear();
        for (auto& [name, vec] : ctx.f32Cols) vec.clear();
        for (auto& [name, vec] : ctx.stringCols) vec.clear();
        ctx.u32ColsGPU.clear();
        ctx.f32ColsGPU.clear();
        ctx.activeRowsGPU = nullptr;
        ctx.activeRowsCountGPU = 0;
        ctx.activeRows.clear();
        ctx.rowCount = 0;
        return;
    }

    // Compact GPU u32 columns (async — no per-column sync)
    for (auto& [name, buf] : ctx.u32ColsGPU) {
        if (!buf) continue;
        uint32_t bufRows = (uint32_t)(buf->length() / sizeof(uint32_t));
        if (bufRows > count) {
            GpuBuffer compacted = GpuOps::gatherU32(buf, ctx.activeRowsGPU, count, false);
            if (compacted) {
                buf = std::move(compacted);
            }
        }
    }
    // Compact GPU f32 columns (async — no per-column sync)
    for (auto& [name, buf] : ctx.f32ColsGPU) {
        if (!buf) continue;
        uint32_t bufRows = (uint32_t)(buf->length() / sizeof(float));
        if (bufRows > count) {
            GpuBuffer compacted = GpuOps::gatherF32(buf, ctx.activeRowsGPU, count, false);
            if (compacted) {
                buf = std::move(compacted);
            }
        }
    }
    // GPU-native dict compaction: gather dict IDs on GPU (async)
    for (auto& [name, dict] : ctx.dictCols) {
        if (dict.idsGPU) {
            uint32_t bufRows = (uint32_t)(dict.idsGPU->length() / sizeof(uint32_t));
            if (bufRows > count) {
                GpuBuffer compacted = GpuOps::gatherU32(dict.idsGPU, ctx.activeRowsGPU, count, false);
                if (compacted) {
                    dict.idsGPU = std::move(compacted);
                    dict.rowCount = count;
                    dict.ids.clear();  // Invalidate CPU mirror (lazy sync)
                }
            }
        } else if (dict.ids.size() > count) {
            // CPU-only fallback (no GPU buffer) — needs sync first
            GpuOps::sync();
            uint32_t* indices2 = static_cast<uint32_t*>(ctx.activeRowsGPU->contents());
            std::vector<uint32_t> c;
            c.reserve(count);
            for (uint32_t i = 0; i < count; ++i)
                c.push_back(indices2[i] < (uint32_t)dict.ids.size() ? dict.ids[indices2[i]] : 0u);
            dict.ids = std::move(c);
            dict.rowCount = count;
        }
    }
    // Sync all async GPU gathers before reading results on CPU
    GpuOps::sync();
    {
        auto& s = GpuColumnStore::instance();
        // Phase 1: Dispatch CPU-only column gathers async
        struct PendingU32 { std::string name; GpuBuffer dst; MTL::Buffer* tmpSrc; };
        struct PendingF32 { std::string name; GpuBuffer dst; MTL::Buffer* tmpSrc; };
        std::vector<PendingU32> pendU32;
        std::vector<PendingF32> pendF32;
        for (auto& [name, vec] : ctx.u32Cols) {
            if (vec.size() > count) {
                auto itGpu = ctx.u32ColsGPU.find(name);
                if (itGpu != ctx.u32ColsGPU.end() && itGpu->second) {
                    vec.clear();
                } else {
                    MTL::Buffer* src = s.device()->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    GpuBuffer dst = GpuOps::gatherU32(src, ctx.activeRowsGPU, count, false);
                    pendU32.push_back({name, std::move(dst), src});
                }
            }
        }
        for (auto& [name, vec] : ctx.f32Cols) {
            if (vec.size() > count) {
                auto itGpu = ctx.f32ColsGPU.find(name);
                if (itGpu != ctx.f32ColsGPU.end() && itGpu->second) {
                    vec.clear();
                } else {
                    MTL::Buffer* src = s.device()->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                    GpuBuffer dst = GpuOps::gatherF32(src, ctx.activeRowsGPU, count, false);
                    pendF32.push_back({name, std::move(dst), src});
                }
            }
        }
        // Phase 2: Sync + download
        if (!pendU32.empty() || !pendF32.empty()) GpuOps::sync();
        for (auto& p : pendU32) {
            if (p.dst) {
                auto& vec = ctx.u32Cols[p.name];
                vec.resize(count);
                std::memcpy(vec.data(), p.dst->contents(), count * sizeof(uint32_t));
            }
            if (p.tmpSrc) p.tmpSrc->release();
        }
        for (auto& p : pendF32) {
            if (p.dst) {
                auto& vec = ctx.f32Cols[p.name];
                vec.resize(count);
                std::memcpy(vec.data(), p.dst->contents(), count * sizeof(float));
            }
            if (p.tmpSrc) p.tmpSrc->release();
        }
    }
    // Compact CPU string columns: prefer invalidation if dictCol exists, else GPU/CPU gather
    {
        for (auto& [name, vec] : ctx.stringCols) {
            if (ctx.dictCols.count(name) && ctx.dictCols[name].valid()) {
                vec.clear();  // Will be rebuilt from dict on demand
            } else if (vec.size() > count) {
                // Try GPU gather via flatStringCols
                auto fit = ctx.flatStringCols.find(name);
                if (fit != ctx.flatStringCols.end() && fit->second.chars && ctx.activeRowsGPU) {
                    auto r = GpuOps::gatherFlatString(
                        fit->second.chars, fit->second.offsets, fit->second.lengths,
                        ctx.activeRowsGPU, count, true);
                    if (r.chars) {
                        // Skip CPU string reconstruction — flatStringCol is authoritative
                        vec.clear();
                        // Update flatStringCols to compacted version
                        fit->second.takeFrom(r.chars, r.offsets, r.lengths,
                                             r.rowCount, r.totalBytes);
                        continue;
                    }
                }
                // CPU fallback
                uint32_t* indices = static_cast<uint32_t*>(ctx.activeRowsGPU->contents());
                std::vector<std::string> c;
                c.reserve(count);
                for (uint32_t i = 0; i < count; ++i)
                    c.push_back(indices[i] < (uint32_t)vec.size() ? vec[indices[i]] : std::string());
                vec = std::move(c);
            }
        }
    }
    // Compact flatStringCols that don't have matching stringCols (handled above)
    if (ctx.activeRowsGPU) {
        for (auto& [name, flat] : ctx.flatStringCols) {
            if (!ctx.stringCols.count(name) && flat.chars && flat.rowCount > count) {
                auto r = GpuOps::gatherFlatString(flat.chars, flat.offsets, flat.lengths,
                                                   ctx.activeRowsGPU, count, true);
                if (r.chars) {
                    flat.takeFrom(r.chars, r.offsets, r.lengths,
                                  r.rowCount, r.totalBytes);
                }
            }
        }
    }
    // Clear selection vector — data is now dense
    ctx.activeRowsGPU = nullptr;
    ctx.activeRowsCountGPU = 0;
    ctx.activeRows.clear();
    ctx.rowCount = count;
}

static void applyPostJoinFilter(
    const TypedExprPtr& postJoinFilter,
    const EvalContext& leftCtx, const EvalContext& rightCtx,
    const std::unordered_map<std::string, std::string>& rightColumnMapping,
    EvalContext& outCtx, bool debug
) {
    if (debug) {
        LOG_INFO("Exec", "Join: applying post-join filter, current rows=" << outCtx.rowCount);
    }

    // Rewrite column references in the residual filter to resolve to actual outCtx columns.
    auto findColInCtx = [](const EvalContext& ctx, const std::string& baseName) -> std::string {
        // Exact match first
        if (ctx.u32Cols.count(baseName) || ctx.f32Cols.count(baseName)) return baseName;
        if (ctx.stringCols.count(baseName)) return baseName;
        // Try instance-suffixed versions (e.g., l_suppkey_1, l_suppkey_2, ...)
        for (const auto& [name, _] : ctx.u32Cols) {
            auto pos = name.rfind('_');
            if (pos != std::string::npos) {
                std::string suffix = name.substr(pos + 1);
                bool allDigits = !suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit);
                if (allDigits && name.substr(0, pos) == baseName) return name;
            }
        }
        for (const auto& [name, _] : ctx.f32Cols) {
            auto pos = name.rfind('_');
            if (pos != std::string::npos) {
                std::string suffix = name.substr(pos + 1);
                bool allDigits = !suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit);
                if (allDigits && name.substr(0, pos) == baseName) return name;
            }
        }
        return "";
    };

    std::function<TypedExprPtr(const TypedExprPtr&)> rewriteResidualCols;
    rewriteResidualCols = [&](const TypedExprPtr& expr) -> TypedExprPtr {
        if (!expr) return expr;
        if (expr->kind == TypedExpr::Kind::Compare) {
            auto& cmp = expr->asCompare();
            bool bothCols = (cmp.left && cmp.right &&
                             cmp.left->kind == TypedExpr::Kind::Column &&
                             cmp.right->kind == TypedExpr::Kind::Column);
            if (bothCols) {
                const std::string& lName = cmp.left->asColumn().column;
                const std::string& rName = cmp.right->asColumn().column;

                // Find actual column names in left/right contexts
                std::string leftActual = findColInCtx(leftCtx, lName);
                std::string rightActual = findColInCtx(rightCtx, rName);

                // Map right-side column through rightColumnMapping (rename after join)
                std::string rightInOut = rightActual;
                if (!rightActual.empty() && rightColumnMapping.count(rightActual)) {
                    rightInOut = rightColumnMapping.at(rightActual);
                }

                // Use the actual names if we found them
                std::string newLName = !leftActual.empty() ? leftActual : lName;
                std::string newRName = !rightInOut.empty() ? rightInOut : rName;

                if (newLName != lName || newRName != rName) {
                    if (debug) {
                        LOG_INFO("Exec", "Join: rewrote residual col: "  << lName << " -> " << newLName << ", " << rName << " -> " << newRName);
                    }
                    return TypedExpr::compare(cmp.op, 
                                              TypedExpr::column(newLName), 
                                              TypedExpr::column(newRName));
                }
            }
            return expr;
        }
        if (expr->kind == TypedExpr::Kind::Binary) {
            auto& bin = expr->asBinary();
            auto newLeft = rewriteResidualCols(bin.left);
            auto newRight = rewriteResidualCols(bin.right);
            if (newLeft != bin.left || newRight != bin.right) {
                return TypedExpr::binary(bin.op, newLeft, newRight);
            }
            return expr;
        }
        return expr;
    };

    TypedExprPtr rewrittenFilter = rewriteResidualCols(postJoinFilter);
    if (debug && rewrittenFilter != postJoinFilter) {
        LOG_INFO("Exec", "Join: rewrote post-join filter column references\n");
    }

    // Upload CPU columns to GPU for filtering (they were gathered from join)
    for (const auto& [name, vec] : outCtx.u32Cols) {
        if (!vec.empty() && outCtx.u32ColsGPU.find(name) == outCtx.u32ColsGPU.end()) {
            GpuBuffer buf = GpuOps::createBuffer(vec.data(), vec.size() * sizeof(uint32_t));
            if (buf) outCtx.u32ColsGPU[name].reset(buf);
        }
    }
    for (const auto& [name, vec] : outCtx.f32Cols) {
        if (!vec.empty() && outCtx.f32ColsGPU.find(name) == outCtx.f32ColsGPU.end()) {
            GpuBuffer buf = GpuOps::createBuffer(vec.data(), vec.size() * sizeof(float));
            if (buf) outCtx.f32ColsGPU[name].reset(buf);
        }
    }

    // Use the regular filter infrastructure
    IRFilter postFilter{rewrittenFilter, ""};
    bool filterOk = GpuExecutor::executeFilter(postFilter, outCtx);
    if (debug) {
        LOG_ERROR("Exec", "Join: post-join filter " << (filterOk?"ok":"failed")  << ", rows after=" << outCtx.rowCount);
    }
}

// Detects cross-joins (1=1, lit-vs-lit, non-equality) and separates residual
// non-equality conditions into a post-join filter.
struct JoinKeyExtraction {
    std::vector<std::pair<std::string, std::string>> keyPairs;
    bool isCrossJoin = false;
    bool hasPostJoinFilter = false;
    TypedExprPtr postJoinFilter = nullptr;
    bool failed = false;  // true → caller should return false
};

static JoinKeyExtraction extractJoinConditionKeys(const IRJoin& join, bool debug) {
    JoinKeyExtraction out;

    if (join.conditionStr == "1=1") {
        out.isCrossJoin = true;
    } else if (join.condition && join.condition->kind == TypedExpr::Kind::Compare) {
        const auto& cmp = join.condition->asCompare();
        if (cmp.op == CompareOp::Eq &&
            cmp.left->kind == TypedExpr::Kind::Literal &&
            cmp.right->kind == TypedExpr::Kind::Literal) {
            out.isCrossJoin = true;
        } else if (cmp.op != CompareOp::Eq) {
            out.isCrossJoin = true;
            out.hasPostJoinFilter = true;
            out.postJoinFilter = join.condition;
            if (debug) {
                LOG_INFO("Exec", "Join: detected non-equality condition, treating as cross-join + filter\n");
                LOG_INFO("Exec", "Join: conditionStr=" << join.conditionStr);
            }
        }
    }

    if (join.condition && !out.isCrossJoin) {
        std::vector<TypedExprPtr> residuals;
        extractJoinKeyPairs(join.condition, out.keyPairs, &residuals);
        if (!out.keyPairs.empty() && !residuals.empty()) {
            out.hasPostJoinFilter = true;
            if (residuals.size() == 1) {
                out.postJoinFilter = residuals[0];
            } else {
                TypedExprPtr combined = residuals[0];
                for (size_t ri = 1; ri < residuals.size(); ++ri)
                    combined = TypedExpr::binary(BinaryOp::And, combined, residuals[ri]);
                out.postJoinFilter = combined;
            }
            if (debug)
                LOG_INFO("Exec", "Join: extracted " << residuals.size() << " residual condition(s) as post-join filter\n");
        }
    }

    // Fallback: parse from condition string
    if (!out.isCrossJoin && out.keyPairs.empty()) {
        std::string cond = join.conditionStr;
        size_t pos = 0;
        while (pos < cond.size()) {
            size_t andPos = std::string::npos;
            for (size_t j = pos; j + 5 <= cond.size(); ++j) {
                if (cond[j] == ' ' &&
                    (cond[j+1] == 'A' || cond[j+1] == 'a') &&
                    (cond[j+2] == 'N' || cond[j+2] == 'n') &&
                    (cond[j+3] == 'D' || cond[j+3] == 'd') &&
                    cond[j+4] == ' ') {
                    andPos = j;
                    break;
                }
            }
            std::string part;
            if (andPos != std::string::npos) {
                part = cond.substr(pos, andPos - pos);
                pos = andPos + 5;
            } else {
                part = cond.substr(pos);
                pos = cond.size();
            }
            auto eq = part.find('=');
            if (eq != std::string::npos) {
                std::string left = base_ident(part.substr(0, eq));
                std::string right = base_ident(part.substr(eq + 1));
                if (!left.empty() && !right.empty())
                    out.keyPairs.emplace_back(left, right);
            }
        }
    }

    // Complex non-equality fallback
    if (!out.isCrossJoin && out.keyPairs.empty() && join.condition) {
        LOG_DEBUG("Exec", "Join: no equi-join keys found but has condition, treating as cross-join + filter\n");
        out.isCrossJoin = true;
        out.hasPostJoinFilter = true;
        out.postJoinFilter = join.condition;
    }

    if (!out.isCrossJoin && out.keyPairs.empty()) {
        LOG_DEBUG("Exec", "Join: no key pairs found\n");
        out.failed = true;
    }

    return out;
}

// ---------- postProcessSemiAntiJoin ----------
// Deduplicates SEMI join results and finds unmatched rows for ANTI joins.
static void postProcessSemiAntiJoin(
    JoinResult& jRes, uint32_t& resCount,
    uint32_t lCount, uint32_t rCount,
    bool isSemiJoin, bool isAntiJoin, bool gpuSemiAntiDone,
    const IRJoin& join, bool /*debug*/)
{
    auto& store = GpuColumnStore::instance();

    // SEMI JOIN: deduplicate probeIndices, keeping first match per probe row.
    if (isSemiJoin && resCount > 0 && jRes.probeIndices && !gpuSemiAntiDone) {
        LOG_DEBUG("Exec", "Semi Join: Deduplicating " << resCount << " probe indices\n");
        std::vector<MTL::Buffer*> dedupKeys = { jRes.probeIndices };
        uint32_t uniqueCount = 0;
        GpuBuffer uniqueIdx = GpuOps::dedupByKeys(dedupKeys, resCount, uniqueCount);
        if (uniqueIdx && uniqueCount > 0) {
            auto newProbe = GpuOps::gatherU32(jRes.probeIndices, uniqueIdx, uniqueCount, false);
            auto newBuild = GpuOps::gatherU32(jRes.buildIndices, uniqueIdx, uniqueCount, false);
            GpuOps::sync();
            if (newProbe && newBuild) {
                jRes.probeIndices = std::move(newProbe);
                jRes.buildIndices = std::move(newBuild);
                resCount = uniqueCount;
                jRes.count = uniqueCount;
            }
        } else if (uniqueIdx) {
        }
        LOG_DEBUG("Exec", "Semi Join: After dedup: " << resCount << " unique rows\n");
    }

    // ANTI JOIN: Find rows that did NOT match.
    if (isAntiJoin && rCount > 0 && jRes.probeIndices && !gpuSemiAntiDone) {
        if (join.rightVariant) {
            // RIGHT ANTI: find build (right) rows not matched
            LOG_DEBUG("Exec", "Right Anti Join: Finding non-matching rows from " << rCount << " right rows, " << resCount << " matches\n");
            auto antiRes = GpuOps::findUnmatchedIndices(jRes.buildIndices, resCount, rCount);
            uint32_t antiCount = antiRes.count;
            LOG_DEBUG("Exec", "Right Anti Join: " << antiCount << " non-matching right rows\n");
            jRes.buildIndices = std::move(antiRes.indices);
            jRes.probeIndices.reset(store.device()->newBuffer(
                std::max(antiCount, 1u) * sizeof(uint32_t), MTL::ResourceStorageModeShared));
            std::memset(jRes.probeIndices->contents(), 0, std::max(antiCount, 1u) * sizeof(uint32_t));
            resCount = antiCount;
            jRes.count = antiCount;
        } else {
            // LEFT ANTI (default): find probe (left) rows not matched
            LOG_DEBUG("Exec", "Anti Join: Finding non-matching rows from " << lCount << " left rows, " << resCount << " matches\n");
            auto antiRes = GpuOps::findUnmatchedIndices(jRes.probeIndices, resCount, lCount);
            uint32_t antiCount = antiRes.count;
            LOG_DEBUG("Exec", "Anti Join: " << antiCount << " non-matching rows\n");
            jRes.probeIndices = std::move(antiRes.indices);
            jRes.buildIndices.reset(store.device()->newBuffer(
                std::max(antiCount, 1u) * sizeof(uint32_t), MTL::ResourceStorageModeShared));
            std::memset(jRes.buildIndices->contents(), 0, std::max(antiCount, 1u) * sizeof(uint32_t));
            resCount = antiCount;
            jRes.count = antiCount;
        }
    }
}

// ---------------------------------------------------------------------------
// Resolve join key columns from keyPairs into (leftCol, rightCol) pairs.
// Returns resolved pairs or empty + failed=true on error.
// ---------------------------------------------------------------------------
struct ResolvedJoinKeys {
    std::vector<std::pair<std::string, std::string>> keys;
    bool failed = false;
};

static ResolvedJoinKeys resolveJoinKeyColumns(
    std::vector<std::pair<std::string, std::string>>& keyPairs,
    EvalContext& leftCtx, EvalContext& rightCtx, bool debug)
{
    ResolvedJoinKeys result;
    std::unordered_set<std::string> usedLeftCols, usedRightCols;

    for (auto& [k1, k2] : keyPairs) {
        if (k1 == "supplier_no") k1 = "l_suppkey";
        if (k2 == "supplier_no") k2 = "l_suppkey";

        std::string k1Left  = findColWithSuffix(leftCtx, k1);
        std::string k2Right = findColWithSuffix(rightCtx, k2);
        std::string k2Left  = findColWithSuffix(leftCtx, k2);
        std::string k1Right = findColWithSuffix(rightCtx, k1);

        bool k1InLeft  = !k1Left.empty();
        bool k2InRight = !k2Right.empty();
        bool k2InLeft  = !k2Left.empty();
        bool k1InRight = !k1Right.empty();

        if (k1InLeft && k2InRight) {
            result.keys.emplace_back(k1Left, k2Right);
            usedLeftCols.insert(k1Left);
            usedRightCols.insert(k2Right);
        } else if (k2InLeft && k1InRight) {
            result.keys.emplace_back(k2Left, k1Right);
            usedLeftCols.insert(k2Left);
            usedRightCols.insert(k1Right);
        } else {
            std::string leftResolved, rightResolved;

            if (k1InRight) {
                rightResolved = k1Right;
                leftResolved = fuzzyResolveColumn(leftCtx, k2, usedLeftCols);
            } else if (k2InRight) {
                rightResolved = k2Right;
                leftResolved = fuzzyResolveColumn(leftCtx, k1, usedLeftCols);
            }

            if (leftResolved.empty() && rightResolved.empty()) {
                if (k1InLeft) {
                    leftResolved = k1Left;
                    rightResolved = fuzzyResolveColumn(rightCtx, k2, usedRightCols);
                } else if (k2InLeft) {
                    leftResolved = k2Left;
                    rightResolved = fuzzyResolveColumn(rightCtx, k1, usedRightCols);
                }
            }

            if (!leftResolved.empty() && !rightResolved.empty()) {
                result.keys.emplace_back(leftResolved, rightResolved);
                usedLeftCols.insert(leftResolved);
                usedRightCols.insert(rightResolved);
                if (debug) {
                    LOG_INFO("Exec", "Join: fuzzy resolved " << k1 << "=" << k2 << " to (" << leftResolved << ", " << rightResolved << ")\n");
                }
            } else {
                if (debug) {
                    LOG_WARN("Exec", "Join: cannot resolve key pair " << k1 << "=" << k2 << " k1InLeft=" << k1InLeft << " k2InRight=" << k2InRight << " k2InLeft=" << k2InLeft << " k1InRight=" << k1InRight);
                }
                result.failed = true;
                return result;
            }
        }
    }

    LOG_DEBUG("Exec", "Join: resolved " << result.keys.size() << " key pair(s)\n");
    return result;
}

// -- Extracted: buildCrossJoinIndices --
// Creates Cartesian product index buffers for a cross join on GPU.
static JoinResult buildCrossJoinIndices(
    EvalContext& leftCtx, EvalContext& rightCtx,
    uint32_t lCount, uint32_t rCount, bool /*debug*/) {
    LOG_DEBUG("Exec", "GPU Join: Cross Join 1=1 (" << lCount << " x " << rCount << ")\n");
    uint64_t totalCount = (uint64_t)lCount * (uint64_t)rCount;

    auto device = GpuColumnStore::instance().device();
    if (totalCount > UINT32_MAX) {
        LOG_WARN("Exec", "WARNING: Cross join produces " << totalCount << " rows, exceeding uint32_t max. Clamping to " << UINT32_MAX << ".\n");
        totalCount = UINT32_MAX;
    }
    JoinResult jRes;
    jRes.count = (uint32_t)totalCount;
    jRes.probeIndices.reset(device->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    jRes.buildIndices.reset(device->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared));

    MTL::Buffer* lIndicesGPU = leftCtx.activeRowsGPU;
    MTL::Buffer* rIndicesGPU = rightCtx.activeRowsGPU;
    bool createdL = false, createdR = false;
    if (!lIndicesGPU) { lIndicesGPU = GpuOps::iotaU32(lCount).detach(); createdL = true; }
    if (!rIndicesGPU) { rIndicesGPU = GpuOps::iotaU32(rCount).detach(); createdR = true; }

    GpuOps::crossProduct(lIndicesGPU, rIndicesGPU,
                         jRes.probeIndices, jRes.buildIndices,
                         lCount, rCount);

    if (createdL) lIndicesGPU->release();
    if (createdR) rIndicesGPU->release();
    return jRes;
}

// -- Extracted: swapDelimCorrelationColumns --
// Swaps data in outCtx so DELIM correlation columns keep their primary (un-suffixed) names.
static void swapDelimCorrelationColumns(
    EvalContext& outCtx,
    const std::unordered_set<std::string>& delimCols,
    const std::unordered_map<std::string, std::string>& rightColumnMapping,
    bool /*debug*/) {
    for (const auto& col : delimCols) {
        auto rmIt = rightColumnMapping.find(col);
        if (rmIt != rightColumnMapping.end() && rmIt->second != col) {
            const std::string& renamedCol = rmIt->second;
            if (outCtx.u32Cols.count(col) && outCtx.u32Cols.count(renamedCol)) {
                std::swap(outCtx.u32Cols[col], outCtx.u32Cols[renamedCol]);
                LOG_DEBUG("Exec", "Join: DELIM priority swap: " << col << " <-> " << renamedCol);
            }
            if (outCtx.u32ColsGPU.count(col) && outCtx.u32ColsGPU.count(renamedCol))
                std::swap(outCtx.u32ColsGPU[col], outCtx.u32ColsGPU[renamedCol]);
            if (outCtx.f32Cols.count(col) && outCtx.f32Cols.count(renamedCol))
                std::swap(outCtx.f32Cols[col], outCtx.f32Cols[renamedCol]);
            if (outCtx.f32ColsGPU.count(col) && outCtx.f32ColsGPU.count(renamedCol))
                std::swap(outCtx.f32ColsGPU[col], outCtx.f32ColsGPU[renamedCol]);
            if (outCtx.dictCols.count(col) && outCtx.dictCols.count(renamedCol))
                std::swap(outCtx.dictCols[col], outCtx.dictCols[renamedCol]);
            if (outCtx.stringCols.count(col) && outCtx.stringCols.count(renamedCol))
                std::swap(outCtx.stringCols[col], outCtx.stringCols[renamedCol]);
        }
    }
    outCtx.isDelimCorrelation.clear();
}

bool GpuExecutor::executeJoin(const IRJoin& join,
                                   EvalContext& leftCtx, EvalContext& rightCtx, EvalContext& outCtx) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    // Supported: INNER, LEFT, RIGHT, SEMI, ANTI, MARK (MARK treated as SEMI)
    if (join.type != JoinType::Inner && join.type != JoinType::Left &&
        join.type != JoinType::Right && join.type != JoinType::Semi && 
        join.type != JoinType::Anti && join.type != JoinType::Mark) {
        LOG_DEBUG("Exec", "Join: unsupported join type\n");
        return false;
    }
    
    const bool isLeftJoin = (join.type == JoinType::Left);
    const bool isRightJoin = (join.type == JoinType::Right);
    const bool isSemiJoin = (join.type == JoinType::Semi || join.type == JoinType::Mark);
    const bool isAntiJoin = (join.type == JoinType::Anti);

    // Early exit: if either side has zero rows, the join result is trivially empty
    // (ANTI join with empty left still produces 0 rows; LEFT with empty left = 0 rows)
    {
        uint32_t earlyL = leftCtx.activeRowsCountGPU > 0 ? leftCtx.activeRowsCountGPU : (uint32_t)leftCtx.rowCount;
        uint32_t earlyR = rightCtx.activeRowsCountGPU > 0 ? rightCtx.activeRowsCountGPU : (uint32_t)rightCtx.rowCount;
        if (earlyL == 0 || earlyR == 0) {
            LOG_DEBUG("Exec", "Join: empty input (left=" << earlyL << " right=" << earlyR << "), returning 0-row result\n");
            outCtx.rowCount = 0;
            outCtx.activeRows.clear();
            outCtx.activeRowsGPU.reset();
            outCtx.activeRowsCountGPU = 0;
            return true;
        }
    }
    
    if (debug) {
        LOG_INFO("Exec", "Join: type=" << static_cast<int>(join.type)  << " isLeft=" << isLeftJoin << " isRight=" << isRightJoin  << " isSemi=" << isSemiJoin << " isAnti=" << isAntiJoin);
        LOG_DEBUG("Exec", "Join: leftCtx has " << leftCtx.u32Cols.size() << " u32 cols, " << leftCtx.f32Cols.size() << " f32 cols, " << leftCtx.rowCount << " rows");
        if (debug) for (const auto& [k, v] : leftCtx.u32Cols) std::cerr << " " << k;
        LOG_DEBUG("JOIN", "\n");
        LOG_DEBUG("Exec", "Join: rightCtx has " << rightCtx.u32Cols.size() << " u32 cols, " << rightCtx.f32Cols.size() << " f32 cols, " << rightCtx.rowCount << " rows");
        if (debug) for (const auto& [k, v] : rightCtx.u32Cols) std::cerr << " " << k;
        LOG_DEBUG("JOIN", "\n");
    }
    
    // Extract all join key pairs from the condition
    auto keyExtraction = extractJoinConditionKeys(join, debug);
    if (keyExtraction.failed) return false;
    
    std::vector<std::pair<std::string, std::string>>& keyPairs = keyExtraction.keyPairs;
    bool isCrossJoin = keyExtraction.isCrossJoin;
    bool hasPostJoinFilter = keyExtraction.hasPostJoinFilter;
    TypedExprPtr postJoinFilter = std::move(keyExtraction.postJoinFilter);
    
    if (debug) {
        LOG_INFO("Exec", "Join: " << keyPairs.size() << " key pair(s):\n");
        for (const auto& [l, r] : keyPairs) {
            LOG_INFO("Exec", "Join:   " << l << " = " << r);
        }
        LOG_DEBUG("Exec", "Join: leftCtx has " << leftCtx.u32Cols.size() << " u32 cols, "  << leftCtx.f32Cols.size() << " f32 cols, " << leftCtx.rowCount << " rows");
        if (debug) for (const auto& [n,_] : leftCtx.u32Cols) std::cerr << " " << n;
        LOG_DEBUG("JOIN", std::endl);
        LOG_DEBUG("Exec", "Join: rightCtx has " << rightCtx.u32Cols.size() << " u32 cols, " << rightCtx.f32Cols.size() << " f32 cols, " << rightCtx.rowCount << " rows");
        if (debug) for (const auto& [n,_] : rightCtx.u32Cols) std::cerr << " " << n;
        LOG_DEBUG("JOIN", std::endl);
    }
    
    // Resolve join key columns (with suffix fallback, fuzzy matching, f32→u32 conversion)
    auto resolved = resolveJoinKeyColumns(keyPairs, leftCtx, rightCtx, debug);
    if (resolved.failed) return false;

    std::vector<std::pair<std::string, std::string>>& resolvedKeys = resolved.keys;

    // Get vectors for all keys (lazy-fetch from GPU if CPU vec is empty)
    std::vector<const std::vector<uint32_t>*> leftKeyVecs, rightKeyVecs;
    for (const auto& [lk, rk] : resolvedKeys) {
        // Lazy fetch left key from GPU if needed
        if (leftCtx.u32Cols[lk].empty() && leftCtx.u32ColsGPU.count(lk)) {
            MTL::Buffer* buf = leftCtx.u32ColsGPU.at(lk);
            size_t count = buf->length() / sizeof(uint32_t);
            if (count > 0) {
                std::vector<uint32_t> down(count);
                std::memcpy(down.data(), buf->contents(), count * sizeof(uint32_t));
                leftCtx.u32Cols[lk] = std::move(down);
            }
        }
        // Lazy fetch right key from GPU if needed
        if (rightCtx.u32Cols[rk].empty() && rightCtx.u32ColsGPU.count(rk)) {
            MTL::Buffer* buf = rightCtx.u32ColsGPU.at(rk);
            size_t count = buf->length() / sizeof(uint32_t);
            if (count > 0) {
                std::vector<uint32_t> down(count);
                std::memcpy(down.data(), buf->contents(), count * sizeof(uint32_t));
                rightCtx.u32Cols[rk] = std::move(down);
            }
        }
        leftKeyVecs.push_back(&leftCtx.u32Cols.at(lk));
        rightKeyVecs.push_back(&rightCtx.u32Cols.at(rk));
    }

    if (!isCrossJoin && resolvedKeys.empty()) return false;

    if (!isCrossJoin && resolvedKeys.size() > 2) ENGINE_THROW("GPU Join > 2 columns not implemented");

    auto& store = GpuColumnStore::instance();
    
    // ensureGPU is now the static function ensureColumnOnGPU (extracted above)
    auto ensureGPU = [&](EvalContext& ctx, const std::string& col) -> MTL::Buffer* {
        return ensureColumnOnGPU(ctx, col, debug);
    };
    
    // --- Materialize contexts before join (compact to dense form) ---
    materializeJoinContext(leftCtx, "left", debug);
    materializeJoinContext(rightCtx, "right", debug);
    
    uint32_t rCount = (uint32_t)rightCtx.rowCount;
    uint32_t lCount = (uint32_t)leftCtx.rowCount;

    MTL::Buffer* lBuf = nullptr;
    MTL::Buffer* rBuf = nullptr;
    
    JoinResult jRes;
    
    if (lCount > 0 && rCount > 0) {
        if (isCrossJoin) {
            jRes = buildCrossJoinIndices(leftCtx, rightCtx, lCount, rCount, debug);
        } else if (resolvedKeys.size() == 2) {
            if (debug) {
                LOG_INFO("Exec", "Multi-Key Join (2 keys)\n");
                LOG_INFO("Exec", "Multi-Key Join: key0=(" << resolvedKeys[0].first << ", " << resolvedKeys[0].second << ")\n");
                LOG_INFO("Exec", "Multi-Key Join: key1=(" << resolvedKeys[1].first << ", " << resolvedKeys[1].second << ")\n");
            }
            MTL::Buffer* l1 = ensureGPU(leftCtx, resolvedKeys[0].first);
            MTL::Buffer* r1 = ensureGPU(rightCtx, resolvedKeys[0].second);
            MTL::Buffer* l2 = ensureGPU(leftCtx, resolvedKeys[1].first);
            MTL::Buffer* r2 = ensureGPU(rightCtx, resolvedKeys[1].second);
            if(!l1||!r1||!l2||!r2) ENGINE_THROW("Missing GPU col data for multi-key join");
            
            uint32_t lSize = (uint32_t)leftCtx.rowCount;
            uint32_t rSize = (uint32_t)rightCtx.rowCount;
            
            LOG_DEBUG("Exec", "Multi-Key Join: packing left (" << lSize << " rows)...\n");
            lBuf = GpuOps::packU32ToU64(l1, l2, lSize).detach();
            LOG_DEBUG("Exec", "Multi-Key Join: packing right (" << rSize << " rows)...\n");
            rBuf = GpuOps::packU32ToU64(r1, r2, rSize).detach();
            LOG_DEBUG("Exec", "Multi-Key Join: packing done.\n");
        } else {
            lBuf = ensureGPU(leftCtx, resolvedKeys[0].first);
            rBuf = ensureGPU(rightCtx, resolvedKeys[0].second);
        }

        if (!isCrossJoin && (!lBuf || !rBuf)) ENGINE_THROW("Missing GPU column data for Join");
        
        // Multi-match hash join; right=build, left=probe.
        
        if (!isCrossJoin && debug) LOG_DEBUG("Exec", "GPU Join: Build (" << rCount << "), Probe (" << lCount << ")\n");
        if (debug) {
            LOG_ERROR("Exec", "GPU Join: leftCtx.activeRowsGPU=" << (leftCtx.activeRowsGPU ? "SET" : "NULL") << " rightCtx.activeRowsGPU=" << (rightCtx.activeRowsGPU ? "SET" : "NULL"));
            if (leftCtx.activeRowsGPU) {
                uint32_t* leftIndices = static_cast<uint32_t*>(leftCtx.activeRowsGPU->contents());
                LOG_DEBUG("Exec", "GPU Join: leftActiveIndices first 5: ");
                if (debug) for (uint32_t i = 0; i < std::min(5u, leftCtx.activeRowsCountGPU); ++i) std::cerr << leftIndices[i] << " ";
                LOG_DEBUG("JOIN", "\n");
            }
        }
        
        if (!isCrossJoin) {
            MTL::Buffer* buildActiveRows = rightCtx.activeRowsGPU;
            MTL::Buffer* probeActiveRows = leftCtx.activeRowsGPU;
            
            // GPU ANTI join shortcut: build HT from right, probe left, invert mask
            if (isAntiJoin && resolvedKeys.size() == 1) {
                if (join.rightVariant) {
                    // RIGHT ANTI: find right rows NOT matching left
                    auto antiRes = GpuOps::hashJoinAntiU32(rBuf, rCount, lBuf, lCount);
                    if (!antiRes) ENGINE_THROW("GPU hashJoinAntiU32 failed");
                    jRes.count = antiRes->count;
                    jRes.buildIndices = std::move(antiRes->indices);
                    jRes.probeIndices.reset(store.device()->newBuffer(
                        std::max(antiRes->count, 1u) * sizeof(uint32_t), MTL::ResourceStorageModeShared));
                    std::memset(jRes.probeIndices->contents(), 0,
                                std::max(antiRes->count, 1u) * sizeof(uint32_t));
                } else {
                    // LEFT ANTI: find left rows NOT matching right
                    auto antiRes = GpuOps::hashJoinAntiU32(lBuf, lCount, rBuf, rCount);
                    if (!antiRes) ENGINE_THROW("GPU hashJoinAntiU32 failed");
                    jRes.count = antiRes->count;
                    jRes.probeIndices = std::move(antiRes->indices);
                    jRes.buildIndices.reset(store.device()->newBuffer(
                        std::max(antiRes->count, 1u) * sizeof(uint32_t), MTL::ResourceStorageModeShared));
                    std::memset(jRes.buildIndices->contents(), 0,
                                std::max(antiRes->count, 1u) * sizeof(uint32_t));
                }
                LOG_DEBUG("Exec", "GPU Anti Join (direct): " << jRes.count << " non-matching rows\n");
            }
            // GPU SEMI join shortcut: build HT from right, probe left, compact matches
            else if (isSemiJoin && resolvedKeys.size() == 1) {
                auto semiRes = GpuOps::hashJoinSemiU32(lBuf, lCount, rBuf, rCount);
                if (!semiRes) ENGINE_THROW("GPU hashJoinSemiU32 failed");
                jRes.count = semiRes->count;
                jRes.probeIndices = std::move(semiRes->indices);
                // Build indices: use iota as placeholder (not needed for semi-only output,
                // but downstream might gather right columns for MARK join)
                jRes.buildIndices.reset(store.device()->newBuffer(
                    std::max(semiRes->count, 1u) * sizeof(uint32_t), MTL::ResourceStorageModeShared));
                std::memset(jRes.buildIndices->contents(), 0,
                            std::max(semiRes->count, 1u) * sizeof(uint32_t));
                LOG_DEBUG("Exec", "GPU Semi Join (direct): " << jRes.count << " matched left rows\n");
            }
            else if (resolvedKeys.size() == 2) {
                 jRes = GpuOps::joinHashU64(rBuf, buildActiveRows, rCount, lBuf, probeActiveRows, lCount);
                 lBuf->release(); rBuf->release();
            } else {
                 jRes = GpuOps::joinHash(rBuf, rCount, lBuf, lCount);
            }
            

        }
    } else {
        if (lCount > 0 && rCount == 0 && (isAntiJoin || isLeftJoin)) {
             LOG_DEBUG("Exec", "GPU Join: Empty Build side for Anti/Left Join -> Returning all " << lCount << " left rows.\n");
             jRes.count = lCount;
             
             if (leftCtx.activeRowsGPU) {
                 MTL::Buffer* src = leftCtx.activeRowsGPU;
                 jRes.probeIndices.reset(store.device()->newBuffer(src->contents(), src->length(), MTL::ResourceStorageModeShared));
             } else {
                 std::vector<uint32_t> seq(lCount);
                 std::iota(seq.begin(), seq.end(), 0);
                 jRes.probeIndices.reset(store.device()->newBuffer(seq.data(), seq.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared));
             }
             
             // Placeholder build indices (required non-null)
             jRes.buildIndices.reset(store.device()->newBuffer(lCount * sizeof(uint32_t), MTL::ResourceStorageModeShared));
             std::memset(jRes.buildIndices->contents(), 0, lCount * sizeof(uint32_t));
        } else {
            LOG_DEBUG("Exec", "GPU Join: Skipping (Build=" << rCount << ", Probe=" << lCount << ")\n");
            jRes.count = 0;
            jRes.buildIndices = nullptr;
            jRes.probeIndices = nullptr;
        }
    }
                                       
    if ((lCount > 0 && rCount > 0) && !jRes.buildIndices) ENGINE_THROW("GPU Join Kernel Failed");
    
    LOG_DEBUG("Exec", "GPU Join Success: Result " << jRes.count << " rows.\n");
    
    uint32_t resCount = jRes.count;
    
    // Flag: skip CPU-based SEMI/ANTI post-processing if GPU shortcut was used
    bool gpuSemiAntiDone = (lCount > 0 && rCount > 0 && !isCrossJoin && resolvedKeys.size() == 1 &&
                            (isAntiJoin || isSemiJoin));
    
    // SEMI/ANTI join post-processing (dedup + unmatched row finding)
    postProcessSemiAntiJoin(jRes, resCount, lCount, rCount,
                            isSemiJoin, isAntiJoin, gpuSemiAntiDone, join, debug);
    
    // RIGHT ANTI: Only gather from right side (the build side)
    bool rightAntiGather = (isAntiJoin && join.rightVariant);
    
    // Scatter/gather all columns into output context
    std::unordered_map<std::string, std::string> rightColumnMapping;
    bool earlyReturn = scatterJoinOutputColumns(
        leftCtx, rightCtx, outCtx, jRes, resCount, lCount, rCount,
        isAntiJoin, isSemiJoin, rightAntiGather, rightColumnMapping, debug);
    if (earlyReturn) return true;

    if (debug) {
        LOG_INFO("Exec", "Join: After string gather, outCtx.stringCols sizes:\n");
        for (const auto& [name, vec] : outCtx.stringCols) {
            LOG_INFO("Exec", "stringCol " << name << " size=" << vec.size());
        }
    }

    // LEFT JOIN: Append unmatched left rows with NULL/0 for right columns
    if (isLeftJoin && rCount > 0 && resCount > 0) {
        appendUnmatchedLeftRows(leftCtx, outCtx, jRes, resCount, lCount, rightColumnMapping, debug);
    }

    // RIGHT JOIN: Append unmatched right (build) rows with NULL/0 for left columns
    if (isRightJoin && rCount > 0 && resCount > 0) {
        appendUnmatchedRightRows(rightCtx, outCtx, jRes, resCount, rCount, rightColumnMapping, debug);
    }

    // jRes goes out of scope — GpuBuffer handles release
    
    // If the right side had DELIM correlation columns that were renamed due to collision,
    // swap the data so the correlation columns keep their primary (un-suffixed) names.
    if (!rightCtx.isDelimCorrelation.empty()) {
        swapDelimCorrelationColumns(outCtx, rightCtx.isDelimCorrelation, rightColumnMapping, debug);
    }
    
    // Apply post-join filter for non-equality conditions (e.g., l_suppkey != l_suppkey)
    if (hasPostJoinFilter && postJoinFilter && outCtx.rowCount > 0) {
        applyPostJoinFilter(postJoinFilter, leftCtx, rightCtx, rightColumnMapping, outCtx, debug);
    }
    
    // Rebuild flat string buffers and dictionary encoding for downstream operators
    // Skip columns that already have a valid dictCol (GPU-native path)
    if (outCtx.rowCount > 0) {
        for (const auto& [name, vec] : outCtx.stringCols) {
            if (!vec.empty() && !outCtx.hasDictCol(name)) {
                flattenStringCol(outCtx, name);
                buildDictCol(outCtx, name);
            }
        }
    }
    
    return true;
}

} // namespace engine
