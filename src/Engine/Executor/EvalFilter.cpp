#include "GpuExecutorDetail.hpp"
#include "EngineError.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include "Logger.hpp"

namespace engine {

// ============================================================================
// Operator Implementations
// ============================================================================

static std::optional<engine::GpuFilterOp> mapToGpuFilterOp(engine::CompareOp op) {
    switch (op) {
        case engine::CompareOp::Eq: return engine::GpuFilterOp::EQ;
        case engine::CompareOp::Ne: return engine::GpuFilterOp::NE;
        case engine::CompareOp::Lt: return engine::GpuFilterOp::LT;
        case engine::CompareOp::Le: return engine::GpuFilterOp::LE;
        case engine::CompareOp::Gt: return engine::GpuFilterOp::GT;
        case engine::CompareOp::Ge: return engine::GpuFilterOp::GE;
        default: return std::nullopt;
    }
}

// ============================================================================
// Extracted helpers for executeFilterRecursive / evaluateExpression
// ============================================================================

// Handles IN operator by rewriting to a chain of ORed equality comparisons.
// Supports substring(col, start, len) IN (...) via GPU flat-buffer fast path.
static bool filterCompareInList(const CompareExpr& cmp, EvalContext& ctx,
                                bool debug) {
    const TypedExpr* leftExprRaw = unwrapExpr(cmp.left.get());
    if (cmp.inList.empty()) {
        // IN () -> False / Empty
        ctx.activeRowsGPU= GpuOps::createBuffer(nullptr, 4);
        ctx.activeRowsCountGPU = 0;
        return true;
    }

    bool isCol = (leftExprRaw->kind == TypedExpr::Kind::Column);
    std::string colName;
    if (isCol) colName = leftExprRaw->asColumn().column;
    else if (leftExprRaw->kind == TypedExpr::Kind::Function) {
        const auto& fn = leftExprRaw->asFunction();
        std::string fnName = fn.name;
        std::transform(fnName.begin(), fnName.end(), fnName.begin(), ::tolower);

        if ((fnName == "substring" || fnName == "substr") && fn.args.size() >= 1) {
            const TypedExpr* arg0 = unwrapExpr(fn.args[0].get());
            if (arg0->kind == TypedExpr::Kind::Column) {
                std::string baseCol = arg0->asColumn().column;
                const std::vector<std::string>* vec = nullptr;
                ctx.ensureStringCol(baseCol);
                if (ctx.stringCols.count(baseCol)) vec = &ctx.stringCols.at(baseCol);
                else {
                    std::string resolved = ctx.resolveColName(baseCol);
                    if (!resolved.empty()) {
                        ctx.ensureStringCol(resolved);
                        if (ctx.stringCols.count(resolved)) { vec = &ctx.stringCols.at(resolved); baseCol = resolved; }
                    }
                }

                int start = 1, len = -1;
                if (fn.args.size() >= 2 && fn.args[1]->kind == TypedExpr::Kind::Literal) {
                    if (std::holds_alternative<int64_t>(fn.args[1]->asLiteral().value))
                        start = (int)std::get<int64_t>(fn.args[1]->asLiteral().value);
                }
                if (fn.args.size() >= 3 && fn.args[2]->kind == TypedExpr::Kind::Literal) {
                    if (std::holds_alternative<int64_t>(fn.args[2]->asLiteral().value))
                        len = (int)std::get<int64_t>(fn.args[2]->asLiteral().value);
                }

                if (vec) {
                    // Prefer GPU substring via flat buffers (zero-copy offset adjustment)
                    auto fit = ctx.flatStringCols.find(baseCol);
                    if (fit != ctx.flatStringCols.end() && fit->second.rowCount == vec->size()) {
                        uint32_t gpuStart = (uint32_t)start;
                        uint32_t gpuLen = (len == -1) ? UINT32_MAX : (uint32_t)len;
                        auto [newOff, newLen] = GpuOps::substringFlat(
                            fit->second.offsets, fit->second.lengths,
                            gpuStart, gpuLen, (uint32_t)vec->size());
                        if (newOff && newLen) {
                            colName = "tmp_sub_" + baseCol;
                            ctx.stringCols[colName].resize(vec->size(), "");
                            FlatStringCol subFlat;
                            subFlat.chars = fit->second.chars;
                            subFlat.offsets = std::move(newOff);
                            subFlat.lengths = std::move(newLen);
                            subFlat.rowCount = (uint32_t)vec->size();
                            subFlat.totalBytes = fit->second.totalBytes;
                            ctx.flatStringCols[colName] = std::move(subFlat);
                            isCol = true;
                        }
                    }
                    if (!isCol) {
                        // CPU fallback: per-row substring
                        std::vector<std::string> newVec;
                        newVec.reserve(vec->size());
                        for (const auto& s : *vec) {
                            if (start > (int)s.size()) newVec.push_back("");
                            else {
                                int realLen = (len == -1) ? (s.size() - start + 1) : len;
                                if (start + realLen - 1 > (int)s.size()) realLen = s.size() - start + 1;
                                newVec.push_back(s.substr(start - 1, realLen));
                            }
                        }
                        colName = "tmp_sub_" + baseCol;
                        ctx.stringCols[colName] = std::move(newVec);
                        isCol = true;
                    }
                }
            }
        }
    }

    if (!isCol) {
        if (debug) {
            LOG_INFO("Exec", "IN only supported on Columns. Kind=" << (int)leftExprRaw->kind);
            if (leftExprRaw->kind == TypedExpr::Kind::Function) {
                LOG_INFO("Exec", "Function: " << leftExprRaw->asFunction().name);
                LOG_INFO("Exec", "Available string cols: ");
                for (const auto& kv : ctx.stringCols) std::cerr << kv.first << " ";
                LOG_INFO("EVAL", "\n");
            }
        }
        return false;
    }

    // Rewrite: (col = val1) OR (col = val2) ...
    TypedExprPtr root;
    DataType infType = DataType::String;
    if (leftExprRaw->kind == TypedExpr::Kind::Column) {
        infType = leftExprRaw->asColumn().inferredType;
    }

    for (const auto& valExpr : cmp.inList) {
        auto l = TypedExpr::column(colName);
        l->asColumn().inferredType = infType;
        auto eqExpr = TypedExpr::compare(engine::CompareOp::Eq, l, valExpr);
        if (!root) root = eqExpr;
        else root = TypedExpr::binary(BinaryOp::Or, root, eqExpr);
    }

    return GpuExecutor::executeFilterRecursive(root, ctx);
}

// Constant-fold Literal vs Literal comparisons (e.g. 1=1).
// Returns true if both sides are literals (handled), false otherwise.
static bool filterCompareLiteralVsLiteral(const TypedExpr* leftRaw,
                                          const TypedExpr* rightRaw,
                                          const CompareExpr& cmp,
                                          EvalContext& ctx, bool /*debug*/) {
    if (!(leftRaw && rightRaw &&
          leftRaw->kind == TypedExpr::Kind::Literal &&
          rightRaw->kind == TypedExpr::Kind::Literal))
        return false;

    LOG_DEBUG("Exec", "DEBUG: Literal vs Literal comparison\n");
    bool result = false;
    const auto& lv = leftRaw->asLiteral().value;
    const auto& rv = rightRaw->asLiteral().value;
    if      (cmp.op == engine::CompareOp::Eq) result = (lv == rv);
    else if (cmp.op == engine::CompareOp::Ne) result = (lv != rv);
    else if (cmp.op == engine::CompareOp::Lt) result = (lv < rv);
    else if (cmp.op == engine::CompareOp::Le) result = (lv <= rv);
    else if (cmp.op == engine::CompareOp::Gt) result = (lv > rv);
    else if (cmp.op == engine::CompareOp::Ge) result = (lv >= rv);

    if (result) {
        if (!ctx.activeRowsGPU && ctx.activeRowsCountGPU == 0)
            ctx.activeRowsCountGPU = ctx.rowCount;
    } else {
        ctx.activeRowsGPU= GpuOps::createBuffer(nullptr, 4);
        ctx.activeRowsCountGPU = 0;
    }
    return true;
}

// Handle Column vs Column comparison on GPU.
// Returns true if both sides are columns and the filter was executed.
static bool filterCompareColVsCol(const TypedExpr* leftRaw,
                                  const TypedExpr* rightRaw,
                                  const CompareExpr& cmp,
                                  EvalContext& ctx, bool /*debug*/) {
    if (!(leftRaw && rightRaw &&
          leftRaw->kind == TypedExpr::Kind::Column &&
          rightRaw->kind == TypedExpr::Kind::Column))
        return false;

    std::string lName = leftRaw->asColumn().column;
    std::string rName = rightRaw->asColumn().column;
    std::string lActual = ctx.resolveColName(lName);
    std::string rActual = ctx.resolveColName(rName);
    if (lActual.empty() || rActual.empty()) return false;

    LOG_DEBUG("Exec", "Col vs Col: " << lActual << " vs " << rActual);

    bool lIsF32 = ctx.f32Cols.count(lActual);
    bool rIsF32 = ctx.f32Cols.count(rActual);

    int opInt = 0;
    if      (cmp.op == engine::CompareOp::Eq) opInt = (int)engine::GpuFilterOp::EQ;
    else if (cmp.op == engine::CompareOp::Ne) opInt = (int)engine::GpuFilterOp::NE;
    else if (cmp.op == engine::CompareOp::Lt) opInt = (int)engine::GpuFilterOp::LT;
    else if (cmp.op == engine::CompareOp::Le) opInt = (int)engine::GpuFilterOp::LE;
    else if (cmp.op == engine::CompareOp::Gt) opInt = (int)engine::GpuFilterOp::GT;
    else if (cmp.op == engine::CompareOp::Ge) opInt = (int)engine::GpuFilterOp::GE;
    else return false;

    MTL::Buffer* rootA = lIsF32 ? ctx.f32ColsGPU.at(lActual) : ctx.u32ColsGPU.at(lActual);
    MTL::Buffer* rootB = rIsF32 ? ctx.f32ColsGPU.at(rActual) : ctx.u32ColsGPU.at(rActual);

    MTL::Buffer* finalA = rootA;
    MTL::Buffer* finalB = rootB;
    bool freeFinalA = false;
    bool freeFinalB = false;
    uint32_t workingCount = ctx.activeRowsGPU ? ctx.activeRowsCountGPU : ctx.rowCount;
    bool useF32 = lIsF32 || rIsF32;

    // 1. Cast if needed
    if (useF32 && !lIsF32) { finalA = GpuOps::castU32ToF32(rootA, ctx.rowCount).detach(); freeFinalA = true; }
    if (useF32 && !rIsF32) { finalB = GpuOps::castU32ToF32(rootB, ctx.rowCount).detach(); freeFinalB = true; }

    // 2. Gather if active rows
    if (ctx.activeRowsGPU) {
        MTL::Buffer* gA = useF32 ? GpuOps::gatherF32(finalA, ctx.activeRowsGPU, workingCount).detach()
                                 : GpuOps::gatherU32(finalA, ctx.activeRowsGPU, workingCount).detach();
        if (freeFinalA) finalA->release();
        finalA = gA; freeFinalA = true;

        MTL::Buffer* gB = useF32 ? GpuOps::gatherF32(finalB, ctx.activeRowsGPU, workingCount).detach()
                                 : GpuOps::gatherU32(finalB, ctx.activeRowsGPU, workingCount).detach();
        if (freeFinalB) finalB->release();
        finalB = gB; freeFinalB = true;
    }

    // 3. Execute
    std::optional<FilterResult> res;
    if (useF32) res = GpuOps::filterColColF32(finalA, finalB, workingCount, opInt);
    else        res = GpuOps::filterColColU32(finalA, finalB, workingCount, opInt);

    // 4. Cleanup
    if (freeFinalA && finalA) finalA->release();
    if (freeFinalB && finalB) finalB->release();

    if (res) {
        if (ctx.activeRowsGPU) {
            GpuBuffer newActive = GpuOps::gatherU32(ctx.activeRowsGPU, res->indices, res->count);
            ctx.activeRowsGPU.reset(newActive);
            ctx.activeRowsCountGPU = res->count;
        } else {
            ctx.activeRowsGPU = std::move(res->indices);
            ctx.activeRowsCountGPU = res->count;
        }
        return true;
    }
    return false;
}

// Handle String Filter (LIKE or EQ) — GPU string matching via flat buffers.
// Returns true if the comparison is a string Col op StringLiteral and was handled.
static bool filterCompareStringCol(const CompareExpr& cmp, EvalContext& ctx, bool debug) {
    if (cmp.op != engine::CompareOp::Like && cmp.op != engine::CompareOp::Eq)
        return false;

    const TypedExpr* left = unwrapExpr(cmp.left.get());
    const TypedExpr* right = unwrapExpr(cmp.right.get());
    if (!(left->kind == TypedExpr::Kind::Column &&
          right->kind == TypedExpr::Kind::Literal &&
          std::holds_alternative<std::string>(right->asLiteral().value)))
        return false;

    std::string colName = left->asColumn().column;
    std::string pat = std::get<std::string>(right->asLiteral().value);

    const std::vector<std::string>* vec = nullptr;
    ctx.ensureStringCol(colName);
    if (ctx.stringCols.count(colName)) {
        vec = &ctx.stringCols.at(colName);
    } else {
        std::string resolved = ctx.resolveColName(colName);
        if (!resolved.empty()) {
            ctx.ensureStringCol(resolved);
            if (ctx.stringCols.count(resolved)) { vec = &ctx.stringCols.at(resolved); colName = resolved; }
        }
    }

    if (!vec) return false;

    if (debug)
        LOG_INFO("Exec", "Found string col " << colName << " size " << vec->size() << " pattern '" << pat << "'");

    MTL::Buffer *fChars = nullptr, *fOff = nullptr, *fLen = nullptr;
    auto fit = ctx.flatStringCols.find(colName);
    if (fit != ctx.flatStringCols.end() && fit->second.rowCount == vec->size()) {
        fChars = fit->second.chars; fOff = fit->second.offsets; fLen = fit->second.lengths;
    }

    engine::GpuFilterOp op = (cmp.op == engine::CompareOp::Like) ? engine::GpuFilterOp::LIKE_PATTERN : engine::GpuFilterOp::EQ;
    auto res = GpuOps::filterString(colName, *vec, op, pat, fChars, fOff, fLen);

    if (res) {
        if (ctx.activeRowsGPU) {
            auto joinRes = GpuOps::joinHash(ctx.activeRowsGPU, ctx.activeRowsCountGPU, res->indices, res->count);
            GpuBuffer newActive = GpuOps::gatherU32(ctx.activeRowsGPU, joinRes.buildIndices, joinRes.count);
            ctx.activeRowsGPU.reset(newActive);
            ctx.activeRowsCountGPU = joinRes.count;
        } else {
            ctx.activeRowsGPU = std::move(res->indices);
            ctx.activeRowsCountGPU = res->count;
        }
        return true;
    }
    return false;
}

// Hash-based string equality filter: Col(StringHash) = "Literal".
// Uses FNV1a hash of the literal to compare against pre-hashed u32 column.
// Returns true if handled.
static bool filterCompareStringHash(const CompareExpr& cmp, EvalContext& ctx, bool debug) {
    if (cmp.op != engine::CompareOp::Eq && cmp.op != engine::CompareOp::Ne)
        return false;

    const TypedExpr* left = unwrapExpr(cmp.left.get());
    const TypedExpr* right = unwrapExpr(cmp.right.get());
    if (!(left->kind == TypedExpr::Kind::Column &&
          right->kind == TypedExpr::Kind::Literal &&
          std::holds_alternative<std::string>(right->asLiteral().value)))
        return false;

    std::string colName = left->asColumn().column;
    std::string pat = std::get<std::string>(right->asLiteral().value);
    std::string actualCol = ctx.resolveColName(colName);
    if (actualCol.empty()) return false;

    // Detect SingleChar columns — compare char code, not FNV1a hash
    bool isSingleChar = false;
    {
        std::string tbl = tableForColumn(actualCol);
        if (!tbl.empty())
            isSingleChar = SchemaRegistry::instance().isSingleCharColumn(tbl, actualCol);
    }
    uint32_t hashVal;
    if (isSingleChar && pat.size() == 1) {
        hashVal = static_cast<uint32_t>(static_cast<unsigned char>(pat[0]));
        if (debug)
            LOG_INFO("Exec", "Found SingleChar col " << actualCol << " for pattern '" << pat << "' (charCode=" << hashVal << ")");
    } else {
        hashVal = GpuOps::fnv1a32(pat);
        if (debug)
            LOG_INFO("Exec", "Found StringHash col " << actualCol << " for pattern '" << pat << "' (hashing=" << hashVal << ")");
    }
    engine::GpuFilterOp op = (cmp.op == engine::CompareOp::Eq) ? engine::GpuFilterOp::EQ : engine::GpuFilterOp::NE;

    MTL::Buffer* colBuf = ctx.u32ColsGPU.at(actualCol);
    uint32_t count = (ctx.activeRowsGPU) ? ctx.activeRowsCountGPU : ctx.rowCount;

    std::optional<FilterResult> res;
    if (ctx.activeRowsGPU) {
        res = GpuOps::filterU32Indexed(actualCol, colBuf, ctx.activeRowsGPU, count, op, hashVal);
    } else {
        res = GpuOps::filterU32(actualCol, colBuf, count, op, hashVal);
    }

    if (res) {
        ctx.activeRowsGPU = std::move(res->indices);
        ctx.activeRowsCountGPU = res->count;
        return true;
    }
    return false;
}

// ---------- filterCompareGenericExpression ----------
// Handles comparison filters where one or both sides are computed expressions
// (not simple column references). Evaluates via GPU expression evaluation.
static bool filterCompareGenericExpression(
    const TypedExprPtr& left, const TypedExprPtr& right,
    const TypedExpr* leftUnwrapped, const TypedExpr* rightUnwrapped,
    engine::GpuFilterOp op, EvalContext& ctx, bool /*debug*/)
{
    LOG_DEBUG("Exec", "GPU Filter: Generic Expression Path\n");
    uint32_t count = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
    if (count == 0) return true;

    MTL::Buffer* leftBuf = nullptr;
    float leftLitVal = 0.0f;
    bool leftIsLit = false;

    if (leftUnwrapped->kind == TypedExpr::Kind::Literal) {
        leftIsLit = true;
        const auto& lit = leftUnwrapped->asLiteral();
        if (std::holds_alternative<double>(lit.value)) leftLitVal = (float)std::get<double>(lit.value);
        else if (std::holds_alternative<int64_t>(lit.value)) leftLitVal = (float)std::get<int64_t>(lit.value);
    } else {
        leftBuf = GpuExecutor::evaluateExpression(left, ctx);
        if (!leftBuf) return false;
    }

    MTL::Buffer* rightBuf = nullptr;
    float rightLitVal = 0.0f;
    bool rightIsLit = false;

    if (rightUnwrapped->kind == TypedExpr::Kind::Literal) {
        rightIsLit = true;
        const auto& lit = rightUnwrapped->asLiteral();
        if (std::holds_alternative<double>(lit.value)) rightLitVal = (float)std::get<double>(lit.value);
        else if (std::holds_alternative<int64_t>(lit.value)) rightLitVal = (float)std::get<int64_t>(lit.value);
    } else {
        rightBuf = GpuExecutor::evaluateExpression(right, ctx);
        if (!rightBuf) { if (leftBuf) leftBuf->release(); return false; }
    }

    std::optional<FilterResult> res;

    if (leftIsLit && rightBuf) {
        engine::GpuFilterOp flipped;
        bool valid = true;
        switch (op) {
            case engine::GpuFilterOp::EQ: flipped = engine::GpuFilterOp::EQ; break;
            case engine::GpuFilterOp::NE: flipped = engine::GpuFilterOp::NE; break;
            case engine::GpuFilterOp::LT: flipped = engine::GpuFilterOp::GT; break;
            case engine::GpuFilterOp::LE: flipped = engine::GpuFilterOp::GE; break;
            case engine::GpuFilterOp::GT: flipped = engine::GpuFilterOp::LT; break;
            case engine::GpuFilterOp::GE: flipped = engine::GpuFilterOp::LE; break;
            default: valid = false; break;
        }
        if (valid) res = GpuOps::filterF32("expr", rightBuf, count, flipped, leftLitVal);
    }
    else if (leftBuf && rightIsLit) {
        res = GpuOps::filterF32("expr", leftBuf, count, op, rightLitVal);
    }
    else if (leftBuf && rightBuf) {
        res = GpuOps::filterColColF32(leftBuf, rightBuf, count, (int)op);
    }

    if (leftBuf) leftBuf->release();
    if (rightBuf) rightBuf->release();

    if (res) {
        LOG_DEBUG("Exec", "Generic Filter Result: " << res->count << " rows\n");
        if (ctx.activeRowsGPU) {
            LOG_DEBUG("Exec", "Intersecting Generic with " << ctx.activeRowsCountGPU << " rows\n");
            GpuBuffer newActive = GpuOps::gatherU32(ctx.activeRowsGPU, res->indices, res->count);
            ctx.activeRowsGPU.reset(newActive);
        } else {
            ctx.activeRowsGPU = std::move(res->indices);
        }
        ctx.activeRowsCountGPU = res->count;
        return true;
    }
    return false;
}

// -- Extracted: filterCompareColColResolved --
// Handles the late Col-vs-Col comparison path with buffer resolution,
// aggregate fallback, _rhs_ suffix stripping, gather, cast, and filterColCol dispatch.
static bool filterCompareColColResolved(
    const std::string& colName1, const std::string& colName2,
    engine::GpuFilterOp op, EvalContext& ctx, bool debug) {

    std::string c1 = colName1, c2 = colName2;
    LOG_DEBUG("Exec", "DEBUG ColCol: " << c1 << " vs " << c2);

    auto resolveBuf = [&](std::string& n, bool& isF) -> MTL::Buffer* {
        std::string resolved = ctx.resolveGpuColName(n);
        if (!resolved.empty()) {
            n = resolved;
            if (ctx.f32ColsGPU.count(resolved)) { isF=true; return ctx.f32ColsGPU[resolved]; }
            if (ctx.u32ColsGPU.count(resolved)) { isF=false; return ctx.u32ColsGPU[resolved]; }
        }
        size_t rhsPos = n.rfind("_rhs_");
        if (rhsPos != std::string::npos) {
            std::string base = n.substr(0, rhsPos);
            resolved = ctx.resolveGpuColName(base);
            if (!resolved.empty()) {
                n = resolved;
                if (ctx.f32ColsGPU.count(resolved)) { isF=true; return ctx.f32ColsGPU[resolved]; }
                if (ctx.u32ColsGPU.count(resolved)) { isF=false; return ctx.u32ColsGPU[resolved]; }
            }
        }
        if (n.find("min(") != std::string::npos || n.find("max(") != std::string::npos ||
            n.find("sum(") != std::string::npos || n.find("avg(") != std::string::npos ||
            n.find("count(") != std::string::npos || n.find("MIN(") != std::string::npos ||
            n.find("MAX(") != std::string::npos || n.find("SUM(") != std::string::npos) {
            std::string posKey = "#" + std::to_string(ctx.aggregateCounter);
            if (ctx.f32ColsGPU.count(posKey)) { n=posKey; isF=true; ctx.aggregateCounter++; return ctx.f32ColsGPU[posKey]; }
            if (ctx.u32ColsGPU.count(posKey)) { n=posKey; isF=false; ctx.aggregateCounter++; return ctx.u32ColsGPU[posKey]; }
        }
        return nullptr;
    };

    bool f1=false, f2=false;
    MTL::Buffer* b1 = resolveBuf(c1, f1);
    MTL::Buffer* b2 = resolveBuf(c2, f2);

    if (!b1 || !b2) {
        if (debug) {
            if (!b1) LOG_DEBUG("EVAL", "Failed to resolve " << c1);
            if (!b2) LOG_DEBUG("EVAL", "Failed to resolve " << c2);
            LOG_INFO("EVAL", "Available F32 Cols: ");
            for(const auto& kv : ctx.f32ColsGPU) std::cerr << "'" << kv.first << "' ";
            LOG_INFO("EVAL", "\nAvailable U32 Cols: ");
            for(const auto& kv : ctx.u32ColsGPU) std::cerr << "'" << kv.first << "' ";
            LOG_INFO("EVAL", "\n");
        }
        return false;
    }

    uint32_t currentCount = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
    MTL::Buffer* input1 = b1;
    MTL::Buffer* input2 = b2;
    std::vector<GpuBuffer> temp;

    if (ctx.activeRowsGPU) {
        temp.push_back(f1 ? GpuOps::gatherF32(b1, ctx.activeRowsGPU, currentCount)
                          : GpuOps::gatherU32(b1, ctx.activeRowsGPU, currentCount));
        input1 = temp.back();
        temp.push_back(f2 ? GpuOps::gatherF32(b2, ctx.activeRowsGPU, currentCount)
                          : GpuOps::gatherU32(b2, ctx.activeRowsGPU, currentCount));
        input2 = temp.back();
    }

    std::optional<FilterResult> res;
    if (f1 || f2) {
        if (!f1) { temp.push_back(GpuOps::castU32ToF32(input1, currentCount)); input1 = temp.back(); }
        if (!f2) { temp.push_back(GpuOps::castU32ToF32(input2, currentCount)); input2 = temp.back(); }
        res = GpuOps::filterColColF32(input1, input2, currentCount, (int)op);
    } else {
        res = GpuOps::filterColColU32(input1, input2, currentCount, (int)op);
    }

    // temp vector destructor handles cleanup via GpuBuffer RAII

    if (res) {
        if (ctx.activeRowsGPU) {
            GpuBuffer newActive = GpuOps::gatherU32(ctx.activeRowsGPU, res->indices, res->count);
            ctx.activeRowsGPU.reset(newActive);
        } else {
            ctx.activeRowsGPU = std::move(res->indices);
        }
        ctx.activeRowsCountGPU = res->count;
        return true;
    }
    return false;
}

// -- Extracted: resolveFilterGpuBuffer --
// Resolves a column name to its GPU buffer, trying direct lookup, suffix resolution,
// and aggregate positional fallback (count->~#1, sum->~#0).
// Returns the buffer and sets isF32/colName accordingly, or nullptr if not found.
static MTL::Buffer* resolveFilterGpuBuffer(
    std::string& colName, bool& isF32, EvalContext& ctx, bool /*debug*/)
{
    MTL::Buffer* buf = nullptr;
    if (ctx.f32ColsGPU.count(colName)) {
        buf = ctx.f32ColsGPU[colName];
        isF32 = true;
    } else if (ctx.u32ColsGPU.count(colName)) {
        buf = ctx.u32ColsGPU[colName];
        isF32 = false;
    } else {
        LOG_DEBUG("Exec", "DEBUG FIND BUF: colName " << colName << " not found in GPU cols\n");
        // Fallback for suffixed columns
        {
            std::string resolved = ctx.resolveGpuColName(colName);
            if (!resolved.empty()) {
                colName = resolved;
                if (ctx.f32ColsGPU.count(resolved)) { buf = ctx.f32ColsGPU[resolved]; isF32=true; }
                else { buf = ctx.u32ColsGPU[resolved]; isF32=false; }
            }
        }

        // Fallback for aggregations (count_star() -> #1, sum -> #0 heuristic)
        if (!buf) {
             if (colName.find("count") != std::string::npos || colName.find("COUNT") != std::string::npos) {
                 if (ctx.u32ColsGPU.count("#1")) { buf = ctx.u32ColsGPU["#1"]; isF32=false; colName="#1"; }
                 else if (ctx.f32ColsGPU.count("#1")) { buf = ctx.f32ColsGPU["#1"]; isF32=true; colName="#1"; }
                 else if (ctx.u32ColsGPU.count("#0")) { buf = ctx.u32ColsGPU["#0"]; isF32=false; colName="#0"; }
                 else if (ctx.f32ColsGPU.count("#0")) { buf = ctx.f32ColsGPU["#0"]; isF32=true; colName="#0"; }
             } else if (colName.find("sum") != std::string::npos || colName.find("SUM") != std::string::npos) {
                 if (ctx.f32ColsGPU.count("#0")) { buf = ctx.f32ColsGPU["#0"]; isF32=true; colName="#0"; }
                 else if (ctx.u32ColsGPU.count("#0")) { buf = ctx.u32ColsGPU["#0"]; isF32=false; colName="#0"; }
             }
        }

        if (!buf) {
            LOG_DEBUG("Exec", "DEBUG FIND BUF: FAILED for colName " << colName);
        }
    }
    return buf;
}

// -- Extracted: applyLiteralFilter --
// Applies a GPU filter comparing a column buffer against a literal value.
// Handles f32/u32 paths, scalar broadcast, date conversion, and activeRows.
// Returns true on success (ctx.activeRowsGPU updated), false on failure.
static bool applyLiteralFilter(
    MTL::Buffer* buf, const std::string& colName, bool isF32,
    const Literal& lit, engine::GpuFilterOp op,
    EvalContext& ctx, bool debug)
{
    std::optional<FilterResult> res;
    uint32_t currentCount = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;

    if (isF32) {
        float val = 0.0f;
        if (std::holds_alternative<double>(lit.value)) val = (float)std::get<double>(lit.value);
        else if (std::holds_alternative<int64_t>(lit.value)) val = (float)std::get<int64_t>(lit.value);
        else if (std::holds_alternative<std::string>(lit.value)) {
            try {
                val = std::stof(std::get<std::string>(lit.value));
            } catch(...) {
                // Literal string not a valid float — val stays 0
            }
        }

        // Handle scalar broadcast filter (single value buffer vs multiple rows)
        if (buf && buf->length() == sizeof(float) && currentCount > 1) {
            float colVal = *static_cast<const float*>(buf->contents());
            bool pass = false;
            switch(op) {
                case engine::GpuFilterOp::EQ: pass = (colVal == val); break;
                case engine::GpuFilterOp::NE: pass = (colVal != val); break;
                case engine::GpuFilterOp::LT: pass = (colVal < val); break;
                case engine::GpuFilterOp::LE: pass = (colVal <= val); break;
                case engine::GpuFilterOp::GT: pass = (colVal > val); break;
                case engine::GpuFilterOp::GE: pass = (colVal >= val); break;
                default: break;
            }
            if (!pass) {
                ctx.activeRowsGPU= GpuOps::createBuffer(nullptr, 4);
                ctx.activeRowsCountGPU = 0;
            }
            return true;
        }

        try {
            if (ctx.activeRowsGPU) {
                res = GpuOps::filterF32Indexed(colName, buf, ctx.activeRowsGPU, currentCount, op, val);
            } else {
                res = GpuOps::filterF32(colName, buf, currentCount, op, val);
            }
        } catch(...) {
            LOG_DEBUG("Exec", "Exception in filterF32\n");
            res = std::nullopt;
        }

        if (!res) {
            LOG_DEBUG("Exec", "DEBUG: filterF32 failed. bufLen=" << buf->length() << " count=" << currentCount << " val=" << val << " op=" << (int)op);
            ENGINE_THROW("GPU F32 Filter failed: " + colName);
        }
    } else {
        uint32_t val = 0;
        if (std::holds_alternative<int64_t>(lit.value)) val = (uint32_t)std::get<int64_t>(lit.value);
        else if (std::holds_alternative<double>(lit.value)) val = (uint32_t)std::get<double>(lit.value);
        else if (std::holds_alternative<std::string>(lit.value)) {
            std::string s = std::get<std::string>(lit.value);
            // Handle Date Literal format YYYY-MM-DD -> YYYYMMDD
            if (s.length() == 10 && s[4] == '-' && s[7] == '-') {
                 std::string d = s.substr(0,4) + s.substr(5,2) + s.substr(8,2);
                 try { val = std::stoul(d); } catch(...) {
                     // Malformed date string — val stays 0
                 }
            } else {
                 // Check for single char column
                 std::string tableName = tableForColumn(colName);
                 bool isSingleChar = false;
                 if (!tableName.empty()) {
                     const auto& schema = SchemaRegistry::instance();
                     isSingleChar = schema.isSingleCharColumn(tableName, colName);
                 }

                 if (isSingleChar && s.size() == 1) {
                     val = static_cast<uint32_t>(static_cast<unsigned char>(s[0]));
                 } else {
                     try { 
                         size_t idx = 0;
                         val = std::stoul(s, &idx);
                         if (idx != s.size()) {
                             val = GpuOps::fnv1a32(s);
                         }
                     } catch(...) {
                        // Not a valid integer — fall back to FNV hash of the string
                         val = GpuOps::fnv1a32(s);
                     }
                 }
            }
        }

        // Check for Date column with Days-Since-Epoch literal (small integer)
        if (val > 0 && val < engine::config::kDateFormatThreshold) {
            std::string tableNameD = tableForColumn(colName);
            if (!tableNameD.empty()) {
                 const auto& schema = SchemaRegistry::instance();
                 if (schema.getColumnType(tableNameD, colName) == ColumnType::Date) {
                     using namespace std::chrono;
                     sys_days sd = sys_days(days(val));
                     year_month_day ymd{sd};
                     val = (int)ymd.year() * 10000 + (unsigned)ymd.month() * 100 + (unsigned)ymd.day();
                 }
            }
        }

        // Handle scalar broadcast filter (U32)
        if (buf && buf->length() == sizeof(uint32_t) && currentCount > 1) {
            LOG_DEBUG("Exec", "DEBUG: Scalar broadcast detected. bufLen=" << buf->length() << " currentCount=" << currentCount);
            uint32_t colVal = *static_cast<const uint32_t*>(buf->contents());
            bool pass = false;
            switch(op) {
                case engine::GpuFilterOp::EQ: pass = (colVal == val); break;
                case engine::GpuFilterOp::NE: pass = (colVal != val); break;
                case engine::GpuFilterOp::LT: pass = (colVal < val); break;
                case engine::GpuFilterOp::LE: pass = (colVal <= val); break;
                case engine::GpuFilterOp::GT: pass = (colVal > val); break;
                case engine::GpuFilterOp::GE: pass = (colVal >= val); break;
                default: break;
            }
            if (!pass) {
                ctx.activeRowsGPU= GpuOps::createBuffer(nullptr, 4);
                ctx.activeRowsCountGPU = 0;
            }
            return true;
        }

        LOG_DEBUG("Exec", "DEBUG: About to filter. activeRowsGPU=" << (ctx.activeRowsGPU ? "set" : "null") << " bufLen=" << buf->length() << " count=" << currentCount << " val=" << val << " op=" << (int)op);

        if (ctx.activeRowsGPU) {
            res = GpuOps::filterU32Indexed(colName, buf, ctx.activeRowsGPU, currentCount, op, val);
        } else {
            res = GpuOps::filterU32(colName, buf, currentCount, op, val);
        }

        if (!res) {
            LOG_DEBUG("Exec", "DEBUG: filterU32 failed. bufLen=" << buf->length() << " count=" << currentCount << " val=" << val << " op=" << (int)op);
            ENGINE_THROW("GPU U32 Filter failed: " + colName);
        }
    }

    if (res) {
        ctx.activeRowsGPU = std::move(res->indices);
        ctx.activeRowsCountGPU = res->count;
        if (debug && res->indices) {
            uint32_t* idx = static_cast<uint32_t*>(res->indices->contents());
            LOG_INFO("Exec", "Filter result indices first 5: ");
            for (uint32_t i = 0; i < std::min(5u, res->count); ++i) std::cerr << idx[i] << " ";
            LOG_DEBUG("EVAL", "\n");
        }
        return true;
    }
    return false;
}

bool filterCompare(const TypedExprPtr& expr, EvalContext& ctx) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
if (expr->kind == TypedExpr::Kind::Compare) {
    const auto& cmp = expr->asCompare();

    const TypedExpr* leftRaw = unwrapExpr(cmp.left.get());
    const TypedExpr* rightRaw = unwrapExpr(cmp.right.get());

    // Handle Literal vs Literal (e.g. 1=1)
    if (filterCompareLiteralVsLiteral(leftRaw, rightRaw, cmp, ctx, debug))
        return true;

    // Handle Col vs Col
    if (filterCompareColVsCol(leftRaw, rightRaw, cmp, ctx, debug))
        return true;

    // Identify column and literal
    std::string colName;
    bool isF32 = false;

    // Normalize: Col op Lit
    const TypedExpr* colExpr = nullptr;
    const TypedExpr* litExpr = nullptr;

    // Handle String Filter (Like OR Eq) via extracted helper
    if (filterCompareStringCol(cmp, ctx, debug)) return true;
    if (cmp.op == engine::CompareOp::Like) return false;

    // Optimized Hash Filter Fallback: Col(StringHash) = "Literal"
    if (filterCompareStringHash(cmp, ctx, debug)) return true;

    // Handle IN Operator via extracted helper
    if (cmp.op == engine::CompareOp::In) {
        return filterCompareInList(cmp, ctx, debug);
    }

    if (!mapToGpuFilterOp(cmp.op)) return false;
    engine::GpuFilterOp op = *mapToGpuFilterOp(cmp.op);


    const TypedExpr* leftUnwrapped = unwrapExpr(cmp.left.get());
    const TypedExpr* rightUnwrapped = unwrapExpr(cmp.right.get());

    LOG_DEBUG("Exec", "DEBUG CMP Kinds: " << (int)leftUnwrapped->kind << " vs " << (int)rightUnwrapped->kind);

    // Check for Function-as-Column (e.g. count_star())
    std::string funcColName;
    auto getFuncCol = [&](const TypedExpr* e) -> std::string {
         if (e->kind != TypedExpr::Kind::Function) return "";
         const auto& fn = e->asFunction();
         std::string n = fn.name;
         std::string candidates[] = {n, n+"()"};
         for(auto& c : candidates) {
             if (ctx.f32ColsGPU.count(c) || ctx.u32ColsGPU.count(c)) return c;
         }
         std::string l = n; std::transform(l.begin(), l.end(), l.begin(), ::tolower);
         std::string candidatesL[] = {l, l+"()"};
         for(auto& c : candidatesL) {
             if (ctx.f32ColsGPU.count(c) || ctx.u32ColsGPU.count(c)) return c;
         }
         return "";
    };

    // Handle "col = lit"
    if (leftUnwrapped->kind == TypedExpr::Kind::Column && rightUnwrapped->kind == TypedExpr::Kind::Literal) {
        colExpr = leftUnwrapped;
        litExpr = rightUnwrapped;
    } 
    // Handle "lit = col"
    else if (leftUnwrapped->kind == TypedExpr::Kind::Literal && rightUnwrapped->kind == TypedExpr::Kind::Column) {
        colExpr = rightUnwrapped;
        litExpr = leftUnwrapped;
        // Flip operator
        switch (op) {
            case engine::GpuFilterOp::LT: op = engine::GpuFilterOp::GT; break;
            case engine::GpuFilterOp::LE: op = engine::GpuFilterOp::GE; break;
            case engine::GpuFilterOp::GT: op = engine::GpuFilterOp::LT; break;
            case engine::GpuFilterOp::GE: op = engine::GpuFilterOp::LE; break;
            default: break;
        }
    }
    else if ((funcColName = getFuncCol(leftUnwrapped)) != "" && rightUnwrapped->kind == TypedExpr::Kind::Literal) {
         litExpr = rightUnwrapped;
    }
    else if ((funcColName = getFuncCol(rightUnwrapped)) != "" && leftUnwrapped->kind == TypedExpr::Kind::Literal) {
         litExpr = leftUnwrapped;
         switch (op) {
            case engine::GpuFilterOp::LT: op = engine::GpuFilterOp::GT; break;
            case engine::GpuFilterOp::LE: op = engine::GpuFilterOp::GE; break;
            case engine::GpuFilterOp::GT: op = engine::GpuFilterOp::LT; break;
            case engine::GpuFilterOp::GE: op = engine::GpuFilterOp::LE; break;
            default: break;
        }
    }
     else if (leftUnwrapped->kind == TypedExpr::Kind::Column && rightUnwrapped->kind == TypedExpr::Kind::Column) {
         return filterCompareColColResolved(
             leftUnwrapped->asColumn().column, rightUnwrapped->asColumn().column,
             op, ctx, debug);
    } else {
         return filterCompareGenericExpression(cmp.left, cmp.right, leftUnwrapped, rightUnwrapped, op, ctx, debug);
    }

    if (colExpr) colName = colExpr->asColumn().column;
    else if (!funcColName.empty()) colName = funcColName;
    LOG_DEBUG("Exec", "GPU Filter checking col: " << colName);

    // Check string columns
    const std::vector<std::string>* strVec = nullptr;
    if (ctx.stringCols.count(colName)) {
         strVec = &ctx.stringCols.at(colName);
    } else {
         // check suffixes for strings
         std::string resolved = ctx.resolveColName(colName);
         if (!resolved.empty() && ctx.stringCols.count(resolved)) {
             strVec = &ctx.stringCols.at(resolved); colName = resolved;
         }
    }

    if (strVec) {
         std::string pat = "";
         if (std::holds_alternative<std::string>(litExpr->asLiteral().value)) {
              pat = std::get<std::string>(litExpr->asLiteral().value);
         }
         // Check for pre-flattened Arrow-style buffers
         MTL::Buffer *fChars = nullptr, *fOff = nullptr, *fLen = nullptr;
         auto fit = ctx.flatStringCols.find(colName);
         if (fit != ctx.flatStringCols.end() && fit->second.rowCount == strVec->size()) {
             fChars = fit->second.chars; fOff = fit->second.offsets; fLen = fit->second.lengths;
         }
         auto res = GpuOps::filterString(colName, *strVec, op, pat, fChars, fOff, fLen);
         if (res) {
              if (ctx.activeRowsGPU) {
                  auto joinRes = GpuOps::joinHash(
                      ctx.activeRowsGPU, ctx.activeRowsCountGPU,
                      res->indices, res->count
                  );

                  GpuBuffer newActive = GpuOps::gatherU32(ctx.activeRowsGPU, joinRes.buildIndices, joinRes.count);

                  ctx.activeRowsGPU.reset(newActive);

                  ctx.activeRowsCountGPU = joinRes.count;
              } else {
                  ctx.activeRowsGPU = std::move(res->indices);
                  ctx.activeRowsCountGPU = res->count;
              }
              return true;
         }
         return false;
    }

    // Find buffer via extracted helper
    MTL::Buffer* buf = resolveFilterGpuBuffer(colName, isF32, ctx, debug);
    if (!buf) return false;

    LOG_DEBUG("Exec", "GPU Filter checking col: " << colName << " isF32=" << isF32);

    // Apply literal filter via extracted helper
    return applyLiteralFilter(buf, colName, isF32, litExpr->asLiteral(), op, ctx, debug);
}
    return false;
}

bool filterColumnAsBool(const TypedExprPtr& expr, EvalContext& ctx) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    if (expr->kind == TypedExpr::Kind::Column) {
    std::string colName = expr->asColumn().column;
    LOG_DEBUG("Exec", "GPU Filter checking col (boolean): " << colName);

    bool isF32 = false;
    MTL::Buffer* buf = nullptr;
    // Resolve buffer logic
    if (ctx.f32ColsGPU.count(colName)) { buf = ctx.f32ColsGPU.at(colName); isF32 = true; }
    else if (ctx.u32ColsGPU.count(colName)) { buf = ctx.u32ColsGPU.at(colName); isF32 = false; }
    else {
         // check suffixes
         std::string resolved = ctx.resolveGpuColName(colName);
         if (!resolved.empty()) {
              colName = resolved;
              if (ctx.f32ColsGPU.count(resolved)) { buf=ctx.f32ColsGPU.at(resolved); isF32=true; }
              else { buf=ctx.u32ColsGPU.at(resolved); isF32=false; }
         }
    }

    if (!buf) {
         // Check for aggregates heuristics
         if (colName.find("count") != std::string::npos || colName.find("COUNT") != std::string::npos) {
             if (ctx.u32ColsGPU.count("#1")) { buf = ctx.u32ColsGPU["#1"]; isF32=false; colName="#1"; }
             else if (ctx.f32ColsGPU.count("#1")) { buf = ctx.f32ColsGPU["#1"]; isF32=true; colName="#1"; }
             else if (ctx.u32ColsGPU.count("#0")) { buf = ctx.u32ColsGPU["#0"]; isF32=false; colName="#0"; }
             else if (ctx.f32ColsGPU.count("#0")) { buf = ctx.f32ColsGPU["#0"]; isF32=true; colName="#0"; }
         } else if (colName.find("sum") != std::string::npos || colName.find("SUM") != std::string::npos) {
             if (ctx.f32ColsGPU.count("#0")) { buf = ctx.f32ColsGPU["#0"]; isF32=true; colName="#0"; }
             else if (ctx.u32ColsGPU.count("#0")) { buf = ctx.u32ColsGPU["#0"]; isF32=false; colName="#0"; }
         }
    }

    if (!buf && debug) {
         LOG_ERROR("Exec", "DEBUG: Bool Col Lookup Failed: '" << colName << "'\n");
         LOG_DEBUG("EVAL", "Available F32 (" << ctx.f32ColsGPU.size() << "): ");
         if (debug) for(auto& kv : ctx.f32ColsGPU) std::cerr << "'" << kv.first << "' ";
         LOG_DEBUG("EVAL", "\nAvailable U32 (" << ctx.u32ColsGPU.size() << "): ");
         if (debug) for(auto& kv : ctx.u32ColsGPU) std::cerr << "'" << kv.first << "' ";
         LOG_DEBUG("EVAL", "\n");
    }

    if (buf) {
         uint32_t currentCount = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;

         // Scalar broadcast check
         if (buf->length() <= 8 && currentCount > 1) { 
             bool pass = false;
             if (isF32) {
                 float v = *static_cast<const float*>(buf->contents());
                 pass = (v != 0.0f);
             } else {
                 uint32_t v = *static_cast<const uint32_t*>(buf->contents());
                 pass = (v != 0);
             }
             if (!pass) {
                ctx.activeRowsGPU= GpuOps::createBuffer(nullptr, 4);
                ctx.activeRowsCountGPU = 0;
             }
             return true;
         }

         std::optional<FilterResult> res;
         if (isF32) {
             if (ctx.activeRowsGPU) res = GpuOps::filterF32Indexed(colName, buf, ctx.activeRowsGPU, currentCount, engine::GpuFilterOp::NE, 0.0f);
             else res = GpuOps::filterF32(colName, buf, currentCount, engine::GpuFilterOp::NE, 0.0f);
         } else {
             if (ctx.activeRowsGPU) res = GpuOps::filterU32Indexed(colName, buf, ctx.activeRowsGPU, currentCount, engine::GpuFilterOp::NE, 0);
             else res = GpuOps::filterU32(colName, buf, currentCount, engine::GpuFilterOp::NE, 0);
         }

         if (res) {
            ctx.activeRowsGPU = std::move(res->indices);
            ctx.activeRowsCountGPU = res->count;
            return true;
         }
    }

    // Q16 Fix: NOT prefix(col, 'pat')
    if (colName.rfind("NOT prefix(", 0) == 0) {
         // Parse: NOT prefix(p_type, 'MEDIUM POLISHED')
         // 11 chars start
         size_t comma = colName.find(',');
         size_t endParen = colName.rfind(')');
         if (comma != std::string::npos && endParen != std::string::npos && comma > 11) {
             std::string c = colName.substr(11, comma - 11);
             // trim c
             c.erase(0, c.find_first_not_of(" "));
             c.erase(c.find_last_not_of(" ") + 1);

             std::string pat = colName.substr(comma + 1, endParen - comma - 1);
             // trim pat '...'
             size_t q1 = pat.find('\'');
             size_t q2 = pat.rfind('\'');
             if (q1 != std::string::npos && q2 != std::string::npos && q2 > q1) {
                 pat = pat.substr(q1+1, q2-q1-1);
             }

             LOG_DEBUG("Exec", "Corrected Q16: " << c << " NOT LIKE " << pat << "%\n");

             const std::vector<std::string>* vec = nullptr;
             if (ctx.stringCols.count(c)) vec = &ctx.stringCols.at(c);

             if (vec) {
                  // Apply NOT LIKE prefix
                  // prefix matching: Like 'pat%'
                  // Check for pre-flattened Arrow-style buffers
                  MTL::Buffer *fChars = nullptr, *fOff = nullptr, *fLen = nullptr;
                  auto fit = ctx.flatStringCols.find(c);
                  if (fit != ctx.flatStringCols.end() && fit->second.rowCount == vec->size()) {
                      fChars = fit->second.chars; fOff = fit->second.offsets; fLen = fit->second.lengths;
                  }
                  std::optional<FilterResult> res = GpuOps::filterStringPrefix(c, *vec, pat, true, fChars, fOff, fLen);

                  if (res) {
                      if (ctx.activeRowsGPU) {
                          auto joinRes = GpuOps::joinHash(
                              ctx.activeRowsGPU, ctx.activeRowsCountGPU,
                              res->indices, res->count
                          );
                          GpuBuffer newActive = GpuOps::gatherU32(ctx.activeRowsGPU, joinRes.buildIndices, joinRes.count);

                          ctx.activeRowsGPU.reset(newActive);

                          ctx.activeRowsCountGPU = joinRes.count;
                      } else {
                          ctx.activeRowsGPU = std::move(res->indices);
                          ctx.activeRowsCountGPU = res->count;
                      }
                      return true;
                  }
             }
         }
    }

    } // end if (expr->kind == Column)

    return false;
}

} // namespace engine
