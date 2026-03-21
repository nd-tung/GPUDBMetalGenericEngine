#include "GpuExecutorDetail.hpp"
#include "EngineError.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include "Logger.hpp"

namespace engine {

// Forward declarations (implemented in EvalFilter.cpp)
bool filterCompare(const TypedExprPtr& expr, EvalContext& ctx);
bool filterColumnAsBool(const TypedExprPtr& expr, EvalContext& ctx);

static std::string normalizeFuzzy(const std::string& input) {
    std::string result;
    result.reserve(input.size());
    for (size_t i = 0; i < input.size(); ) {
        unsigned char uc = static_cast<unsigned char>(input[i]);
        char c = static_cast<char>(std::tolower(uc));
        if (std::isspace(uc) || c == '(' || c == ')') { i++; continue; }
        if (std::isdigit(uc)) {
            // Consume integer part
            size_t numStart = result.size();
            while (i < input.size() && std::isdigit(static_cast<unsigned char>(input[i]))) { result += static_cast<char>(std::tolower(static_cast<unsigned char>(input[i]))); i++; }
            // Check for decimal part
            if (i < input.size() && input[i] == '.' && i + 1 < input.size() && std::isdigit(static_cast<unsigned char>(input[i + 1]))) {
                result += '.'; i++; // skip '.'
                while (i < input.size() && std::isdigit(static_cast<unsigned char>(input[i]))) { result += static_cast<char>(std::tolower(static_cast<unsigned char>(input[i]))); i++; }
                // Strip trailing zeros after decimal point
                while (result.size() > numStart + 1 && result.back() == '0') result.pop_back();
                // Strip trailing decimal point
                if (result.back() == '.') result.pop_back();
            }
        } else {
            result += c; i++;
        }
    }
    return result;
}

// Helper: Get the true column capacity (max possible index + 1) from GPU buffers.
// After executeFilter shrinks ctx.rowCount, GPU column buffers may still hold the
// original (larger) data. Mask operations (OR, NOT) need the full capacity so that
// global indices are not silently truncated.
static uint32_t getTotalRowCapacity(const EvalContext& ctx) {
    uint32_t cap = ctx.rowCount;
    for (const auto& [n, b] : ctx.u32ColsGPU) {
        if (b) { cap = std::max(cap, (uint32_t)(b->length() / sizeof(uint32_t))); break; }
    }
    if (cap == ctx.rowCount) {
        for (const auto& [n, b] : ctx.f32ColsGPU) {
            if (b) { cap = std::max(cap, (uint32_t)(b->length() / sizeof(float))); break; }
        }
    }
    return cap;
}

static GpuBuffer evalColumnExpr(const TypedExprPtr& expr, EvalContext& ctx, uint32_t count) {
    if (expr->kind == TypedExpr::Kind::Column) {
    std::string col = expr->asColumn().column;
    MTL::Buffer* buf = nullptr;
    bool isU32 = false;

    // For aggregate columns (sum, avg, etc.), prefer columns with varying values over scalar broadcasts
    // This handles cross-join cases where LHS has grouped results and RHS has scalar
    bool isAggCol = (col.find("sum(") != std::string::npos || col.find("SUM(") != std::string::npos ||
                     col.find("avg(") != std::string::npos || col.find("AVG(") != std::string::npos ||
                     col.find("min(") != std::string::npos || col.find("MIN(") != std::string::npos ||
                     col.find("max(") != std::string::npos || col.find("MAX(") != std::string::npos ||
                     col.find("count(") != std::string::npos || col.find("COUNT(") != std::string::npos);

    if (isAggCol && ctx.rowCount > 1) {
        // For aggregates, first look for a column with varying values (not scalar broadcast)
        std::string colLower = col;
        std::transform(colLower.begin(), colLower.end(), colLower.begin(), ::tolower);
        colLower.erase(std::remove_if(colLower.begin(), colLower.end(), ::isspace), colLower.end());

        for (const auto& [name, b] : ctx.f32ColsGPU) {
            if (!b || b->length() / sizeof(float) < 2) continue;
            std::string nameLower = name;
            std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
            nameLower.erase(std::remove_if(nameLower.begin(), nameLower.end(), ::isspace), nameLower.end());
            if (colLower == nameLower) {
                // Check if values vary (not a scalar broadcast)
                float* ptr = static_cast<float*>(b->contents());
                size_t n = b->length() / sizeof(float);
                bool varying = false;
                float first = ptr[0];
                for (size_t i = 1; i < std::min(n, engine::config::kColumnSampleSize); ++i) {
                    if (ptr[i] != first) { varying = true; break; }
                }
                if (varying) {
                    LOG_DEBUG("Exec", "evaluateExpression: agg match varying col '" << name << "' for '" << col << "'\n");
                    buf = b;
                    break;
                }
            }
        }
    }

    // Standard exact match
    if (!buf && ctx.f32ColsGPU.count(col)) {
        buf = ctx.f32ColsGPU[col];
    } else if (!buf && ctx.u32ColsGPU.count(col)) {
        buf = ctx.u32ColsGPU[col];
        isU32 = true;
    } else if (ctx.f32Cols.count(col) && !ctx.f32Cols[col].empty()) {
        const auto& vec = ctx.f32Cols[col];
        // Handle correct gathering from CPU vector
        if (vec.size() == 1 && count > 1) { // Scalar broadcast
            float val = vec[0];
            return GpuOps::createFilledF32(val, count);
        } else {
            GpuBuffer rawBuf = GpuOps::createBuffer(vec.data(), vec.size() * sizeof(float));
            // Only gather if data is not already compacted to avoid OOB reads
            if (ctx.activeRowsGPU && vec.size() != count) {
                return GpuOps::gatherF32(rawBuf, ctx.activeRowsGPU, count);
            }
            return rawBuf;
        }
    } else if (ctx.u32Cols.count(col) && !ctx.u32Cols[col].empty()) {
        const auto& vec = ctx.u32Cols[col];
        if (vec.size() == 1 && count > 1) { // Scalar broadcast
            float val = (float)vec[0];
            return GpuOps::createFilledF32(val, count);
        } else {
            GpuBuffer rawBuf = GpuOps::createBuffer(vec.data(), vec.size() * sizeof(uint32_t));

            // Only gather if data is not already compacted
            if (ctx.activeRowsGPU && vec.size() != count) {
                GpuBuffer gathered = GpuOps::gatherU32(rawBuf, ctx.activeRowsGPU, count);
                return GpuOps::castU32ToF32(gathered, count);
            }

            return GpuOps::castU32ToF32(rawBuf, count);
        }
    } else {
         // Suffix + RHS search via resolveGpuColName
         std::string resolved = ctx.resolveGpuColName(col);
         if (!resolved.empty()) {
             if (ctx.f32ColsGPU.count(resolved)) { buf = ctx.f32ColsGPU[resolved]; }
             else if (ctx.u32ColsGPU.count(resolved)) { buf = ctx.u32ColsGPU[resolved]; isU32=true; }
         }
    }

    // Fallback: Heuristic for Scalar Aggregates mismatch (sum(...) vs #0)
    // First try fuzzy matching by removing spaces and lowercasing
    if (!buf) {
        std::string colLower = normalizeFuzzy(col);

        if (env_truthy("GPUDB_DEBUG_OPS") && col.find("sum") != std::string::npos) {
            LOG_INFO("Exec", "evaluateExpression: fuzzy search for col='" << col << "' normalized='" << colLower << "'\n");
        }

        // Search for matching column in f32ColsGPU
        for (const auto& [name, b] : ctx.f32ColsGPU) {
            if (!b) continue;
            std::string nameLower = normalizeFuzzy(name);

            if (colLower == nameLower) {
                LOG_DEBUG("Exec", "evaluateExpression: fuzzy matched '" << col << "' to '" << name << "'\n");
                buf = b;
                break;
            }
        }
        // Also try u32ColsGPU
        if (!buf) {
            for (const auto& [name, b] : ctx.u32ColsGPU) {
                if (!b) continue;
                std::string nameLower = normalizeFuzzy(name);
                if (colLower == nameLower) {
                    LOG_DEBUG("Exec", "evaluateExpression: fuzzy matched (u32) '" << col << "' to '" << name << "'\n");
                    buf = b;
                    isU32 = true;
                    break;
                }
            }
        }
    }

    // Fallback: positional #N heuristic for aggregates
    if (!buf) {
        bool hasHash0 = (ctx.f32ColsGPU.count("#0") || ctx.f32Cols.count("#0") || ctx.u32Cols.count("#0"));
        if (hasHash0) {
            if (col.find("sum(") != std::string::npos || col.find("SUM(") != std::string::npos ||
                col.find("avg(") != std::string::npos || col.find("AVG(") != std::string::npos ||
                col.find("min(") != std::string::npos || col.find("MIN(") != std::string::npos ||
                col.find("max(") != std::string::npos || col.find("MAX(") != std::string::npos ||
                col.find("count(") != std::string::npos || col.find("COUNT(") != std::string::npos) {

                std::string posKey = "#" + std::to_string(ctx.aggregateCounter);
                LOG_DEBUG("Exec", "evaluateExpression: heuristic mapping " << col << " to " << posKey);

                if (ctx.f32ColsGPU.count(posKey)) { buf = ctx.f32ColsGPU[posKey]; ctx.aggregateCounter++; }
                else if (ctx.u32ColsGPU.count(posKey)) { buf = ctx.u32ColsGPU[posKey]; isU32=true; ctx.aggregateCounter++; }
                else if (ctx.f32Cols.count(posKey) && !ctx.f32Cols[posKey].empty()) {
                    float val = ctx.f32Cols[posKey][0];
                    ctx.aggregateCounter++;
                    return GpuOps::createFilledF32(val, count);
                }
                else if (ctx.u32Cols.count(posKey) && !ctx.u32Cols[posKey].empty()) {
                    float val = (float)ctx.u32Cols[posKey][0];
                    ctx.aggregateCounter++;
                    return GpuOps::createFilledF32(val, count);
                }
            }
        }
    }

    if (!buf) return GpuBuffer();

    // If U32, cast to F32 (and gather if needed)
    if (isU32) {
         if (ctx.activeRowsGPU) {
             // Gather to compact U32, then cast
             GpuBuffer gathered = GpuOps::gatherU32(buf, ctx.activeRowsGPU, count);
             return GpuOps::castU32ToF32(gathered, count);
         } else {
             return GpuOps::castU32ToF32(buf, count);
         }
    }

    // If F32
    if (ctx.activeRowsGPU) {
        return GpuOps::gatherF32(buf, ctx.activeRowsGPU, count);
    } else {
        buf->retain();
        return GpuBuffer(buf);
    }
}
    return GpuBuffer(); // unreachable: call site guarantees expr->kind == Column
}

static GpuBuffer evalCaseExpr(const TypedExprPtr& expr, EvalContext& ctx, uint32_t count) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
if (expr->kind == TypedExpr::Kind::Case) {
    LOG_DEBUG("Exec", "evaluateExpression: Entering CASE expression handler\n");

    // Safely access CaseExpr data
    if (!std::holds_alternative<CaseExpr>(expr->data)) {
        LOG_ERROR("Exec", "ERROR: CASE expression kind but data is not CaseExpr!\n");
        return GpuBuffer();
    }
    const auto& c = expr->asCase();

    if (debug) {
        LOG_INFO("Exec", "CASE: " << c.cases.size() << " WHEN branches, hasElse=" << (c.elseExpr ? "yes" : "no"));
    }

    // 1. Initialize output buffer based on ELSE
    GpuBuffer outBuf;

    if (c.elseExpr) {
        LOG_DEBUG("Exec", "CASE: elseExpr kind=" << static_cast<int>(c.elseExpr->kind));
        if (c.elseExpr->kind == TypedExpr::Kind::Literal) {
            float elseVal = 0.0f;
            const auto& lit = c.elseExpr->asLiteral();
            if (std::holds_alternative<int64_t>(lit.value)) elseVal = (float)std::get<int64_t>(lit.value);
            else if (std::holds_alternative<double>(lit.value)) elseVal = (float)std::get<double>(lit.value);

            outBuf = GpuOps::createFilledF32(elseVal, count);
        } else {
            GpuBuffer elseBuf = GpuExecutor::evaluateExpression(c.elseExpr, ctx);
            if (!elseBuf) return GpuBuffer();

            // Copy elseBuf to outBuf (always copy to avoid modifying source columns)
            outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
            memcpy(outBuf->contents(), elseBuf->contents(), count * sizeof(float));
            // elseBuf auto-releases via GpuBuffer destructor
        }
    } else {
         // Default 0.0
         outBuf = GpuOps::createFilledF32(0.0f, count);
    } 

    // Helper to check if an expression contains DuckDB's "error" guard function
    auto containsErrorFunction = [](const TypedExprPtr& expr) -> bool {
        if (!expr) return false;
        if (expr->kind == TypedExpr::Kind::Function) {
            const auto& fn = expr->asFunction();
            // DuckDB uses "error" function for runtime checks (e.g., scalar subquery validation)
            std::string lower = fn.name;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower == "error" || lower == "\"error\"") return true;
        }
        return false;
    };

    // 2. Process WHEN clauses
    // Logic: For each WHEN, find filter mask, scatter THEN value
    for (const auto& w : c.cases) {
         // Skip WHEN clauses that contain error guard functions (DuckDB scalar subquery validation)
         // These are never meant to execute in valid queries
         if (containsErrorFunction(w.then)) {
             LOG_DEBUG("Exec", "CASE: skipping WHEN branch with error() guard function\n");
             continue;
         }

         // Clone context to isolate filter effects
         EvalContext subCtx = ctx; // GpuBuffer copy ctor auto-retains activeRowsGPU

         if (GpuExecutor::executeFilterRecursive(w.when, subCtx)) {
             // subCtx.activeRowsGPU now holds the indices where condition is true.
             // Get value for THEN
             float thenVal = 0.0f;
             bool literalThen = false;

             if (w.then->kind == TypedExpr::Kind::Literal) {
                 const auto& lit = w.then->asLiteral();
                 if (std::holds_alternative<int64_t>(lit.value)) thenVal = (float)std::get<int64_t>(lit.value);
                 else if (std::holds_alternative<double>(lit.value)) thenVal = (float)std::get<double>(lit.value);
                 literalThen = true;
             }

             if (subCtx.activeRowsCountGPU > 0) {
                 if (literalThen) {
                     GpuOps::scatterConstantF32(outBuf, subCtx.activeRowsGPU, subCtx.activeRowsCountGPU, thenVal);
                 } else {
                     GpuBuffer thenBuf = GpuExecutor::evaluateExpression(w.then, subCtx);
                     if (thenBuf) {
                         GpuOps::scatterF32(thenBuf, outBuf, subCtx.activeRowsGPU, subCtx.activeRowsCountGPU);
                         // thenBuf auto-releases via GpuBuffer destructor
                     } else {
                         // If evaluation fails (e.g. error function), and we have active rows, we can't proceed on GPU
                         LOG_DEBUG("Exec", "CASE THEN non-literal GPU eval failed\n");
                         return GpuBuffer(); // outBuf auto-releases
                     }
                 }
             }

             // subCtx goes out of scope here; GpuBuffer destructor releases activeRowsGPU
         } else {
             LOG_DEBUG("Exec", "CASE condition eval failed on GPU\n");
             // subCtx goes out of scope; GpuBuffer destructor releases activeRowsGPU
             return GpuBuffer(); // outBuf auto-releases
         }
    }
    return outBuf;
}
    return GpuBuffer(); // unreachable: call site guarantees expr->kind == Case
}

static GpuBuffer evalFunctionExpr(const TypedExprPtr& expr, EvalContext& ctx, uint32_t count) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
if (expr->kind == TypedExpr::Kind::Function) {
    const auto& fn = expr->asFunction();
    std::string fnName = fn.name;

    // Check if this "Function" is actively a Column in the context
    // Try exact name or name()
    std::vector<std::string> candidates = {fnName, fnName + "()"};
    // Case-insensitive versions
    std::string lower = fnName; std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower != fnName) { candidates.push_back(lower); candidates.push_back(lower + "()"); }

    for (const auto& candidate : candidates) {
        if (ctx.f32ColsGPU.count(candidate)) {
            MTL::Buffer* buf = ctx.f32ColsGPU.at(candidate);
            // Respect activeRowsGPU
            if (ctx.activeRowsGPU) {
                 return GpuOps::gatherF32(buf, ctx.activeRowsGPU, count);
            } else {
                 buf->retain();
                 return GpuBuffer(buf);  // reuse existing GPU buffer
            }
        }
        if (ctx.u32ColsGPU.count(candidate)) {
            MTL::Buffer* buf = ctx.u32ColsGPU.at(candidate);
            // Cast U32 -> F32
            if (ctx.activeRowsGPU) {
                 GpuBuffer gathered = GpuOps::gatherU32(buf, ctx.activeRowsGPU, count);
                 return GpuOps::castU32ToF32(gathered, count);
            } else {
                 return GpuOps::castU32ToF32(buf, count);
            }
        }
    }

    std::transform(fnName.begin(), fnName.end(), fnName.begin(), ::toupper);

    // Handle "FIRST" aggregate - DuckDB uses this for scalar subquery results
    // FIRST returns a SCALAR value (the first/only row from a subquery)
    // When evaluating in post-join context, look for pre-computed first(...) column to avoid
    // accidentally matching varying grouped columns instead of the scalar broadcast
    if (fnName == "FIRST" || fnName == "\"FIRST\"") {
        LOG_DEBUG("Exec", "evaluateExpression: FIRST aggregate, evaluating argument\n");
        if (fn.args.size() >= 1 && fn.args[0]) {
            // Look for any pre-computed "first"(...) column that has uniform values (scalar broadcast)
            // This avoids accidentally picking varying grouped columns
            for (const auto& [colName, buf] : ctx.f32ColsGPU) {
                if (!buf) continue;
                std::string colLower = colName;
                std::transform(colLower.begin(), colLower.end(), colLower.begin(), ::tolower);
                // Check if it's a first(...) column
                if (colLower.find("\"first\"") == 0 || colLower.find("first(") == 0) {
                    // Check if values are uniform (scalar broadcast) - this is what FIRST should return
                    float* ptr = static_cast<float*>(buf->contents());
                    size_t n = buf->length() / sizeof(float);
                    bool uniform = true;
                    if (n > 1) {
                        float first = ptr[0];
                        for (size_t i = 1; i < std::min(n, (size_t)10); ++i) {
                            if (std::abs(ptr[i] - first) > 1e-6f) { uniform = false; break; }
                        }
                    }
                    if (uniform) {
                        LOG_DEBUG("Exec", "evaluateExpression: FIRST using scalar column '" << colName << "' val=" << ptr[0]);
                        buf->retain();
                        return GpuBuffer(buf);
                    }
                }
            }

            // Fallback to recursive evaluation (shouldn't be needed if column exists)
            return GpuExecutor::evaluateExpression(fn.args[0], ctx);
        }
    }

    // For aggregates (min, max, sum...), first try to find matching column by name
    if (fnName == "MIN" || fnName == "MAX" || fnName == "SUM" || fnName == "AVG" || fnName == "COUNT") {
        // Build the expected column name pattern like "sum(" (lowercase)
        std::string lowerFn = fnName;
        std::transform(lowerFn.begin(), lowerFn.end(), lowerFn.begin(), ::tolower);
        std::string fnPrefix = lowerFn + "(";

        // Search f32ColsGPU for matching aggregate column
        for (const auto& [colName, buf] : ctx.f32ColsGPU) {
            std::string lowerCol = colName;
            std::transform(lowerCol.begin(), lowerCol.end(), lowerCol.begin(), ::tolower);
            // Check if column starts with the aggregate function
            if (lowerCol.find(fnPrefix) == 0 && buf) {
                LOG_DEBUG("Exec", "evaluateExpression: Found aggregate col '" << colName << "' for " << fnName);
                buf->retain();
                return GpuBuffer(buf);
            }
        }
        // Also check u32ColsGPU
        for (const auto& [colName, buf] : ctx.u32ColsGPU) {
            std::string lowerCol = colName;
            std::transform(lowerCol.begin(), lowerCol.end(), lowerCol.begin(), ::tolower);
            if (lowerCol.find(fnPrefix) == 0 && buf) {
                LOG_DEBUG("Exec", "evaluateExpression: Found aggregate col (u32) '" << colName << "' for " << fnName << ", casting to f32\n");
                if (ctx.activeRowsGPU) {
                    GpuBuffer gathered = GpuOps::gatherU32(buf, ctx.activeRowsGPU, count);
                    return GpuOps::castU32ToF32(gathered, count);
                } else {
                    return GpuOps::castU32ToF32(buf, buf->length() / sizeof(uint32_t));
                }
            }
        }

        // Fallback: positional heuristic for #N columns
        std::string posKey = "#" + std::to_string(ctx.aggregateCounter);
         LOG_DEBUG("Exec", "evaluateExpression: Function heuristic mapping " << fnName << " to " << posKey);

         if (ctx.f32ColsGPU.count(posKey)) {
             MTL::Buffer* buf = ctx.f32ColsGPU[posKey];
             buf->retain(); 
             ctx.aggregateCounter++; 
             return GpuBuffer(buf);
         }
         if (ctx.u32ColsGPU.count(posKey)) {
             MTL::Buffer* buf = ctx.u32ColsGPU[posKey];
             ctx.aggregateCounter++;
             if (ctx.activeRowsGPU) {
                  GpuBuffer gathered = GpuOps::gatherU32(buf, ctx.activeRowsGPU, count);
                  return GpuOps::castU32ToF32(gathered, count);
             } else {
                  return GpuOps::castU32ToF32(buf, count);
             }
         }
         if (ctx.f32Cols.count(posKey) && !ctx.f32Cols[posKey].empty()) {
             float val = ctx.f32Cols[posKey][0];
             ctx.aggregateCounter++;
             return GpuOps::createFilledF32(val, count);
         }
         if (ctx.u32Cols.count(posKey) && !ctx.u32Cols[posKey].empty()) {
             float val = (float)ctx.u32Cols[posKey][0];
             ctx.aggregateCounter++;
             return GpuOps::createFilledF32(val, count);
         }
    }

    // Handle explicit error throws (from scalar subquery checks) or "error" calls
    if (fnName == "ERROR" || fnName == "\"ERROR\"") {
         return GpuOps::createFilledF32(0.0f, count);
    }

    if (fnName == "EXTRACT" && fn.args.size() == 2) {
         const auto& unitArg = fn.args[0];
         const auto& valArg = fn.args[1];

         std::string unitStr;
         if (unitArg->kind == TypedExpr::Kind::Literal) {
             const auto& l = unitArg->asLiteral();
             if (std::holds_alternative<std::string>(l.value)) {
                 unitStr = std::get<std::string>(l.value);
             }
         } else if (unitArg->kind == TypedExpr::Kind::Column) {
             unitStr = unitArg->asColumn().column;
         }

         std::transform(unitStr.begin(), unitStr.end(), unitStr.begin(), ::toupper);

         if (unitStr == "YEAR") {
             GpuBuffer inBuf = GpuExecutor::evaluateExpression(valArg, ctx);
             if (!inBuf) {
                 if (debug) {
                     LOG_ERROR("Exec", "EXTRACT failed: could not evaluate valArg. Kind=" << (int)valArg->kind);
                     if (valArg->kind == TypedExpr::Kind::Column) LOG_DEBUG("EVAL", "  Col: " << valArg->asColumn().column);
                 }
                 return GpuBuffer();
             }

             // EXTRACT(YEAR) logic: floor(val / 10000)
             GpuBuffer divBuf = GpuOps::arithDivF32ColScalar(inBuf, 10000.0f, count);
             GpuBuffer floorBuf = GpuOps::mathFloorF32(divBuf, count);

             // inBuf auto-releases via GpuBuffer destructor
             return floorBuf;
         }
    }

    LOG_DEBUG("Exec", "Unsupported GPU function: " << fn.name);
    return GpuBuffer(); 
}
    return GpuBuffer(); // unreachable: call site guarantees expr->kind == Function
}

// -- Extracted: filterStringFunction --
// Handles LIKE, NOTLIKE, SUFFIX, PREFIX, CONTAINS filter functions via GPU string ops.
static bool filterStringFunction(
    const std::string& fnName, const engine::FunctionCall& fn,
    EvalContext& ctx, bool debug)
{
    engine::GpuFilterOp op = engine::GpuFilterOp::EQ;
    if (fnName == "NOTLIKE") op = engine::GpuFilterOp::NE;
    else if (fnName == "LIKE" || fnName == "SUFFIX" || fnName == "CONTAINS") op = engine::GpuFilterOp::LIKE_PATTERN;

    const TypedExpr* left = unwrapExpr(fn.args[0].get());
    const TypedExpr* right = unwrapExpr(fn.args[1].get());

    if (left->kind != TypedExpr::Kind::Column || right->kind != TypedExpr::Kind::Literal)
        return false;

    std::string colName = left->asColumn().column;
    std::string pat;
    if (std::holds_alternative<std::string>(right->asLiteral().value))
         pat = std::get<std::string>(right->asLiteral().value);

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

    if (!vec && debug) {
        LOG_ERROR("Exec", "DEBUG: String lookup failed for " << colName << ". Available keys: ");
        for (const auto& kv : ctx.stringCols) std::cerr << kv.first << " ";
        LOG_INFO("EVAL", "\n");
    }

    if (!vec) return false;

    LOG_DEBUG("Exec", "Found string col " << colName << " size " << vec->size() << " pattern '" << pat << "'");

    // Check for pre-flattened Arrow-style buffers
    MTL::Buffer *fChars = nullptr, *fOff = nullptr, *fLen = nullptr;
    auto fit = ctx.flatStringCols.find(colName);
    if (fit != ctx.flatStringCols.end() && fit->second.rowCount == vec->size()) {
        fChars = fit->second.chars; fOff = fit->second.offsets; fLen = fit->second.lengths;
    }

    std::optional<FilterResult> res;
    if (fnName == "PREFIX") {
        res = GpuOps::filterStringPrefix(colName, *vec, pat, false, fChars, fOff, fLen);
    } else {
        res = GpuOps::filterString(colName, *vec, op, pat, fChars, fOff, fLen);
    }

    if (!res) return false;

    LOG_DEBUG("Exec", "String Filter Result Count: " << res->count);

    if (ctx.activeRowsGPU) {
        LOG_DEBUG("Exec", "Intersecting with existing " << ctx.activeRowsCountGPU << " rows\n");
        auto joinRes = GpuOps::joinHash(
            ctx.activeRowsGPU, ctx.activeRowsCountGPU,
            res->indices, res->count);
        LOG_DEBUG("Exec", "Intersection Result: " << joinRes.count << " rows\n");
        GpuBuffer newActive = GpuOps::gatherU32(ctx.activeRowsGPU, joinRes.buildIndices, joinRes.count);
        ctx.activeRowsGPU = std::move(newActive);
        ctx.activeRowsCountGPU = joinRes.count;
    } else {
        ctx.activeRowsGPU = std::move(res->indices);
        ctx.activeRowsCountGPU = res->count;
    }
    return true;
}

bool GpuExecutor::executeFilterRecursive(const TypedExprPtr& expr, EvalContext& ctx) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    if (!expr) return true;
    
    // Trivial check: If input is empty, filter does nothing and result is empty
    uint32_t currentInputCount = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
    if (currentInputCount == 0) return true;

    LOG_DEBUG("Exec", "DEBUG REC: Kind=" << (int)expr->kind);

    if (expr->kind == TypedExpr::Kind::Cast) {
        return executeFilterRecursive(expr->asCast().expr, ctx);
    }
    if (expr->kind == TypedExpr::Kind::Alias) {
        return executeFilterRecursive(expr->asAlias().expr, ctx);
    }
    
    if (expr->kind == TypedExpr::Kind::Unary) {
        const auto& un = expr->asUnary();
        if (un.op == UnaryOp::Not) {
             // 1. Capture Input Indices
             MTL::Buffer* inputIndices = ctx.activeRowsGPU;
             uint32_t inputCount = ctx.activeRowsCountGPU;
             if (inputIndices) inputIndices->retain();

             // Handle implicit identity (0..totalRows) if inputIndices is null
             // Use full column capacity, not ctx.rowCount which may have been
             // shrunk by a prior executeFilter call.
             uint32_t totalRows = getTotalRowCapacity(ctx);
             if (!inputIndices && totalRows > 0) {
                 inputCount = totalRows;
             }

             // 2. Execute Operand (Filter for "True")
             if (!executeFilterRecursive(un.operand, ctx)) {
                  if (inputIndices) inputIndices->release();
                  return false; 
             }
             MTL::Buffer* resultIndices = ctx.activeRowsGPU;
             uint32_t resultCount = ctx.activeRowsCountGPU;
             if (resultIndices) resultIndices->retain();

             // 3. Perform Set Difference on GPU: Input - Result
             // Convert both to masks, then do AND NOT, then compact
             MTL::Buffer* inputMask = nullptr;
             if (inputIndices) {
                 inputMask = GpuOps::indicesToMask(inputIndices, inputCount, totalRows).detach();
             } else {
                 // All rows active - create all-ones mask on GPU
                 inputMask = GpuOps::createFilledU32(1, totalRows).detach();
             }

             GpuBuffer resultMask = GpuOps::indicesToMask(resultIndices, resultCount, totalRows);
             
             // diffMask = inputMask AND NOT resultMask
             GpuBuffer diffMask = GpuOps::logicAndNotU32(inputMask, resultMask, totalRows);
             
             // Compact mask back to indices
             auto [diffIndices, diffCount] = GpuOps::compactU32Mask(diffMask, totalRows);
             
             // Cleanup
             if (inputIndices) inputIndices->release();
             if (resultIndices) resultIndices->release();
             if (inputMask) inputMask->release();

             ctx.activeRowsGPU = std::move(diffIndices);
             ctx.activeRowsCountGPU = diffCount;
             return true;
        }
        return false;
    }
    
    if (expr->kind == TypedExpr::Kind::Function) {
        const auto& fn = expr->asFunction();
        LOG_DEBUG("Exec", "DEBUG FUNC: " << fn.name << " args=" << fn.args.size());
        
        // Handle IN function (workaround for truncated plans)
        if (fn.name == "IN") {
            // Workaround for DuckDB truncated plan "IN (...)" -> parsed as Function IN(Column("..."))
            if (fn.args.size() == 1) {
                 const TypedExpr* arg0 = unwrapExpr(fn.args[0].get());
                 if (arg0 && arg0->kind == TypedExpr::Kind::Column && arg0->asColumn().column == "...") {
                     LOG_DEBUG("Exec", "WARNING: Ignoring truncated IN (...) filter. Assuming handled by scan.");
                     return true;
                 }
            }

            if (fn.args.size() >= 2) {
            // Rewrite as Ors: arg[0] IN (arg[1], arg[2]...)
            const TypedExpr* left = unwrapExpr(fn.args[0].get());
            LOG_DEBUG("Exec", "DEBUG IN: left kind=" << (int)left->kind);
            
            TypedExprPtr root = nullptr;
            for (size_t i = 1; i < fn.args.size(); ++i) {
                const TypedExpr* right = unwrapExpr(fn.args[i].get());
                LOG_DEBUG("Exec", "DEBUG IN: arg " << i << " kind=" << (int)right->kind);
                
                if (left->kind == TypedExpr::Kind::Column && right->kind == TypedExpr::Kind::Literal) {
                     auto lCol = std::make_shared<TypedExpr>();
                     lCol->kind = TypedExpr::Kind::Column;
                     lCol->data = engine::ColumnRef{"", left->asColumn().column};
                     
                     auto rLit = std::make_shared<TypedExpr>();
                     rLit->kind = TypedExpr::Kind::Literal;
                     rLit->data = engine::Literal{right->asLiteral().value};

                     auto eq = std::make_shared<TypedExpr>();
                     eq->kind = TypedExpr::Kind::Compare;
                     eq->data = engine::CompareExpr{engine::CompareOp::Eq, lCol, rLit, {}};
                     
                     if (!root) {
                         root = eq;
                     } else {
                         auto orExpr = std::make_shared<TypedExpr>();
                         orExpr->kind = TypedExpr::Kind::Binary;
                         orExpr->data = engine::BinaryExpr{engine::BinaryOp::Or, root, eq};
                         root = orExpr;
                     }
                }
            }
            if (root) return executeFilterRecursive(root, ctx);
            return false;
            }
        }

        // Handle LIKE, NOTLIKE, SUFFIX, PREFIX, CONTAINS
        if ((fn.name == "LIKE" || fn.name == "NOTLIKE" || fn.name == "SUFFIX" || fn.name == "PREFIX" || fn.name == "CONTAINS") && fn.args.size() == 2) {
             return filterStringFunction(fn.name, fn, ctx, debug);
        }
        return false;
    }

    if (expr->kind == TypedExpr::Kind::Binary) {
        const auto& bin = expr->asBinary();
        if (bin.op == BinaryOp::And) {
            // Sequential filtering updates activeRowsGPU
            if (!executeFilterRecursive(bin.left, ctx)) return false;
            return executeFilterRecursive(bin.right, ctx);
        }
        if (bin.op == BinaryOp::Or) {
             // 1. Capture current state (Input Indices)
             MTL::Buffer* inputIndices = ctx.activeRowsGPU;
             uint32_t inputCount = ctx.activeRowsCountGPU;
             // Use full column capacity for mask dimensions — ctx.rowCount may
             // have been shrunk by a prior executeFilter, but indices still
             // reference original (larger) column positions.
             uint32_t totalRows = getTotalRowCapacity(ctx);
             if (inputIndices) inputIndices->retain();

             // 2. Run Left
             if (!executeFilterRecursive(bin.left, ctx)) {
                  if (inputIndices) inputIndices->release();
                  return false;
             }
             MTL::Buffer* leftRes = ctx.activeRowsGPU;
             uint32_t leftCount = ctx.activeRowsCountGPU;
             if (leftRes) leftRes->retain();

             // 3. Restore Input for Right
             ctx.activeRowsGPU.reset(inputIndices); 
             ctx.activeRowsCountGPU = inputCount;
             
             // 4. Run Right
             if (!executeFilterRecursive(bin.right, ctx)) {
                  if (leftRes) leftRes->release();
                  return false;
             }
             MTL::Buffer* rightRes = ctx.activeRowsGPU;
             uint32_t rightCount = ctx.activeRowsCountGPU;
             if (rightRes) rightRes->retain();

             // 5. Union leftRes and rightRes on GPU
             // Convert both to masks, OR them, then compact
             GpuBuffer leftMask = GpuOps::indicesToMask(leftRes, leftCount, totalRows);
             GpuBuffer rightMask = GpuOps::indicesToMask(rightRes, rightCount, totalRows);
             
             GpuBuffer unionMask = GpuOps::logicOrU32(leftMask, rightMask, totalRows);
             
             // Compact mask back to indices
             auto [unionIndices, unionCount] = GpuOps::compactU32Mask(unionMask, totalRows);
             
             // Cleanup
             if (leftRes) leftRes->release();
             if (rightRes) rightRes->release();
             ctx.activeRowsGPU = std::move(unionIndices);
             ctx.activeRowsCountGPU = unionCount;
             return true;
        }
        return false; // Other binary ops not supported
    }
    
    if (expr->kind == TypedExpr::Kind::Compare) {
    return filterCompare(expr, ctx);
    }

    if (expr->kind == TypedExpr::Kind::Column) {
    return filterColumnAsBool(expr, ctx);
    }

    return false;
}

// ============================================================================
// Expression Evaluation
// ============================================================================

// -- Extracted: evalAggregateExpr --
// Resolves an aggregate expression by alias, positional key (#N), or standard
// aggregate prefix (SUM_#, COUNT_#, etc.), searching GPU and CPU columns.
static GpuBuffer evalAggregateExpr(
    const engine::AggregateExpr& agg, EvalContext& ctx, uint32_t count, [[maybe_unused]] bool debug)
{
    // Try alias
    if (!agg.alias.empty() && ctx.f32ColsGPU.count(agg.alias)) {
        MTL::Buffer* buf = ctx.f32ColsGPU[agg.alias];
        buf->retain(); return GpuBuffer(buf);
    }

    // Try positional scalar aggregate lookup (#N)
    bool hasScalarAggregates = (ctx.f32Cols.count("#0") || ctx.f32ColsGPU.count("#0") ||
                                ctx.u32Cols.count("#0") || ctx.u32ColsGPU.count("#0"));

    if (hasScalarAggregates) {
         std::string posKey = "#" + std::to_string(ctx.aggregateCounter);
         if (ctx.f32ColsGPU.count(posKey)) {
             MTL::Buffer* buf = ctx.f32ColsGPU[posKey];
             buf->retain(); ctx.aggregateCounter++; return GpuBuffer(buf);
         }
         if (ctx.f32Cols.count(posKey) && !ctx.f32Cols[posKey].empty()) {
             float val = ctx.f32Cols[posKey][0];
             ctx.aggregateCounter++;
             return GpuOps::createFilledF32(val, count);
         }
         if (ctx.u32Cols.count(posKey) && !ctx.u32Cols[posKey].empty()) {
             float val = (float)ctx.u32Cols[posKey][0];
             ctx.aggregateCounter++;
             return GpuOps::createFilledF32(val, count);
         }
         if (ctx.u32ColsGPU.count(posKey)) {
             ctx.aggregateCounter++;
             return GpuOps::castU32ToF32(ctx.u32ColsGPU[posKey], count);
         }
    }

    // Try standard aggregate prefixes
    std::string prefix;
    switch (agg.func) {
        case AggFunc::Sum: prefix = "SUM_#"; break;
        case AggFunc::Count:
        case AggFunc::CountStar: prefix = "COUNT_#"; break;
        case AggFunc::Avg: prefix = "AVG_#"; break;
        case AggFunc::Min: prefix = "MIN_#"; break;
        case AggFunc::Max: prefix = "MAX_#"; break;
        default: prefix = "AGG_#"; break;
    }

    for (const auto& [name, buf] : ctx.f32ColsGPU) {
        if (name.rfind(prefix, 0) == 0) { buf->retain(); return GpuBuffer(buf.get()); }
    }
    for (const auto& [name, vec] : ctx.f32Cols) {
        if (name.rfind(prefix, 0) == 0 && !vec.empty())
            return GpuOps::createFilledF32(vec[0], count);
    }
    for (const auto& [name, vec] : ctx.u32Cols) {
        if (name.rfind(prefix, 0) == 0 && !vec.empty())
            return GpuOps::createFilledF32((float)vec[0], count);
    }
    for (const auto& [name, buf] : ctx.u32ColsGPU) {
        if (name.rfind(prefix, 0) == 0)
            return GpuOps::castU32ToF32(buf, count);
    }

    return GpuBuffer();
}

GpuBuffer GpuExecutor::evaluateExpression(const TypedExprPtr& expr, EvalContext& ctx) {
    // RAII guard: batch GPU arithmetic ops within expression evaluation.
    // Top-level call enables batching; nested recursive calls are no-ops.
    // On top-level destructor, flushes the GPU queue so all results are ready.
    struct BatchGuard {
        bool topLevel;
        BatchGuard() : topLevel(!GpuOps::isBatchActive()) {
            if (topLevel) GpuOps::beginBatch();
        }
        ~BatchGuard() {
            if (topLevel) GpuOps::endBatch();
        }
    } batchGuard;

    if (!expr) return GpuBuffer();
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");

    // Determine effective row count for output buffer
    uint32_t count = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
    // Fallback if row count seems wrong
    if (count == 0 && ctx.rowCount > 0 && !ctx.activeRowsGPU) count = ctx.rowCount;
    if (count == 0) return GpuBuffer();
    
    if (expr->kind == TypedExpr::Kind::Column) {
    return evalColumnExpr(expr, ctx, count);
    }
    
    if (expr->kind == TypedExpr::Kind::Literal) {
        float val = 0.0f;
        const auto& lit = expr->asLiteral();
        if (std::holds_alternative<double>(lit.value)) val = (float)std::get<double>(lit.value);
        else if (std::holds_alternative<int64_t>(lit.value)) val = (float)std::get<int64_t>(lit.value);
        
        return GpuOps::createFilledF32(val, count);
    }

    if (expr->kind == TypedExpr::Kind::Aggregate) {
        GpuBuffer result = evalAggregateExpr(expr->asAggregate(), ctx, count, debug);
        if (result) return result;
    }

    if (expr->kind == TypedExpr::Kind::Compare) {
        // Debug info
        if (debug) {
              const auto& c = expr->asCompare();
              LOG_INFO("Exec", "Eval Compare: LeftKind=" << (c.left ? (int)c.left->kind : -1) << " RightKind=" << (c.right ? (int)c.right->kind : -1));
              if (c.left && c.left->kind == TypedExpr::Kind::Column) LOG_DEBUG("EVAL", "  LeftCol: " << c.left->asColumn().column);
        }

        // Evaluate predicate and cast to float (1.0 for true, 0.0 for false)
        EvalContext subCtx = ctx; // GpuBuffer copy ctor auto-retains activeRowsGPU

        if (executeFilterRecursive(std::const_pointer_cast<TypedExpr>(expr), subCtx)) {
            // subCtx.activeRowsGPU matches condition
            GpuBuffer outBuf = GpuOps::createFilledF32(0.0f, count);

            if (subCtx.activeRowsCountGPU > 0) {
                 // Scatter 1.0 to matching rows
                 GpuOps::scatterConstantF32(outBuf, subCtx.activeRowsGPU, subCtx.activeRowsCountGPU, 1.0f);
            }
            
            return outBuf;
        } else {
            // Filter eval failed -> return nullptr
            LOG_DEBUG("Exec", "Eval Compare: executeFilterRecursive failed\n");
            return GpuBuffer();
        }
    }

    if (expr->kind == TypedExpr::Kind::Binary) {
        const auto& bin = expr->asBinary();
        if (bin.op == BinaryOp::Mul || bin.op == BinaryOp::Sub || bin.op == BinaryOp::Add || bin.op == BinaryOp::Div) {
            float leftVal = 0; bool leftIsLit = false;
            float rightVal = 0; bool rightIsLit = false;
            bool isMul = (bin.op == BinaryOp::Mul);
            bool isAdd = (bin.op == BinaryOp::Add);
            bool isDiv = (bin.op == BinaryOp::Div);

            if (bin.left->kind == TypedExpr::Kind::Literal) {
                 auto& lit = bin.left->asLiteral();
                 if (std::holds_alternative<double>(lit.value)) leftVal = (float)std::get<double>(lit.value);
                 else if (std::holds_alternative<int64_t>(lit.value)) leftVal = (float)std::get<int64_t>(lit.value);
                 leftIsLit = true;
            }
            
            if (bin.right->kind == TypedExpr::Kind::Literal) {
                 auto& lit = bin.right->asLiteral();
                 if (std::holds_alternative<double>(lit.value)) rightVal = (float)std::get<double>(lit.value);
                 else if (std::holds_alternative<int64_t>(lit.value)) rightVal = (float)std::get<int64_t>(lit.value);
                 rightIsLit = true;
            }
            
            if (leftIsLit) {
                // Lit op Right — GpuBuffer auto-releases rightBuf
                GpuBuffer rightBuf = evaluateExpression(bin.right, ctx);
                if (!rightBuf) return GpuBuffer();
                if (isMul) return GpuOps::arithMulF32ColScalar(rightBuf, leftVal, count);
                else if (isAdd) return GpuOps::arithAddF32ColScalar(rightBuf, leftVal, count);
                else if (isDiv) return GpuOps::arithDivF32ScalarCol(leftVal, rightBuf, count);
                else return GpuOps::arithSubF32ScalarCol(leftVal, rightBuf, count);
            } else if (rightIsLit) {
                // Left op Lit — GpuBuffer auto-releases leftBuf
                GpuBuffer leftBuf = evaluateExpression(bin.left, ctx);
                if (!leftBuf) return GpuBuffer();
                if (isMul) return GpuOps::arithMulF32ColScalar(leftBuf, rightVal, count);
                else if (isAdd) return GpuOps::arithAddF32ColScalar(leftBuf, rightVal, count);
                else if (isDiv) return GpuOps::arithDivF32ColScalar(leftBuf, rightVal, count);
                else return GpuOps::arithSubF32ColScalar(leftBuf, rightVal, count);
            } else {
                // Left op Right — GpuBuffers auto-release on scope exit
                GpuBuffer leftBuf = evaluateExpression(bin.left, ctx);
                GpuBuffer rightBuf = evaluateExpression(bin.right, ctx);
                if (!leftBuf || !rightBuf) return GpuBuffer();
                if (isMul) return GpuOps::arithMulF32ColCol(leftBuf, rightBuf, count);
                else if (isAdd) return GpuOps::arithAddF32ColCol(leftBuf, rightBuf, count);
                else if (isDiv) return GpuOps::arithDivF32ColCol(leftBuf, rightBuf, count);
                else return GpuOps::arithSubF32ColCol(leftBuf, rightBuf, count);
            }
        }
    }
    
    if (expr->kind == TypedExpr::Kind::Alias) {
        return evaluateExpression(expr->asAlias().expr, ctx);
    }

    if (expr->kind == TypedExpr::Kind::Cast) {
        return evaluateExpression(expr->asCast().expr, ctx);
    }
    
    if (expr->kind == TypedExpr::Kind::Case) {
    return evalCaseExpr(expr, ctx, count);
    }

    if (expr->kind == TypedExpr::Kind::Function) {
    return evalFunctionExpr(expr, ctx, count);
    }

    return GpuBuffer();
}

} // namespace engine
