#include "GpuExecutorPriv.hpp"
#include "Operators.hpp" // Included for GpuOps::filter*
#include "Predicate.hpp"    // Included for engine::expr::CompOp
#include "ColumnStoreGPU.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace engine {

// ============================================================================
// Operator Implementations
// ============================================================================

static std::optional<engine::expr::CompOp> mapCompOp(engine::CompareOp op) {
    switch (op) {
        case engine::CompareOp::Eq: return engine::expr::CompOp::EQ;
        case engine::CompareOp::Ne: return engine::expr::CompOp::NE;
        case engine::CompareOp::Lt: return engine::expr::CompOp::LT;
        case engine::CompareOp::Le: return engine::expr::CompOp::LE;
        case engine::CompareOp::Gt: return engine::expr::CompOp::GT;
        case engine::CompareOp::Ge: return engine::expr::CompOp::GE;
        default: return std::nullopt;
    }
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

bool GpuExecutor::executeGPUFilterRecursive(const TypedExprPtr& expr, EvalContext& ctx) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    if (!expr) return true;
    
    // Trivial check: If input is empty, filter does nothing and result is empty
    uint32_t currentInputCount = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
    if (currentInputCount == 0) return true;

    if (debug) std::cerr << "[Exec] DEBUG REC: Kind=" << (int)expr->kind << "\n";

    // Helper to unwrap Cast/Alias
    auto unwrap = [](const TypedExpr* e) -> const TypedExpr* {
        while (e) {
            if (e->kind == TypedExpr::Kind::Cast) {
                e = e->asCast().expr.get();
            } else if (e->kind == TypedExpr::Kind::Alias) {
                e = e->asAlias().expr.get();
            } else {
                break;
            }
        }
        return e;
    };

    if (expr->kind == TypedExpr::Kind::Cast) {
        return executeGPUFilterRecursive(expr->asCast().expr, ctx);
    }
    if (expr->kind == TypedExpr::Kind::Alias) {
        return executeGPUFilterRecursive(expr->asAlias().expr, ctx);
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
             if (!executeGPUFilterRecursive(un.operand, ctx)) {
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
                 inputMask = GpuOps::indicesToMask(inputIndices, inputCount, totalRows);
             } else {
                 // All rows active - create all-ones mask on GPU
                 inputMask = GpuOps::createFilledU32(1, totalRows);
             }

             MTL::Buffer* resultMask = GpuOps::indicesToMask(resultIndices, resultCount, totalRows);
             
             // diffMask = inputMask AND NOT resultMask
             MTL::Buffer* diffMask = GpuOps::logicAndNotU32(inputMask, resultMask, totalRows);
             
             // Compact mask back to indices
             auto [diffIndices, diffCount] = GpuOps::compactU32Mask(diffMask, totalRows);
             
             // Cleanup
             if (inputIndices) inputIndices->release();
             if (resultIndices) resultIndices->release();
             if (inputMask) inputMask->release();
             if (resultMask) resultMask->release();
             if (diffMask) diffMask->release();
             if (ctx.activeRowsGPU) ctx.activeRowsGPU->release();

             ctx.activeRowsGPU = diffIndices;
             ctx.activeRowsCountGPU = diffCount;
             return true;
        }
        return false;
    }
    
    if (expr->kind == TypedExpr::Kind::Function) {
        const auto& fn = expr->asFunction();
        if (debug) std::cerr << "[Exec] DEBUG FUNC: " << fn.name << " args=" << fn.args.size() << "\n";
        
        // Handle IN function (workaround for truncated plans)
        if (fn.name == "IN") {
            // Workaround for DuckDB truncated plan "IN (...)" -> parsed as Function IN(Column("..."))
            if (fn.args.size() == 1) {
                 const TypedExpr* arg0 = unwrap(fn.args[0].get());
                 if (arg0 && arg0->kind == TypedExpr::Kind::Column && arg0->asColumn().column == "...") {
                     if (debug) std::cerr << "[Exec] WARNING: Ignoring truncated IN (...) filter. Assuming handled by scan." << std::endl;
                     return true;
                 }
            }

            if (fn.args.size() >= 2) {
            // Rewrite as Ors: arg[0] IN (arg[1], arg[2]...)
            const TypedExpr* left = unwrap(fn.args[0].get());
            if (debug) std::cerr << "[Exec] DEBUG IN: left kind=" << (int)left->kind << "\n";
            
            TypedExprPtr root = nullptr;
            for (size_t i = 1; i < fn.args.size(); ++i) {
                const TypedExpr* right = unwrap(fn.args[i].get());
                if (debug) std::cerr << "[Exec] DEBUG IN: arg " << i << " kind=" << (int)right->kind << "\n";
                
                if (left->kind == TypedExpr::Kind::Column && right->kind == TypedExpr::Kind::Literal) {
                     auto lCol = std::make_shared<TypedExpr>();
                     lCol->kind = TypedExpr::Kind::Column;
                     lCol->data = engine::ColumnRef{left->asColumn().column};
                     
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
            if (root) return executeGPUFilterRecursive(root, ctx);
            return false;
            }
        }

        // Handle LIKE, NOTLIKE, SUFFIX, PREFIX, CONTAINS
        if ((fn.name == "LIKE" || fn.name == "NOTLIKE" || fn.name == "SUFFIX" || fn.name == "PREFIX" || fn.name == "CONTAINS") && fn.args.size() == 2) {
             engine::expr::CompOp op = engine::expr::CompOp::EQ;
             if (fn.name == "NOTLIKE") op = engine::expr::CompOp::NE;
             else if (fn.name == "LIKE" || fn.name == "SUFFIX" || fn.name == "CONTAINS") op = (engine::expr::CompOp)999;

             const TypedExpr* left = unwrap(fn.args[0].get());
             const TypedExpr* right = unwrap(fn.args[1].get());
             
             if (left->kind == TypedExpr::Kind::Column && right->kind == TypedExpr::Kind::Literal) {
                 std::string colName = left->asColumn().column;
                 std::string pat = "";
                 if (std::holds_alternative<std::string>(right->asLiteral().value)) {
                      pat = std::get<std::string>(right->asLiteral().value);
                 }
                 
                 const std::vector<std::string>* vec = nullptr;
                 if (ctx.stringCols.count(colName)) {
                     vec = &ctx.stringCols.at(colName);
                 } else {
                     // Try suffixed
                     for(int i=1; i<=9; ++i) {
                         std::string s = colName+"_"+std::to_string(i);
                         if (ctx.stringCols.count(s)) { vec = &ctx.stringCols.at(s); colName=s; break; }
                     }
                 }

                 if (!vec && debug) {
                     std::cerr << "[Exec] DEBUG: String lookup failed for " << colName << ". Available keys: ";
                     for(const auto& kv : ctx.stringCols) std::cerr << kv.first << " ";
                     std::cerr << "\n";
                 }

                 if (vec) {
                      if (debug) {
                          std::cerr << "[Exec] Found string col " << colName << " size " << vec->size() << " pattern '" << pat << "'" << std::endl;
                      }
                      
                      std::optional<FilterResult> res;
                      // Prefer Arrow-style flat string path (avoids re-flattening)
                      auto flatIt = ctx.flatStringColsGPU.find(colName);
                      if (flatIt != ctx.flatStringColsGPU.end() && flatIt->second.valid()) {
                          if (fn.name == "PREFIX") {
                              res = GpuOps::filterStringPrefixFlat(colName, flatIt->second, pat, false);
                          } else {
                              res = GpuOps::filterStringFlat(colName, flatIt->second, op, pat);
                          }
                      } else {
                          if (fn.name == "PREFIX") {
                              res = GpuOps::filterStringPrefix(colName, *vec, pat, false);
                          } else {
                              res = GpuOps::filterString(colName, *vec, op, pat);
                          }
                      }
                      
                      if (res) {
                          if (debug) std::cerr << "[Exec] String Filter Result Count: " << res->count << "\n";
                          // Handle context intersection (GPU Join logic)
                          if (ctx.activeRowsGPU) {
                              if (debug) std::cerr << "[Exec] Intersecting with existing " << ctx.activeRowsCountGPU << " rows\n";
                              auto joinRes = GpuOps::joinHash(
                                  ctx.activeRowsGPU, nullptr, ctx.activeRowsCountGPU,
                                  res->indices, nullptr, res->count
                              );
                              if (debug) std::cerr << "[Exec] Intersection Result: " << joinRes.count << " rows\n";
                              
                              MTL::Buffer* newActive = GpuOps::gatherU32(ctx.activeRowsGPU, joinRes.buildIndices, joinRes.count);
                              
                              if (ctx.activeRowsGPU) ctx.activeRowsGPU->release();
                              if (res->indices) res->indices->release();
                              if (joinRes.buildIndices) joinRes.buildIndices->release();
                              if (joinRes.probeIndices) joinRes.probeIndices->release();
                              
                              ctx.activeRowsGPU = newActive;
                              ctx.activeRowsCountGPU = joinRes.count;
                          } else {
                              ctx.activeRowsGPU = res->indices;
                              ctx.activeRowsCountGPU = res->count;
                          }
                          return true;
                      }
                 }
             }
        }
        return false;
    }

    if (expr->kind == TypedExpr::Kind::Binary) {
        const auto& bin = expr->asBinary();
        if (bin.op == BinaryOp::And) {
            // Sequential filtering updates activeRowsGPU
            if (!executeGPUFilterRecursive(bin.left, ctx)) return false;
            return executeGPUFilterRecursive(bin.right, ctx);
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
             if (!executeGPUFilterRecursive(bin.left, ctx)) {
                  if (inputIndices) inputIndices->release();
                  return false;
             }
             MTL::Buffer* leftRes = ctx.activeRowsGPU;
             uint32_t leftCount = ctx.activeRowsCountGPU;
             if (leftRes) leftRes->retain();

             // 3. Restore Input for Right
             if (ctx.activeRowsGPU) ctx.activeRowsGPU->release(); 
             ctx.activeRowsGPU = inputIndices; 
             ctx.activeRowsCountGPU = inputCount;
             
             // 4. Run Right
             if (!executeGPUFilterRecursive(bin.right, ctx)) {
                  if (leftRes) leftRes->release();
                  return false;
             }
             MTL::Buffer* rightRes = ctx.activeRowsGPU;
             uint32_t rightCount = ctx.activeRowsCountGPU;
             if (rightRes) rightRes->retain();

             // 5. Union leftRes and rightRes on GPU
             // Convert both to masks, OR them, then compact
             MTL::Buffer* leftMask = GpuOps::indicesToMask(leftRes, leftCount, totalRows);
             MTL::Buffer* rightMask = GpuOps::indicesToMask(rightRes, rightCount, totalRows);
             
             MTL::Buffer* unionMask = GpuOps::logicOrU32(leftMask, rightMask, totalRows);
             
             // Compact mask back to indices
             auto [unionIndices, unionCount] = GpuOps::compactU32Mask(unionMask, totalRows);
             
             // Cleanup
             if (leftRes) leftRes->release();
             if (rightRes) rightRes->release();
             if (leftMask) leftMask->release();
             if (rightMask) rightMask->release();
             if (unionMask) unionMask->release();
             if (ctx.activeRowsGPU) ctx.activeRowsGPU->release();

             ctx.activeRowsGPU = unionIndices;
             ctx.activeRowsCountGPU = unionCount;
             return true;
        }
        return false; // Other binary ops not supported
    }
    
    if (expr->kind == TypedExpr::Kind::Compare) {
        const auto& cmp = expr->asCompare();

        // Helper to unwrap Cast/Alias
        auto unwrap = [](const TypedExpr* e) -> const TypedExpr* {
            while (e) {
                if (e->kind == TypedExpr::Kind::Cast) e = e->asCast().expr.get();
                else if (e->kind == TypedExpr::Kind::Alias) e = e->asAlias().expr.get();
                else break;
            }
            return e;
        };

        const TypedExpr* leftRaw = unwrap(cmp.left.get());
        const TypedExpr* rightRaw = unwrap(cmp.right.get());

        // Handle Literal vs Literal (e.g. 1=1)
        if (leftRaw && rightRaw && 
            leftRaw->kind == TypedExpr::Kind::Literal && 
            rightRaw->kind == TypedExpr::Kind::Literal) {
            
            if (debug) std::cerr << "[Exec] DEBUG: Literal vs Literal comparison\n";
            bool result = false;
            // Use variant equality
            if (cmp.op == engine::CompareOp::Eq) {
                result = (leftRaw->asLiteral().value == rightRaw->asLiteral().value);
            } else if (cmp.op == engine::CompareOp::Ne) {
                result = (leftRaw->asLiteral().value != rightRaw->asLiteral().value);
            }
            
            if (result) {
                // Ensure count is consistent for implicit "all rows" state
                if (!ctx.activeRowsGPU && ctx.activeRowsCountGPU == 0) {
                     ctx.activeRowsCountGPU = ctx.rowCount;
                }
                return true; // Keep all rows
            } else {
                // Clear all rows
                if(ctx.activeRowsGPU) ctx.activeRowsGPU->release();
                ctx.activeRowsGPU = GpuOps::createBuffer(nullptr, 4);
                ctx.activeRowsCountGPU = 0;
                return true;
            }
        }

        // Handle Col vs Col
        if (leftRaw && rightRaw && 
            leftRaw->kind == TypedExpr::Kind::Column && 
            rightRaw->kind == TypedExpr::Kind::Column) {
            
             std::string lName = leftRaw->asColumn().column;
             std::string rName = rightRaw->asColumn().column;

             // Helper to resolve column name
             auto resolveCol = [&](const std::string& name) -> std::string {
                 if (ctx.u32Cols.count(name) || ctx.f32Cols.count(name)) return name;
                 for(int i=1; i<=9; ++i) {
                     std::string s = name+"_"+std::to_string(i);
                     if (ctx.u32Cols.count(s) || ctx.f32Cols.count(s)) return s;
                 }
                 std::string rhsPattern = name + "_rhs_";
                 for (const auto& [n, _] : ctx.u32Cols) if (n.find(rhsPattern) == 0) return n;
                 for (const auto& [n, _] : ctx.f32Cols) if (n.find(rhsPattern) == 0) return n;
                 return "";
             };

             std::string lActual = resolveCol(lName);
             std::string rActual = resolveCol(rName);

             if (!lActual.empty() && !rActual.empty()) {
                  if (debug) std::cerr << "[Exec] Col vs Col: " << lActual << " vs " << rActual << "\n";
                  
                  // Determine type. Prefer F32 if mismatch? Or assume same.
                  bool lIsF32 = ctx.f32Cols.count(lActual);
                  bool rIsF32 = ctx.f32Cols.count(rActual);
                  
                  // Map Op
                  int opInt = 0;
                  if (cmp.op == engine::CompareOp::Eq) opInt = (int)engine::expr::CompOp::EQ;
                  else if (cmp.op == engine::CompareOp::Ne) opInt = (int)engine::expr::CompOp::NE;
                  else if (cmp.op == engine::CompareOp::Lt) opInt = (int)engine::expr::CompOp::LT;
                  else if (cmp.op == engine::CompareOp::Le) opInt = (int)engine::expr::CompOp::LE;
                  else if (cmp.op == engine::CompareOp::Gt) opInt = (int)engine::expr::CompOp::GT;
                  else if (cmp.op == engine::CompareOp::Ge) opInt = (int)engine::expr::CompOp::GE;
                  else return false;

                  MTL::Buffer* rootA = (lIsF32) ? ctx.f32ColsGPU.at(lActual) : ctx.u32ColsGPU.at(lActual);
                  MTL::Buffer* rootB = (rIsF32) ? ctx.f32ColsGPU.at(rActual) : ctx.u32ColsGPU.at(rActual);
                  
                  MTL::Buffer* finalA = rootA;
                  MTL::Buffer* finalB = rootB;
                  bool freeFinalA = false;
                  bool freeFinalB = false;
                  uint32_t workingCount = (ctx.activeRowsGPU) ? ctx.activeRowsCountGPU : ctx.rowCount;

                  bool useF32 = lIsF32 || rIsF32;
                  
                  // 1. Cast if needed (Global scale)
                  if (useF32 && !lIsF32) {
                      finalA = GpuOps::castU32ToF32(rootA, ctx.rowCount);
                      freeFinalA = true;
                  }
                  if (useF32 && !rIsF32) {
                      finalB = GpuOps::castU32ToF32(rootB, ctx.rowCount);
                      freeFinalB = true;
                  }

                  // 2. Gather if active rows
                  if (ctx.activeRowsGPU) {
                       MTL::Buffer* gA = (useF32) ? GpuOps::gatherF32(finalA, ctx.activeRowsGPU, workingCount) 
                                                  : GpuOps::gatherU32(finalA, ctx.activeRowsGPU, workingCount);
                       if (freeFinalA) finalA->release(); // Release intermediate
                       finalA = gA;
                       freeFinalA = true; // Gathered buffer needs release

                       MTL::Buffer* gB = (useF32) ? GpuOps::gatherF32(finalB, ctx.activeRowsGPU, workingCount) 
                                                  : GpuOps::gatherU32(finalB, ctx.activeRowsGPU, workingCount);
                       if (freeFinalB) finalB->release();
                       finalB = gB;
                       freeFinalB = true;
                  }
                  
                  // 3. Execute
                  std::optional<FilterResult> res;
                  if (useF32) res = GpuOps::filterColColF32(finalA, finalB, workingCount, opInt);
                  else res = GpuOps::filterColColU32(finalA, finalB, workingCount, opInt);

                  // 4. Cleanup
                  if (freeFinalA && finalA) finalA->release();
                  if (freeFinalB && finalB) finalB->release();

                  if (res) {
                      if (ctx.activeRowsGPU) {
                           // Combine indices
                           MTL::Buffer* newActive = GpuOps::gatherU32(ctx.activeRowsGPU, res->indices, res->count);
                           if (ctx.activeRowsGPU) ctx.activeRowsGPU->release();
                           if (res->indices) res->indices->release();
                           ctx.activeRowsGPU = newActive;
                           ctx.activeRowsCountGPU = res->count;
                      } else {
                           ctx.activeRowsGPU = res->indices;
                           ctx.activeRowsCountGPU = res->count;
                      }
                      return true;
                  }
             }
        }
        
        // Identify column and literal
        std::string colName;
        bool isF32 = false;
        
        // Normalize: Col op Lit
        const TypedExpr* colExpr = nullptr;
        const TypedExpr* litExpr = nullptr;


        // Handle String Filter (Like OR Eq)
        if (cmp.op == engine::CompareOp::Like || cmp.op == engine::CompareOp::Eq) {
             const TypedExpr* left = unwrap(cmp.left.get());
             const TypedExpr* right = unwrap(cmp.right.get());
             if (left->kind == TypedExpr::Kind::Column && right->kind == TypedExpr::Kind::Literal && 
                 std::holds_alternative<std::string>(right->asLiteral().value)) {
                 
                 std::string colName = left->asColumn().column;
                 std::string pat = std::get<std::string>(right->asLiteral().value);
                 
                 const std::vector<std::string>* vec = nullptr;
                 if (ctx.stringCols.count(colName)) {
                     vec = &ctx.stringCols.at(colName);
                 } else {
                     // Try suffixed
                     for(int i=1; i<=9; ++i) {
                         std::string s = colName+"_"+std::to_string(i);
                         if (ctx.stringCols.count(s)) { vec = &ctx.stringCols.at(s); colName=s; break; }
                     }
                     // Try _rhs_
                     if (!vec) {
                         std::string rhsPattern = colName + "_rhs_";
                         for (const auto& [name, v] : ctx.stringCols) {
                             if (name.find(rhsPattern) == 0) { vec = &v; colName = name; break; }
                         }
                     }
                 }

                 if (vec) {
                      if (std::getenv("GPUDB_DEBUG_OPS")) {
                          std::cerr << "[Exec] Found string col " << colName << " size " << vec->size() << " pattern '" << pat << "'" << std::endl;
                      }
                      
                      engine::expr::CompOp op = (cmp.op == engine::CompareOp::Like) ? (engine::expr::CompOp)999 : engine::expr::CompOp::EQ;
                      // Prefer flat string path
                      std::optional<FilterResult> res;
                      auto flatIt = ctx.flatStringColsGPU.find(colName);
                      if (flatIt != ctx.flatStringColsGPU.end() && flatIt->second.valid()) {
                          res = GpuOps::filterStringFlat(colName, flatIt->second, op, pat);
                      } else {
                          res = GpuOps::filterString(colName, *vec, op, pat);
                      }
                      
                      if (res) {
                          if (ctx.activeRowsGPU) {
                              auto joinRes = GpuOps::joinHash( ctx.activeRowsGPU, nullptr, ctx.activeRowsCountGPU, res->indices, nullptr, res->count);
                              MTL::Buffer* newActive = GpuOps::gatherU32(ctx.activeRowsGPU, joinRes.buildIndices, joinRes.count);
                              
                              if (ctx.activeRowsGPU) ctx.activeRowsGPU->release();
                              if (res->indices) res->indices->release();
                              if (joinRes.buildIndices) joinRes.buildIndices->release();
                              if (joinRes.probeIndices) joinRes.probeIndices->release();
                              
                              ctx.activeRowsGPU = newActive;
                              ctx.activeRowsCountGPU = joinRes.count;
                          } else {
                              ctx.activeRowsGPU = res->indices;
                              ctx.activeRowsCountGPU = res->count;
                          }
                          return true;
                      }
                 }
             }
             if (cmp.op == engine::CompareOp::Like) return false;
        }

        // Optimized Hash Filter Fallback: Col(StringHash) = "Literal"
        if (cmp.op == engine::CompareOp::Eq || cmp.op == engine::CompareOp::Ne) {
             const TypedExpr* left = unwrap(cmp.left.get());
             const TypedExpr* right = unwrap(cmp.right.get());
             if (left->kind == TypedExpr::Kind::Column && right->kind == TypedExpr::Kind::Literal && 
                 std::holds_alternative<std::string>(right->asLiteral().value)) {
                 
                 std::string colName = left->asColumn().column;
                 std::string pat = std::get<std::string>(right->asLiteral().value);
                 
                 // Try to find the U32 column (Hash) if String column wasn't found above
                 // Resolve actual U32 col name (suffixes/rhs)
                 std::string actualCol = "";
                 if (ctx.u32Cols.count(colName)) actualCol = colName;
                 else {
                     for(int i=1; i<=9; ++i) {
                         if (ctx.u32Cols.count(colName+"_"+std::to_string(i))) { actualCol = colName+"_"+std::to_string(i); break; }
                     }
                     if (actualCol.empty()) {
                         std::string rhsPattern = colName + "_rhs_";
                         for (const auto& [name, _] : ctx.u32Cols) {
                             if (name.find(rhsPattern) == 0) { actualCol = name; break; }
                         }
                     }
                 }
                 
                 if (!actualCol.empty()) {
                     if (std::getenv("GPUDB_DEBUG_OPS")) {
                          std::cerr << "[Exec] Found StringHash col " << actualCol << " for pattern '" << pat << "' (hashing)" << std::endl;
                     }
                     uint32_t hashVal = GpuOps::fnv1a32(pat);
                     engine::expr::CompOp op = (cmp.op == engine::CompareOp::Eq) ? engine::expr::CompOp::EQ : engine::expr::CompOp::NE;
                     
                     MTL::Buffer* colBuf = ctx.u32ColsGPU.at(actualCol);
                     uint32_t count = (ctx.activeRowsGPU) ? ctx.activeRowsCountGPU : ctx.rowCount;
                     
                     std::optional<FilterResult> res;
                     if (ctx.activeRowsGPU) {
                         res = GpuOps::filterU32Indexed(actualCol, colBuf, ctx.activeRowsGPU, count, op, hashVal);
                     } else {
                         res = GpuOps::filterU32(actualCol, colBuf, count, op, hashVal);
                     }
                     
                     if (res) {
                         if (ctx.activeRowsGPU) ctx.activeRowsGPU->release();
                         ctx.activeRowsGPU = res->indices;
                         ctx.activeRowsCountGPU = res->count;
                         return true;
                     }
                 }
             }
        }

        // Handle IN Operator via rewriting to ORs (GPU Path)
        if (cmp.op == engine::CompareOp::In) {
             const TypedExpr* leftExprRaw = unwrap(cmp.left.get());
             if (cmp.inList.empty()) {
                  // IN () -> False / Empty
                  if(ctx.activeRowsGPU) ctx.activeRowsGPU->release();
                  ctx.activeRowsGPU = GpuOps::createBuffer(nullptr, 4);
                  ctx.activeRowsCountGPU = 0;
                  return true;
             }
             
             bool isCol = (leftExprRaw->kind == TypedExpr::Kind::Column);
             std::string colName;
             if(isCol) colName = leftExprRaw->asColumn().column;
             else if (leftExprRaw->kind == TypedExpr::Kind::Function) {
                 const auto& fn = leftExprRaw->asFunction();
                 // Handle substring(col, start, len) IN (...)
                 // Note: DuckDB often calls it "substring" or "substr"
                 std::string fnName = fn.name;
                 std::transform(fnName.begin(), fnName.end(), fnName.begin(), ::tolower);
                 
                 if ((fnName == "substring" || fnName == "substr") && fn.args.size() >= 1) {
                     const TypedExpr* arg0 = unwrap(fn.args[0].get());
                     if (arg0->kind == TypedExpr::Kind::Column) {
                         std::string baseCol = arg0->asColumn().column;
                         // Resolve base column
                         const std::vector<std::string>* vec = nullptr;
                         if (ctx.stringCols.count(baseCol)) vec = &ctx.stringCols.at(baseCol);
                         else {
                             // Try suffixed
                             for(int i=1; i<=9; ++i) {
                                 std::string s = baseCol+"_"+std::to_string(i);
                                 if (ctx.stringCols.count(s)) { vec = &ctx.stringCols.at(s); baseCol = s; break; }
                             }
                             // Try _rhs_
                             if (!vec) {
                                 std::string rhsCheck = baseCol + "_rhs_";
                                 for (const auto& [name, v] : ctx.stringCols) {
                                     if (name.find(rhsCheck) == 0) { vec = &v; baseCol = name; break; }
                                 }
                             }
                         }
                         
                         int start = 1, len = -1;
                         if (fn.args.size()>=2 && fn.args[1]->kind==TypedExpr::Kind::Literal) {
                             if (std::holds_alternative<int64_t>(fn.args[1]->asLiteral().value))
                                start = (int)std::get<int64_t>(fn.args[1]->asLiteral().value);
                         }
                         if (fn.args.size()>=3 && fn.args[2]->kind==TypedExpr::Kind::Literal) {
                             if (std::holds_alternative<int64_t>(fn.args[2]->asLiteral().value))
                                len = (int)std::get<int64_t>(fn.args[2]->asLiteral().value);
                         }

                         if (vec) {
                             // Perform CPU substring
                             std::vector<std::string> newVec;
                             newVec.reserve(vec->size());
                             for(const auto& s : *vec) {
                                 if (start > (int)s.size()) newVec.push_back("");
                                 else {
                                    int realLen = (len == -1) ? (s.size() - start + 1) : len;
                                    if (start + realLen - 1 > (int)s.size()) realLen = s.size() - start + 1;
                                    newVec.push_back(s.substr(start-1, realLen));
                                 }
                             }
                             // Register temp column
                             colName = "tmp_sub_" + baseCol;
                             ctx.stringCols[colName] = std::move(newVec);
                             // Create flat string column for GPU filter path
                             {
                                 auto& cstoreTmp = ColumnStoreGPU::instance();
                                 ctx.flatStringColsGPU[colName] = makeFlatStringColumn(cstoreTmp.device(), ctx.stringCols[colName]);
                             }
                             isCol = true;
                         }
                     }
                 }
             }
             
             if (!isCol) {
                 if (debug) {
                     std::cerr << "[Exec] IN only supported on Columns. Kind=" << (int)leftExprRaw->kind << "\n";
                     if (leftExprRaw->kind == TypedExpr::Kind::Function) {
                         std::cerr << "[Exec] Function: " << leftExprRaw->asFunction().name << "\n";
                         // Debug available string cols if we failed to find base col
                         std::cerr << "[Exec] Available string cols: ";
                         for(const auto& kv : ctx.stringCols) std::cerr << kv.first << " ";
                         std::cerr << "\n";
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
                 // Create Column Expr using factory
                 auto l = TypedExpr::column(colName);
                 // asColumn() is mutable
                 l->asColumn().inferredType = infType;
                 
                 // Create Compare Expr using factory
                 auto eqExpr = TypedExpr::compare(engine::CompareOp::Eq, l, valExpr);
                 
                 if (!root) {
                     root = eqExpr;
                 } else {
                     root = TypedExpr::binary(BinaryOp::Or, root, eqExpr);
                 }
             }
             
             return executeGPUFilterRecursive(root, ctx);
        }

        if (!mapCompOp(cmp.op)) return false;
        engine::expr::CompOp op = *mapCompOp(cmp.op);

        
        const TypedExpr* leftUnwrapped = unwrap(cmp.left.get());
        const TypedExpr* rightUnwrapped = unwrap(cmp.right.get());

        if (debug) std::cerr << "[Exec] DEBUG CMP Kinds: " << (int)leftUnwrapped->kind << " vs " << (int)rightUnwrapped->kind << "\n";
        
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
                case engine::expr::CompOp::LT: op = engine::expr::CompOp::GT; break;
                case engine::expr::CompOp::LE: op = engine::expr::CompOp::GE; break;
                case engine::expr::CompOp::GT: op = engine::expr::CompOp::LT; break;
                case engine::expr::CompOp::GE: op = engine::expr::CompOp::LE; break;
                default: break;
            }
        }
        else if ((funcColName = getFuncCol(leftUnwrapped)) != "" && rightUnwrapped->kind == TypedExpr::Kind::Literal) {
             litExpr = rightUnwrapped;
        }
        else if ((funcColName = getFuncCol(rightUnwrapped)) != "" && leftUnwrapped->kind == TypedExpr::Kind::Literal) {
             litExpr = leftUnwrapped;
             switch (op) {
                case engine::expr::CompOp::LT: op = engine::expr::CompOp::GT; break;
                case engine::expr::CompOp::LE: op = engine::expr::CompOp::GE; break;
                case engine::expr::CompOp::GT: op = engine::expr::CompOp::LT; break;
                case engine::expr::CompOp::GE: op = engine::expr::CompOp::LE; break;
                default: break;
            }
        }
         else if (leftUnwrapped->kind == TypedExpr::Kind::Column && rightUnwrapped->kind == TypedExpr::Kind::Column) {
             std::string c1 = leftUnwrapped->asColumn().column;
             std::string c2 = rightUnwrapped->asColumn().column;
             if (debug) std::cerr << "[Exec] DEBUG ColCol: " << c1 << " vs " << c2 << "\n";

             auto resolveBuf = [&](std::string& n, bool& isF) -> MTL::Buffer* {
                 if (ctx.f32ColsGPU.count(n)) { isF=true; return ctx.f32ColsGPU[n]; }
                 if (ctx.u32ColsGPU.count(n)) { isF=false; return ctx.u32ColsGPU[n]; }
                 // Suffix check
                 for(int i=1; i<=9; ++i) {
                     std::string s = n+"_"+std::to_string(i);
                     if (ctx.f32ColsGPU.count(s)) { n=s; isF=true; return ctx.f32ColsGPU[s]; }
                     if (ctx.u32ColsGPU.count(s)) { n=s; isF=false; return ctx.u32ColsGPU[s]; }
                 }
                 // Try stripping _rhs_ suffix (e.g. col_rhs_23 -> col)
                 size_t rhsPos = n.rfind("_rhs_");
                 if (rhsPos != std::string::npos) {
                     std::string base = n.substr(0, rhsPos);
                     if (ctx.f32ColsGPU.count(base)) { n=base; isF=true; return ctx.f32ColsGPU[base]; }
                     if (ctx.u32ColsGPU.count(base)) { n=base; isF=false; return ctx.u32ColsGPU[base]; }
                     
                     // Also try suffixes on base
                     for(int i=1; i<=9; ++i) {
                         std::string s = base+"_"+std::to_string(i);
                         if (ctx.f32ColsGPU.count(s)) { n=s; isF=true; return ctx.f32ColsGPU[s]; }
                         if (ctx.u32ColsGPU.count(s)) { n=s; isF=false; return ctx.u32ColsGPU[s]; }
                     }
                 }
                 
                 // Fallback for aggregates in Col=Col comparison
                 if (n.find("min(") != std::string::npos || n.find("max(") != std::string::npos ||
                     n.find("sum(") != std::string::npos || n.find("avg(") != std::string::npos ||
                     n.find("count(") != std::string::npos || n.find("MIN(") != std::string::npos ||
                     n.find("MAX(") != std::string::npos || n.find("SUM(") != std::string::npos) {
                     
                     std::string posKey = "#" + std::to_string(g_aggregateCounter);
                     if (ctx.f32ColsGPU.count(posKey)) { n=posKey; isF=true; g_aggregateCounter++; return ctx.f32ColsGPU[posKey]; }
                     if (ctx.u32ColsGPU.count(posKey)) { n=posKey; isF=false; g_aggregateCounter++; return ctx.u32ColsGPU[posKey]; }
                 }

                 return nullptr;
             };

             bool f1=false, f2=false;
             MTL::Buffer* b1 = resolveBuf(c1, f1);
             MTL::Buffer* b2 = resolveBuf(c2, f2);
             
             if (!b1 || !b2) {
                 if (debug) {
                     if (!b1) std::cerr << "[Exec] Failed to resolve " << c1 << "\n";
                     if (!b2) std::cerr << "[Exec] Failed to resolve " << c2 << "\n";
                     
                     std::cerr << "Available F32 Cols: ";
                     for(const auto& kv : ctx.f32ColsGPU) std::cerr << "'" << kv.first << "' ";
                     std::cerr << "\n";
                     std::cerr << "Available U32 Cols: ";
                     for(const auto& kv : ctx.u32ColsGPU) std::cerr << "'" << kv.first << "' ";
                     std::cerr << "\n";
                 }
                 return false;
             }

             uint32_t currentCount = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
             MTL::Buffer* input1 = b1;
             MTL::Buffer* input2 = b2;
             std::vector<MTL::Buffer*> temp;

             if (ctx.activeRowsGPU) {
                 input1 = f1 ? GpuOps::gatherF32(b1, ctx.activeRowsGPU, currentCount)
                             : GpuOps::gatherU32(b1, ctx.activeRowsGPU, currentCount);
                 temp.push_back(input1);
                 
                 input2 = f2 ? GpuOps::gatherF32(b2, ctx.activeRowsGPU, currentCount)
                             : GpuOps::gatherU32(b2, ctx.activeRowsGPU, currentCount);
                 temp.push_back(input2);
             }
             
             std::optional<FilterResult> res;
             if (f1 || f2) {
                 if (!f1) { MTL::Buffer* t = GpuOps::castU32ToF32(input1, currentCount); temp.push_back(t); input1 = t; }
                 if (!f2) { MTL::Buffer* t = GpuOps::castU32ToF32(input2, currentCount); temp.push_back(t); input2 = t; }
                 res = GpuOps::filterColColF32(input1, input2, currentCount, (int)op);
             } else {
                 res = GpuOps::filterColColU32(input1, input2, currentCount, (int)op);
             }

             for(auto* t : temp) if(t) t->release();

             if (res) {
                 if (ctx.activeRowsGPU) {
                      MTL::Buffer* newActive = GpuOps::gatherU32(ctx.activeRowsGPU, res->indices, res->count);
                      ctx.activeRowsGPU->release();
                      ctx.activeRowsGPU = newActive;
                      res->indices->release();
                 } else {
                      if (ctx.activeRowsGPU) ctx.activeRowsGPU->release();
                      ctx.activeRowsGPU = res->indices;
                 }
                 ctx.activeRowsCountGPU = res->count;
                 return true;
             }
             return false;
        } else {
             if (debug) std::cerr << "[Exec] GPU Filter: Generic Expression Path\n";
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
                 leftBuf = evalExprFloatGPU(cmp.left, ctx);
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
                 rightBuf = evalExprFloatGPU(cmp.right, ctx); 
                 if (!rightBuf) { if(leftBuf) leftBuf->release(); return false; }
             }
             
             std::optional<FilterResult> res;
             
             if (leftIsLit && rightBuf) {
                 // Lit op Col -> Flip Op -> Col flippedOp Lit
                 engine::expr::CompOp flipped;
                 bool valid = true;
                 switch(op) {
                     case engine::expr::CompOp::EQ: flipped = engine::expr::CompOp::EQ; break;
                     case engine::expr::CompOp::NE: flipped = engine::expr::CompOp::NE; break;
                     case engine::expr::CompOp::LT: flipped = engine::expr::CompOp::GT; break; 
                     case engine::expr::CompOp::LE: flipped = engine::expr::CompOp::GE; break;
                     case engine::expr::CompOp::GT: flipped = engine::expr::CompOp::LT; break;
                     case engine::expr::CompOp::GE: flipped = engine::expr::CompOp::LE; break;
                     default: valid = false; break;
                 }
                 if(valid) res = GpuOps::filterF32("expr", rightBuf, count, flipped, leftLitVal);
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
                 if (debug) std::cerr << "[Exec] Generic Filter Result (likely p_size): " << res->count << " rows\n";
                 if (ctx.activeRowsGPU) {
                      if (debug) std::cerr << "[Exec] Intersecting Generic with " << ctx.activeRowsCountGPU << " rows\n";
                      MTL::Buffer* newActive = GpuOps::gatherU32(ctx.activeRowsGPU, res->indices, res->count);
                      ctx.activeRowsGPU->release();
                      ctx.activeRowsGPU = newActive;
                      res->indices->release();
                 } else {
                      if (ctx.activeRowsGPU) ctx.activeRowsGPU->release();
                      ctx.activeRowsGPU = res->indices;
                 }
                 ctx.activeRowsCountGPU = res->count;
                 return true;
             }
             return false; 
        }
        
        if (colExpr) colName = colExpr->asColumn().column;
        else if (!funcColName.empty()) colName = funcColName;
        if (debug) std::cerr << "[Exec] GPU Filter checking col: " << colName << "\n";
        
        // Check string columns
        const std::vector<std::string>* strVec = nullptr;
        if (ctx.stringCols.count(colName)) {
             strVec = &ctx.stringCols.at(colName);
        } else {
             // check suffixes for strings
             for(int i=1; i<=9; ++i) {
                 std::string s = colName+"_"+std::to_string(i);
                 if (ctx.stringCols.count(s)) { strVec = &ctx.stringCols.at(s); colName=s; break; }
             }
        }
        
        if (strVec) {
             std::string pat = "";
             if (std::holds_alternative<std::string>(litExpr->asLiteral().value)) {
                  pat = std::get<std::string>(litExpr->asLiteral().value);
             }
             // Prefer flat string path
             std::optional<FilterResult> res;
             auto flatIt = ctx.flatStringColsGPU.find(colName);
             if (flatIt != ctx.flatStringColsGPU.end() && flatIt->second.valid()) {
                 res = GpuOps::filterStringFlat(colName, flatIt->second, op, pat);
             } else {
                 res = GpuOps::filterString(colName, *strVec, op, pat);
             }
             if (res) {
                  if (ctx.activeRowsGPU) {
                      auto joinRes = GpuOps::joinHash(
                          ctx.activeRowsGPU, nullptr, ctx.activeRowsCountGPU,
                          res->indices, nullptr, res->count
                      );
                      
                      MTL::Buffer* newActive = GpuOps::gatherU32(ctx.activeRowsGPU, joinRes.buildIndices, joinRes.count);
                      
                      ctx.activeRowsGPU->release();
                      res->indices->release();
                      if (joinRes.buildIndices) joinRes.buildIndices->release();
                      if (joinRes.probeIndices) joinRes.probeIndices->release();
                      
                      ctx.activeRowsGPU = newActive;
                      ctx.activeRowsCountGPU = joinRes.count;
                  } else {
                      ctx.activeRowsGPU = res->indices;
                      ctx.activeRowsCountGPU = res->count;
                  }
                  return true;
             }
             return false;
        }

        // Find buffer
        MTL::Buffer* buf = nullptr;
        if (ctx.f32ColsGPU.count(colName)) {
            buf = ctx.f32ColsGPU[colName];
            isF32 = true;
        } else if (ctx.u32ColsGPU.count(colName)) {
            buf = ctx.u32ColsGPU[colName];
            isF32 = false;
        } else {
            if (debug) std::cerr << "[Exec] DEBUG FIND BUF: colName " << colName << " not found in GPU cols\n";
            // Fallback for suffixed columns
            if (!ctx.u32ColsGPU.empty() || !ctx.f32ColsGPU.empty()) {
                for(int i=1; i<=9; ++i) {
                    std::string s = colName+"_"+std::to_string(i);
                    if (ctx.u32ColsGPU.count(s)) { buf = ctx.u32ColsGPU[s]; colName=s; isF32=false; break; }
                    if (ctx.f32ColsGPU.count(s)) { buf = ctx.f32ColsGPU[s]; colName=s; isF32=true; break; }
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
                if (debug) std::cerr << "[Exec] DEBUG FIND BUF: FAILED for colName " << colName << "\n";
                return false;
            }
        }
        
        // Extract literal
        const auto& lit = litExpr->asLiteral();
        std::optional<FilterResult> res;
        
        uint32_t currentCount = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
        
        if (isF32) {
            float val = 0.0f;
            if (std::holds_alternative<double>(lit.value)) val = (float)std::get<double>(lit.value);
            else if (std::holds_alternative<int64_t>(lit.value)) val = (float)std::get<int64_t>(lit.value);
            else if (std::holds_alternative<std::string>(lit.value)) {
                try {
                    val = std::stof(std::get<std::string>(lit.value));
                } catch(...) {}
            }
            
            // Handle scalar broadcast filter (single value buffer vs multiple rows)
            if (buf && buf->length() == sizeof(float) && currentCount > 1) {
                float colVal = *static_cast<const float*>(buf->contents());
                bool pass = false;
                switch(op) {
                    case engine::expr::CompOp::EQ: pass = (colVal == val); break;
                    case engine::expr::CompOp::NE: pass = (colVal != val); break;
                    case engine::expr::CompOp::LT: pass = (colVal < val); break;
                    case engine::expr::CompOp::LE: pass = (colVal <= val); break;
                    case engine::expr::CompOp::GT: pass = (colVal > val); break;
                    case engine::expr::CompOp::GE: pass = (colVal >= val); break;
                }
                if (!pass) {
                    if(ctx.activeRowsGPU) ctx.activeRowsGPU->release();
                    ctx.activeRowsGPU = GpuOps::createBuffer(nullptr, 4);
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
                if(debug) std::cerr << "[Exec] Exception in filterF32\n";
                res = std::nullopt;
            }
            
            if (!res && debug) {
                std::cerr << "[Exec] DEBUG: filterF32 failed. bufLen=" << buf->length() << " count=" << currentCount << " val=" << val << " op=" << (int)op << "\n";
            }
            
            if (!res) {
                 if(debug) std::cerr << "[Exec] GPU Filter F32 failed for " << colName << ". CPU fallback disabled.\n";
                 throw std::runtime_error("GPU F32 Filter failed: " + colName);
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
                     try { val = std::stoul(d); } catch(...) {}
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
                             val = GpuOps::fnv1a32(s);
                         }
                     }
                }
            }
            
            // Check for Date column with Days-Since-Epoch literal (small integer)
            // Planner often converts DATE 'YYYY-MM-DD' to integer days-since-epoch
            // But engine stores dates as YYYYMMDD integers. We must convert if needed.
            if (val > 0 && val < 100000) {
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
                if (debug) std::cerr << "[Exec] DEBUG: Scalar broadcast detected. bufLen=" << buf->length() << " currentCount=" << currentCount << "\n";
                uint32_t colVal = *static_cast<const uint32_t*>(buf->contents());
                bool pass = false;
                switch(op) {
                    case engine::expr::CompOp::EQ: pass = (colVal == val); break;
                    case engine::expr::CompOp::NE: pass = (colVal != val); break;
                    case engine::expr::CompOp::LT: pass = (colVal < val); break;
                    case engine::expr::CompOp::LE: pass = (colVal <= val); break;
                    case engine::expr::CompOp::GT: pass = (colVal > val); break;
                    case engine::expr::CompOp::GE: pass = (colVal >= val); break;
                }
                if (!pass) {
                    if(ctx.activeRowsGPU) ctx.activeRowsGPU->release();
                    ctx.activeRowsGPU = GpuOps::createBuffer(nullptr, 4);
                    ctx.activeRowsCountGPU = 0;
                }
                return true;
            }

            if (debug) std::cerr << "[Exec] DEBUG: About to filter. activeRowsGPU=" << (ctx.activeRowsGPU ? "set" : "null") << " bufLen=" << buf->length() << " count=" << currentCount << " val=" << val << " op=" << (int)op << "\n";

            if (ctx.activeRowsGPU) {
                res = GpuOps::filterU32Indexed(colName, buf, ctx.activeRowsGPU, currentCount, op, val);
            } else {
                res = GpuOps::filterU32(colName, buf, currentCount, op, val);
            }
            if (!res && debug) {
                std::cerr << "[Exec] DEBUG: filterU32 failed. bufLen=" << buf->length() << " count=" << currentCount << " val=" << val << " op=" << (int)op << "\n";
            }
            
            if (!res) {
                 if(debug) std::cerr << "[Exec] GPU Filter U32 failed for " << colName << ". CPU fallback disabled.\n";
                 throw std::runtime_error("GPU U32 Filter failed: " + colName);
            }
        }
        
        if (res) {
            if (ctx.activeRowsGPU) ctx.activeRowsGPU->release();
            ctx.activeRowsGPU = res->indices; // Transfer ownership
            ctx.activeRowsCountGPU = res->count;
            if (debug && res->indices) {
                uint32_t* idx = (uint32_t*)res->indices->contents();
                std::cerr << "[Exec] Filter result indices first 5: ";
                for (int i = 0; i < std::min(5u, res->count); ++i) std::cerr << idx[i] << " ";
                std::cerr << "\n";
            }
            return true;
        }
        return false;
    }

    if (expr->kind == TypedExpr::Kind::Column) {
        std::string colName = expr->asColumn().column;
        if (debug) std::cerr << "[Exec] GPU Filter checking col (boolean): " << colName << "\n";
        
        bool isF32 = false;
        MTL::Buffer* buf = nullptr;
        // Resolve buffer logic
        if (ctx.f32ColsGPU.count(colName)) { buf = ctx.f32ColsGPU.at(colName); isF32 = true; }
        else if (ctx.u32ColsGPU.count(colName)) { buf = ctx.u32ColsGPU.at(colName); isF32 = false; }
        else {
             // check suffixes
             for(int i=1; i<=9; ++i) {
                  std::string s = colName+"_"+std::to_string(i);
                  if (ctx.f32ColsGPU.count(s)) { buf=ctx.f32ColsGPU.at(s); colName=s; isF32=true; break; }
                  if (ctx.u32ColsGPU.count(s)) { buf=ctx.u32ColsGPU.at(s); colName=s; isF32=false; break; }
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
             std::cerr << "[Exec] DEBUG: Bool Col Lookup Failed: '" << colName << "'\n";
             std::cerr << "Available F32 (" << ctx.f32ColsGPU.size() << "): ";
             for(auto& kv : ctx.f32ColsGPU) std::cerr << "'" << kv.first << "' ";
             std::cerr << "\nAvailable U32 (" << ctx.u32ColsGPU.size() << "): ";
             for(auto& kv : ctx.u32ColsGPU) std::cerr << "'" << kv.first << "' ";
             std::cerr << "\n";
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
                    if(ctx.activeRowsGPU) ctx.activeRowsGPU->release();
                    ctx.activeRowsGPU = GpuOps::createBuffer(nullptr, 4);
                    ctx.activeRowsCountGPU = 0;
                 }
                 return true;
             }
             
             std::optional<FilterResult> res;
             if (isF32) {
                 if (ctx.activeRowsGPU) res = GpuOps::filterF32Indexed(colName, buf, ctx.activeRowsGPU, currentCount, engine::expr::CompOp::NE, 0.0f);
                 else res = GpuOps::filterF32(colName, buf, currentCount, engine::expr::CompOp::NE, 0.0f);
             } else {
                 if (ctx.activeRowsGPU) res = GpuOps::filterU32Indexed(colName, buf, ctx.activeRowsGPU, currentCount, engine::expr::CompOp::NE, 0);
                 else res = GpuOps::filterU32(colName, buf, currentCount, engine::expr::CompOp::NE, 0);
             }
             
             if (res) {
                if (ctx.activeRowsGPU) ctx.activeRowsGPU->release();
                ctx.activeRowsGPU = res->indices;
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
                 
                 if (debug) std::cerr << "[Exec] Corrected Q16: " << c << " NOT LIKE " << pat << "%\n";
                 
                 const std::vector<std::string>* vec = nullptr;
                 if (ctx.stringCols.count(c)) vec = &ctx.stringCols.at(c);
                 
                 if (vec) {
                      // Apply NOT LIKE prefix (prefer flat path)
                      std::optional<FilterResult> res;
                      auto flatIt = ctx.flatStringColsGPU.find(c);
                      if (flatIt != ctx.flatStringColsGPU.end() && flatIt->second.valid()) {
                          res = GpuOps::filterStringPrefixFlat(c, flatIt->second, pat, true);
                      } else {
                          res = GpuOps::filterStringPrefix(c, *vec, pat, true);
                      }
                      
                      if (res) {
                          if (ctx.activeRowsGPU) {
                              auto joinRes = GpuOps::joinHash(
                                  ctx.activeRowsGPU, nullptr, ctx.activeRowsCountGPU,
                                  res->indices, nullptr, res->count
                              );
                              MTL::Buffer* newActive = GpuOps::gatherU32(ctx.activeRowsGPU, joinRes.buildIndices, joinRes.count);
                              
                              if (ctx.activeRowsGPU) ctx.activeRowsGPU->release();
                              if (res->indices) res->indices->release();
                              if (joinRes.buildIndices) joinRes.buildIndices->release();
                              if (joinRes.probeIndices) joinRes.probeIndices->release();
                              
                              ctx.activeRowsGPU = newActive;
                              ctx.activeRowsCountGPU = joinRes.count;
                          } else {
                              ctx.activeRowsGPU = res->indices;
                              ctx.activeRowsCountGPU = res->count;
                          }
                          return true;
                      }
                 }
             }
        }
        
        return false;
    }

    if (expr->kind == TypedExpr::Kind::Cast) {
        return executeGPUFilterRecursive(expr->asCast().expr, ctx);
    }
    if (expr->kind == TypedExpr::Kind::Alias) {
        return executeGPUFilterRecursive(expr->asAlias().expr, ctx);
    }

    if (expr->kind == TypedExpr::Kind::Function) {
        const auto& func = expr->asFunction();
        // Handle contains/CONTAINS (from LIKE)
        if ((func.name == "contains" || func.name == "CONTAINS") && func.args.size() >= 2) {
             auto unwrap = [](const TypedExpr* e) -> const TypedExpr* {
                while (e) {
                    if (e->kind == TypedExpr::Kind::Cast) {
                        e = e->asCast().expr.get();
                    } else if (e->kind == TypedExpr::Kind::Alias) {
                        e = e->asAlias().expr.get();
                    } else {
                        break;
                    }
                }
                return e;
            };

            const TypedExpr* c = unwrap(func.args[0].get());
            const TypedExpr* l = unwrap(func.args[1].get());
            
            if (c->kind == TypedExpr::Kind::Column && l->kind == TypedExpr::Kind::Literal) {
                 std::string colName = c->asColumn().column;
                 std::string pat = "";
                 if (std::holds_alternative<std::string>(l->asLiteral().value)) {
                     pat = std::get<std::string>(l->asLiteral().value);
                 }
                 
                 const std::vector<std::string>* vec = nullptr;
                 if (ctx.stringCols.count(colName)) {
                     vec = &ctx.stringCols.at(colName);
                 } else {
                     for(int i=1; i<=9; ++i) {
                         std::string s = colName+"_"+std::to_string(i);
                         if (ctx.stringCols.count(s)) { vec = &ctx.stringCols.at(s); colName=s; break; }
                     }
                 }

                 if (vec) {
                      auto res = GpuOps::filterString(colName, *vec, engine::expr::CompOp::EQ, pat);
                      if (res) {
                          if (ctx.activeRowsGPU) ctx.activeRowsGPU->release();
                          ctx.activeRowsGPU = res->indices;
                          ctx.activeRowsCountGPU = res->count;
                          return true;
                      }
                 }
            }
        }
    }
    
    return false;
}

// ============================================================================
// Expression Evaluation
// ============================================================================

// CPU fallback stubs (disabled)
std::vector<float> GpuExecutor::evalExprFloat(const TypedExprPtr&, const EvalContext&) {
    throw std::runtime_error("CPU evalExprFloat called. CPU fallback is strictly disabled.");
    return {};
}

std::vector<uint32_t> GpuExecutor::evalExprU32(const TypedExprPtr&, const EvalContext&) {
    throw std::runtime_error("CPU evalExprU32 called. CPU fallback is strictly disabled.");
    return {};
}

std::vector<bool> GpuExecutor::evalPredicate(const TypedExprPtr&, const EvalContext&) {
    throw std::runtime_error("CPU evalPredicate called. CPU fallback is strictly disabled.");
    return {};
}
MTL::Buffer* GpuExecutor::evalExprFloatGPU(const TypedExprPtr& expr, EvalContext& ctx) {
    if (!expr) return nullptr;
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");

    // Determine effective row count for output buffer
    uint32_t count = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
    // Fallback if row count seems wrong
    if (count == 0 && ctx.rowCount > 0 && !ctx.activeRowsGPU) count = ctx.rowCount;
    if (count == 0) return nullptr;
    
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
                    float* ptr = (float*)b->contents();
                    size_t n = b->length() / sizeof(float);
                    bool varying = false;
                    float first = ptr[0];
                    for (size_t i = 1; i < std::min(n, (size_t)100); ++i) {  // Check first 100 values
                        if (ptr[i] != first) { varying = true; break; }
                    }
                    if (varying) {
                        if (debug) std::cerr << "[Exec] evalExprFloatGPU: agg match varying col '" << name << "' for '" << col << "'\n";
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
                MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
                float* ptr = (float*)outBuf->contents();
                std::fill(ptr, ptr + count, val);
                return outBuf;
            } else {
                MTL::Buffer* rawBuf = GpuOps::createBuffer(vec.data(), vec.size() * sizeof(float));
                // Only gather if data is not already compacted to avoid OOB reads
                if (ctx.activeRowsGPU && vec.size() != count) {
                    MTL::Buffer* gathered = GpuOps::gatherF32(rawBuf, ctx.activeRowsGPU, count);
                    rawBuf->release();
                    return gathered;
                }
                return rawBuf;
            }
        } else if (ctx.u32Cols.count(col) && !ctx.u32Cols[col].empty()) {
            const auto& vec = ctx.u32Cols[col];
            if (vec.size() == 1 && count > 1) { // Scalar broadcast
                float val = (float)vec[0];
                MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
                float* ptr = (float*)outBuf->contents();
                std::fill(ptr, ptr + count, val);
                return outBuf;
            } else {
                MTL::Buffer* rawBuf = GpuOps::createBuffer(vec.data(), vec.size() * sizeof(uint32_t));
                MTL::Buffer* targetBuf = rawBuf;
                bool temp = false;
                
                // Only gather if data is not already compacted
                if (ctx.activeRowsGPU && vec.size() != count) {
                    targetBuf = GpuOps::gatherU32(rawBuf, ctx.activeRowsGPU, count);
                    temp = true;
                }
                
                MTL::Buffer* f32Buf = GpuOps::castU32ToF32(targetBuf, count);
                
                if (temp) targetBuf->release();
                rawBuf->release();
                return f32Buf;
            }
        } else {
             // Suffix search
             for(int i=1; i<=9; ++i) {
                 std::string s = col + "_" + std::to_string(i);
                 if (ctx.f32ColsGPU.count(s)) { buf = ctx.f32ColsGPU[s]; break; }
                 if (ctx.u32ColsGPU.count(s)) { buf = ctx.u32ColsGPU[s]; isU32=true; break; }
             }
             // RHS Suffix search
             if (!buf) {
                 std::string rhsPattern = col + "_rhs_";
                 for (const auto& [name, b] : ctx.f32ColsGPU) {
                     if (name.find(rhsPattern) == 0) { buf = b; break; }
                 }
                 if (!buf) {
                     for (const auto& [name, b] : ctx.u32ColsGPU) {
                         if (name.find(rhsPattern) == 0) { buf = b; isU32=true; break; }
                     }
                 }
             }
        }

        // Fallback: Heuristic for Scalar Aggregates mismatch (sum(...) vs #0)
        // First try fuzzy matching by removing spaces and lowercasing
        if (!buf) {
            std::string colLower = col;
            std::transform(colLower.begin(), colLower.end(), colLower.begin(), ::tolower);
            // Remove spaces for fuzzy compare
            colLower.erase(std::remove_if(colLower.begin(), colLower.end(), ::isspace), colLower.end());
            
            if (env_truthy("GPUDB_DEBUG_OPS") && col.find("sum") != std::string::npos) {
                std::cerr << "[Exec] evalExprFloatGPU: fuzzy search for col='" << col << "' length=" << col.length() << " normalized length=" << colLower.length() << "\n";
            }
            
            // Search for matching column in f32ColsGPU
            for (const auto& [name, b] : ctx.f32ColsGPU) {
                if (!b) continue;
                std::string nameLower = name;
                std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
                nameLower.erase(std::remove_if(nameLower.begin(), nameLower.end(), ::isspace), nameLower.end());
                
                if (colLower == nameLower) {
                    if (env_truthy("GPUDB_DEBUG_OPS")) std::cerr << "[Exec] evalExprFloatGPU: fuzzy matched '" << col << "' to '" << name << "'\n";
                    buf = b;
                    break;
                }
            }
            // Also try u32ColsGPU
            if (!buf) {
                for (const auto& [name, b] : ctx.u32ColsGPU) {
                    if (!b) continue;
                    std::string nameLower = name;
                    std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
                    nameLower.erase(std::remove_if(nameLower.begin(), nameLower.end(), ::isspace), nameLower.end());
                    if (colLower == nameLower) {
                        if (env_truthy("GPUDB_DEBUG_OPS")) std::cerr << "[Exec] evalExprFloatGPU: fuzzy matched (u32) '" << col << "' to '" << name << "'\n";
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
                    
                    std::string posKey = "#" + std::to_string(g_aggregateCounter);
                    if (env_truthy("GPUDB_DEBUG_OPS")) std::cerr << "[Exec] evalExprFloatGPU: heuristic mapping " << col << " to " << posKey << "\n";
                    
                    if (ctx.f32ColsGPU.count(posKey)) { buf = ctx.f32ColsGPU[posKey]; g_aggregateCounter++; }
                    else if (ctx.u32ColsGPU.count(posKey)) { buf = ctx.u32ColsGPU[posKey]; isU32=true; g_aggregateCounter++; }
                    else if (ctx.f32Cols.count(posKey) && !ctx.f32Cols[posKey].empty()) {
                        float val = ctx.f32Cols[posKey][0];
                        MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
                        float* ptr = (float*)outBuf->contents();
                        std::fill(ptr, ptr + count, val);
                        g_aggregateCounter++;
                        return outBuf;
                    }
                    else if (ctx.u32Cols.count(posKey) && !ctx.u32Cols[posKey].empty()) {
                        float val = (float)ctx.u32Cols[posKey][0];
                        MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
                        float* ptr = (float*)outBuf->contents();
                        std::fill(ptr, ptr + count, val);
                        g_aggregateCounter++;
                        return outBuf;
                    }
                }
            }
        }
        
        if (!buf) return nullptr;
        
        // If U32, cast to F32 (and gather if needed)
        if (isU32) {
             if (ctx.activeRowsGPU) {
                 // Gather to compact U32, then cast
                 MTL::Buffer* gathered = GpuOps::gatherU32(buf, ctx.activeRowsGPU, count);
                 MTL::Buffer* casted = GpuOps::castU32ToF32(gathered, count);
                 gathered->release(); // Release gathering intermediate
                 return casted;
             } else {
                 return GpuOps::castU32ToF32(buf, count);
             }
        }
        
        // If F32
        if (ctx.activeRowsGPU) {
            return GpuOps::gatherF32(buf, ctx.activeRowsGPU, count);
        } else {
            buf->retain();
            return buf; 
        }
    }
    
    if (expr->kind == TypedExpr::Kind::Literal) {
        float val = 0.0f;
        const auto& lit = expr->asLiteral();
        if (std::holds_alternative<double>(lit.value)) val = (float)std::get<double>(lit.value);
        else if (std::holds_alternative<int64_t>(lit.value)) val = (float)std::get<int64_t>(lit.value);
        
        MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
        float* ptr = (float*)outBuf->contents();
        std::fill(ptr, ptr + count, val); 
        return outBuf;
    }

    if (expr->kind == TypedExpr::Kind::Aggregate) {
        const auto& agg = expr->asAggregate();
        // Look for alias
        if (!agg.alias.empty() && ctx.f32ColsGPU.count(agg.alias)) {
            MTL::Buffer* buf = ctx.f32ColsGPU[agg.alias];
            buf->retain(); return buf;
        }

        // Try positional scalar aggregate lookup (#N)
        // Check if environment has scalar aggregates
        bool hasScalarAggregates = false;
        if (ctx.f32Cols.count("#0") || ctx.f32ColsGPU.count("#0") || ctx.u32Cols.count("#0") || ctx.u32ColsGPU.count("#0")) hasScalarAggregates = true;
        
        if (hasScalarAggregates) {
             std::string posKey = "#" + std::to_string(g_aggregateCounter);
             
             // Check if posKey exists in any map
             if (ctx.f32ColsGPU.count(posKey)) {
                 MTL::Buffer* buf = ctx.f32ColsGPU[posKey];
                 buf->retain(); g_aggregateCounter++; return buf;
             }
             if (ctx.f32Cols.count(posKey) && !ctx.f32Cols[posKey].empty()) {
                 float val = ctx.f32Cols[posKey][0];
                 MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
                 float* ptr = (float*)outBuf->contents();
                 std::fill(ptr, ptr + count, val);
                 g_aggregateCounter++;
                 return outBuf;
             }
             if (ctx.u32Cols.count(posKey) && !ctx.u32Cols[posKey].empty()) {
                 float val = (float)ctx.u32Cols[posKey][0];
                 MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
                 float* ptr = (float*)outBuf->contents();
                 std::fill(ptr, ptr + count, val);
                 g_aggregateCounter++;
                 return outBuf;
             }
             if (ctx.u32ColsGPU.count(posKey)) {
                 g_aggregateCounter++;
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
        
        // Also enable heuristic lookup for #0, #1 etc if we failed positional logic but they exist
        // This is a safety net for when g_aggregateCounter gets desynchronized
        if (ctx.f32ColsGPU.count("#0")) {
             // Only if we haven't found match, checking #0 as fallback
             // But tricky if multiple aggregates.
        }
        
        for (const auto& [name, buf] : ctx.f32ColsGPU) {
            if (name.rfind(prefix, 0) == 0) {
                buf->retain();
                return buf;
            }
        }
        
        // Check CPU columns (ctx.f32Cols) - especially for scalar aggregates
        for (const auto& [name, vec] : ctx.f32Cols) {
            if (name.rfind(prefix, 0) == 0 && !vec.empty()) {
                float val = vec[0];
                // Broadcast scalar to all rows
                MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
                float* ptr = (float*)outBuf->contents();
                std::fill(ptr, ptr + count, val);
                return outBuf;
            }
        }

        // Check U32 (e.g. COUNT) columns on CPU
        for (const auto& [name, vec] : ctx.u32Cols) {
            if (name.rfind(prefix, 0) == 0 && !vec.empty()) {
                float val = (float)vec[0];
                MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
                float* ptr = (float*)outBuf->contents();
                std::fill(ptr, ptr + count, val);
                return outBuf;
            }
        }

        // Check U32 (e.g. COUNT) and cast
         for (const auto& [name, buf] : ctx.u32ColsGPU) {
            if (name.rfind(prefix, 0) == 0) {
                return GpuOps::castU32ToF32(buf, count);
            }
        }
    }

    if (expr->kind == TypedExpr::Kind::Compare) {
        // Debug info
        if (env_truthy("GPUDB_DEBUG_OPS")) {
              const auto& c = expr->asCompare();
              std::cerr << "[Exec] Eval Compare: LeftKind=" << (c.left ? (int)c.left->kind : -1) << " RightKind=" << (c.right ? (int)c.right->kind : -1) << "\n";
              if (c.left && c.left->kind == TypedExpr::Kind::Column) std::cerr << "  LeftCol: " << c.left->asColumn().column << "\n";
        }

        // Evaluate predicate and cast to float (1.0 for true, 0.0 for false)
        EvalContext subCtx = ctx; 
        if (ctx.activeRowsGPU) ctx.activeRowsGPU->retain();

        if (executeGPUFilterRecursive(std::const_pointer_cast<TypedExpr>(expr), subCtx)) {
            // subCtx.activeRowsGPU matches condition
            MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
            float* ptr = (float*)outBuf->contents();
            std::fill(ptr, ptr + count, 0.0f); // Default 0.0

            if (subCtx.activeRowsCountGPU > 0) {
                 // Scatter 1.0 to matching rows
                 GpuOps::scatterConstantF32(outBuf, subCtx.activeRowsGPU, subCtx.activeRowsCountGPU, 1.0f);
            }
            
            if (subCtx.activeRowsGPU) subCtx.activeRowsGPU->release();
            return outBuf;
        } else {
            // Filter eval failed -> return nullptr
            if (env_truthy("GPUDB_DEBUG_OPS")) std::cerr << "[Exec] Eval Compare: executeGPUFilterRecursive failed\n";
            if (subCtx.activeRowsGPU) subCtx.activeRowsGPU->release();
            return nullptr;
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
                // Lit op Right
                MTL::Buffer* rightBuf = evalExprFloatGPU(bin.right, ctx);
                if (!rightBuf) return nullptr;
                if (isMul) return GpuOps::arithMulF32ColScalar(rightBuf, leftVal, count);
                else if (isAdd) return GpuOps::arithAddF32ColScalar(rightBuf, leftVal, count); // Commutative
                else if (isDiv) return GpuOps::arithDivF32ScalarCol(leftVal, rightBuf, count);
                else return GpuOps::arithSubF32ScalarCol(leftVal, rightBuf, count);
            } else if (rightIsLit) {
                // Left op Lit
                MTL::Buffer* leftBuf = evalExprFloatGPU(bin.left, ctx);
                if (!leftBuf) return nullptr;
                if (isMul) return GpuOps::arithMulF32ColScalar(leftBuf, rightVal, count);
                else if (isAdd) return GpuOps::arithAddF32ColScalar(leftBuf, rightVal, count);
                else if (isDiv) return GpuOps::arithDivF32ColScalar(leftBuf, rightVal, count);
                else return GpuOps::arithSubF32ColScalar(leftBuf, rightVal, count);
            } else {
                // Left op Right
                MTL::Buffer* leftBuf = evalExprFloatGPU(bin.left, ctx);
                MTL::Buffer* rightBuf = evalExprFloatGPU(bin.right, ctx);
                if (!leftBuf || !rightBuf) return nullptr;
                if (isMul) return GpuOps::arithMulF32ColCol(leftBuf, rightBuf, count);
                else if (isAdd) return GpuOps::arithAddF32ColCol(leftBuf, rightBuf, count);
                else if (isDiv) return GpuOps::arithDivF32ColCol(leftBuf, rightBuf, count);
                else return GpuOps::arithSubF32ColCol(leftBuf, rightBuf, count);
            }
        }
    }
    
    if (expr->kind == TypedExpr::Kind::Alias) {
        return evalExprFloatGPU(expr->asAlias().expr, ctx);
    }

    if (expr->kind == TypedExpr::Kind::Cast) {
        return evalExprFloatGPU(expr->asCast().expr, ctx);
    }
    
    if (expr->kind == TypedExpr::Kind::Case) {
        if (debug) std::cerr << "[Exec] evalExprFloatGPU: Entering CASE expression handler\n";
        
        // Safely access CaseExpr data
        if (!std::holds_alternative<CaseExpr>(expr->data)) {
            std::cerr << "[Exec] ERROR: CASE expression kind but data is not CaseExpr!\n";
            return nullptr;
        }
        const auto& c = expr->asCase();
        
        if (debug) {
            std::cerr << "[Exec] CASE: " << c.cases.size() << " WHEN branches, hasElse=" << (c.elseExpr ? "yes" : "no") << "\n";
        }
        
        // 1. Initialize output buffer based on ELSE
        MTL::Buffer* outBuf = nullptr;
        
        if (c.elseExpr) {
            if (debug) std::cerr << "[Exec] CASE: elseExpr kind=" << static_cast<int>(c.elseExpr->kind) << "\n";
            if (c.elseExpr->kind == TypedExpr::Kind::Literal) {
                float elseVal = 0.0f;
                const auto& lit = c.elseExpr->asLiteral();
                if (std::holds_alternative<int64_t>(lit.value)) elseVal = (float)std::get<int64_t>(lit.value);
                else if (std::holds_alternative<double>(lit.value)) elseVal = (float)std::get<double>(lit.value);
                
                outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
                float* raw = (float*)outBuf->contents();
                std::fill(raw, raw + count, elseVal);
            } else {
                MTL::Buffer* elseBuf = evalExprFloatGPU(c.elseExpr, ctx);
                if (!elseBuf) return nullptr;
                
                // Copy elseBuf to outBuf (always copy to avoid modifying source columns)
                outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
                memcpy(outBuf->contents(), elseBuf->contents(), count * sizeof(float));
                
                // Release elseBuf if it's an intermediate (not in context columns)
                bool isBorrowed = false;
                for (const auto& [n, b] : ctx.f32ColsGPU) { if (b == elseBuf) { isBorrowed = true; break; } }
                if (!isBorrowed) {
                    for (const auto& [n, b] : ctx.u32ColsGPU) { if (b == elseBuf) { isBorrowed = true; break; } }
                }
                if (!isBorrowed) elseBuf->release();
            }
        } else {
             // Default 0.0
             outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
             float* raw = (float*)outBuf->contents();
             std::fill(raw, raw + count, 0.0f);
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
                 if (debug) std::cerr << "[Exec] CASE: skipping WHEN branch with error() guard function\n";
                 continue;
             }
             
             // Clone context to isolate filter effects
             EvalContext subCtx = ctx; 
             // Retain activeRowsGPU as subCtx takes ownership of a reference
             if (ctx.activeRowsGPU) ctx.activeRowsGPU->retain(); 
             
             if (executeGPUFilterRecursive(w.when, subCtx)) {
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
                         MTL::Buffer* thenBuf = evalExprFloatGPU(w.then, subCtx);
                         if (thenBuf) {
                             GpuOps::scatterF32(thenBuf, outBuf, subCtx.activeRowsGPU, subCtx.activeRowsCountGPU);
                             thenBuf->release();
                         } else {
                             // If evaluation fails (e.g. error function), and we have active rows, we can't proceed on GPU
                             if (debug) std::cerr << "[Exec] CASE THEN non-literal GPU eval failed\n";
                             if (subCtx.activeRowsGPU) subCtx.activeRowsGPU->release();
                             outBuf->release();
                             return nullptr;
                         }
                     }
                 }
                 
                 // Cleanup subCtx result
                 if (subCtx.activeRowsGPU) subCtx.activeRowsGPU->release();
             } else {
                 if (debug) std::cerr << "[Exec] CASE condition eval failed on GPU\n";
                 // Cleanup original reference
                 if (subCtx.activeRowsGPU) subCtx.activeRowsGPU->release();
                 outBuf->release(); 
                 return nullptr;
             }
        }
        return outBuf;
    }

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
                     return GpuOps::createBuffer(buf->contents(), count * sizeof(float)); // Copy
                }
            }
            if (ctx.u32ColsGPU.count(candidate)) {
                MTL::Buffer* buf = ctx.u32ColsGPU.at(candidate);
                // Cast U32 -> F32
                if (ctx.activeRowsGPU) {
                     MTL::Buffer* gathered = GpuOps::gatherU32(buf, ctx.activeRowsGPU, count);
                     MTL::Buffer* asFloat = GpuOps::castU32ToF32(gathered, count);
                     gathered->release();
                     return asFloat;
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
            if (debug) std::cerr << "[Exec] evalExprFloatGPU: FIRST aggregate, evaluating argument\n";
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
                        float* ptr = (float*)buf->contents();
                        size_t n = buf->length() / sizeof(float);
                        bool uniform = true;
                        if (n > 1) {
                            float first = ptr[0];
                            for (size_t i = 1; i < std::min(n, (size_t)10); ++i) {
                                if (std::abs(ptr[i] - first) > 1e-6f) { uniform = false; break; }
                            }
                        }
                        if (uniform) {
                            if (debug) std::cerr << "[Exec] evalExprFloatGPU: FIRST using scalar column '" << colName << "' val=" << ptr[0] << "\n";
                            buf->retain();
                            return buf;
                        }
                    }
                }
                
                // Fallback to recursive evaluation (shouldn't be needed if column exists)
                return evalExprFloatGPU(fn.args[0], ctx);
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
                    if (debug) std::cerr << "[Exec] evalExprFloatGPU: Found aggregate col '" << colName << "' for " << fnName << "\n";
                    buf->retain();
                    return buf;
                }
            }
            // Also check u32ColsGPU
            for (const auto& [colName, buf] : ctx.u32ColsGPU) {
                std::string lowerCol = colName;
                std::transform(lowerCol.begin(), lowerCol.end(), lowerCol.begin(), ::tolower);
                if (lowerCol.find(fnPrefix) == 0 && buf) {
                    if (debug) std::cerr << "[Exec] evalExprFloatGPU: Found aggregate col (u32) '" << colName << "' for " << fnName << ", casting to f32\n";
                    if (ctx.activeRowsGPU) {
                        MTL::Buffer* gathered = GpuOps::gatherU32(buf, ctx.activeRowsGPU, count);
                        MTL::Buffer* casted = GpuOps::castU32ToF32(gathered, count);
                        gathered->release();
                        return casted;
                    } else {
                        return GpuOps::castU32ToF32(buf, buf->length() / sizeof(uint32_t));
                    }
                }
            }
            
            // Fallback: positional heuristic for #N columns
            std::string posKey = "#" + std::to_string(g_aggregateCounter);
             if (debug) std::cerr << "[Exec] evalExprFloatGPU: Function heuristic mapping " << fnName << " to " << posKey << "\n";
             
             if (ctx.f32ColsGPU.count(posKey)) {
                 MTL::Buffer* buf = ctx.f32ColsGPU[posKey];
                 buf->retain(); 
                 g_aggregateCounter++; 
                 return buf;
             }
             if (ctx.u32ColsGPU.count(posKey)) {
                 MTL::Buffer* buf = ctx.u32ColsGPU[posKey];
                 g_aggregateCounter++;
                 if (ctx.activeRowsGPU) {
                      MTL::Buffer* gathered = GpuOps::gatherU32(buf, ctx.activeRowsGPU, count);
                      MTL::Buffer* casted = GpuOps::castU32ToF32(gathered, count);
                      gathered->release();
                      return casted;
                 } else {
                      return GpuOps::castU32ToF32(buf, count);
                 }
             }
             if (ctx.f32Cols.count(posKey) && !ctx.f32Cols[posKey].empty()) {
                 float val = ctx.f32Cols[posKey][0];
                 MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
                 float* ptr = (float*)outBuf->contents();
                 std::fill(ptr, ptr + count, val);
                 g_aggregateCounter++;
                 return outBuf;
             }
             if (ctx.u32Cols.count(posKey) && !ctx.u32Cols[posKey].empty()) {
                 float val = (float)ctx.u32Cols[posKey][0];
                 MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
                 float* ptr = (float*)outBuf->contents();
                 std::fill(ptr, ptr + count, val);
                 g_aggregateCounter++;
                 return outBuf;
             }
        }
        
        // Handle explicit error throws (from scalar subquery checks) or "error" calls
        if (fnName == "ERROR" || fnName == "\"ERROR\"") {
             MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(float));
             float* ptr = (float*)outBuf->contents();
             std::fill(ptr, ptr + count, 0.0f); // Zero-fill; CASE expression handles validity
             return outBuf;
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
                 MTL::Buffer* inBuf = evalExprFloatGPU(valArg, ctx);
                 if (!inBuf) {
                     if (debug) {
                         std::cerr << "[Exec] EXTRACT failed: could not evaluate valArg. Kind=" << (int)valArg->kind << "\n";
                         if (valArg->kind == TypedExpr::Kind::Column) std::cerr << "  Col: " << valArg->asColumn().column << "\n";
                     }
                     return nullptr;
                 }
                 
                 // EXTRACT(YEAR) logic: floor(val / 10000)
                 MTL::Buffer* divBuf = GpuOps::arithDivF32ColScalar(inBuf, 10000.0f, count);
                 MTL::Buffer* floorBuf = GpuOps::mathFloorF32(divBuf, count);
                 
                 divBuf->release();
                 
                 bool isCtx = false;
                 for(auto& [k,v] : ctx.f32ColsGPU) if(v==inBuf) { isCtx=true; break; }
                 for(auto& [k,v] : ctx.u32ColsGPU) if(v==inBuf) { isCtx=true; break; }
                 if(!isCtx) inBuf->release();
                 
                 return floorBuf;
             }
        }
        
        if (debug) std::cerr << "[Exec] Unsupported GPU function: " << fn.name << "\n";
        return nullptr; 
    }

    if (expr->kind == TypedExpr::Kind::Aggregate) {
        // Fallback or specific logic
        if (expr->asAggregate().func == AggFunc::Sum) {
             return evalExprFloatGPU(expr, ctx);
        }
    }
    
    return nullptr;
}

MTL::Buffer* GpuExecutor::evalExprU32GPU(const TypedExprPtr& expr, EvalContext& ctx) {

    if (!expr) return nullptr;
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    uint32_t count = (ctx.activeRowsGPU != nullptr) ? ctx.activeRowsCountGPU : ctx.rowCount;
    if (count == 0 && ctx.rowCount > 0 && !ctx.activeRowsGPU) count = ctx.rowCount;
    if (count == 0) return nullptr;

    if (expr->kind == TypedExpr::Kind::Column) {
        std::string col = expr->asColumn().column;
        
        MTL::Buffer* buf = nullptr;
        if (ctx.u32ColsGPU.count(col)) {
            buf = ctx.u32ColsGPU[col];
        } else if (ctx.u32Cols.count(col) && !ctx.u32Cols[col].empty()) {
            const auto& vec = ctx.u32Cols[col];
            if (vec.size() >= count) { // Full column
                MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(uint32_t));
                memcpy(outBuf->contents(), vec.data(), count * sizeof(uint32_t));
                // Return gathered/copied buffer directly
                if (ctx.activeRowsGPU) {
                     MTL::Buffer* tmp = GpuOps::createBuffer(nullptr, count * sizeof(uint32_t)); 
                     memcpy(tmp->contents(), vec.data(), count * sizeof(uint32_t));
                     // Gather
                     MTL::Buffer* gathered = GpuOps::gatherU32(tmp, ctx.activeRowsGPU, count);
                     tmp->release();
                     return gathered;
                } else {
                     MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(uint32_t)); 
                     memcpy(outBuf->contents(), vec.data(), count * sizeof(uint32_t));
                     return outBuf;
                }
            } else {
                // Scalar broadcast
                uint32_t val = vec[0];
                MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(uint32_t));
                uint32_t* ptr = (uint32_t*)outBuf->contents();
                std::fill(ptr, ptr + count, val);
                return outBuf;
            }
        } else {
            // Suffix search
            for(int i=1; i<=9; ++i) {
                std::string s = col + "_" + std::to_string(i);
                if (ctx.u32ColsGPU.count(s)) { buf = ctx.u32ColsGPU[s]; break; }
            }
            // RHS Suffix search
            if (!buf) {
                 std::string rhsPattern = col + "_rhs_";
                 for (const auto& [name, b] : ctx.u32ColsGPU) {
                     if (name.find(rhsPattern) == 0) { buf = b; break; }
                 }
            }
        }
        
        if (buf) {
            if (ctx.activeRowsGPU) {
                return GpuOps::gatherU32(buf, ctx.activeRowsGPU, count);
            } else {
                buf->retain(); // Caller takes ownership
                return buf;
            }
        }
    }
    
    // Literal
    if (expr->kind == TypedExpr::Kind::Literal) {
        uint32_t val = 0;
        const auto& lit = expr->asLiteral();
        if (std::holds_alternative<int64_t>(lit.value)) val = (uint32_t)std::get<int64_t>(lit.value);
        else if (std::holds_alternative<std::string>(lit.value)) val = GpuOps::fnv1a32(std::get<std::string>(lit.value));
        else if (std::holds_alternative<double>(lit.value)) val = (uint32_t)std::get<double>(lit.value);

        MTL::Buffer* outBuf = GpuOps::createBuffer(nullptr, count * sizeof(uint32_t));
        uint32_t* ptr = (uint32_t*)outBuf->contents();
        std::fill(ptr, ptr + count, val);
        return outBuf;
    }
    
    if (debug) std::cerr << "[Exec] evalExprU32GPU not implemented for this kind: " << (int)expr->kind << "\n";
    return nullptr;
}
} // namespace engine
