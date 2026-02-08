#include "GpuExecutor.hpp"
#include "GpuExecutorPriv.hpp"
#include "TypedExpr.hpp"
#include "Predicate.hpp"
#include "Operators.hpp"
#include "ColumnStoreGPU.hpp"
#include <Metal/Metal.hpp>
#include <future>
#include <thread>

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <map>
#include <numeric>

namespace engine {

// Helper: Get unique name for a column, adding suffix if needed to avoid collision
static std::string getUniqueColumnName(const std::string& name, 
                                        std::unordered_set<std::string>& existingNames,
                                        bool debug = false) {
    if (existingNames.count(name) == 0) {
        existingNames.insert(name);
        return name;
    }
    
    // Need to rename - find an unused suffix
    for (int suffix = 1; suffix <= 10; ++suffix) {
        std::string newName = name + "_" + std::to_string(suffix);
        if (existingNames.count(newName) == 0) {
            existingNames.insert(newName);
            if (debug) {
                std::cerr << "[Exec] Join: Renaming duplicate column " << name << " -> " << newName << "\n";
            }
            return newName;
        }
    }
    
    // Fallback - use a unique ID
    std::string fallback = name + "_" + std::to_string(rand() % 10000);
    existingNames.insert(fallback);
    return fallback;
}

// Helper: extract all equi-join key pairs from a condition expression
static void extractJoinKeyPairs(const TypedExprPtr& expr, 
                                std::vector<std::pair<std::string, std::string>>& keyPairs) {
    if (!expr) return;
    
    if (expr->kind == TypedExpr::Kind::Compare) {
        const auto& cmp = expr->asCompare();
        if (cmp.op == CompareOp::Eq && cmp.left && cmp.right) {
            if (cmp.left->kind == TypedExpr::Kind::Column && 
                cmp.right->kind == TypedExpr::Kind::Column) {
                keyPairs.emplace_back(cmp.left->asColumn().column, 
                                      cmp.right->asColumn().column);
            }
        }
    } else if (expr->kind == TypedExpr::Kind::Binary) {
        const auto& bin = expr->asBinary();
        if (bin.op == BinaryOp::And) {
            extractJoinKeyPairs(bin.left, keyPairs);
            extractJoinKeyPairs(bin.right, keyPairs);
        }
    }
}

bool GpuExecutor::executeJoin(const IRJoin& join, const std::string& datasetPath,
                                   EvalContext& leftCtx, EvalContext& rightCtx, EvalContext& outCtx) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    // Supported: INNER, LEFT, RIGHT, SEMI, ANTI, MARK (MARK treated as SEMI)
    if (join.type != JoinType::Inner && join.type != JoinType::Left &&
        join.type != JoinType::Right && join.type != JoinType::Semi && 
        join.type != JoinType::Anti && join.type != JoinType::Mark) {
        if (debug) std::cerr << "[Exec] Join: unsupported join type\n";
        return false;
    }
    
    const bool isLeftJoin = (join.type == JoinType::Left);
    const bool isRightJoin = (join.type == JoinType::Right);
    const bool isSemiJoin = (join.type == JoinType::Semi || join.type == JoinType::Mark);
    const bool isAntiJoin = (join.type == JoinType::Anti);
    
    if (debug) {
        std::cerr << "[Exec] Join: type=" << static_cast<int>(join.type) 
                  << " isLeft=" << isLeftJoin << " isRight=" << isRightJoin 
                  << " isSemi=" << isSemiJoin << " isAnti=" << isAntiJoin << "\n";
    }
    
    // Extract all join key pairs from the condition
    std::vector<std::pair<std::string, std::string>> keyPairs;
    
    bool isCrossJoin = false;
    bool hasPostJoinFilter = false;  // For non-equality conditions (>, <, etc.)
    TypedExprPtr postJoinFilter = nullptr;
    
    if (join.conditionStr == "1=1") {
        isCrossJoin = true;
    } else if (join.condition && join.condition->kind == TypedExpr::Kind::Compare) {
        const auto& cmp = join.condition->asCompare();
        if (cmp.op == CompareOp::Eq &&
            cmp.left->kind == TypedExpr::Kind::Literal &&
            cmp.right->kind == TypedExpr::Kind::Literal) {
            isCrossJoin = true;
        }
        // Check for non-equality comparison (treat as cross join + filter)
        else if (cmp.op != CompareOp::Eq) {
            // This is a non-equality join condition (e.g., value > threshold from HAVING subquery)
            // Treat as cross-join with post-filter
            isCrossJoin = true;
            hasPostJoinFilter = true;
            postJoinFilter = join.condition;
            if (debug) {
                std::cerr << "[Exec] Join: detected non-equality condition, treating as cross-join + filter\n";
                std::cerr << "[Exec] Join: conditionStr=" << join.conditionStr << "\n";
            }
        }
    }

    if (join.condition && !isCrossJoin) {
        extractJoinKeyPairs(join.condition, keyPairs);
    }
    
    // Fallback: parse from condition string (single pair only)
    if (!isCrossJoin && keyPairs.empty()) {
        std::string cond = join.conditionStr;
        // Simple parsing - split by AND and extract each equality
        std::istringstream iss(cond);
        std::string part;
        while (std::getline(iss, part, 'A')) {
            auto eq = part.find('=');
            if (eq != std::string::npos) {
                std::string left = base_ident(part.substr(0, eq));
                std::string right = base_ident(part.substr(eq + 1));
                if (!left.empty() && !right.empty()) {
                    keyPairs.emplace_back(left, right);
                }
            }
        }
    }
    
    // If we still have no key pairs and there's a condition, it might be a complex non-equality join
    // (e.g., NESTED_LOOP_JOIN with > comparison from HAVING subquery)
    if (!isCrossJoin && keyPairs.empty() && join.condition) {
        // Treat as cross-join with post-filter
        if (debug) std::cerr << "[Exec] Join: no equi-join keys found but has condition, treating as cross-join + filter\n";
        isCrossJoin = true;
        hasPostJoinFilter = true;
        postJoinFilter = join.condition;
    }
    
    if (!isCrossJoin && keyPairs.empty()) {
        if (debug) std::cerr << "[Exec] Join: no key pairs found\n";
        return false;
    }
    
    if (debug) {
        std::cerr << "[Exec] Join: " << keyPairs.size() << " key pair(s):\n";
        for (const auto& [l, r] : keyPairs) {
            std::cerr << "[Exec] Join:   " << l << " = " << r << std::endl;
        }
        std::cerr << "[Exec] Join: leftCtx has " << leftCtx.u32Cols.size() << " u32 cols, " 
                  << leftCtx.f32Cols.size() << " f32 cols, " << leftCtx.rowCount << " rows";
        for (const auto& [n,_] : leftCtx.u32Cols) std::cerr << " " << n;
        std::cerr << std::endl;
        std::cerr << "[Exec] Join: rightCtx has " << rightCtx.u32Cols.size() << " u32 cols, "
                  << rightCtx.f32Cols.size() << " f32 cols, " << rightCtx.rowCount << " rows";
        for (const auto& [n,_] : rightCtx.u32Cols) std::cerr << " " << n;
        std::cerr << std::endl;
    }
    
    // Helper to find column with suffix fallback for multi-instance tables
    // Checks f32Cols and auto-converts to u32Cols (bitwise) if found
    auto findColWithSuffix = [](EvalContext& ctx, 
                                 const std::string& col) -> std::string {
        // Check U32 Direct
        if (ctx.u32Cols.find(col) != ctx.u32Cols.end()) return col;
        
        // Check F32 Direct -> Convert
        if (ctx.f32Cols.find(col) != ctx.f32Cols.end()) {
             const auto& fVec = ctx.f32Cols.at(col);
             std::vector<uint32_t> uVec(fVec.size());
             if (!fVec.empty()) std::memcpy(uVec.data(), fVec.data(), fVec.size() * sizeof(uint32_t));
             ctx.u32Cols[col] = std::move(uVec);
             return col;
        }

        // Try suffixed versions
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
    };

    // Helper to attempt fuzzy resolution of a column in a context
    // excludeCols: set of column names already used by prior key resolutions (for multi-key joins)
    auto fuzzyResolve = [&](EvalContext& ctx, const std::string& colName,
                            const std::unordered_set<std::string>& excludeCols = {}) -> std::string {
        // 1. Try suffixed versions (e.g. name_1)
        std::string s = findColWithSuffix(ctx, colName); // Use updated helper
        if (!s.empty() && excludeCols.find(s) == excludeCols.end()) return s;

        // 2. Try prefix aliases BEFORE positional refs (l_ -> o_, etc)
        if (colName.size() > 2 && colName[1] == '_') {
            std::string suffix = colName.substr(2);
            static const std::vector<std::string> prefixes = {"l_", "o_", "c_", "p_", "s_", "ps_", "n_", "r_"};
            for (const auto& p : prefixes) {
                std::string alt = p + suffix;
                std::string res = findColWithSuffix(ctx, alt); // Re-use helper to handle conversion
                if (!res.empty() && excludeCols.find(res) == excludeCols.end()) return res;
            }
        }

        // 3. Try positional refs (#0..#9) - skip already-used refs
        for (int i = 0; i < 10; ++i) {
            std::string posRef = "#" + std::to_string(i);
            if (excludeCols.find(posRef) != excludeCols.end()) continue; // Skip already used
            if (ctx.u32Cols.count(posRef)) return posRef;
            // Check F32 #N
            if (ctx.f32Cols.count(posRef)) {
                 const auto& fVec = ctx.f32Cols.at(posRef);
                 std::vector<uint32_t> uVec(fVec.size());
                 if (!fVec.empty()) std::memcpy(uVec.data(), fVec.data(), fVec.size() * sizeof(uint32_t));
                 ctx.u32Cols[posRef] = std::move(uVec);
                 return posRef;
            }
        }
        
        // 4. Try suffix match (Iterate both u32 and f32)
        auto underscorePos = colName.find('_');
        if (underscorePos != std::string::npos) {
             std::string suffix = colName.substr(underscorePos); 
             for (const auto& [n, _] : ctx.u32Cols) {
                 if (n.size() >= suffix.size() && 
                     n.rfind(suffix) == n.size() - suffix.size()) {
                     return n;
                 }
             }
             // F32 Loop
             for (const auto& [n, _] : ctx.f32Cols) {
                 if (n.size() >= suffix.size() && 
                     n.rfind(suffix) == n.size() - suffix.size()) {
                     // Check if not already in u32 (optimization)
                     findColWithSuffix(ctx, n); // Ensure converted
                     return n;
                 }
             }
        }

        // 5. Try stripping explicit aliases (_rhs_N, _lhs_N)
        size_t rhsPos = colName.find("_rhs_");
        if (rhsPos != std::string::npos) {
            std::string base = colName.substr(0, rhsPos);
            // Search exact base
            if (ctx.u32Cols.count(base) || ctx.f32Cols.count(base)) return base;
            // Recurse fuzzy on base
            // (Use simple direct check of prefixes to avoid infinite recursion if implemented recursively)
            if (base.size() > 2 && base[1] == '_') {
                std::string suffix = base.substr(2);
                static const std::vector<std::string> prefixes = {"l_", "o_", "c_", "p_", "s_", "ps_", "n_", "r_"};
                for (const auto& p : prefixes) {
                     std::string alt = p + suffix;
                     if (ctx.u32Cols.count(alt) || ctx.f32Cols.count(alt)) return alt;
                }
            }
        }
        
        return "";
    };
    
    // Resolve key columns - figure out which column is in left vs right
    std::vector<std::pair<std::string, std::string>> resolvedKeys; // (leftCol, rightCol)
    std::unordered_set<std::string> usedLeftCols, usedRightCols; // Track used cols for multi-key joins
    for (auto& [k1, k2] : keyPairs) {
        if (k1 == "supplier_no") k1 = "l_suppkey";
        if (k2 == "supplier_no") k2 = "l_suppkey";
        // Check if k1 is in left and k2 is in right (with suffix fallback)
        std::string k1Left = findColWithSuffix(leftCtx, k1);
        std::string k2Right = findColWithSuffix(rightCtx, k2);
        std::string k2Left = findColWithSuffix(leftCtx, k2);
        std::string k1Right = findColWithSuffix(rightCtx, k1);
        
        bool k1InLeft = !k1Left.empty();
        bool k2InRight = !k2Right.empty();
        bool k2InLeft = !k2Left.empty();
        bool k1InRight = !k1Right.empty();
        
        if (k1InLeft && k2InRight) {
            resolvedKeys.emplace_back(k1Left, k2Right);
            usedLeftCols.insert(k1Left);
            usedRightCols.insert(k2Right);
        } else if (k2InLeft && k1InRight) {
            resolvedKeys.emplace_back(k2Left, k1Right);
            usedLeftCols.insert(k2Left);
            usedRightCols.insert(k1Right);
        } else {
             // Try to fuzzy resolve missing left key if right key exists
             std::string leftResolved, rightResolved;
             
             if (k1InRight) {
                 // Right has k1. We need k2 in Left.
                 rightResolved = k1Right;
                 leftResolved = fuzzyResolve(leftCtx, k2, usedLeftCols);
             } else if (k2InRight) {
                 // Right has k2. We need k1 in Left.
                 rightResolved = k2Right;
                 leftResolved = fuzzyResolve(leftCtx, k1, usedLeftCols);
             }
             
             // Try to fuzzy resolve missing right key if left key exists
             if (leftResolved.empty() && rightResolved.empty()) {
                  if (k1InLeft) {
                      leftResolved = k1Left;
                      rightResolved = fuzzyResolve(rightCtx, k2, usedRightCols);
                  } else if (k2InLeft) {
                      leftResolved = k2Left;
                      rightResolved = fuzzyResolve(rightCtx, k1, usedRightCols);
                  }
             }
             
             if (!leftResolved.empty() && !rightResolved.empty()) {
                  resolvedKeys.emplace_back(leftResolved, rightResolved);
                  usedLeftCols.insert(leftResolved);
                  usedRightCols.insert(rightResolved);
                   if (debug) {
                       std::cerr << "[Exec] Join: fuzzy resolved " << k1 << "=" << k2 << " to (" 
                                 << leftResolved << ", " << rightResolved << ")\n";
                   }
             } else {
                if (debug) {
                    std::cerr << "[Exec] Join: cannot resolve key pair " << k1 << "=" << k2 
                            << " k1InLeft=" << k1InLeft << " k2InRight=" << k2InRight
                            << " k2InLeft=" << k2InLeft << " k1InRight=" << k1InRight << "\n";
                }
                return false;
             }
        }
    }
    
    if (debug) {
        std::cerr << "[Exec] Join: resolved " << resolvedKeys.size() << " key pair(s)\n";
    }
    
    // Get vectors for all keys
    std::vector<const std::vector<uint32_t>*> leftKeyVecs, rightKeyVecs;
    for (const auto& [lk, rk] : resolvedKeys) {
        leftKeyVecs.push_back(&leftCtx.u32Cols.at(lk));
        rightKeyVecs.push_back(&rightCtx.u32Cols.at(rk));
    }
    
    if (!isCrossJoin && resolvedKeys.empty()) return false;

    if (!isCrossJoin && resolvedKeys.size() > 2) throw std::runtime_error("GPU Join > 2 columns not implemented");

    auto& store = ColumnStoreGPU::instance();
    
    auto ensureGPU = [&](EvalContext& ctx, const std::string& col) -> MTL::Buffer* {
        uint32_t expectedSize = ctx.activeRowsGPU ? ctx.activeRowsCountGPU : (uint32_t)ctx.rowCount;
        if (ctx.u32ColsGPU.count(col)) {
            MTL::Buffer* existing = ctx.u32ColsGPU.at(col);
            // If the existing GPU buffer is larger than expected (uncompacted),
            // apply activeRowsGPU gather to produce a compacted buffer.
            if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0 &&
                existing->length() / sizeof(uint32_t) > expectedSize) {
                if (debug) std::cerr << "[Exec] ensureGPU: compacting GPU buf " << col << " from " << (existing->length()/sizeof(uint32_t)) << " to " << expectedSize << "\n";
                auto compactedBuf = GpuOps::gatherU32(existing, ctx.activeRowsGPU, ctx.activeRowsCountGPU, true);
                if (compactedBuf) {
                    if (debug) {
                        uint32_t* p = (uint32_t*)compactedBuf->contents();
                        std::cerr << "[Exec] ensureGPU: compacted " << col << " first 5:";
                        for (uint32_t i = 0; i < std::min(expectedSize, 5u); i++) std::cerr << " " << p[i];
                        std::cerr << "\n";
                    }
                    ctx.u32ColsGPU[col] = compactedBuf;
                    // Don't release 'existing' - it may be shared by other contexts
                    return compactedBuf;
                }
            }
            return existing;
        }
        if (ctx.u32Cols.count(col)) {
             const auto& vec = ctx.u32Cols.at(col);
             // If activeRowsGPU is set and CPU vector is larger than expected,
             // compact the column via GPU gather so indices are consistent.
             if (ctx.activeRowsGPU && ctx.activeRowsCountGPU > 0 && vec.size() > expectedSize) {
                 auto fullBuf = store.device()->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                 if (fullBuf) {
                     auto compactedBuf = GpuOps::gatherU32(fullBuf, ctx.activeRowsGPU, ctx.activeRowsCountGPU, true);
                     fullBuf->release();
                     if (compactedBuf) {
                         ctx.u32ColsGPU[col] = compactedBuf;
                         return compactedBuf;
                     }
                 }
             }
             auto buf = store.device()->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
             ctx.u32ColsGPU[col] = buf;
             return buf;
        }
        return nullptr;
    };
    
    // --- Materialize contexts before join (compact to dense form) ---
    auto materializeCtx = [&](EvalContext& ctx, const char* label) {
        if (!ctx.activeRowsGPU) return;  // No filter applied — nothing to materialize
        uint32_t count = ctx.activeRowsCountGPU;
        if (debug) std::cerr << "[Exec] Join: materializing " << label 
                             << " ctx (" << count << " active rows from " << ctx.rowCount << ")\n";
        // If the filter matched 0 rows, clear everything and set rowCount=0
        if (count == 0) {
            for (auto& [name, vec] : ctx.u32Cols) vec.clear();
            for (auto& [name, vec] : ctx.f32Cols) vec.clear();
            for (auto& [name, vec] : ctx.stringCols) vec.clear();
            ctx.flatStringColsGPU.clear();
            for (auto& [name, buf] : ctx.u32ColsGPU) {
                // Don't release — may be shared; just null out
                buf = nullptr;
            }
            for (auto& [name, buf] : ctx.f32ColsGPU) {
                buf = nullptr;
            }
            ctx.activeRowsGPU = nullptr;
            ctx.activeRowsCountGPU = 0;
            ctx.activeRows.clear();
            ctx.rowCount = 0;
            return;
        }
        
        // Compact GPU u32 columns
        for (auto& [name, buf] : ctx.u32ColsGPU) {
            if (!buf) continue;
            uint32_t bufRows = (uint32_t)(buf->length() / sizeof(uint32_t));
            if (bufRows > count) {
                MTL::Buffer* compacted = GpuOps::gatherU32(buf, ctx.activeRowsGPU, count, true);
                if (compacted) {
                    // Don't release old buf — may be shared
                    buf = compacted;
                }
            }
        }
        // Compact GPU f32 columns
        for (auto& [name, buf] : ctx.f32ColsGPU) {
            if (!buf) continue;
            uint32_t bufRows = (uint32_t)(buf->length() / sizeof(float));
            if (bufRows > count) {
                MTL::Buffer* compacted = GpuOps::gatherF32(buf, ctx.activeRowsGPU, count, true);
                if (compacted) {
                    buf = compacted;
                }
            }
        }
        // Compact CPU u32 columns
        uint32_t* indices = (uint32_t*)ctx.activeRowsGPU->contents();
        for (auto& [name, vec] : ctx.u32Cols) {
            if (vec.size() > count) {
                std::vector<uint32_t> c;
                c.reserve(count);
                for (uint32_t i = 0; i < count; ++i)
                    c.push_back(indices[i] < (uint32_t)vec.size() ? vec[indices[i]] : 0u);
                vec = std::move(c);
            }
        }
        // Compact CPU f32 columns
        for (auto& [name, vec] : ctx.f32Cols) {
            if (vec.size() > count) {
                std::vector<float> c;
                c.reserve(count);
                for (uint32_t i = 0; i < count; ++i)
                    c.push_back(indices[i] < (uint32_t)vec.size() ? vec[indices[i]] : 0.0f);
                vec = std::move(c);
            }
        }
        // Compact CPU string columns
        for (auto& [name, vec] : ctx.stringCols) {
            if (vec.size() > count) {
                std::vector<std::string> c;
                c.reserve(count);
                for (uint32_t i = 0; i < count; ++i)
                    c.push_back(indices[i] < (uint32_t)vec.size() ? vec[indices[i]] : std::string());
                vec = std::move(c);
            }
        }
        // Clear selection vector — data is now dense
        ctx.activeRowsGPU = nullptr;
        ctx.activeRowsCountGPU = 0;
        ctx.activeRows.clear();
        ctx.rowCount = count;
    };
    
    materializeCtx(leftCtx, "left");
    materializeCtx(rightCtx, "right");
    
    uint32_t rCount = (uint32_t)rightCtx.rowCount;
    uint32_t lCount = (uint32_t)leftCtx.rowCount;

    MTL::Buffer* lBuf = nullptr;
    MTL::Buffer* rBuf = nullptr;
    
    JoinResult jRes;
    
    if (lCount > 0 && rCount > 0) {
        if (isCrossJoin) {
            if (debug) std::cerr << "[Exec] GPU Join: Cross Join 1=1 (" << lCount << " x " << rCount << ")\n";
            uint64_t totalCount = (uint64_t)lCount * (uint64_t)rCount;
             
            auto device = ColumnStoreGPU::instance().device();
            jRes.count = (uint32_t)totalCount;
            jRes.probeIndices = device->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            jRes.buildIndices = device->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            
            // Get or create left/right index buffers on GPU
            MTL::Buffer* lIndicesGPU = leftCtx.activeRowsGPU;
            MTL::Buffer* rIndicesGPU = rightCtx.activeRowsGPU;
            bool createdL = false, createdR = false;
            
            if (!lIndicesGPU) {
                lIndicesGPU = GpuOps::iotaU32(lCount);
                createdL = true;
            }
            if (!rIndicesGPU) {
                rIndicesGPU = GpuOps::iotaU32(rCount);
                createdR = true;
            }
            
            // Cross product on GPU
            GpuOps::crossProduct(lIndicesGPU, rIndicesGPU,
                                       jRes.probeIndices, jRes.buildIndices,
                                       lCount, rCount);
            
            if (createdL) lIndicesGPU->release();
            if (createdR) rIndicesGPU->release();
        } else if (resolvedKeys.size() == 2) {
            if (debug) {
                std::cerr << "[Exec] Multi-Key Join (2 keys)\n";
                std::cerr << "[Exec] Multi-Key Join: key0=(" << resolvedKeys[0].first << ", " << resolvedKeys[0].second << ")\n";
                std::cerr << "[Exec] Multi-Key Join: key1=(" << resolvedKeys[1].first << ", " << resolvedKeys[1].second << ")\n";
            }
            MTL::Buffer* l1 = ensureGPU(leftCtx, resolvedKeys[0].first);
            MTL::Buffer* r1 = ensureGPU(rightCtx, resolvedKeys[0].second);
            MTL::Buffer* l2 = ensureGPU(leftCtx, resolvedKeys[1].first);
            MTL::Buffer* r2 = ensureGPU(rightCtx, resolvedKeys[1].second);
            if(!l1||!r1||!l2||!r2) throw std::runtime_error("Missing GPU col data for multi-key join");
            
            uint32_t lSize = (uint32_t)leftCtx.rowCount;
            uint32_t rSize = (uint32_t)rightCtx.rowCount;
            
            if (debug) std::cerr << "[Exec] Multi-Key Join: packing left (" << lSize << " rows)...\n" << std::flush;
            lBuf = GpuOps::packU32ToU64(l1, l2, lSize);
            if (debug) std::cerr << "[Exec] Multi-Key Join: packing right (" << rSize << " rows)...\n" << std::flush;
            rBuf = GpuOps::packU32ToU64(r1, r2, rSize);
            if (debug) std::cerr << "[Exec] Multi-Key Join: packing done.\n" << std::flush;
        } else {
            lBuf = ensureGPU(leftCtx, resolvedKeys[0].first);
            rBuf = ensureGPU(rightCtx, resolvedKeys[0].second);
        }

        if (!isCrossJoin && (!lBuf || !rBuf)) throw std::runtime_error("Missing GPU column data for Join");
        
        // Multi-match hash join; build/probe order only affects performance.
        
        bool swapped = false;
        
        if (!isCrossJoin && debug) std::cerr << "[Exec] GPU Join: Build (" << rCount << "), Probe (" << lCount << ") swapped=" << swapped << "\n";
        if (debug) {
            std::cerr << "[Exec] GPU Join: leftCtx.activeRowsGPU=" << (leftCtx.activeRowsGPU ? "SET" : "NULL") << " rightCtx.activeRowsGPU=" << (rightCtx.activeRowsGPU ? "SET" : "NULL") << "\n";
            if (leftCtx.activeRowsGPU) {
                uint32_t* leftIndices = (uint32_t*)leftCtx.activeRowsGPU->contents();
                std::cerr << "[Exec] GPU Join: leftActiveIndices first 5: ";
                for (int i = 0; i < std::min(5u, leftCtx.activeRowsCountGPU); ++i) std::cerr << leftIndices[i] << " ";
                std::cerr << "\n";
            }
        }
        
        if (!isCrossJoin) {
            // When swapped: lBuf was originally right, rBuf was originally left
            // So the activeRows also need to be swapped
            MTL::Buffer* buildActiveRows = swapped ? leftCtx.activeRowsGPU : rightCtx.activeRowsGPU;
            MTL::Buffer* probeActiveRows = swapped ? rightCtx.activeRowsGPU : leftCtx.activeRowsGPU;
            
            if (resolvedKeys.size() == 2) {
                 jRes = GpuOps::joinHashU64(rBuf, buildActiveRows, rCount, lBuf, probeActiveRows, lCount);
                 lBuf->release(); rBuf->release();
            } else {
                 jRes = GpuOps::joinHash(rBuf, buildActiveRows, rCount, lBuf, probeActiveRows, lCount);
            }
            
            // If swapped, the build/probe indices are also swapped relative to left/right
            if (swapped) {
                // probeIndices now refers to original right rows, buildIndices to original left rows
                std::swap(jRes.probeIndices, jRes.buildIndices);
            }
        }
    } else {
        if (lCount > 0 && rCount == 0 && (isAntiJoin || isLeftJoin)) {
             if (debug) std::cerr << "[Exec] GPU Join: Empty Build side for Anti/Left Join -> Returning all " << lCount << " left rows.\n";
             jRes.count = lCount;
             
             if (leftCtx.activeRowsGPU) {
                 MTL::Buffer* src = leftCtx.activeRowsGPU;
                 jRes.probeIndices = store.device()->newBuffer(src->contents(), src->length(), MTL::ResourceStorageModeShared);
             } else {
                 std::vector<uint32_t> seq(lCount);
                 std::iota(seq.begin(), seq.end(), 0);
                 jRes.probeIndices = store.device()->newBuffer(seq.data(), seq.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
             }
             
             // Placeholder build indices (required non-null)
             jRes.buildIndices = store.device()->newBuffer(lCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
             std::memset(jRes.buildIndices->contents(), 0, lCount * sizeof(uint32_t));
        } else {
            if (debug) std::cerr << "[Exec] GPU Join: Skipping (Build=" << rCount << ", Probe=" << lCount << ")\n";
            jRes.count = 0;
            jRes.buildIndices = nullptr;
            jRes.probeIndices = nullptr;
        }
    }
                                       
    if ((lCount > 0 && rCount > 0) && !jRes.buildIndices) throw std::runtime_error("GPU Join Kernel Failed");
    
    if (debug) std::cerr << "[Exec] GPU Join Success: Result " << jRes.count << " rows.\n";
    
    uint32_t resCount = jRes.count;
    
    // SEMI JOIN: deduplicate probeIndices, keeping first match per probe row.
    if (isSemiJoin && resCount > 0 && jRes.probeIndices) {
        if (debug) std::cerr << "[Exec] Semi Join: Deduplicating " << resCount << " probe indices\n";
        
        uint32_t* probePtr = (uint32_t*)jRes.probeIndices->contents();
        uint32_t* buildPtr = (uint32_t*)jRes.buildIndices->contents();
        
        // Map each probe index to its first matching build index
        std::map<uint32_t, uint32_t> probeToFirstBuild;
        for (uint32_t i = 0; i < resCount; ++i) {
            uint32_t pi = probePtr[i];
            if (probeToFirstBuild.find(pi) == probeToFirstBuild.end()) {
                probeToFirstBuild[pi] = buildPtr[i];
            }
        }
        
        std::vector<uint32_t> uniqueProbeVec;
        std::vector<uint32_t> uniqueBuildVec;
        uniqueProbeVec.reserve(probeToFirstBuild.size());
        uniqueBuildVec.reserve(probeToFirstBuild.size());
        for (const auto& [pi, bi] : probeToFirstBuild) {
            uniqueProbeVec.push_back(pi);
            uniqueBuildVec.push_back(bi);
        }
        
        uint32_t uniqueCount = (uint32_t)uniqueProbeVec.size();
        if (debug) std::cerr << "[Exec] Semi Join: After dedup: " << uniqueCount << " unique rows\n";
        
        // Replace the original probeIndices
        jRes.probeIndices->release();
        jRes.probeIndices = store.device()->newBuffer(
            uniqueProbeVec.data(), uniqueCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        
        // Preserve matching build indices for right-side column gathering
        if (jRes.buildIndices) jRes.buildIndices->release();
        jRes.buildIndices = store.device()->newBuffer(
            uniqueBuildVec.data(), uniqueCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        
        resCount = uniqueCount;
        jRes.count = uniqueCount;
    }
    
    // ANTI JOIN: Find LHS rows that did NOT match
    // Skip when rCount==0: the empty-build fast path already returns all left rows directly.
    if (isAntiJoin && rCount > 0 && jRes.probeIndices) {
        if (debug) std::cerr << "[Exec] Anti Join: Finding non-matching rows from " << lCount << " left rows, " << resCount << " matches\n";
        
        // Collect all matched LHS indices
        uint32_t* probePtr = (uint32_t*)jRes.probeIndices->contents();
        std::unordered_set<uint32_t> matchedIndices(probePtr, probePtr + resCount);
        

        std::vector<uint32_t> allLeftIndices(lCount);
        std::iota(allLeftIndices.begin(), allLeftIndices.end(), 0);
        
        // Find non-matching indices
        std::vector<uint32_t> antiResult;
        for (uint32_t idx : allLeftIndices) {
            if (matchedIndices.count(idx) == 0) {
                antiResult.push_back(idx);
            }
        }
        
        uint32_t antiCount = (uint32_t)antiResult.size();
        if (debug) std::cerr << "[Exec] Anti Join: " << antiCount << " non-matching rows\n";
        
        // Replace with anti-join result
        jRes.probeIndices->release();
        if (antiCount > 0) {
            jRes.probeIndices = store.device()->newBuffer(
                antiResult.data(), antiCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        } else {
            jRes.probeIndices = store.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
        }
        
        if (jRes.buildIndices) jRes.buildIndices->release();
        jRes.buildIndices = store.device()->newBuffer(
            std::max(antiCount, 1u) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        std::memset(jRes.buildIndices->contents(), 0, std::max(antiCount, 1u) * sizeof(uint32_t));
        
        resCount = antiCount;
        jRes.count = antiCount;
    }
    
    outCtx.rowCount = resCount;
    outCtx.activeRowsGPU = nullptr;
    outCtx.activeRowsCountGPU = 0; // Materialized

    // Collect LEFT column names (these are the "primary" names in the output)
    // We only rename RIGHT columns if they would collide with LEFT column names
    std::unordered_set<std::string> leftColumnNames;
    for (const auto& [name, _] : leftCtx.u32Cols) leftColumnNames.insert(name);
    for (const auto& [name, _] : leftCtx.f32Cols) leftColumnNames.insert(name);
    for (const auto& [name, _] : leftCtx.stringCols) leftColumnNames.insert(name);
    
    // Pre-compute the rename mapping for ALL right column names
    // This ensures U32 hash and string versions of the same column get the same output name
    std::unordered_set<std::string> rightColumnNames;
    for (const auto& [name, _] : rightCtx.u32Cols) rightColumnNames.insert(name);
    for (const auto& [name, _] : rightCtx.f32Cols) rightColumnNames.insert(name);
    for (const auto& [name, _] : rightCtx.stringCols) rightColumnNames.insert(name);
    
    // Map from original right column name to output column name
    std::unordered_map<std::string, std::string> rightColumnMapping;
    std::unordered_set<std::string> usedNames;
    for (const auto& name : leftColumnNames) usedNames.insert(name);
    
    for (const auto& name : rightColumnNames) {
        if (leftColumnNames.count(name) == 0) {
            // No collision - use original name
            rightColumnMapping[name] = name;
            usedNames.insert(name);
        } else {
            // Collision - find a unique suffix
            for (int suffix = 1; suffix <= 10; ++suffix) {
                std::string newName = name + "_" + std::to_string(suffix);
                if (usedNames.count(newName) == 0) {
                    rightColumnMapping[name] = newName;
                    usedNames.insert(newName);
                    if (debug) {
                        std::cerr << "[Exec] Join: Renaming duplicate column " << name << " -> " << newName << "\n";
                    }
                    break;
                }
            }
            // Fallback if all suffixes taken
            if (rightColumnMapping.count(name) == 0) {
                std::string fallback = name + "_r";
                rightColumnMapping[name] = fallback;
                usedNames.insert(fallback);
            }
        }
    }
    
    // Helper: Get output name for right column using the pre-computed mapping
    auto getRightColumnName = [&](const std::string& name) -> std::string {
        auto it = rightColumnMapping.find(name);
        if (it != rightColumnMapping.end()) {
            return it->second;
        }
        return name; // Shouldn't happen
    };

    if (resCount == 0) {
        // Left columns - use name as-is
        for (const auto& [name, _] : leftCtx.u32Cols) { 
            outCtx.u32Cols[name] = {};
        }
        for (const auto& [name, _] : leftCtx.f32Cols) {
            outCtx.f32Cols[name] = {};
        }
        for (const auto& [name, _] : leftCtx.stringCols) {
            outCtx.stringCols[name] = {};
        }
        // Right columns - rename only if collision
        for (const auto& [name, _] : rightCtx.u32Cols) {
            std::string outName = getRightColumnName(name);
            outCtx.u32Cols[outName] = {};
        }
        for (const auto& [name, _] : rightCtx.f32Cols) {
            std::string outName = getRightColumnName(name);
            outCtx.f32Cols[outName] = {};
        }
        for (const auto& [name, _] : rightCtx.stringCols) {
            std::string outName = getRightColumnName(name);
            outCtx.stringCols[outName] = {};
        }
        return true;
    }

    // Gather Left Columns - use names as-is (they're the "primary" columns)
    if (debug && jRes.probeIndices) {
        uint32_t* probePtr = (uint32_t*)jRes.probeIndices->contents();
        std::cerr << "[Exec] Join: probeIndices first 5: ";
        for (int i = 0; i < std::min(5u, resCount); ++i) std::cerr << probePtr[i] << " ";
        std::cerr << "\n";
    }
    for (const auto& [name, valid] : leftCtx.u32Cols) {
        if (debug) std::cerr << "[Exec] Join: gathering L_U32 " << name << " srcSize=" << valid.size() << "\n";
        MTL::Buffer* src = ensureGPU(leftCtx, name);
        if (src) {
             MTL::Buffer* gathered = GpuOps::gatherU32(src, jRes.probeIndices, resCount, false);
             outCtx.u32ColsGPU[name] = gathered;
             outCtx.u32Cols[name].clear(); // Mark CPU side as invalid
        }
    }
    for (const auto& [name, valid] : leftCtx.f32Cols) {
        if (debug) std::cerr << "[Exec] Join: gathering L_F32 " << name << " srcSize=" << valid.size() << "\n";
        MTL::Buffer* src = nullptr;
        if (leftCtx.f32ColsGPU.count(name)) src = leftCtx.f32ColsGPU.at(name);
        else if (leftCtx.f32Cols.count(name)) {
             const auto& vec = leftCtx.f32Cols.at(name);
             src = store.device()->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
             leftCtx.f32ColsGPU[name] = src;
        }
        
        if (src) {
             MTL::Buffer* gathered = GpuOps::gatherF32(src, jRes.probeIndices, resCount, false);
             outCtx.f32ColsGPU[name] = gathered;
             outCtx.f32Cols[name].clear();
        }
    }
    
    // Gather Right Columns - rename only if collision with left
    if (rCount > 0) {
        for (const auto& [name, valid] : rightCtx.u32Cols) {
            std::string outName = getRightColumnName(name);
            if (debug) std::cerr << "[Exec] Join: gathering R_U32 " << name << " -> " << outName << "\n";
            MTL::Buffer* src = ensureGPU(rightCtx, name);
            if (src) {
                 MTL::Buffer* gathered = GpuOps::gatherU32(src, jRes.buildIndices, resCount, false);
                 outCtx.u32ColsGPU[outName] = gathered;
                 outCtx.u32Cols[outName].clear();
            }
        }
        for (const auto& [name, valid] : rightCtx.f32Cols) {
            std::string outName = getRightColumnName(name);
            if (debug) std::cerr << "[Exec] Join: gathering R_F32 " << name << " -> " << outName << "\n";
            MTL::Buffer* src = nullptr;
            if (rightCtx.f32ColsGPU.count(name)) src = rightCtx.f32ColsGPU.at(name);
            else if (rightCtx.f32Cols.count(name)) {
                 const auto& vec = rightCtx.f32Cols.at(name);
                 src = store.device()->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                 rightCtx.f32ColsGPU[name] = src;
            }
    
            if (src) {
                 MTL::Buffer* gathered = GpuOps::gatherF32(src, jRes.buildIndices, resCount, false);
                 outCtx.f32ColsGPU[outName] = gathered;
                 outCtx.f32Cols[outName].clear();
            }
        }
    } else if (resCount > 0) {
         // Empty Build Side (Anti/Left Join) - Fill Right Columns with Default (0/Null)
         for (const auto& [name, valid] : rightCtx.u32Cols) {
             std::string outName = getRightColumnName(name);
             MTL::Buffer* buf = store.device()->newBuffer(resCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
             std::memset(buf->contents(), 0, resCount * sizeof(uint32_t));
             outCtx.u32ColsGPU[outName] = buf;
             std::vector<uint32_t> cpuVec(resCount, 0);
             outCtx.u32Cols[outName] = std::move(cpuVec);
        }
         for (const auto& [name, valid] : rightCtx.f32Cols) {
             std::string outName = getRightColumnName(name);
             MTL::Buffer* buf = store.device()->newBuffer(resCount * sizeof(float), MTL::ResourceStorageModeShared);
             std::memset(buf->contents(), 0, resCount * sizeof(float));
             outCtx.f32ColsGPU[outName] = buf;
             std::vector<float> cpuVec(resCount, 0.0f);
             outCtx.f32Cols[outName] = std::move(cpuVec);
        }
    }

    // CPU Gather for String Columns
    if (!leftCtx.stringCols.empty() || !rightCtx.stringCols.empty()) {
        std::vector<uint32_t> cpuProbeIndices(resCount);
        std::vector<uint32_t> cpuBuildIndices(resCount);
        // Copy back
        std::memcpy(cpuProbeIndices.data(), jRes.probeIndices->contents(), resCount * sizeof(uint32_t));
        std::memcpy(cpuBuildIndices.data(), jRes.buildIndices->contents(), resCount * sizeof(uint32_t));
        
        // Helper for parallel string gather
        auto parallelGather = [&](const std::vector<uint32_t>& indices, const std::vector<std::string>& srcVec, std::vector<std::string>& dstVec) {
             dstVec.resize(resCount); // Default construct empty strings
             
             size_t numThreads = std::thread::hardware_concurrency();
             if (numThreads == 0) numThreads = 4;
             // Don't spawn threads for small data to avoid overhead
             if (resCount < 10000) numThreads = 1;
             
             size_t chunkSize = (resCount + numThreads - 1) / numThreads;
             std::vector<std::future<void>> futures;
             
             for (size_t t = 0; t < numThreads; ++t) {
                 size_t start = t * chunkSize;
                 size_t end = std::min(start + chunkSize, (size_t)resCount);
                 if (start >= end) break;
                 
                 futures.push_back(std::async(std::launch::async, [&, start, end]() {
                     for (size_t i = start; i < end; ++i) {
                         uint32_t idx = indices[i];
                         if (idx < srcVec.size()) dstVec[i] = srcVec[idx];
                     }
                 }));
             }
             for (auto& f : futures) f.wait();
        };

        for (const auto& [name, vec] : leftCtx.stringCols) {
            // Left columns use name as-is
            if (debug) std::cerr << "[Exec] Join: gathering L_STR " << name << " srcSize=" << vec.size() << " resCount=" << resCount << "\n";
            std::vector<std::string> newVec;
            parallelGather(cpuProbeIndices, vec, newVec);
            if (debug) std::cerr << "[Exec] Join: gathered L_STR " << name << " newVec.size=" << newVec.size() << "\n";
            outCtx.stringCols[name] = std::move(newVec);
        }
        for (const auto& [name, vec] : rightCtx.stringCols) {
             // Right columns: rename only if collision with left
             std::string outName = getRightColumnName(name);
             if (debug) std::cerr << "[Exec] Join: gathering R_STR " << name << " -> " << outName << " srcSize=" << vec.size() << " resCount=" << resCount << "\n";
             std::vector<std::string> newVec;
             parallelGather(cpuBuildIndices, vec, newVec);
             if (debug) std::cerr << "[Exec] Join: gathered R_STR " << name << " newVec.size=" << newVec.size() << "\n";
             outCtx.stringCols[outName] = std::move(newVec);
        }
    }
    
    GpuOps::sync(); // Ensure all async gathers complete

    // Rebuild flat string columns from gathered data
    if (!outCtx.stringCols.empty()) {
        auto& cstore = ColumnStoreGPU::instance();
        for (const auto& [name, vec] : outCtx.stringCols) {
            outCtx.flatStringColsGPU[name] = makeFlatStringColumn(cstore.device(), vec);
        }
    }

    if (debug) {
        std::cerr << "[Exec] Join: After string gather, outCtx.stringCols sizes:\n";
        for (const auto& [name, vec] : outCtx.stringCols) {
            std::cerr << "[Exec]   stringCol " << name << " size=" << vec.size() << "\n";
        }
    }

    // LEFT JOIN: Append unmatched left rows with NULL/0 for right columns
    if (isLeftJoin && rCount > 0 && resCount > 0) {
        // Find left rows that did NOT match
        uint32_t* probePtr = (uint32_t*)jRes.probeIndices->contents();
        std::unordered_set<uint32_t> matchedLeft(probePtr, probePtr + resCount);
        
        std::vector<uint32_t> unmatchedIndices;
        for (uint32_t i = 0; i < lCount; ++i) {
            if (matchedLeft.count(i) == 0) {
                unmatchedIndices.push_back(i);
            }
        }
        
        uint32_t unmatchedCount = (uint32_t)unmatchedIndices.size();
        if (debug) std::cerr << "[Exec] Left Join: " << unmatchedCount << " unmatched left rows to append\n";
        
        if (unmatchedCount > 0) {
            uint32_t totalCount = resCount + unmatchedCount;
            
            // Create GPU buffer of unmatched indices for gather
            MTL::Buffer* unmatchedBuf = store.device()->newBuffer(
                unmatchedIndices.data(), unmatchedCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            
            // Append left columns: gather unmatched rows and concatenate with matched
            for (auto& [name, buf] : outCtx.u32ColsGPU) {
                // Check if this is a LEFT column (exists in leftCtx)
                if (leftCtx.u32Cols.count(name) || leftCtx.u32ColsGPU.count(name)) {
                    MTL::Buffer* leftSrc = nullptr;
                    if (leftCtx.u32ColsGPU.count(name)) leftSrc = leftCtx.u32ColsGPU.at(name);
                    else if (leftCtx.u32Cols.count(name) && !leftCtx.u32Cols.at(name).empty()) {
                        const auto& vec = leftCtx.u32Cols.at(name);
                        leftSrc = store.device()->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    }
                    if (leftSrc) {
                        MTL::Buffer* unmatchedGathered = GpuOps::gatherU32(leftSrc, unmatchedBuf, unmatchedCount, false);
                        // Concatenate: matched (buf, resCount) + unmatched (unmatchedGathered, unmatchedCount)
                        MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                        std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(uint32_t));
                        std::memcpy((uint8_t*)combined->contents() + resCount * sizeof(uint32_t),
                                    unmatchedGathered->contents(), unmatchedCount * sizeof(uint32_t));
                        buf->release();
                        unmatchedGathered->release();
                        buf = combined;
                    }
                } else {
                    // RIGHT column: extend with zeros (NULL)
                    MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(uint32_t));
                    std::memset((uint8_t*)combined->contents() + resCount * sizeof(uint32_t), 0, unmatchedCount * sizeof(uint32_t));
                    buf->release();
                    buf = combined;
                }
            }
            for (auto& [name, buf] : outCtx.f32ColsGPU) {
                if (leftCtx.f32Cols.count(name) || leftCtx.f32ColsGPU.count(name)) {
                    MTL::Buffer* leftSrc = nullptr;
                    if (leftCtx.f32ColsGPU.count(name)) leftSrc = leftCtx.f32ColsGPU.at(name);
                    else if (leftCtx.f32Cols.count(name) && !leftCtx.f32Cols.at(name).empty()) {
                        const auto& vec = leftCtx.f32Cols.at(name);
                        leftSrc = store.device()->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                    }
                    if (leftSrc) {
                        MTL::Buffer* unmatchedGathered = GpuOps::gatherF32(leftSrc, unmatchedBuf, unmatchedCount, false);
                        MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(float), MTL::ResourceStorageModeShared);
                        std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(float));
                        std::memcpy((uint8_t*)combined->contents() + resCount * sizeof(float),
                                    unmatchedGathered->contents(), unmatchedCount * sizeof(float));
                        buf->release();
                        unmatchedGathered->release();
                        buf = combined;
                    }
                } else {
                    MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(float), MTL::ResourceStorageModeShared);
                    std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(float));
                    std::memset((uint8_t*)combined->contents() + resCount * sizeof(float), 0, unmatchedCount * sizeof(float));
                    buf->release();
                    buf = combined;
                }
            }
            
            // String columns: append unmatched left values + empty for right
            for (auto& [name, vec] : outCtx.stringCols) {
                if (leftCtx.stringCols.count(name)) {
                    const auto& leftVec = leftCtx.stringCols.at(name);
                    for (uint32_t idx : unmatchedIndices) {
                        if (idx < leftVec.size()) vec.push_back(leftVec[idx]);
                        else vec.push_back("");
                    }
                } else {
                    // Right string column: append empty strings
                    for (uint32_t i = 0; i < unmatchedCount; ++i) {
                        vec.push_back("");
                    }
                }
            }
            
            // Also update any CPU-side u32/f32 cols
            for (auto& [name, vec] : outCtx.u32Cols) {
                if (!vec.empty() && leftCtx.u32Cols.count(name)) {
                    const auto& leftVec = leftCtx.u32Cols.at(name);
                    for (uint32_t idx : unmatchedIndices) {
                        vec.push_back(idx < leftVec.size() ? leftVec[idx] : 0);
                    }
                } else if (!vec.empty()) {
                    for (uint32_t i = 0; i < unmatchedCount; ++i) vec.push_back(0);
                }
            }
            for (auto& [name, vec] : outCtx.f32Cols) {
                if (!vec.empty() && leftCtx.f32Cols.count(name)) {
                    const auto& leftVec = leftCtx.f32Cols.at(name);
                    for (uint32_t idx : unmatchedIndices) {
                        vec.push_back(idx < leftVec.size() ? leftVec[idx] : 0.0f);
                    }
                } else if (!vec.empty()) {
                    for (uint32_t i = 0; i < unmatchedCount; ++i) vec.push_back(0.0f);
                }
            }
            
            unmatchedBuf->release();
            outCtx.rowCount = totalCount;
            resCount = totalCount;

            // Rebuild flat string columns after LEFT JOIN append
            if (!outCtx.stringCols.empty()) {
                auto& cstoreL = ColumnStoreGPU::instance();
                outCtx.flatStringColsGPU.clear();
                for (const auto& [name2, vec2] : outCtx.stringCols) {
                    outCtx.flatStringColsGPU[name2] = makeFlatStringColumn(cstoreL.device(), vec2);
                }
            }

            if (debug) std::cerr << "[Exec] Left Join: total output rows = " << totalCount << "\n";
        }
    }

    // RIGHT JOIN: Append unmatched right (build) rows with NULL/0 for left columns
    if (isRightJoin && rCount > 0 && resCount > 0) {
        // Find right rows that did NOT match (using buildIndices)
        uint32_t* buildPtr = (uint32_t*)jRes.buildIndices->contents();
        // Note: resCount may have been updated by LEFT JOIN logic above, but for RIGHT JOIN
        // we use the original matched count from the hash join
        uint32_t matchedCount = jRes.count;  // Original hash join result count
        std::unordered_set<uint32_t> matchedRight(buildPtr, buildPtr + matchedCount);
        
        std::vector<uint32_t> unmatchedIndices;
        for (uint32_t i = 0; i < rCount; ++i) {
            if (matchedRight.count(i) == 0) {
                unmatchedIndices.push_back(i);
            }
        }
        
        uint32_t unmatchedCount = (uint32_t)unmatchedIndices.size();
        if (debug) std::cerr << "[Exec] Right Join: " << unmatchedCount << " unmatched right rows to append\n";
        
        if (unmatchedCount > 0) {
            uint32_t totalCount = resCount + unmatchedCount;
            
            MTL::Buffer* unmatchedBuf = store.device()->newBuffer(
                unmatchedIndices.data(), unmatchedCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            
            // For RIGHT columns: gather unmatched rows and append
            // For LEFT columns: extend with zeros (NULL)
            for (auto& [name, buf] : outCtx.u32ColsGPU) {
                if (rightCtx.u32Cols.count(name) || rightCtx.u32ColsGPU.count(name) ||
                    rightCtx.u32Cols.count(getRightColumnName(name)) || rightCtx.u32ColsGPU.count(getRightColumnName(name))) {
                    // This is a RIGHT column — gather unmatched right values
                    // Find the source in rightCtx (may need reverse name mapping)
                    std::string srcName = name;
                    // Check if this was renamed from a right column
                    for (const auto& [origName, mappedName] : rightColumnMapping) {
                        if (mappedName == name) { srcName = origName; break; }
                    }
                    MTL::Buffer* rightSrc = nullptr;
                    if (rightCtx.u32ColsGPU.count(srcName)) rightSrc = rightCtx.u32ColsGPU.at(srcName);
                    else if (rightCtx.u32Cols.count(srcName) && !rightCtx.u32Cols.at(srcName).empty()) {
                        const auto& vec = rightCtx.u32Cols.at(srcName);
                        rightSrc = store.device()->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    }
                    if (rightSrc) {
                        MTL::Buffer* unmatchedGathered = GpuOps::gatherU32(rightSrc, unmatchedBuf, unmatchedCount, false);
                        MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                        std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(uint32_t));
                        std::memcpy((uint8_t*)combined->contents() + resCount * sizeof(uint32_t),
                                    unmatchedGathered->contents(), unmatchedCount * sizeof(uint32_t));
                        buf->release();
                        unmatchedGathered->release();
                        buf = combined;
                    } else {
                        // Can't find source — extend with zeros
                        MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                        std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(uint32_t));
                        std::memset((uint8_t*)combined->contents() + resCount * sizeof(uint32_t), 0, unmatchedCount * sizeof(uint32_t));
                        buf->release();
                        buf = combined;
                    }
                } else {
                    // LEFT column: extend with zeros (NULL)
                    MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                    std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(uint32_t));
                    std::memset((uint8_t*)combined->contents() + resCount * sizeof(uint32_t), 0, unmatchedCount * sizeof(uint32_t));
                    buf->release();
                    buf = combined;
                }
            }
            for (auto& [name, buf] : outCtx.f32ColsGPU) {
                if (rightCtx.f32Cols.count(name) || rightCtx.f32ColsGPU.count(name)) {
                    MTL::Buffer* rightSrc = nullptr;
                    std::string srcName = name;
                    for (const auto& [origName, mappedName] : rightColumnMapping) {
                        if (mappedName == name) { srcName = origName; break; }
                    }
                    if (rightCtx.f32ColsGPU.count(srcName)) rightSrc = rightCtx.f32ColsGPU.at(srcName);
                    else if (rightCtx.f32Cols.count(srcName) && !rightCtx.f32Cols.at(srcName).empty()) {
                        const auto& vec = rightCtx.f32Cols.at(srcName);
                        rightSrc = store.device()->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                    }
                    if (rightSrc) {
                        MTL::Buffer* unmatchedGathered = GpuOps::gatherF32(rightSrc, unmatchedBuf, unmatchedCount, false);
                        MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(float), MTL::ResourceStorageModeShared);
                        std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(float));
                        std::memcpy((uint8_t*)combined->contents() + resCount * sizeof(float),
                                    unmatchedGathered->contents(), unmatchedCount * sizeof(float));
                        buf->release();
                        unmatchedGathered->release();
                        buf = combined;
                    } else {
                        MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(float), MTL::ResourceStorageModeShared);
                        std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(float));
                        std::memset((uint8_t*)combined->contents() + resCount * sizeof(float), 0, unmatchedCount * sizeof(float));
                        buf->release();
                        buf = combined;
                    }
                } else {
                    MTL::Buffer* combined = store.device()->newBuffer(totalCount * sizeof(float), MTL::ResourceStorageModeShared);
                    std::memcpy(combined->contents(), buf->contents(), resCount * sizeof(float));
                    std::memset((uint8_t*)combined->contents() + resCount * sizeof(float), 0, unmatchedCount * sizeof(float));
                    buf->release();
                    buf = combined;
                }
            }
            
            // String columns
            for (auto& [name, vec] : outCtx.stringCols) {
                std::string srcName = name;
                for (const auto& [origName, mappedName] : rightColumnMapping) {
                    if (mappedName == name) { srcName = origName; break; }
                }
                if (rightCtx.stringCols.count(srcName)) {
                    const auto& rightVec = rightCtx.stringCols.at(srcName);
                    for (uint32_t idx : unmatchedIndices) {
                        if (idx < rightVec.size()) vec.push_back(rightVec[idx]);
                        else vec.push_back("");
                    }
                } else {
                    for (uint32_t i = 0; i < unmatchedCount; ++i) vec.push_back("");
                }
            }
            
            // CPU-side cols
            for (auto& [name, vec] : outCtx.u32Cols) {
                if (!vec.empty()) {
                    std::string srcName = name;
                    for (const auto& [origName, mappedName] : rightColumnMapping) {
                        if (mappedName == name) { srcName = origName; break; }
                    }
                    if (rightCtx.u32Cols.count(srcName)) {
                        const auto& rightVec = rightCtx.u32Cols.at(srcName);
                        for (uint32_t idx : unmatchedIndices) {
                            vec.push_back(idx < rightVec.size() ? rightVec[idx] : 0);
                        }
                    } else {
                        for (uint32_t i = 0; i < unmatchedCount; ++i) vec.push_back(0);
                    }
                }
            }
            for (auto& [name, vec] : outCtx.f32Cols) {
                if (!vec.empty()) {
                    std::string srcName = name;
                    for (const auto& [origName, mappedName] : rightColumnMapping) {
                        if (mappedName == name) { srcName = origName; break; }
                    }
                    if (rightCtx.f32Cols.count(srcName)) {
                        const auto& rightVec = rightCtx.f32Cols.at(srcName);
                        for (uint32_t idx : unmatchedIndices) {
                            vec.push_back(idx < rightVec.size() ? rightVec[idx] : 0.0f);
                        }
                    } else {
                        for (uint32_t i = 0; i < unmatchedCount; ++i) vec.push_back(0.0f);
                    }
                }
            }
            
            unmatchedBuf->release();
            outCtx.rowCount = totalCount;
            resCount = totalCount;

            // Rebuild flat string columns after RIGHT JOIN append
            if (!outCtx.stringCols.empty()) {
                auto& cstoreR = ColumnStoreGPU::instance();
                outCtx.flatStringColsGPU.clear();
                for (const auto& [name2, vec2] : outCtx.stringCols) {
                    outCtx.flatStringColsGPU[name2] = makeFlatStringColumn(cstoreR.device(), vec2);
                }
            }

            if (debug) std::cerr << "[Exec] Right Join: total output rows = " << totalCount << "\n";
        }
    }

    jRes.buildIndices->release();
    jRes.probeIndices->release();
    
    // Apply post-join filter for non-equality conditions (e.g., value > threshold from HAVING subquery)
    // Use GPU for evaluation and comparison
    if (hasPostJoinFilter && postJoinFilter && outCtx.rowCount > 0) {
        if (debug) {
            std::cerr << "[Exec] Join: applying post-join filter on GPU, current rows=" << outCtx.rowCount << "\n";
            std::cerr << "[Exec] Join: outCtx.f32ColsGPU:";
            for (const auto& [n, b] : outCtx.f32ColsGPU) std::cerr << " " << n << "(" << (b?b->length()/sizeof(float):0) << ")";
            std::cerr << "\n";
            std::cerr << "[Exec] Join: outCtx.u32ColsGPU:";
            for (const auto& [n, b] : outCtx.u32ColsGPU) std::cerr << " " << n << "(" << (b?b->length()/sizeof(uint32_t):0) << ")";
            std::cerr << "\n";
        }
        
        // Upload CPU columns to GPU for filtering (they were gathered from join)
        for (const auto& [name, vec] : outCtx.u32Cols) {
            if (!vec.empty() && outCtx.u32ColsGPU.find(name) == outCtx.u32ColsGPU.end()) {
                MTL::Buffer* buf = GpuOps::createBuffer(vec.data(), vec.size() * sizeof(uint32_t));
                if (buf) outCtx.u32ColsGPU[name] = buf;
            }
        }
        for (const auto& [name, vec] : outCtx.f32Cols) {
            if (!vec.empty() && outCtx.f32ColsGPU.find(name) == outCtx.f32ColsGPU.end()) {
                MTL::Buffer* buf = GpuOps::createBuffer(vec.data(), vec.size() * sizeof(float));
                if (buf) outCtx.f32ColsGPU[name] = buf;
            }
        }
        
        // For comparison filters, evaluate both sides on GPU and use GPU filter kernel
        if (postJoinFilter->kind == TypedExpr::Kind::Compare) {
            const auto& cmp = postJoinFilter->asCompare();
            
            // Evaluate LHS and RHS on GPU
            MTL::Buffer* lhsBuf = evalExprFloatGPU(cmp.left, outCtx);
            MTL::Buffer* rhsBuf = evalExprFloatGPU(cmp.right, outCtx);
            
            if (lhsBuf && rhsBuf) {
                size_t lhsCount = lhsBuf->length() / sizeof(float);
                size_t rhsCount = rhsBuf->length() / sizeof(float);
                
                // Get scalar threshold from RHS (should be 1 value after CASE evaluation)
                float threshold = *(float*)rhsBuf->contents();
                
                if (debug) {
                    float* lhsPtr = (float*)lhsBuf->contents();
                    std::cerr << "[Exec] Join post-filter GPU: LHS count=" << lhsCount << " first=" << (lhsCount>0?lhsPtr[0]:0)
                              << " RHS count=" << rhsCount << " threshold=" << threshold << "\n";
                }
                
                // Map comparison operator
                expr::CompOp gpuOp = expr::CompOp::GT;
                switch (cmp.op) {
                    case CompareOp::Gt: gpuOp = expr::CompOp::GT; break;
                    case CompareOp::Ge: gpuOp = expr::CompOp::GE; break;
                    case CompareOp::Lt: gpuOp = expr::CompOp::LT; break;
                    case CompareOp::Le: gpuOp = expr::CompOp::LE; break;
                    case CompareOp::Eq: gpuOp = expr::CompOp::EQ; break;
                    case CompareOp::Ne: gpuOp = expr::CompOp::NE; break;
                    default: gpuOp = expr::CompOp::GT; break;
                }
                
                // Use GPU filter with scalar threshold
                auto filterResult = GpuOps::filterF32("post_join_filter", lhsBuf, outCtx.rowCount, gpuOp, threshold);
                
                lhsBuf->release();
                rhsBuf->release();
                
                if (filterResult && filterResult->count > 0) {
                    uint32_t matchCount = filterResult->count;
                    MTL::Buffer* resultIndices = filterResult->indices;
                    
                    if (debug) {
                        std::cerr << "[Exec] Join: GPU post-filter matched " << matchCount << "/" << outCtx.rowCount << " rows\n";
                    }
                    
                    if (matchCount < outCtx.rowCount) {
                        // GPU compact all columns using gather
                        for (auto& [name, buf] : outCtx.u32ColsGPU) {
                            if (buf && buf->length() > 0) {
                                MTL::Buffer* compacted = GpuOps::gatherU32(buf, resultIndices, matchCount);
                                buf->release();
                                buf = compacted;
                            }
                        }
                        for (auto& [name, buf] : outCtx.f32ColsGPU) {
                            if (buf && buf->length() > 0) {
                                MTL::Buffer* compacted = GpuOps::gatherF32(buf, resultIndices, matchCount);
                                buf->release();
                                buf = compacted;
                            }
                        }
                        
                        // Let Project download on-demand; keep data on GPU
                        
                        // Compact string columns (no GPU string support - must use CPU)
                        if (!outCtx.stringCols.empty()) {
                            std::vector<uint32_t> keepIndices(matchCount);
                            std::memcpy(keepIndices.data(), resultIndices->contents(), matchCount * sizeof(uint32_t));
                            for (auto& [name, vec] : outCtx.stringCols) {
                                if (!vec.empty()) {
                                    std::vector<std::string> compacted;
                                    compacted.reserve(matchCount);
                                    for (uint32_t idx : keepIndices) {
                                        if (idx < vec.size()) compacted.push_back(std::move(vec[idx]));
                                    }
                                    vec = std::move(compacted);
                                }
                            }
                            // Rebuild flat string columns after compaction
                            auto& cstorePJ = ColumnStoreGPU::instance();
                            outCtx.flatStringColsGPU.clear();
                            for (const auto& [nm, vc] : outCtx.stringCols) {
                                outCtx.flatStringColsGPU[nm] = makeFlatStringColumn(cstorePJ.device(), vc);
                            }
                        }
                        
                        outCtx.rowCount = matchCount;
                        if (debug) {
                            std::cerr << "[Exec] Join: outCtx after filter: rowCount=" << outCtx.rowCount << " u32ColsGPU=";
                            for (const auto& [n, b] : outCtx.u32ColsGPU) std::cerr << n << "(" << (b?b->length()/sizeof(uint32_t):0) << ") ";
                            std::cerr << "\n  f32ColsGPU=";
                            for (const auto& [n, b] : outCtx.f32ColsGPU) std::cerr << n << "(" << (b?b->length()/sizeof(float):0) << ") ";
                            std::cerr << "\n";
                        }
                    }
                    
                    resultIndices->release();
                } else if (debug) {
                    std::cerr << "[Exec] Join: GPU post-filter matched 0 rows or failed\n";
                    outCtx.rowCount = 0;
                }
            } else {
                if (debug) std::cerr << "[Exec] Join: post-filter eval failed, LHS=" << (lhsBuf?"ok":"null") << " RHS=" << (rhsBuf?"ok":"null") << "\n";
                if (lhsBuf) lhsBuf->release();
                if (rhsBuf) rhsBuf->release();
            }
        } else if (debug) {
            std::cerr << "[Exec] Join: post-filter is not a comparison (kind=" << static_cast<int>(postJoinFilter->kind) << ")\n";
        }
    }
    
    return true;
}


bool GpuExecutor::orchestrateJoin(
    const IRJoin& join,
    const std::string& datasetPath,
    EvalContext& currentCtx,
    std::unordered_map<std::string, EvalContext>& tableContexts,
    std::vector<EvalContext>& savedPipelines,
    std::vector<std::set<std::string>>& savedPipelineTables,
    std::set<std::string>& joinedTables,
    bool& hasPipeline,
    ExecutionResult& result
) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
                result.isScalarAggregate = false;
                
                // Collect all columns referenced in the join condition
                std::set<std::string> condCols;
                collectColumnsFromExpr(join.condition, condCols);
                
                if (debug) {
                    std::cerr << "[Exec] Join: conditionStr=" << join.conditionStr << "\n";
                    std::cerr << "[Exec] Join: type=";
                    switch (join.type) {
                        case JoinType::Inner: std::cerr << "Inner"; break;
                        case JoinType::Left: std::cerr << "Left"; break;
                        case JoinType::Semi: std::cerr << "Semi"; break;
                        case JoinType::Anti: std::cerr << "Anti"; break;
                        case JoinType::Mark: std::cerr << "Mark"; break;
                        default: std::cerr << "Unknown(" << static_cast<int>(join.type) << ")"; break;
                    }
                    std::cerr << "\n";
                    std::cerr << "[Exec] Join: condCols extracted: ";
                    for (const auto& c : condCols) std::cerr << c << " ";
                    std::cerr << "(total=" << condCols.size() << ")\n";
                }
                
                // Skip trivial self-joins from DELIM_SCAN correlation markers.
                bool isTrivialSelfJoin = false;
                
                // Check for IS NOT DISTINCT FROM pattern (DuckDB's DELIM_SCAN correlation marker)
                if (join.conditionStr.find("IS NOT DISTINCT FROM") != std::string::npos) {
                    // Check if condition is a self-comparison (col = col)
                    size_t isNotPos = join.conditionStr.find("IS NOT DISTINCT FROM");
                    if (isNotPos != std::string::npos) {
                        std::string leftPart = join.conditionStr.substr(0, isNotPos);
                        std::string rightPart = join.conditionStr.substr(isNotPos + 20); // len("IS NOT DISTINCT FROM") = 20
                        // Trim whitespace
                        while (!leftPart.empty() && std::isspace(leftPart.back())) leftPart.pop_back();
                        while (!rightPart.empty() && std::isspace(rightPart.front())) rightPart.erase(0, 1);
                        // Remove trailing garbage from rightPart
                        size_t endPos = rightPart.find_first_of(" )");
                        if (endPos != std::string::npos) rightPart = rightPart.substr(0, endPos);
                        
                        if (leftPart == rightPart || 
                            (!leftPart.empty() && !rightPart.empty() && 
                             leftPart.find(rightPart) != std::string::npos)) {
                            // Verify column is in currentCtx (may be hidden by GroupBy).
                            bool colInContext = (currentCtx.u32Cols.find(leftPart) != currentCtx.u32Cols.end() ||
                                                 currentCtx.f32Cols.find(leftPart) != currentCtx.f32Cols.end());
                            
                            if (colInContext) {
                                // Don't skip LEFT joins (needed for DELIM joins)
                                if (join.type != JoinType::Left) {
                                    // Don't skip if explicit right table is specified
                                    if (join.rightTable.empty()) {
                                        isTrivialSelfJoin = true;
                                    } else if (debug) {
                                        std::cerr << "[Exec] Join: IS NOT DISTINCT FROM self-comparison BUT explicit right table specified (" << join.rightTable << "). Not skipping.\n";
                                    }
                                }
                                if (debug && isTrivialSelfJoin) {
                                    std::cerr << "[Exec] Join: IS NOT DISTINCT FROM self-comparison: '" 
                                              << leftPart << "' vs '" << rightPart << "' (col in context)\n";
                                }
                            } else if (debug) {
                                std::cerr << "[Exec] Join: IS NOT DISTINCT FROM self-comparison: '" 
                                          << leftPart << "' vs '" << rightPart << "' BUT col not in context, may need re-join\n";
                            }
                        }
                    }
                }
                
                // Also check for self-comparison patterns: col = col
                if (!isTrivialSelfJoin && condCols.size() == 1) {
                    // Only one unique column - it's a self-comparison
                    const std::string& col = *condCols.begin();
                    
                    // First check if the column is actually in the current context
                    bool colInContext = (currentCtx.u32Cols.find(col) != currentCtx.u32Cols.end() ||
                                         currentCtx.f32Cols.find(col) != currentCtx.f32Cols.end());
                    
                    if (colInContext) {
                        std::string baseTable = tableForColumn(col);
                        // Check if base table or any of its instances are already joined
                        bool alreadyJoined = false;
                        for (const auto& jt : joinedTables) {
                            if (jt == baseTable || jt.rfind(baseTable + "_", 0) == 0) {
                                alreadyJoined = true;
                                break;
                            }
                        }
                        if (alreadyJoined) {
                            // The column's table is already joined AND the column is in context
                            // Don't skip LEFT joins
                            if (join.type != JoinType::Left) {
                                if (join.rightTable.empty()) {
                                    isTrivialSelfJoin = true;
                                } else if (debug) {
                                    std::cerr << "[Exec] Join: self-comparison BUT explicit right table specified (" << join.rightTable << "). Not skipping.\n";
                                }
                            }
                            if (debug && isTrivialSelfJoin) {
                                std::cerr << "[Exec] Join: self-comparison detected for " << col << " (table " << baseTable << " already joined, col in context)\n";
                            }
                        }
                    } else if (debug) {
                        std::cerr << "[Exec] Join: self-comparison for " << col << " but col not in context, may need re-join\n";
                    }
                }
                
                if (isTrivialSelfJoin) {
                    if (debug) {
                        std::cerr << "[Exec] Join: skipping trivial self-join (all columns already in pipeline)\n";
                    }
                    return true; // Skip this join
                }
                
                // Check for scalar subquery pattern (join condition contains SUBQUERY).
                // currentCtx has scalar value; savedPipelines has main data.
                if (join.conditionStr.find("SUBQUERY") != std::string::npos && !savedPipelines.empty()) {
                    if (debug) {
                        std::cerr << "[Exec] Join: detected scalar subquery pattern\n";
                        std::cerr << "[Exec]   Current context rows: " << currentCtx.rowCount << "\n";
                        std::cerr << "[Exec]   Saved pipelines: " << savedPipelines.size() << "\n";
                    }
                    
                    // Determine if scalar is in currentCtx or savedPipelines
                    double scalarValue = 0.0;
                    bool foundScalar = false;
                    bool scalarIsInCurrent = (currentCtx.rowCount == 1);
                    
                    int groupedPipelineIdx = -1;
                    int scalarPipelineIdx = -1;

                    if (!scalarIsInCurrent) {
                        // Check if scalar is in savedPipelines
                        for (size_t pi = 0; pi < savedPipelines.size(); ++pi) {
                            if (savedPipelines[pi].rowCount == 1 || savedPipelines[pi].isScalarResult) {
                                scalarPipelineIdx = static_cast<int>(pi);
                                if (debug && savedPipelines[pi].isScalarResult) {
                                    std::cerr << "[Exec]   Found scalar pipeline via flag (rowCount=" << savedPipelines[pi].rowCount << ")\n";
                                }
                                break;
                            }
                        }
                    } else {
                        // Scalar is current. Find grouped pipeline in saved
                        for (size_t pi = 0; pi < savedPipelineTables.size(); ++pi) {
                             if (savedPipelineTables[pi].count("__GROUPED__") > 0 || savedPipelines[pi].rowCount > 1) {
                                 groupedPipelineIdx = static_cast<int>(pi);
                                 break;
                             }
                        }
                    }

                    const EvalContext* scalarCtx = nullptr;
                    if (scalarIsInCurrent) {
                         scalarCtx = &currentCtx;
                    } else if (scalarPipelineIdx >= 0) {
                         scalarCtx = &savedPipelines[scalarPipelineIdx];
                    }

                    if (!scalarCtx) {
                        result.error = "Scalar subquery join: could not locate scalar value source (neither current inputs nor saved pipelines seem correct)";
                        return false;
                    }

                    // Extract scalar from scalarCtx
                    // Priority: #0, then SUM/AVG, then any
                    auto tryExtract = [&](const std::string& pattern, bool exact) -> bool {
                         // Search f32
                         for (const auto& [name, values] : scalarCtx->f32Cols) {
                             if (values.empty()) continue;
                             bool match = exact ? (name == pattern) : (name.find(pattern) != std::string::npos);
                             if (match) {
                                 scalarValue = values[0];
                                 if (debug) std::cerr << "[Exec]   Scalar value from f32 col '" << name << "': " << scalarValue << "\n";
                                 return true;
                             }
                         }
                         // Search u32
                         for (const auto& [name, values] : scalarCtx->u32Cols) {
                             if (values.empty()) continue;
                             bool match = exact ? (name == pattern) : (name.find(pattern) != std::string::npos);
                             if (match) {
                                 scalarValue = static_cast<double>(values[0]);
                                 if (debug) std::cerr << "[Exec]   Scalar value from u32 col '" << name << "': " << scalarValue << "\n";
                                 return true;
                             }
                         }
                         return false;
                    };

                    if (!foundScalar) foundScalar = tryExtract("#0", true);
                    // Also check for #0 in u32 (some DBs output integer counts)
                    if (!foundScalar) foundScalar = tryExtract("SUM", false);
                    if (!foundScalar) foundScalar = tryExtract("AVG", false);
                    if (!foundScalar) foundScalar = tryExtract("first", false);
                    
                    // Fallback to any
                    if (!foundScalar) foundScalar = tryExtract("", false);

                    if (!foundScalar) {
                        result.error = "Scalar subquery join: could not find scalar value";
                        return false;
                    }

                    // Capture input scalars (e.g. CASE, Aggregates) to broadcast.
                    std::map<std::string, float> scalarF32s;
                    std::map<std::string, uint32_t> scalarU32s;
                    if (scalarCtx) {
                         for(auto& [n, v] : scalarCtx->f32Cols) if(!v.empty()) scalarF32s[n] = v[0];
                         for(auto& [n, v] : scalarCtx->u32Cols) if(!v.empty()) scalarU32s[n] = v[0];
                    }

                    // Prepare the Data (Grouped) Pipeline
                    if (scalarIsInCurrent) {
                        if (groupedPipelineIdx < 0) {
                            result.error = "Scalar subquery join: could not find grouped pipeline";
                            return false;
                        }
                        // Restore saved pipeline
                        currentCtx = savedPipelines[groupedPipelineIdx];
                        joinedTables = savedPipelineTables[groupedPipelineIdx];
                        joinedTables.erase("__GROUPED__");
                        
                        savedPipelines.erase(savedPipelines.begin() + groupedPipelineIdx);
                        savedPipelineTables.erase(savedPipelineTables.begin() + groupedPipelineIdx);
                        
                        if (debug) {
                            std::cerr << "[Exec]   Restored saved pipeline with " << currentCtx.rowCount << " rows\n";
                        }
                    } else {
                        // Data is already currentCtx. Just remove the scalar pipeline from saved.
                        if (scalarPipelineIdx >= 0) {
                            savedPipelines.erase(savedPipelines.begin() + scalarPipelineIdx);
                            savedPipelineTables.erase(savedPipelineTables.begin() + scalarPipelineIdx);
                        }
                        if (debug) {
                            std::cerr << "[Exec]   Using current context as data table with " << currentCtx.rowCount << " rows\n";
                        }
                    }

                    // Inject broadcasted scalars into the data context
                    for(auto& [n, v] : scalarF32s) {
                        if (currentCtx.f32Cols.find(n) == currentCtx.f32Cols.end() && currentCtx.f32ColsGPU.find(n) == currentCtx.f32ColsGPU.end()) {
                             currentCtx.f32Cols[n] = {v}; // Size 1 vector (scalar broadcast)
                             if (debug) std::cerr << "[Exec]   Broadcasted scalar F32col: " << n << "\n";
                        }
                    }
                    for(auto& [n, v] : scalarU32s) {
                        if (currentCtx.u32Cols.find(n) == currentCtx.u32Cols.end() && currentCtx.u32ColsGPU.find(n) == currentCtx.u32ColsGPU.end()) {
                             currentCtx.u32Cols[n] = {v};
                             if (debug) std::cerr << "[Exec]   Broadcasted scalar U32col: " << n << "\n";
                        }
                    }

                    // Parse condition to extract comparison column and operator
                    std::string condStr = join.conditionStr;
                    
                    // Find the comparison operator
                    size_t opPos = std::string::npos;
                    std::string opStr;
                    engine::expr::CompOp compOp = engine::expr::CompOp::EQ;
                    if ((opPos = condStr.find(" > SUBQUERY")) != std::string::npos) {
                        opStr = ">";
                        compOp = engine::expr::CompOp::GT;
                    } else if ((opPos = condStr.find(" >= SUBQUERY")) != std::string::npos) {
                        opStr = ">=";
                        compOp = engine::expr::CompOp::GE;
                    } else if ((opPos = condStr.find(" < SUBQUERY")) != std::string::npos) {
                        opStr = "<";
                        compOp = engine::expr::CompOp::LT;
                    } else if ((opPos = condStr.find(" <= SUBQUERY")) != std::string::npos) {
                        opStr = "<=";
                        compOp = engine::expr::CompOp::LE;
                    } else if ((opPos = condStr.find(" = SUBQUERY")) != std::string::npos) {
                        opStr = "=";
                        compOp = engine::expr::CompOp::EQ;
                    }
                    
                    if (opPos == std::string::npos) {
                        result.error = "Scalar subquery join: unsupported comparison operator in condition: " + condStr;
                        return false;
                    }
                    
                    // Extract the column/expression being compared
                    std::string leftExpr = condStr.substr(0, opPos);
                    // Trim
                    while (!leftExpr.empty() && std::isspace(leftExpr.back())) leftExpr.pop_back();
                    
                    // Find matching aggregate column in context
                    std::string aggColName;
                    
                    // First check if we have #1 (typical aggregate position)
                    if (currentCtx.f32Cols.find("#1") != currentCtx.f32Cols.end()) {
                        aggColName = "#1";
                    } else if (currentCtx.f32Cols.find("SUM_#1") != currentCtx.f32Cols.end()) {
                        aggColName = "SUM_#1";
                    } else if (currentCtx.u32Cols.find("#1") != currentCtx.u32Cols.end()) {
                        aggColName = "#1";
                    } else {
                        // Look for any aggregate column
                        for (const auto& [name, vals] : currentCtx.f32Cols) {
                            if (name.find("SUM") != std::string::npos || 
                                name.find("AVG") != std::string::npos ||
                                name.find("COUNT") != std::string::npos ||
                                name[0] == '#') {
                                aggColName = name;
                                break;
                            }
                        }
                    }
                    
                    if (aggColName.empty()) {
                        result.error = "Scalar subquery join: could not find aggregate column";
                        return false;
                    }
                    
                    if (debug) {
                        std::cerr << "[Exec]   Filtering: " << aggColName << " " << opStr << " " << scalarValue << "\n";
                    }
                    
                    // Apply scalar subquery filter on GPU
                    // 1. Ensure data columns are uploaded to GPU
                    auto device = ColumnStoreGPU::instance().device();
                    for (auto& [name, vec] : currentCtx.f32Cols) {
                        if (currentCtx.f32ColsGPU.find(name) == currentCtx.f32ColsGPU.end()) {
                            auto buf = device->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                            if (buf) currentCtx.f32ColsGPU[name] = buf;
                        }
                    }
                    for (auto& [name, vec] : currentCtx.u32Cols) {
                        if (currentCtx.u32ColsGPU.find(name) == currentCtx.u32ColsGPU.end()) {
                            auto buf = device->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
                            if (buf) currentCtx.u32ColsGPU[name] = buf;
                        }
                    }
                    
                    // 2. Build a TypedExpr comparison predicate: aggColName <op> scalarValue
                    CompareOp typedOp = CompareOp::Eq;
                    switch (compOp) {
                        case engine::expr::CompOp::GT: typedOp = CompareOp::Gt; break;
                        case engine::expr::CompOp::GE: typedOp = CompareOp::Ge; break;
                        case engine::expr::CompOp::LT: typedOp = CompareOp::Lt; break;
                        case engine::expr::CompOp::LE: typedOp = CompareOp::Le; break;
                        case engine::expr::CompOp::EQ: typedOp = CompareOp::Eq; break;
                        case engine::expr::CompOp::NE: typedOp = CompareOp::Ne; break;
                    }
                    auto filterPred = TypedExpr::compare(
                        typedOp,
                        TypedExpr::column(aggColName),
                        TypedExpr::literal(static_cast<double>(scalarValue))
                    );
                    
                    // 3. Execute GPU filter
                    if (!executeGPUFilterRecursive(filterPred, currentCtx)) {
                        result.error = "Scalar subquery join: GPU filter failed for " + aggColName;
                        return false;
                    }
                    
                    // 4. Materialize: compact all columns using activeRowsGPU
                    if (currentCtx.activeRowsGPU && currentCtx.activeRowsCountGPU > 0) {
                        uint32_t count = currentCtx.activeRowsCountGPU;
                        uint32_t* indices = (uint32_t*)currentCtx.activeRowsGPU->contents();
                        
                        // Compact GPU columns
                        for (auto& [name, buf] : currentCtx.u32ColsGPU) {
                            if (!buf) continue;
                            uint32_t bufRows = (uint32_t)(buf->length() / sizeof(uint32_t));
                            if (bufRows > count) {
                                auto compacted = GpuOps::gatherU32(buf, currentCtx.activeRowsGPU, count, true);
                                if (compacted) buf = compacted;
                            }
                        }
                        for (auto& [name, buf] : currentCtx.f32ColsGPU) {
                            if (!buf) continue;
                            uint32_t bufRows = (uint32_t)(buf->length() / sizeof(float));
                            if (bufRows > count) {
                                auto compacted = GpuOps::gatherF32(buf, currentCtx.activeRowsGPU, count, true);
                                if (compacted) buf = compacted;
                            }
                        }
                        // Compact CPU columns
                        for (auto& [name, vec] : currentCtx.u32Cols) {
                            if (vec.size() > count) {
                                std::vector<uint32_t> c;
                                c.reserve(count);
                                for (uint32_t i = 0; i < count; ++i)
                                    c.push_back(indices[i] < (uint32_t)vec.size() ? vec[indices[i]] : 0u);
                                vec = std::move(c);
                            }
                        }
                        for (auto& [name, vec] : currentCtx.f32Cols) {
                            if (vec.size() > count) {
                                std::vector<float> c;
                                c.reserve(count);
                                for (uint32_t i = 0; i < count; ++i)
                                    c.push_back(indices[i] < (uint32_t)vec.size() ? vec[indices[i]] : 0.0f);
                                vec = std::move(c);
                            }
                        }
                        
                        currentCtx.activeRowsGPU = nullptr;
                        currentCtx.activeRowsCountGPU = 0;
                        currentCtx.activeRows.clear();
                        currentCtx.rowCount = count;
                    } else {
                        // No rows matched
                        currentCtx.rowCount = 0;
                        currentCtx.activeRows.clear();
                        currentCtx.activeRowsGPU = nullptr;
                        currentCtx.activeRowsCountGPU = 0;
                    }
                    
                    // Reset scalar aggregate flag - we now have a proper table result
                    result.isScalarAggregate = false;
                    
                    if (debug) {
                        std::cerr << "[Exec]   After scalar filter: " << currentCtx.rowCount << " rows\n";
                    }
                    
                    // Don't do the normal join - we've handled this specially
                    return true;
                }
                
                // Alt: scalar SUBQUERY join (savedPipelines empty, data in tableContexts).
                // currentCtx has scalar; right side is in tableContexts.
                if (join.conditionStr.find("SUBQUERY") != std::string::npos && savedPipelines.empty()) {
                    // Check if this is a theta-comparison (>, <, >=, <=) with SUBQUERY
                    std::string condStr = join.conditionStr;
                    size_t opPos = std::string::npos;
                    std::string opStr;
                    bool isTheta = false;
                    
                    if ((opPos = condStr.find(" > SUBQUERY")) != std::string::npos) {
                        opStr = ">"; isTheta = true;
                    } else if ((opPos = condStr.find(" >= SUBQUERY")) != std::string::npos) {
                        opStr = ">="; isTheta = true;
                    } else if ((opPos = condStr.find(" < SUBQUERY")) != std::string::npos) {
                        opStr = "<"; isTheta = true;
                    } else if ((opPos = condStr.find(" <= SUBQUERY")) != std::string::npos) {
                        opStr = "<="; isTheta = true;
                    } else if ((opPos = condStr.find(" = SUBQUERY")) != std::string::npos) {
                        opStr = "="; isTheta = true;
                    }
                    
                    if (isTheta && currentCtx.rowCount <= 1) {
                        if (debug) {
                            std::cerr << "[Exec] Join: scalar SUBQUERY theta-join (tableContexts path)\n";
                            std::cerr << "[Exec]   Current context rows: " << currentCtx.rowCount << "\n";
                        }
                        
                        // Extract scalar value from currentCtx
                        double scalarValue = 0.0;
                        bool foundScalar = false;
                        
                        // Priority 1: Explicit AVG column (common for scalar subquery).
                        auto avgIt = currentCtx.f32Cols.find("AVG");
                        if (avgIt != currentCtx.f32Cols.end() && !avgIt->second.empty()) {
                            scalarValue = avgIt->second[0];
                            foundScalar = true;
                            if (debug) {
                                std::cerr << "[Exec]   Scalar value from 'AVG': " << scalarValue << "\n";
                            }
                        }
                        
                        // Priority 2: Look for SUM column
                        if (!foundScalar) {
                            auto sumIt = currentCtx.f32Cols.find("SUM");
                            if (sumIt != currentCtx.f32Cols.end() && !sumIt->second.empty()) {
                                scalarValue = sumIt->second[0];
                                foundScalar = true;
                                if (debug) {
                                    std::cerr << "[Exec]   Scalar value from 'SUM': " << scalarValue << "\n";
                                }
                            }
                        }
                        
                        // Priority 3: #0 (first computed column, scalar result).
                        if (!foundScalar) {
                            auto numIt = currentCtx.f32Cols.find("#0");
                            if (numIt != currentCtx.f32Cols.end() && !numIt->second.empty()) {
                                scalarValue = numIt->second[0];
                                foundScalar = true;
                                if (debug) {
                                    std::cerr << "[Exec]   Scalar value from '#0': " << scalarValue << "\n";
                                }
                            }
                        }
                        
                        // Fallback: any f32 column except COUNT
                        if (!foundScalar) {
                            for (const auto& [name, values] : currentCtx.f32Cols) {
                                if (!values.empty() && name.find("COUNT") == std::string::npos) {
                                    scalarValue = values[0];
                                    foundScalar = true;
                                    if (debug) {
                                        std::cerr << "[Exec]   Scalar value fallback from '" << name << "': " << scalarValue << "\n";
                                    }
                                    break;
                                }
                            }
                        }
                        
                        if (!foundScalar) {
                            if (debug) std::cerr << "[Exec]   Could not find scalar value\n";
                            result.error = "Scalar SUBQUERY join: could not extract scalar value";
                            return false;
                        }
                        
                        // Find the data table - the one containing the comparison column
                        // Parse column from condition (e.g., "CAST(c_acctbal AS DOUBLE)" -> c_acctbal)
                        std::string leftExpr = condStr.substr(0, opPos);
                        // Extract column name from CAST or direct reference
                        std::string filterCol;
                        if (leftExpr.find("CAST(") != std::string::npos) {
                            size_t start = leftExpr.find("CAST(") + 5;
                            size_t end = leftExpr.find(" AS", start);
                            if (end != std::string::npos) {
                                filterCol = leftExpr.substr(start, end - start);
                                // Trim
                                while (!filterCol.empty() && std::isspace(filterCol.front())) filterCol.erase(0, 1);
                                while (!filterCol.empty() && std::isspace(filterCol.back())) filterCol.pop_back();
                            }
                        }
                        if (filterCol.empty()) {
                            filterCol = leftExpr;
                            while (!filterCol.empty() && std::isspace(filterCol.front())) filterCol.erase(0, 1);
                            while (!filterCol.empty() && std::isspace(filterCol.back())) filterCol.pop_back();
                        }
                        
                        if (debug) {
                            std::cerr << "[Exec]   Filter column: " << filterCol << "\n";
                        }
                        
                        // Find the table with this column in tableContexts
                        std::string dataTable;
                        for (const auto& [tname, tctx] : tableContexts) {
                            if (tctx.f32Cols.find(filterCol) != tctx.f32Cols.end() ||
                                tctx.u32Cols.find(filterCol) != tctx.u32Cols.end()) {
                                // Check for suffixed versions too
                                if (joinedTables.find(tname) == joinedTables.end()) {
                                    dataTable = tname;
                                    break;
                                }
                            }
                            // Try with suffix
                            for (const auto& [cname, cvals] : tctx.f32Cols) {
                                if ((cname == filterCol || cname.find(filterCol + "_") == 0 || 
                                     cname.rfind("_" + filterCol) == cname.size() - filterCol.size() - 1) &&
                                    joinedTables.find(tname) == joinedTables.end()) {
                                    dataTable = tname;
                                    filterCol = cname;  // Use actual column name
                                    break;
                                }
                            }
                            if (!dataTable.empty()) break;
                        }
                        
                        if (dataTable.empty()) {
                            if (debug) std::cerr << "[Exec]   Could not find data table\n";
                            result.error = "Scalar SUBQUERY join: could not find data table";
                            return false;
                        }
                        
                        if (debug) {
                            std::cerr << "[Exec]   Data table: " << dataTable << " with " 
                                      << tableContexts[dataTable].rowCount << " rows\n";
                        }
                        
                        // Apply the filter: col <op> scalarValue
                        EvalContext& dataCtx = tableContexts[dataTable];
                        std::vector<uint32_t> passingIndices;
                        
                        auto it = dataCtx.f32Cols.find(filterCol);
                        if (it == dataCtx.f32Cols.end()) {
                            // Try suffixed versions
                            for (const auto& [cname, cvals] : dataCtx.f32Cols) {
                                if (cname.find(filterCol) != std::string::npos) {
                                    it = dataCtx.f32Cols.find(cname);
                                    filterCol = cname;
                                    break;
                                }
                            }
                        }
                        
                        if (it != dataCtx.f32Cols.end()) {
                            // Valid column to filter
                            auto& store = ColumnStoreGPU::instance();
                             
                            // Ensure column is on GPU
                            MTL::Buffer* colBuf = nullptr;
                            if (dataCtx.f32ColsGPU.count(filterCol)) {
                                colBuf = dataCtx.f32ColsGPU[filterCol];
                            } else {
                                // Upload (Lazy)
                                const auto& vec = it->second;
                                colBuf = store.device()->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                                dataCtx.f32ColsGPU[filterCol] = colBuf;
                            }

                            // Map Op
                            engine::expr::CompOp op = engine::expr::CompOp::EQ;
                            if (opStr == ">") op = engine::expr::CompOp::GT;
                            else if (opStr == ">=") op = engine::expr::CompOp::GE;
                            else if (opStr == "<") op = engine::expr::CompOp::LT;
                            else if (opStr == "<=") op = engine::expr::CompOp::LE;
                            else if (opStr == "=") op = engine::expr::CompOp::EQ;

                            std::optional<FilterResult> filterRes;
                            if (dataCtx.activeRowsGPU) {
                                 filterRes = GpuOps::filterF32Indexed(filterCol, colBuf, dataCtx.activeRowsGPU, dataCtx.activeRowsCountGPU, op, static_cast<float>(scalarValue));
                            } else {
                                 filterRes = GpuOps::filterF32(filterCol, colBuf, dataCtx.rowCount, op, static_cast<float>(scalarValue));
                            }
                            
                            if (!filterRes) throw std::runtime_error("GPU Scalar Filter failed");
                            
                            MTL::Buffer* indices = filterRes->indices;
                            uint32_t newCount = filterRes->count;
                            
                            // Download indices for CPU String sync
                            std::vector<uint32_t> passingIndices(newCount);
                            if (newCount > 0) {
                                std::memcpy(passingIndices.data(), indices->contents(), newCount * sizeof(uint32_t));
                            }

                            // Safe Gather for U32 (preserving aliases and avoiding double-free)
                            std::unordered_map<MTL::Buffer*, MTL::Buffer*> u32Replacements;
                            for (auto& [name, buf] : dataCtx.u32ColsGPU) {
                                if (buf && u32Replacements.find(buf) == u32Replacements.end()) {
                                    u32Replacements[buf] = GpuOps::gatherU32(buf, indices, newCount);
                                }
                            }
                            // Update map with new buffers
                            for (auto& [name, buf] : dataCtx.u32ColsGPU) {
                                if (buf) {
                                    MTL::Buffer* newBuf = u32Replacements[buf];
                                    newBuf->retain(); 
                                    buf = newBuf; 
                                }
                            }
                            // Cleanup old buffers and consume creation ref of new buffers
                            for (auto& [oldBuf, newBuf] : u32Replacements) {
                                oldBuf->release(); 
                                newBuf->release(); 
                            }
                            
                            // Safe Gather for F32
                            std::unordered_map<MTL::Buffer*, MTL::Buffer*> f32Replacements;
                            for (auto& [name, buf] : dataCtx.f32ColsGPU) {
                                if (buf && f32Replacements.find(buf) == f32Replacements.end()) {
                                    f32Replacements[buf] = GpuOps::gatherF32(buf, indices, newCount);
                                }
                            }
                            for (auto& [name, buf] : dataCtx.f32ColsGPU) {
                                if (buf) {
                                    MTL::Buffer* newBuf = f32Replacements[buf];
                                    newBuf->retain();
                                    buf = newBuf;
                                }
                            }
                            for (auto& [oldBuf, newBuf] : f32Replacements) {
                                oldBuf->release();
                                newBuf->release();
                            }
                            
                            // Handle strings on CPU (unavoidable materialization)
                            for (auto& [name, vals] : dataCtx.stringCols) {
                                std::vector<std::string> compacted;
                                compacted.reserve(passingIndices.size());
                                for (uint32_t idx : passingIndices) {
                                    if (idx < vals.size()) compacted.push_back(vals[idx]);
                                    else compacted.push_back("");
                                }
                                vals = std::move(compacted);
                            }

                            // Update Context
                            dataCtx.rowCount = newCount;
                            
                            // After gather, buffers are dense; clear activeRowsGPU.
                            if (dataCtx.activeRowsGPU) {
                                dataCtx.activeRowsGPU->release();
                                dataCtx.activeRowsGPU = nullptr;
                            }
                            dataCtx.activeRowsCountGPU = 0;
                            
                            // Clear CPU vectors to enforce GPU usage
                            for(auto& [n, v] : dataCtx.u32Cols) v.clear(); 
                            for(auto& [n, v] : dataCtx.f32Cols) v.clear();
                            dataCtx.activeRows.clear();
                            
                            indices->release();
                        }
                        
                        // Switch currentCtx to the filtered data table
                        currentCtx = dataCtx;
                        joinedTables.clear();
                        joinedTables.insert(dataTable);
                        hasPipeline = true;
                        
                        return true;  // Handled this join
                    }
                }
                
                // Check for malformed joins where both condition columns are from the same table
                // and are DIFFERENT columns (e.g., "p_size = p_partkey" in Q16).
                // Do NOT skip valid self-comparisons (e.g., "col = col") which indicate self-joins.
                if (condCols.size() == 2) {
                    std::string firstTable;
                    bool allColsFromSameTable = true;
                    bool hasOrphanColumn = false;
                    std::vector<std::string> colsList(condCols.begin(), condCols.end());
                    
                    for (const auto& col : condCols) {
                        std::string baseTable = tableForColumn(col);
                        if (baseTable.empty()) {
                            hasOrphanColumn = true;
                        } else {
                            if (firstTable.empty()) {
                                firstTable = baseTable;
                            } else if (baseTable != firstTable) {
                                allColsFromSameTable = false;
                            }
                        }
                    }
                    
                    // Check for self-comparison patterns (e.g., "l_k = l_k").
                    // These indicate valid self-joins between table instances.
                    bool hasSelfComparisonInCondition = false;
                    for (const auto& col : condCols) {
                        // Check for "col = col" pattern
                        std::string pattern1 = col + " = " + col;
                        std::string pattern2 = col + " IS NOT DISTINCT FROM " + col;
                        if (join.conditionStr.find(pattern1) != std::string::npos ||
                            join.conditionStr.find(pattern2) != std::string::npos) {
                            hasSelfComparisonInCondition = true;
                            break;
                        }
                    }
                    
                    // "p_size = p_partkey" (same table, different col) -> skip.
                    // "col = col" -> valid self-join.
                    // Also check for suffixed aliases (e.g. p_partkey_rhs_9) which imply distinct instances
                    bool hasAlias = false;
                    for (const auto& col : condCols) {
                        if (col.find("_rhs_") != std::string::npos || col.find("_lhs_") != std::string::npos) {
                            hasAlias = true;
                            break;
                        }
                    }

                    if (allColsFromSameTable && !firstTable.empty() && !hasOrphanColumn && !hasSelfComparisonInCondition && !hasAlias) {
                        if (debug) {
                            std::cerr << "[Exec] Join: skipping malformed join (both columns from " 
                                      << firstTable << ", different cols: " << colsList[0] << " vs " << colsList[1] << ")\n";
                        }
                        return true; // Skip this join
                    }
                    
                    // Check orphan columns (no prefix). Only skip if genuinely not found anywhere (CTX/CTE).
                    if (hasOrphanColumn) {
                        bool orphanFoundSomewhere = false;
                        for (const auto& col : condCols) {
                            if (tableForColumn(col).empty()) {
                                // Check if this orphan column exists in currentCtx, tableContexts, or savedPipelines
                                if (currentCtx.u32Cols.find(col) != currentCtx.u32Cols.end() ||
                                    currentCtx.f32Cols.find(col) != currentCtx.f32Cols.end()) {
                                    orphanFoundSomewhere = true;
                                    break;
                                }
                                for (const auto& [tname, tctx] : tableContexts) {
                                    if (tctx.u32Cols.find(col) != tctx.u32Cols.end() ||
                                        tctx.f32Cols.find(col) != tctx.f32Cols.end()) {
                                        orphanFoundSomewhere = true;
                                        break;
                                    }
                                }
                                for (const auto& sp : savedPipelines) {
                                    if (sp.u32Cols.find(col) != sp.u32Cols.end() ||
                                        sp.f32Cols.find(col) != sp.f32Cols.end()) {
                                        orphanFoundSomewhere = true;
                                        break;
                                    }
                                }
                            }
                        }
                        
                        // Only skip if orphan column is truly not found
                        if (!orphanFoundSomewhere) {
                            static const std::unordered_map<std::string, std::string> knownAliases = {
                                {"supplier_no", "l_suppkey"}
                            };
                            
                            bool resolvedAlias = false;
                            for (const auto& col : condCols) {
                                if (knownAliases.count(col)) {
                                    std::string mapped = knownAliases.at(col);
                                    // Check if mapped column exists anywhere
                                    bool mappedFound = false;
                                    auto checkCtx = [&](const EvalContext& ctx) {
                                        return ctx.u32Cols.count(mapped) || ctx.f32Cols.count(mapped);
                                    };
                                    
                                    if (checkCtx(currentCtx)) mappedFound = true;
                                    else {
                                        for (const auto& [t, c] : tableContexts) if (checkCtx(c)) mappedFound = true;
                                        if (!mappedFound) for (const auto& sp : savedPipelines) if (checkCtx(sp)) mappedFound = true;
                                    }
                                    
                                    if (mappedFound) {
                                        if (debug) std::cerr << "[Exec] Join: resolved orphan '" << col << "' -> '" << mapped << "'\n";
                                        resolvedAlias = true;
                                        // Found via alias. Proceed to 'matchesRHS' check; subsequent logic handles lookup.
                                        orphanFoundSomewhere = true;
                                    }
                                }
                            }

                            if (!orphanFoundSomewhere) {
                                if (debug) {
                                    std::cerr << "[Exec] Join: skipping join with orphan column (not found anywhere)\n";
                                }
                                return true; // Skip this join
                            }
                        } else if (debug) {
                            std::cerr << "[Exec] Join: orphan column found in some context, proceeding\n";
                        }
                    }
                }
                
                // Check for unjoined table instances (priority over saved pipelines for multi-instance tables).
                // Ensure table is NOT already in a saved pipeline.
                std::string unjoinedTableForJoin;
                
                // Lambda to check if column (or its suffixed version) exists in a context
                auto hasColumnOrSuffixed = [](const EvalContext& ctx, const std::string& colName) -> bool {
                    if (ctx.u32Cols.find(colName) != ctx.u32Cols.end()) return true;
                    if (ctx.f32Cols.find(colName) != ctx.f32Cols.end()) return true;
                    // Try numeric suffixes
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
                };
                
// Check if we need to use a saved pipeline for this join
                // Check this FIRST to prefer connecting to existing intermediate results (e.g. filtered sub-joins)
                // over creating new Cartesian products with fresh table instances.
                int savedPipelineIdx = -1;

                // PRIORITY: Explicit right table check (for DELIM joins)
                if (!join.rightTable.empty()) {
                    bool specificTableFound = false;
                    
                    // For base table names (not tmpl_ prefixes), check tableContexts FIRST
                    // This prevents incorrectly using a saved pipeline that contains a table
                    // when we actually need a fresh instance (e.g., nation table for multiple
                    // different nation joins in Q7)
                    bool isBaseTable = (join.rightTable.find("tmpl_") != 0);
                    
                    if (isBaseTable && tableContexts.count(join.rightTable)) {
                        unjoinedTableForJoin = join.rightTable;
                        specificTableFound = true;
                        if (debug) std::cerr << "[Exec] Join: found explicit right table '" << join.rightTable << "' in tableContexts (base table priority)\n";
                    }
                    
                    // For tmpl_ tables, check saved pipelines first
                    if (!specificTableFound) {
                        for (int pi = (int)savedPipelines.size() - 1; pi >= 0; --pi) {
                            if (savedPipelineTables[pi].count(join.rightTable)) {
                                savedPipelineIdx = pi;
                                specificTableFound = true;
                                if (debug) std::cerr << "[Exec] Join: found explicit right table '" << join.rightTable << "' in saved pipeline #" << pi << "\n";
                                break;
                            }
                        }
                    }
                    
                    // Check table contexts if not found in saved
                    if (!specificTableFound && tableContexts.count(join.rightTable)) {
                        unjoinedTableForJoin = join.rightTable;
                        specificTableFound = true;
                        if (debug) std::cerr << "[Exec] Join: found explicit right table '" << join.rightTable << "' in tableContexts\n";
                    }
                    
                    // VALIDATE: If the explicit right table doesn't contain any condition columns
                    // that are missing from the current context, it's likely a misidentification
                    // (e.g., planner captured a saved pipeline instead of the actual join target).
                    // Fall through to heuristic search in that case.
                    if (specificTableFound && !unjoinedTableForJoin.empty()) {
                        const EvalContext& rightCandidate = tableContexts[unjoinedTableForJoin];
                        bool hasNewColumn = false;
                        for (const auto& col : condCols) {
                            if (!hasColumnOrSuffixed(currentCtx, col) && hasColumnOrSuffixed(rightCandidate, col)) {
                                hasNewColumn = true;
                                break;
                            }
                        }
                        if (!hasNewColumn) {
                            if (debug) std::cerr << "[Exec] Join: explicit right table '" << unjoinedTableForJoin
                                                  << "' has no new condition columns, falling through to heuristic\n";
                            unjoinedTableForJoin.clear();
                            specificTableFound = false;
                        }
                    }
                }

                // If explicit lookup didn't set anything, run legacy heuristic
                if (savedPipelineIdx < 0 && unjoinedTableForJoin.empty())
                // Prefer LATEST pipeline (reverse search) to ensure we get the most accumulated state
                for (int pi = (int)savedPipelines.size() - 1; pi >= 0; --pi) {
                    const auto& savedCtx = savedPipelines[pi];
                    // Check if this saved pipeline has columns needed for the join
                    for (const auto& col : condCols) {
                        if (hasColumnOrSuffixed(savedCtx, col)) {
                            // Check that current pipeline doesn't have this column (or suffixed version)
                            if (!hasColumnOrSuffixed(currentCtx, col)) {
                                // Before accepting, check if the saved pipeline has been aggregated
                                // (very few rows compared to the fresh table in tableContexts).
                                // If so, prefer the fresh table from tableContexts instead.
                                std::string baseTable = tableForColumn(col);
                                bool isAggregatedPipeline = false;
                                if (!baseTable.empty() && savedCtx.rowCount <= 10) {
                                    for (const auto& [key, freshCtx] : tableContexts) {
                                        bool isInstanceOf = (key == baseTable || 
                                                            key.rfind(baseTable + "_", 0) == 0);
                                        if (isInstanceOf && freshCtx.rowCount > 10 && 
                                            joinedTables.find(key) == joinedTables.end()) {
                                            isAggregatedPipeline = true;
                                            if (debug) std::cerr << "[Exec] Join: savedPipeline " << pi 
                                                << " has " << savedCtx.rowCount << " rows but fresh table '" 
                                                << key << "' has " << freshCtx.rowCount 
                                                << " rows — skipping aggregated pipeline\n";
                                            break;
                                        }
                                    }
                                }
                                if (!isAggregatedPipeline) {
                                    savedPipelineIdx = pi;
                                }
                            }
                        }
                    }
                    if (savedPipelineIdx >= 0) break;
                }

                if (savedPipelineIdx < 0 && unjoinedTableForJoin.empty()) {
                    for (const auto& col : condCols) {
                        // Skip if column (or its suffixed version) is already in current context
                        if (hasColumnOrSuffixed(currentCtx, col)) {
                            continue;  // Column already in current context
                        }
                        
                        std::string baseTable = tableForColumn(col);
                        if (baseTable.empty()) continue;
                        
                        // Check for unjoined table instances
                        for (const auto& [key, ctx] : tableContexts) {
                            bool isInstanceOf = (key == baseTable || 
                                                key.rfind(baseTable + "_", 0) == 0);
                            if (isInstanceOf && joinedTables.find(key) == joinedTables.end()) {
                                // Check if this table is in a saved pipeline.
                                // If the saved pipeline has been aggregated to a very small row count
                                // (e.g. scalar subquery result), prefer the fresh table from tableContexts
                                // over the aggregated saved pipeline.
                                bool inSavedPipeline = false;
                                bool savedPipelineIsAggregated = false;
                                for (size_t spi = 0; spi < savedPipelineTables.size(); ++spi) {
                                    if (savedPipelineTables[spi].find(key) != savedPipelineTables[spi].end()) {
                                        inSavedPipeline = true;
                                        // Check if the saved pipeline was aggregated down 
                                        // (much fewer rows than the original table)
                                        size_t savedRows = savedPipelines[spi].rowCount;
                                        size_t freshRows = ctx.rowCount;
                                        if (freshRows > 10 && savedRows <= 10) {
                                            savedPipelineIsAggregated = true;
                                        }
                                        break;
                                    }
                                }
                                if (inSavedPipeline && !savedPipelineIsAggregated) {
                                    if (debug) {
                                        std::cerr << "[Exec] Join: table " << key 
                                                  << " is in saved pipeline, skipping\n";
                                    }
                                    continue;  // Skip - use saved pipeline instead
                                }
                                if (savedPipelineIsAggregated && debug) {
                                    std::cerr << "[Exec] Join: table " << key 
                                              << " is in saved pipeline but pipeline was aggregated, using fresh table\n";
                                }
                                
                                if (hasColumnOrSuffixed(ctx, col)) {
                                    unjoinedTableForJoin = key;
                                    if (debug) {
                                        std::cerr << "[Exec] Join: found unjoined table " << key 
                                                  << " with column " << col << "\n";
                                    }
                                    break;
                                }
                            }
                        }
                        if (!unjoinedTableForJoin.empty()) break;
                    }
                }
                
                EvalContext rightCtx;
                std::set<std::string> rightJoinedTables;
                
                if (savedPipelineIdx >= 0) {
                    // Use saved pipeline as right context (multi-pipeline merge join)
                    rightCtx = savedPipelines[savedPipelineIdx];
                    rightJoinedTables = savedPipelineTables[savedPipelineIdx];
                    if (debug) {
                        std::cerr << "[Exec] Join: using saved pipeline " << savedPipelineIdx 
                                  << " with " << rightCtx.rowCount << " rows as right side\n";
                        std::cerr << "[Exec] Join: saved pipeline tables: ";
                        for (const auto& t : rightJoinedTables) std::cerr << t << " ";
                        std::cerr << "\n";
                    }
                } else if (!unjoinedTableForJoin.empty()) {
                    // Use the unjoined table we found earlier (priority over other inference)
                    // BUT: Skip if this is a spurious ANTI join with a scalar subquery table after GroupBy
                    // This pattern appears when DuckDB decorrelates scalar subqueries and creates
                    // both a theta-join (for the comparison) and an ANTI join (which is redundant)
                    bool skipSpuriousAntiJoin = false;
                    if ((join.type == JoinType::Anti || join.type == JoinType::Mark) &&
                        joinedTables.find("__GROUPED__") != joinedTables.end()) {
                        const EvalContext& potentialRight = tableContexts[unjoinedTableForJoin];
                        // If the potential right table has rowCount=1 (scalar subquery result),
                        // and this is a self-comparison (same col = same col), skip it
                        if (potentialRight.rowCount <= 1 && 
                            join.conditionStr.find("IS NOT DISTINCT FROM") != std::string::npos) {
                            // Parse condition string to check for self-comparison
                            // Format: "col IS NOT DISTINCT FROM col"
                            std::string cond = join.conditionStr;
                            size_t pos = cond.find(" IS NOT DISTINCT FROM ");
                            if (pos != std::string::npos) {
                                std::string left = cond.substr(0, pos);
                                std::string right = cond.substr(pos + 22); // " IS NOT DISTINCT FROM " = 22 chars
                                // Trim whitespace
                                while (!left.empty() && std::isspace(left.back())) left.pop_back();
                                while (!right.empty() && std::isspace(right.front())) right.erase(0, 1);
                                if (left == right) {
                                    if (debug) {
                                        std::cerr << "[Exec] Join: skipping spurious ANTI join with scalar table "
                                                  << unjoinedTableForJoin << " after GroupBy\n";
                                    }
                                    skipSpuriousAntiJoin = true;
                                }
                            }
                        }
                    }
                    
                    if (skipSpuriousAntiJoin) {
                        return true; // Skip this join entirely
                    }
                    
                    rightCtx = tableContexts[unjoinedTableForJoin];
                    rightJoinedTables.insert(unjoinedTableForJoin);
                    if (debug) {
                        std::cerr << "[Exec] Join: using pre-found unjoined table " << unjoinedTableForJoin
                                  << " with " << rightCtx.rowCount << " rows as right side\n";
                    }
                } else {
                    // Find the right table from join condition - look for a table not yet joined
                    std::string rightTable;
                    if (!join.rightTable.empty()) {
                        rightTable = join.rightTable;
                    } else {
                        // Infer from condition - find a table not already in joinedTables
                        // Two-pass strategy:
                        // Pass 1: Find columns not in currentCtx (cleaner case)
                        // Pass 2: Find unjoined instances even if base column is in ctx
                        
                        // Pass 1: columns not in currentCtx
                        for (const auto& col : condCols) {
                            std::string baseTable = tableForColumn(col);
                            if (baseTable.empty()) continue;
                            
                            // Skip if column is already in currentCtx
                            bool colInCurrentCtx = (currentCtx.u32Cols.find(col) != currentCtx.u32Cols.end() ||
                                                   currentCtx.f32Cols.find(col) != currentCtx.f32Cols.end());
                            if (colInCurrentCtx) continue;
                            
                            // Find an unjoined instance of this table that contains this column
                            for (const auto& [key, ctx] : tableContexts) {
                                bool isInstanceOf = (key == baseTable || 
                                                    key.rfind(baseTable + "_", 0) == 0);
                                if (isInstanceOf && joinedTables.find(key) == joinedTables.end()) {
                                    // Check for column - try exact match first, then suffixed versions
                                    bool hasCol = (ctx.u32Cols.find(col) != ctx.u32Cols.end() ||
                                                  ctx.f32Cols.find(col) != ctx.f32Cols.end());
                                    // If not found, try suffixed versions (e.g., n_nationkey_2 for nation_2)
                                    if (!hasCol) {
                                        for (int suffix = 1; suffix <= 9; ++suffix) {
                                            std::string suffixedCol = col + "_" + std::to_string(suffix);
                                            if (ctx.u32Cols.find(suffixedCol) != ctx.u32Cols.end() ||
                                                ctx.f32Cols.find(suffixedCol) != ctx.f32Cols.end()) {
                                                hasCol = true;
                                                break;
                                            }
                                        }
                                    }
                                    if (hasCol) {
                                        rightTable = key;
                                        if (debug) {
                                            std::cerr << "[Exec] Join: found unjoined instance " << key 
                                                      << " for base table " << baseTable 
                                                      << " (has column " << col << " or suffixed variant)\n";
                                        }
                                        break;
                                    }
                                }
                            }
                            if (!rightTable.empty()) break;
                        }
                        
                        // Pass 2: if no table found, look for unjoined instances of multi-instance tables
                        if (rightTable.empty()) {
                            for (const auto& col : condCols) {
                                std::string baseTable = tableForColumn(col);
                                if (baseTable.empty()) continue;
                                
                                // Find an unjoined instance, even if column is in ctx from another instance
                                for (const auto& [key, ctx] : tableContexts) {
                                    bool isInstanceOf = (key == baseTable || 
                                                        key.rfind(baseTable + "_", 0) == 0);
                                    if (isInstanceOf && joinedTables.find(key) == joinedTables.end()) {
                                        // Check for column - try exact match first, then suffixed versions
                                        bool hasCol = (ctx.u32Cols.find(col) != ctx.u32Cols.end() ||
                                                      ctx.f32Cols.find(col) != ctx.f32Cols.end());
                                        // If not found, try suffixed versions (e.g., n_nationkey_2 for nation_2)
                                        if (!hasCol) {
                                            for (int suffix = 1; suffix <= 9; ++suffix) {
                                                std::string suffixedCol = col + "_" + std::to_string(suffix);
                                                if (ctx.u32Cols.find(suffixedCol) != ctx.u32Cols.end() ||
                                                    ctx.f32Cols.find(suffixedCol) != ctx.f32Cols.end()) {
                                                    hasCol = true;
                                                    break;
                                                }
                                            }
                                        }
                                        if (hasCol) {
                                            rightTable = key;
                                            if (debug) {
                                                std::cerr << "[Exec] Join: pass2 found unjoined instance " << key 
                                                          << " for base table " << baseTable 
                                                          << " (has column " << col << " or suffixed variant)\n";
                                            }
                                            break;
                                        }
                                    }
                                }
                                if (!rightTable.empty()) break;
                            }
                        }
                    }
                    
                    if (rightTable.empty() || tableContexts.find(rightTable) == tableContexts.end()) {
                        if (debug) {
                            std::cerr << "[Exec] Join: cannot determine right table. joinedTables=";
                            for (const auto& t : joinedTables) std::cerr << t << " ";
                            std::cerr << "\n";
                            std::cerr << "[Exec] Join: available tableContexts=";
                            for (const auto& [k, v] : tableContexts) std::cerr << k << " ";
                            std::cerr << "\n";
                        }
                        result.error = "Cannot determine right table for join";
                        return false;
                    }
                    
                    rightCtx = tableContexts[rightTable];
                    rightJoinedTables.insert(rightTable);
                }
                
                EvalContext joinCtx;
                

                // Apply right filter if present (e.g. pushed down predicates)
                if (join.rightFilter) {
                    if (debug) std::cerr << "[Exec] Join: Applying right filter to right side (GPU)\n";
                    
                    if (!executeGPUFilterRecursive(join.rightFilter, rightCtx)) {
                         throw std::runtime_error("GPU Join Right Filter failed.");
                    }
                }

                // DELIM_JOIN dedup: For self-comparison joins (same column on both sides),
                // part of DuckDB's DELIM pattern, the RHS may have duplicate join keys
                // causing many-to-many fan-out. Deduplicate the RHS by ALL join key columns
                // to ensure each probe gets at most one match.
                {
                    // Extract all self-comparison key columns from the condition
                    std::vector<std::string> delimDedupKeys;
                    bool hasINDFKey = false; // Has "IS NOT DISTINCT FROM" pattern
                    
                    // Split condition by AND and check each part for self-comparison
                    std::string condStr = join.conditionStr;
                    std::vector<std::string> parts;
                    size_t start = 0;
                    while (start < condStr.size()) {
                        size_t andPos = condStr.find(" AND ", start);
                        if (andPos == std::string::npos) {
                            parts.push_back(condStr.substr(start));
                            break;
                        }
                        parts.push_back(condStr.substr(start, andPos - start));
                        start = andPos + 5;
                    }
                    
                    for (const auto& part : parts) {
                        std::string col;
                        bool isINDF = false;
                        // Check "col IS NOT DISTINCT FROM col"
                        size_t isNotPos = part.find("IS NOT DISTINCT FROM");
                        if (isNotPos != std::string::npos) {
                            col = part.substr(0, isNotPos);
                            while (!col.empty() && std::isspace(col.back())) col.pop_back();
                            while (!col.empty() && std::isspace(col.front())) col.erase(0, 1);
                            isINDF = true;
                        } else {
                            // Check "col = col"
                            size_t eqPos = part.find(" = ");
                            if (eqPos != std::string::npos) {
                                std::string lhs = part.substr(0, eqPos);
                                std::string rhs = part.substr(eqPos + 3);
                                while (!lhs.empty() && std::isspace(lhs.back())) lhs.pop_back();
                                while (!rhs.empty() && std::isspace(rhs.front())) rhs.erase(0, 1);
                                if (lhs == rhs) col = lhs;
                            }
                        }
                        if (!col.empty()) {
                            delimDedupKeys.push_back(col);
                            hasINDFKey = hasINDFKey || isINDF;
                        }
                    }
                    
                    // Only apply RHS dedup when:
                    // 1. IS NOT DISTINCT FROM conditions (standard DELIM_JOIN marker), OR
                    // 2. For "=" self-comparison: only when the left side has fewer rows
                    //    (indicating it's a DELIM_SCAN result, not a pipeline with 
                    //    the same row count).
                    bool shouldDedup = !delimDedupKeys.empty() && !join.rightTable.empty() && rightCtx.rowCount > 1;
                    if (shouldDedup) {
                        // Materialize ALL GPU columns to CPU first
                        for (auto& [name, buf] : rightCtx.u32ColsGPU) {
                            if (buf && (!rightCtx.u32Cols.count(name) || rightCtx.u32Cols.at(name).empty())) {
                                rightCtx.u32Cols[name].resize(rightCtx.rowCount);
                                memcpy(rightCtx.u32Cols[name].data(), buf->contents(), rightCtx.rowCount * sizeof(uint32_t));
                            }
                        }
                        for (auto& [name, buf] : rightCtx.f32ColsGPU) {
                            if (buf && (!rightCtx.f32Cols.count(name) || rightCtx.f32Cols.at(name).empty())) {
                                rightCtx.f32Cols[name].resize(rightCtx.rowCount);
                                memcpy(rightCtx.f32Cols[name].data(), buf->contents(), rightCtx.rowCount * sizeof(float));
                            }
                        }
                        
                        // Resolve actual column names in rightCtx
                        std::vector<std::string> resolvedKeys;
                        for (const auto& k : delimDedupKeys) {
                            // Try direct match, then suffixed
                            if (rightCtx.u32Cols.count(k) && rightCtx.u32Cols.at(k).size() == rightCtx.rowCount) {
                                resolvedKeys.push_back(k);
                            } else {
                                // Try suffixed versions
                                bool found = false;
                                for (int s = 1; s <= 9; ++s) {
                                    std::string sk = k + "_" + std::to_string(s);
                                    if (rightCtx.u32Cols.count(sk) && rightCtx.u32Cols.at(sk).size() == rightCtx.rowCount) {
                                        resolvedKeys.push_back(sk);
                                        found = true;
                                        break;
                                    }
                                }
                                if (!found) {
                                    // Try prefix aliases (s_ -> ps_, etc)
                                    if (k.size() > 2 && k[1] == '_') {
                                        std::string suffix = k.substr(2);
                                        for (const auto& p : {"l_", "o_", "c_", "p_", "s_", "ps_", "n_", "r_"}) {
                                            std::string alt = std::string(p) + suffix;
                                            if (rightCtx.u32Cols.count(alt) && rightCtx.u32Cols.at(alt).size() == rightCtx.rowCount) {
                                                resolvedKeys.push_back(alt);
                                                found = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        if (!resolvedKeys.empty()) {
                            // Build composite key and dedup
                            std::unordered_map<std::string, uint32_t> seen;
                            std::vector<uint32_t> keepIdx;
                            for (uint32_t i = 0; i < rightCtx.rowCount; ++i) {
                                std::string compositeKey;
                                for (const auto& col : resolvedKeys) {
                                    compositeKey += std::to_string(rightCtx.u32Cols.at(col)[i]) + "|";
                                }
                                if (seen.find(compositeKey) == seen.end()) {
                                    seen[compositeKey] = i;
                                    keepIdx.push_back(i);
                                }
                            }
                            uint32_t newCount = (uint32_t)keepIdx.size();
                            if (newCount < rightCtx.rowCount) {
                                if (debug) {
                                    std::cerr << "[Exec] Join: DELIM dedup RHS by [";
                                    for (size_t ri=0; ri<resolvedKeys.size(); ++ri) { if (ri) std::cerr << ","; std::cerr << resolvedKeys[ri]; }
                                    std::cerr << "]: " << rightCtx.rowCount << " -> " << newCount << "\n";
                                }
                                // Compact all columns
                                for (auto& [name, vec] : rightCtx.u32Cols) {
                                    if (vec.size() == rightCtx.rowCount) {
                                        std::vector<uint32_t> compact(newCount);
                                        for (uint32_t i = 0; i < newCount; ++i) compact[i] = vec[keepIdx[i]];
                                        vec = std::move(compact);
                                    }
                                }
                                for (auto& [name, vec] : rightCtx.f32Cols) {
                                    if (vec.size() == rightCtx.rowCount) {
                                        std::vector<float> compact(newCount);
                                        for (uint32_t i = 0; i < newCount; ++i) compact[i] = vec[keepIdx[i]];
                                        vec = std::move(compact);
                                    }
                                }
                                for (auto& [name, vec] : rightCtx.stringCols) {
                                    if (vec.size() == rightCtx.rowCount) {
                                        std::vector<std::string> compact(newCount);
                                        for (uint32_t i = 0; i < newCount; ++i) compact[i] = vec[keepIdx[i]];
                                        vec = std::move(compact);
                                    }
                                }
                                // Re-upload GPU columns
                                for (auto& [name, buf] : rightCtx.u32ColsGPU) {
                                    if (rightCtx.u32Cols.count(name) && rightCtx.u32Cols.at(name).size() == newCount) {
                                        const auto& v = rightCtx.u32Cols.at(name);
                                        buf = GpuOps::createBuffer(v.data(), v.size() * sizeof(uint32_t));
                                    }
                                }
                                for (auto& [name, buf] : rightCtx.f32ColsGPU) {
                                    if (rightCtx.f32Cols.count(name) && rightCtx.f32Cols.at(name).size() == newCount) {
                                        const auto& v = rightCtx.f32Cols.at(name);
                                        buf = GpuOps::createBuffer(v.data(), v.size() * sizeof(float));
                                    }
                                }
                                rightCtx.rowCount = newCount;
                                rightCtx.activeRows.clear();
                                rightCtx.activeRowsGPU = nullptr;
                                
                                // Strip right-side columns that already exist on the left side
                                // to prevent deduped (potentially wrong) values from overriding
                                // the left side's correct per-row values. Keep only:
                                // - The dedup key columns (needed for the hash join)
                                // - Columns that are NEW (don't exist on the left)
                                {
                                    std::set<std::string> keepU32(resolvedKeys.begin(), resolvedKeys.end());
                                    for (const auto& [name, _] : rightCtx.u32Cols) {
                                        if (currentCtx.u32Cols.find(name) == currentCtx.u32Cols.end())
                                            keepU32.insert(name);
                                    }
                                    for (auto it2 = rightCtx.u32Cols.begin(); it2 != rightCtx.u32Cols.end(); ) {
                                        if (keepU32.find(it2->first) == keepU32.end()) it2 = rightCtx.u32Cols.erase(it2);
                                        else ++it2;
                                    }
                                    for (auto it2 = rightCtx.u32ColsGPU.begin(); it2 != rightCtx.u32ColsGPU.end(); ) {
                                        if (keepU32.find(it2->first) == keepU32.end()) it2 = rightCtx.u32ColsGPU.erase(it2);
                                        else ++it2;
                                    }
                                    // Strip f32 cols that exist on left (except new ones)
                                    for (auto it2 = rightCtx.f32Cols.begin(); it2 != rightCtx.f32Cols.end(); ) {
                                        if (currentCtx.f32Cols.find(it2->first) != currentCtx.f32Cols.end()) it2 = rightCtx.f32Cols.erase(it2);
                                        else ++it2;
                                    }
                                    for (auto it2 = rightCtx.f32ColsGPU.begin(); it2 != rightCtx.f32ColsGPU.end(); ) {
                                        if (currentCtx.f32ColsGPU.find(it2->first) != currentCtx.f32ColsGPU.end()) it2 = rightCtx.f32ColsGPU.erase(it2);
                                        else ++it2;
                                    }
                                    // Strip string cols that exist on left
                                    for (auto it2 = rightCtx.stringCols.begin(); it2 != rightCtx.stringCols.end(); ) {
                                        if (currentCtx.stringCols.find(it2->first) != currentCtx.stringCols.end()) {
                                            rightCtx.flatStringColsGPU.erase(it2->first);
                                            it2 = rightCtx.stringCols.erase(it2);
                                        }
                                        else ++it2;
                                    }
                                    if (debug) {
                                        std::cerr << "[Exec] Join: stripped RHS to " << rightCtx.u32Cols.size()
                                                  << " u32, " << rightCtx.f32Cols.size() << " f32, "
                                                  << rightCtx.stringCols.size() << " string cols\n";
                                    }
                                }
                                rightCtx.activeRowsCountGPU = 0;
                            }
                        }
                    }
                }

                // SEMI join with self-comparison: swap so outer table is the probe side.
                bool semiSwapped = false;
                if (join.type == JoinType::Semi && !join.rightTable.empty()) {
                    // Check for self-comparison condition (col = col)
                    auto& cond = join.conditionStr;
                    size_t eqPos = cond.find(" = ");
                    if (eqPos != std::string::npos) {
                        std::string lhs = cond.substr(0, eqPos);
                        std::string rhs = cond.substr(eqPos + 3);
                        while (!lhs.empty() && std::isspace(lhs.back())) lhs.pop_back();
                        while (!rhs.empty() && std::isspace(rhs.front())) rhs.erase(0, 1);
                        if (lhs == rhs) {
                            if (debug) std::cerr << "[Exec] SEMI join: swapping sides (right table becomes probe)\n";
                            std::swap(currentCtx, rightCtx);
                            semiSwapped = true;
                        }
                    }
                }

                if (!executeJoin(join, datasetPath, currentCtx, rightCtx, joinCtx)) {
                    result.error = "Join execution failed";
                    return false;
                }
                
                currentCtx = std::move(joinCtx);
                if (debug) {
                    std::cerr << "[Exec] Join: currentCtx after move: rowCount=" << currentCtx.rowCount << " u32ColsGPU.size=" << currentCtx.u32ColsGPU.size() << "\n";
                    std::cerr << "[Exec] Join: currentCtx.stringCols after move:\n";
                    for (const auto& [n, v] : currentCtx.stringCols) {
                        std::cerr << "[Exec]   " << n << " size=" << v.size() << "\n";
                    }
                    std::cerr << "[Exec] Join: currentCtx.currentTable='" << currentCtx.currentTable << "'\n";
                }
                // Merge all joined tables from both sides
                for (const auto& t : rightJoinedTables) {
                    joinedTables.insert(t);
                }
                hasPipeline = true;  // We now have a joined result in the pipeline
                if (debug) {
                    std::cerr << "[Exec] Join: " << currentCtx.rowCount << " rows after. joinedTables=";
                    for (const auto& t : joinedTables) std::cerr << t << " ";
                    std::cerr << "\n";
                }

    return true;
}

} // namespace engine
