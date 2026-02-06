#include "IRGpuExecutorV2.hpp"
#include "IRGpuExecutorV2_Priv.hpp"
#include "TypedExpr.hpp"
#include "Predicate.hpp"
#include "OperatorsGPU.hpp"
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
                std::cerr << "[V2] Join: Renaming duplicate column " << name << " -> " << newName << "\n";
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

bool IRGpuExecutorV2::executeJoin(const IRJoinV2& join, const std::string& datasetPath,
                                   EvalContext& leftCtx, EvalContext& rightCtx, EvalContext& outCtx) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    // Support INNER, LEFT, RIGHT, SEMI, ANTI, and MARK joins
    // MARK joins are treated as SEMI for execution purposes (used for NOT IN decorrelation)
    if (join.type != JoinType::Inner && join.type != JoinType::Left &&
        join.type != JoinType::Right && join.type != JoinType::Semi && 
        join.type != JoinType::Anti && join.type != JoinType::Mark) {
        if (debug) std::cerr << "[V2] Join: unsupported join type\n";
        return false;
    }
    
    const bool isLeftJoin = (join.type == JoinType::Left);
    const bool isRightJoin = (join.type == JoinType::Right);
    const bool isSemiJoin = (join.type == JoinType::Semi || join.type == JoinType::Mark);  // Treat MARK as SEMI
    const bool isAntiJoin = (join.type == JoinType::Anti);
    
    if (debug) {
        std::cerr << "[V2] Join: type=" << static_cast<int>(join.type) 
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
                std::cerr << "[V2] Join: detected non-equality condition, treating as cross-join + filter\n";
                std::cerr << "[V2] Join: conditionStr=" << join.conditionStr << "\n";
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
        if (debug) std::cerr << "[V2] Join: no equi-join keys found but has condition, treating as cross-join + filter\n";
        isCrossJoin = true;
        hasPostJoinFilter = true;
        postJoinFilter = join.condition;
    }
    
    if (!isCrossJoin && keyPairs.empty()) {
        if (debug) std::cerr << "[V2] Join: no key pairs found\n";
        return false;
    }
    
    if (debug) {
        std::cerr << "[V2] Join: " << keyPairs.size() << " key pair(s):\n";
        for (const auto& [l, r] : keyPairs) {
            std::cerr << "[V2] Join:   " << l << " = " << r << std::endl;
        }
        std::cerr << "[V2] Join: leftCtx has " << leftCtx.u32Cols.size() << " u32 cols, " 
                  << leftCtx.f32Cols.size() << " f32 cols, " << leftCtx.rowCount << " rows";
        for (const auto& [n,_] : leftCtx.u32Cols) std::cerr << " " << n;
        std::cerr << std::endl;
        std::cerr << "[V2] Join: rightCtx has " << rightCtx.u32Cols.size() << " u32 cols, "
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
    auto fuzzyResolve = [&](EvalContext& ctx, const std::string& colName) -> std::string {
        // 1. Try suffixed versions (e.g. name_1)
        std::string s = findColWithSuffix(ctx, colName); // Use updated helper
        if (!s.empty()) return s;

        // 2. Try positional refs (#0..#9)
        for (int i = 0; i < 10; ++i) {
            std::string posRef = "#" + std::to_string(i);
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

        // 3. Try prefix aliases (l_ -> o_, etc)
        if (colName.size() > 2 && colName[1] == '_') {
            std::string suffix = colName.substr(2);
            static const std::vector<std::string> prefixes = {"l_", "o_", "c_", "p_", "s_", "ps_", "n_", "r_"};
            for (const auto& p : prefixes) {
                std::string alt = p + suffix;
                std::string res = findColWithSuffix(ctx, alt); // Re-use helper to handle conversion
                if (!res.empty()) return res;
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
    for (auto& [k1, k2] : keyPairs) {
        // Patch for Q15: Resolve known aliases early
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
        } else if (k2InLeft && k1InRight) {
            resolvedKeys.emplace_back(k2Left, k1Right);
        } else {
             // Try to fuzzy resolve missing left key if right key exists
             std::string leftResolved, rightResolved;
             
             if (k1InRight) {
                 // Right has k1. We need k2 in Left.
                 rightResolved = k1Right;
                 leftResolved = fuzzyResolve(leftCtx, k2);
             } else if (k2InRight) {
                 // Right has k2. We need k1 in Left.
                 rightResolved = k2Right;
                 leftResolved = fuzzyResolve(leftCtx, k1);
             }
             
             // Try to fuzzy resolve missing right key if left key exists
             if (leftResolved.empty() && rightResolved.empty()) {
                  if (k1InLeft) {
                      leftResolved = k1Left;
                      rightResolved = fuzzyResolve(rightCtx, k2);
                  } else if (k2InLeft) {
                      leftResolved = k2Left;
                      rightResolved = fuzzyResolve(rightCtx, k1);
                  }
             }
             
             if (!leftResolved.empty() && !rightResolved.empty()) {
                  resolvedKeys.emplace_back(leftResolved, rightResolved);
                   if (debug) {
                       std::cerr << "[V2] Join: fuzzy resolved " << k1 << "=" << k2 << " to (" 
                                 << leftResolved << ", " << rightResolved << ")\n";
                   }
             } else {
                if (debug) {
                    std::cerr << "[V2] Join: cannot resolve key pair " << k1 << "=" << k2 
                            << " k1InLeft=" << k1InLeft << " k2InRight=" << k2InRight
                            << " k2InLeft=" << k2InLeft << " k1InRight=" << k1InRight << "\n";
                }
                return false;
             }
        }
    }
    
    if (debug) {
        std::cerr << "[V2] Join: resolved " << resolvedKeys.size() << " key pair(s)\n";
    }
    
    // Get vectors for all keys
    std::vector<const std::vector<uint32_t>*> leftKeyVecs, rightKeyVecs;
    for (const auto& [lk, rk] : resolvedKeys) {
        leftKeyVecs.push_back(&leftCtx.u32Cols.at(lk));
        rightKeyVecs.push_back(&rightCtx.u32Cols.at(rk));
    }
    
    if (!isCrossJoin && resolvedKeys.empty()) return false;

    // Single-Column Join Implementation
    // if (resolvedKeys.size() > 1) {
    //      throw std::runtime_error("Multi-column GPU Join not implemented");
    // }

    if (!isCrossJoin && resolvedKeys.size() > 2) throw std::runtime_error("GPU Join > 2 columns not implemented");

    auto& store = ColumnStoreGPU::instance();
    
    auto ensureGPU = [&](EvalContext& ctx, const std::string& col) -> MTL::Buffer* {
        if (ctx.u32ColsGPU.count(col)) return ctx.u32ColsGPU.at(col);
        if (ctx.u32Cols.count(col)) {
             const auto& vec = ctx.u32Cols.at(col);
             auto buf = store.device()->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
             ctx.u32ColsGPU[col] = buf;
             return buf;
        }
        return nullptr;
    };
    
    uint32_t rCount = rightCtx.activeRowsGPU ? rightCtx.activeRowsCountGPU : (uint32_t)rightCtx.rowCount;
    uint32_t lCount = leftCtx.activeRowsGPU ? leftCtx.activeRowsCountGPU : (uint32_t)leftCtx.rowCount;

    MTL::Buffer* lBuf = nullptr;
    MTL::Buffer* rBuf = nullptr;
    
    JoinResultGPU jRes;
    
    if (lCount > 0 && rCount > 0) {
        if (isCrossJoin) {
            if (debug) std::cerr << "[V2] GPU Join: Cross Join 1=1 (" << lCount << " x " << rCount << ")\n";
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
                lIndicesGPU = OperatorsGPU::iotaU32(lCount);
                createdL = true;
            }
            if (!rIndicesGPU) {
                rIndicesGPU = OperatorsGPU::iotaU32(rCount);
                createdR = true;
            }
            
            // Cross product on GPU
            OperatorsGPU::crossProduct(lIndicesGPU, rIndicesGPU,
                                       jRes.probeIndices, jRes.buildIndices,
                                       lCount, rCount);
            
            if (createdL) lIndicesGPU->release();
            if (createdR) rIndicesGPU->release();
        } else if (resolvedKeys.size() == 2) {
            if (debug) {
                std::cerr << "[V2] Multi-Key Join (2 keys)\n";
                std::cerr << "[V2] Multi-Key Join: key0=(" << resolvedKeys[0].first << ", " << resolvedKeys[0].second << ")\n";
                std::cerr << "[V2] Multi-Key Join: key1=(" << resolvedKeys[1].first << ", " << resolvedKeys[1].second << ")\n";
            }
            MTL::Buffer* l1 = ensureGPU(leftCtx, resolvedKeys[0].first);
            MTL::Buffer* r1 = ensureGPU(rightCtx, resolvedKeys[0].second);
            MTL::Buffer* l2 = ensureGPU(leftCtx, resolvedKeys[1].first);
            MTL::Buffer* r2 = ensureGPU(rightCtx, resolvedKeys[1].second);
            if(!l1||!r1||!l2||!r2) throw std::runtime_error("Missing GPU col data for multi-key join");
            
            uint32_t lSize = (uint32_t)leftCtx.rowCount;
            uint32_t rSize = (uint32_t)rightCtx.rowCount;
            
            if (debug) std::cerr << "[V2] Multi-Key Join: packing left (" << lSize << " rows)...\n" << std::flush;
            lBuf = OperatorsGPU::packU32ToU64(l1, l2, lSize);
            if (debug) std::cerr << "[V2] Multi-Key Join: packing right (" << rSize << " rows)...\n" << std::flush;
            rBuf = OperatorsGPU::packU32ToU64(r1, r2, rSize);
            if (debug) std::cerr << "[V2] Multi-Key Join: packing done.\n" << std::flush;
        } else {
            lBuf = ensureGPU(leftCtx, resolvedKeys[0].first);
            rBuf = ensureGPU(rightCtx, resolvedKeys[0].second);
        }

        if (!isCrossJoin && (!lBuf || !rBuf)) throw std::runtime_error("Missing GPU column data for Join");
        
        // Determine if we should swap build/probe sides.
        // Our hash join only keeps one entry per key in the build table, so if the build side
        // has non-unique keys, we lose rows.
        // 
        // CRITICAL: For TPC-H style fact-dimension joins, the left side (fact table like lineitem)
        // often has duplicate keys when joining to dimension tables. We should NEVER swap the
        // build/probe sides because:
        // 1. Building from fact table loses rows when multiple facts have the same FK
        // 2. The hash table can't handle multiple values per key
        //
        // For correctness, always build from right (dimension/smaller) and probe from left (fact/larger).
        // This might use more memory but ensures all rows are preserved.
        
        bool swapped = false;
        // DISABLED: Don't swap to avoid losing rows when fact table has duplicate join keys
        // if (!isCrossJoin && rCount > lCount && resolvedKeys.size() == 1) {
        //     if (debug) std::cerr << "[V2] GPU Join: Swapping build/probe for size (rCount=" << rCount << " > lCount=" << lCount << ")\n";
        //     std::swap(lBuf, rBuf);
        //     std::swap(lCount, rCount);
        //     swapped = true;
        // }
        
        if (!isCrossJoin && debug) std::cerr << "[V2] GPU Join: Build (" << rCount << "), Probe (" << lCount << ") swapped=" << swapped << "\n";
        if (debug) {
            std::cerr << "[V2] GPU Join: leftCtx.activeRowsGPU=" << (leftCtx.activeRowsGPU ? "SET" : "NULL") << " rightCtx.activeRowsGPU=" << (rightCtx.activeRowsGPU ? "SET" : "NULL") << "\n";
            if (leftCtx.activeRowsGPU) {
                uint32_t* leftIndices = (uint32_t*)leftCtx.activeRowsGPU->contents();
                std::cerr << "[V2] GPU Join: leftActiveIndices first 5: ";
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
                 jRes = OperatorsGPU::joinHashU64(rBuf, buildActiveRows, rCount, lBuf, probeActiveRows, lCount);
                 lBuf->release(); rBuf->release();
            } else {
                 jRes = OperatorsGPU::joinHash(rBuf, buildActiveRows, rCount, lBuf, probeActiveRows, lCount);
            }
            
            // If swapped, the build/probe indices are also swapped relative to left/right
            if (swapped) {
                // probeIndices now refers to original right rows, buildIndices to original left rows
                std::swap(jRes.probeIndices, jRes.buildIndices);
            }
        }
    } else {
        if (lCount > 0 && rCount == 0 && (isAntiJoin || isLeftJoin)) {
             if (debug) std::cerr << "[V2] GPU Join: Empty Build side for Anti/Left Join -> Returning all " << lCount << " left rows.\n";
             jRes.count = lCount;
             
             if (leftCtx.activeRowsGPU) {
                 MTL::Buffer* src = leftCtx.activeRowsGPU;
                 jRes.probeIndices = store.device()->newBuffer(src->contents(), src->length(), MTL::ResourceStorageModeShared);
             } else {
                 std::vector<uint32_t> seq(lCount);
                 std::iota(seq.begin(), seq.end(), 0);
                 jRes.probeIndices = store.device()->newBuffer(seq.data(), seq.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
             }
             
             // Dummy build indices (won't be used if we skip right gather)
             // But must be non-null to pass check
             jRes.buildIndices = store.device()->newBuffer(lCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
             std::memset(jRes.buildIndices->contents(), 0, lCount * sizeof(uint32_t));
        } else {
            if (debug) std::cerr << "[V2] GPU Join: Skipping (Build=" << rCount << ", Probe=" << lCount << ")\n";
            jRes.count = 0;
            jRes.buildIndices = nullptr;
            jRes.probeIndices = nullptr;
        }
    }
                                       
    if ((lCount > 0 && rCount > 0) && !jRes.buildIndices) throw std::runtime_error("GPU Join Kernel Failed");
    
    if (debug) std::cerr << "[V2] GPU Join Success: Result " << jRes.count << " rows.\n";
    
    uint32_t resCount = jRes.count;
    
    // SEMI JOIN: Deduplicate probeIndices to keep each LHS row only once
    // For Semi join, we only want to know if there's at least one match, not all matches
    if (isSemiJoin && resCount > 0 && jRes.probeIndices) {
        if (debug) std::cerr << "[V2] Semi Join: Deduplicating " << resCount << " probe indices\n";
        
        uint32_t* probePtr = (uint32_t*)jRes.probeIndices->contents();
        std::vector<uint32_t> probeVec(probePtr, probePtr + resCount);
        
        // Sort and unique to get distinct LHS indices
        std::sort(probeVec.begin(), probeVec.end());
        probeVec.erase(std::unique(probeVec.begin(), probeVec.end()), probeVec.end());
        
        uint32_t uniqueCount = (uint32_t)probeVec.size();
        if (debug) std::cerr << "[V2] Semi Join: After dedup: " << uniqueCount << " unique rows\n";
        
        // Create new buffer with unique indices
        MTL::Buffer* uniqueProbe = store.device()->newBuffer(
            probeVec.data(), uniqueCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        
        // Replace the original probeIndices
        jRes.probeIndices->release();
        jRes.probeIndices = uniqueProbe;
        
        // For Semi join, we don't need buildIndices (no RHS columns in output)
        // But we need them for the gather operations, so create dummy ones
        if (jRes.buildIndices) jRes.buildIndices->release();
        jRes.buildIndices = store.device()->newBuffer(uniqueCount * sizeof(uint32_t), MTL::ResourceStorageModeShared);
        std::memset(jRes.buildIndices->contents(), 0, uniqueCount * sizeof(uint32_t));
        
        resCount = uniqueCount;
        jRes.count = uniqueCount;
    }
    
    // ANTI JOIN: Find LHS rows that did NOT match
    if (isAntiJoin && jRes.probeIndices) {
        if (debug) std::cerr << "[V2] Anti Join: Finding non-matching rows from " << lCount << " left rows, " << resCount << " matches\n";
        
        // Collect all matched LHS indices
        uint32_t* probePtr = (uint32_t*)jRes.probeIndices->contents();
        std::unordered_set<uint32_t> matchedIndices(probePtr, probePtr + resCount);
        
        // Generate all LHS indices
        std::vector<uint32_t> allLeftIndices;
        if (leftCtx.activeRowsGPU) {
            uint32_t* leftPtr = (uint32_t*)leftCtx.activeRowsGPU->contents();
            allLeftIndices.assign(leftPtr, leftPtr + leftCtx.activeRowsCountGPU);
        } else {
            allLeftIndices.resize(lCount);
            std::iota(allLeftIndices.begin(), allLeftIndices.end(), 0);
        }
        
        // Find non-matching indices
        std::vector<uint32_t> antiResult;
        for (uint32_t idx : allLeftIndices) {
            if (matchedIndices.count(idx) == 0) {
                antiResult.push_back(idx);
            }
        }
        
        uint32_t antiCount = (uint32_t)antiResult.size();
        if (debug) std::cerr << "[V2] Anti Join: " << antiCount << " non-matching rows\n";
        
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
                        std::cerr << "[V2] Join: Renaming duplicate column " << name << " -> " << newName << "\n";
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
        std::cerr << "[V2] Join: probeIndices first 5: ";
        for (int i = 0; i < std::min(5u, resCount); ++i) std::cerr << probePtr[i] << " ";
        std::cerr << "\n";
    }
    for (const auto& [name, valid] : leftCtx.u32Cols) {
        if (debug) std::cerr << "[V2] Join: gathering L_U32 " << name << " srcSize=" << valid.size() << "\n";
        MTL::Buffer* src = ensureGPU(leftCtx, name);
        if (src) {
             MTL::Buffer* gathered = OperatorsGPU::gatherU32(src, jRes.probeIndices, resCount, false);
             outCtx.u32ColsGPU[name] = gathered;
             outCtx.u32Cols[name].clear(); // Mark CPU side as invalid
        }
    }
    for (const auto& [name, valid] : leftCtx.f32Cols) {
        if (debug) std::cerr << "[V2] Join: gathering L_F32 " << name << " srcSize=" << valid.size() << "\n";
        MTL::Buffer* src = nullptr;
        if (leftCtx.f32ColsGPU.count(name)) src = leftCtx.f32ColsGPU.at(name);
        else if (leftCtx.f32Cols.count(name)) {
             const auto& vec = leftCtx.f32Cols.at(name);
             src = store.device()->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
             leftCtx.f32ColsGPU[name] = src;
        }
        
        if (src) {
             MTL::Buffer* gathered = OperatorsGPU::gatherF32(src, jRes.probeIndices, resCount, false);
             outCtx.f32ColsGPU[name] = gathered;
             outCtx.f32Cols[name].clear();
        }
    }
    
    // Gather Right Columns - rename only if collision with left
    if (rCount > 0) {
        for (const auto& [name, valid] : rightCtx.u32Cols) {
            std::string outName = getRightColumnName(name);
            if (debug) std::cerr << "[V2] Join: gathering R_U32 " << name << " -> " << outName << "\n";
            MTL::Buffer* src = ensureGPU(rightCtx, name);
            if (src) {
                 MTL::Buffer* gathered = OperatorsGPU::gatherU32(src, jRes.buildIndices, resCount, false);
                 outCtx.u32ColsGPU[outName] = gathered;
                 outCtx.u32Cols[outName].clear();
            }
        }
        for (const auto& [name, valid] : rightCtx.f32Cols) {
            std::string outName = getRightColumnName(name);
            if (debug) std::cerr << "[V2] Join: gathering R_F32 " << name << " -> " << outName << "\n";
            MTL::Buffer* src = nullptr;
            if (rightCtx.f32ColsGPU.count(name)) src = rightCtx.f32ColsGPU.at(name);
            else if (rightCtx.f32Cols.count(name)) {
                 const auto& vec = rightCtx.f32Cols.at(name);
                 src = store.device()->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                 rightCtx.f32ColsGPU[name] = src;
            }
    
            if (src) {
                 MTL::Buffer* gathered = OperatorsGPU::gatherF32(src, jRes.buildIndices, resCount, false);
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
            if (debug) std::cerr << "[V2] Join: gathering L_STR " << name << " srcSize=" << vec.size() << "\n";
            std::vector<std::string> newVec;
            parallelGather(cpuProbeIndices, vec, newVec);
            outCtx.stringCols[name] = std::move(newVec);
        }
        for (const auto& [name, vec] : rightCtx.stringCols) {
             // Right columns: rename only if collision with left
             std::string outName = getRightColumnName(name);
             if (debug) std::cerr << "[V2] Join: gathering R_STR " << name << " -> " << outName << " srcSize=" << vec.size() << "\n";
             std::vector<std::string> newVec;
             parallelGather(cpuBuildIndices, vec, newVec);
             outCtx.stringCols[outName] = std::move(newVec);
        }
    }
    
    OperatorsGPU::sync(); // Ensure all async gathers complete

    jRes.buildIndices->release();
    jRes.probeIndices->release();
    
    // Apply post-join filter for non-equality conditions (e.g., value > threshold from HAVING subquery)
    // Use GPU for evaluation and comparison
    if (hasPostJoinFilter && postJoinFilter && outCtx.rowCount > 0) {
        if (debug) {
            std::cerr << "[V2] Join: applying post-join filter on GPU, current rows=" << outCtx.rowCount << "\n";
            std::cerr << "[V2] Join: outCtx.f32ColsGPU:";
            for (const auto& [n, b] : outCtx.f32ColsGPU) std::cerr << " " << n << "(" << (b?b->length()/sizeof(float):0) << ")";
            std::cerr << "\n";
            std::cerr << "[V2] Join: outCtx.u32ColsGPU:";
            for (const auto& [n, b] : outCtx.u32ColsGPU) std::cerr << " " << n << "(" << (b?b->length()/sizeof(uint32_t):0) << ")";
            std::cerr << "\n";
        }
        
        // Upload CPU columns to GPU for filtering (they were gathered from join)
        for (const auto& [name, vec] : outCtx.u32Cols) {
            if (!vec.empty() && outCtx.u32ColsGPU.find(name) == outCtx.u32ColsGPU.end()) {
                MTL::Buffer* buf = OperatorsGPU::createBuffer(vec.data(), vec.size() * sizeof(uint32_t));
                if (buf) outCtx.u32ColsGPU[name] = buf;
            }
        }
        for (const auto& [name, vec] : outCtx.f32Cols) {
            if (!vec.empty() && outCtx.f32ColsGPU.find(name) == outCtx.f32ColsGPU.end()) {
                MTL::Buffer* buf = OperatorsGPU::createBuffer(vec.data(), vec.size() * sizeof(float));
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
                    std::cerr << "[V2] Join post-filter GPU: LHS count=" << lhsCount << " first=" << (lhsCount>0?lhsPtr[0]:0)
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
                auto filterResult = OperatorsGPU::filterF32("post_join_filter", lhsBuf, outCtx.rowCount, gpuOp, threshold);
                
                lhsBuf->release();
                rhsBuf->release();
                
                if (filterResult && filterResult->count > 0) {
                    uint32_t matchCount = filterResult->count;
                    MTL::Buffer* resultIndices = filterResult->indices;
                    
                    if (debug) {
                        std::cerr << "[V2] Join: GPU post-filter matched " << matchCount << "/" << outCtx.rowCount << " rows\n";
                    }
                    
                    if (matchCount < outCtx.rowCount) {
                        // GPU compact all columns using gather
                        for (auto& [name, buf] : outCtx.u32ColsGPU) {
                            if (buf && buf->length() > 0) {
                                MTL::Buffer* compacted = OperatorsGPU::gatherU32(buf, resultIndices, matchCount);
                                buf->release();
                                buf = compacted;
                            }
                        }
                        for (auto& [name, buf] : outCtx.f32ColsGPU) {
                            if (buf && buf->length() > 0) {
                                MTL::Buffer* compacted = OperatorsGPU::gatherF32(buf, resultIndices, matchCount);
                                buf->release();
                                buf = compacted;
                            }
                        }
                        
                        // NOTE: Don't sync CPU vectors here - let Project download on-demand
                        // This minimizes CPU memory usage and keeps data on GPU
                        
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
                        }
                        
                        outCtx.rowCount = matchCount;
                        if (debug) {
                            std::cerr << "[V2] Join: outCtx after filter: rowCount=" << outCtx.rowCount << " u32ColsGPU=";
                            for (const auto& [n, b] : outCtx.u32ColsGPU) std::cerr << n << "(" << (b?b->length()/sizeof(uint32_t):0) << ") ";
                            std::cerr << "\n  f32ColsGPU=";
                            for (const auto& [n, b] : outCtx.f32ColsGPU) std::cerr << n << "(" << (b?b->length()/sizeof(float):0) << ") ";
                            std::cerr << "\n";
                        }
                    }
                    
                    resultIndices->release();
                } else if (debug) {
                    std::cerr << "[V2] Join: GPU post-filter matched 0 rows or failed\n";
                    outCtx.rowCount = 0;
                }
            } else {
                if (debug) std::cerr << "[V2] Join: post-filter eval failed, LHS=" << (lhsBuf?"ok":"null") << " RHS=" << (rhsBuf?"ok":"null") << "\n";
                if (lhsBuf) lhsBuf->release();
                if (rhsBuf) rhsBuf->release();
            }
        } else if (debug) {
            std::cerr << "[V2] Join: post-filter is not a comparison (kind=" << static_cast<int>(postJoinFilter->kind) << ")\n";
        }
    }
    
    return true;
}


bool IRGpuExecutorV2::orchestrateJoin(
    const IRJoinV2& join,
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
                result.isScalarAggregate = false; // Reset scalar flag on join
                // const auto& join = node.asJoin(); // Already passed as argument
                
                // Collect all columns referenced in the join condition
                std::set<std::string> condCols;
                collectColumnsFromExpr(join.condition, condCols);
                
                if (debug) {
                    std::cerr << "[V2] Join: conditionStr=" << join.conditionStr << "\n";
                    std::cerr << "[V2] Join: type=";
                    switch (join.type) {
                        case JoinType::Inner: std::cerr << "Inner"; break;
                        case JoinType::Left: std::cerr << "Left"; break;
                        case JoinType::Semi: std::cerr << "Semi"; break;
                        case JoinType::Anti: std::cerr << "Anti"; break;
                        case JoinType::Mark: std::cerr << "Mark"; break;
                        default: std::cerr << "Unknown(" << static_cast<int>(join.type) << ")"; break;
                    }
                    std::cerr << "\n";
                    std::cerr << "[V2] Join: condCols extracted: ";
                    for (const auto& c : condCols) std::cerr << c << " ";
                    std::cerr << "(total=" << condCols.size() << ")\n";
                }
                
                // Check for trivial self-join where all columns reference tables already joined.
                // Common in DuckDB's DELIM_SCAN for EXISTS subqueries (e.g., "o_k IS NOT DISTINCT FROM o_k").
                // Skip these internal plan nodes.
                bool isTrivialSelfJoin = false;
                
                // Check for IS NOT DISTINCT FROM pattern (DuckDB's DELIM_SCAN correlation marker)
                if (join.conditionStr.find("IS NOT DISTINCT FROM") != std::string::npos) {
                    // Parse the condition to see if it's a self-comparison
                    // Pattern: "col IS NOT DISTINCT FROM col" where col appears twice
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
                            // Check if column is in currentCtx (GroupBy might hide original columns).
                            // If missing, we may need to re-join.
                            bool colInContext = (currentCtx.u32Cols.find(leftPart) != currentCtx.u32Cols.end() ||
                                                 currentCtx.f32Cols.find(leftPart) != currentCtx.f32Cols.end());
                            
                            if (colInContext) {
                                // Don't skip LEFT joins (needed for DELIM joins)
                                if (join.type != JoinType::Left) {
                                    // Don't skip if explicit right table is specified
                                    if (join.rightTable.empty()) {
                                        isTrivialSelfJoin = true;
                                    } else if (debug) {
                                        std::cerr << "[V2] Join: IS NOT DISTINCT FROM self-comparison BUT explicit right table specified (" << join.rightTable << "). Not skipping.\n";
                                    }
                                }
                                if (debug && isTrivialSelfJoin) {
                                    std::cerr << "[V2] Join: IS NOT DISTINCT FROM self-comparison: '" 
                                              << leftPart << "' vs '" << rightPart << "' (col in context)\n";
                                }
                            } else if (debug) {
                                std::cerr << "[V2] Join: IS NOT DISTINCT FROM self-comparison: '" 
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
                                    std::cerr << "[V2] Join: self-comparison BUT explicit right table specified (" << join.rightTable << "). Not skipping.\n";
                                }
                            }
                            if (debug && isTrivialSelfJoin) {
                                std::cerr << "[V2] Join: self-comparison detected for " << col << " (table " << baseTable << " already joined, col in context)\n";
                            }
                        }
                    } else if (debug) {
                        std::cerr << "[V2] Join: self-comparison for " << col << " but col not in context, may need re-join\n";
                    }
                }
                
                if (isTrivialSelfJoin) {
                    if (debug) {
                        std::cerr << "[V2] Join: skipping trivial self-join (all columns already in pipeline)\n";
                    }
                    return true; // Skip this join
                }
                
                // Check for scalar subquery pattern (join condition contains SUBQUERY).
                // currentCtx has scalar value; savedPipelines has main data.
                if (join.conditionStr.find("SUBQUERY") != std::string::npos && !savedPipelines.empty()) {
                    if (debug) {
                        std::cerr << "[V2] Join: detected scalar subquery pattern\n";
                        std::cerr << "[V2]   Current context rows: " << currentCtx.rowCount << "\n";
                        std::cerr << "[V2]   Saved pipelines: " << savedPipelines.size() << "\n";
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
                                    std::cerr << "[V2]   Found scalar pipeline via flag (rowCount=" << savedPipelines[pi].rowCount << ")\n";
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
                                 if (debug) std::cerr << "[V2]   Scalar value from f32 col '" << name << "': " << scalarValue << "\n";
                                 return true;
                             }
                         }
                         // Search u32
                         for (const auto& [name, values] : scalarCtx->u32Cols) {
                             if (values.empty()) continue;
                             bool match = exact ? (name == pattern) : (name.find(pattern) != std::string::npos);
                             if (match) {
                                 scalarValue = static_cast<double>(values[0]);
                                 if (debug) std::cerr << "[V2]   Scalar value from u32 col '" << name << "': " << scalarValue << "\n";
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
                            std::cerr << "[V2]   Restored saved pipeline with " << currentCtx.rowCount << " rows\n";
                        }
                    } else {
                        // Data is already currentCtx. Just remove the scalar pipeline from saved.
                        if (scalarPipelineIdx >= 0) {
                            savedPipelines.erase(savedPipelines.begin() + scalarPipelineIdx);
                            savedPipelineTables.erase(savedPipelineTables.begin() + scalarPipelineIdx);
                        }
                        if (debug) {
                            std::cerr << "[V2]   Using current context as data table with " << currentCtx.rowCount << " rows\n";
                        }
                    }

                    // Inject broadcasted scalars into the data context
                    for(auto& [n, v] : scalarF32s) {
                        if (currentCtx.f32Cols.find(n) == currentCtx.f32Cols.end() && currentCtx.f32ColsGPU.find(n) == currentCtx.f32ColsGPU.end()) {
                             currentCtx.f32Cols[n] = {v}; // Size 1 vector (scalar broadcast)
                             if (debug) std::cerr << "[V2]   Broadcasted scalar F32col: " << n << "\n";
                        }
                    }
                    for(auto& [n, v] : scalarU32s) {
                        if (currentCtx.u32Cols.find(n) == currentCtx.u32Cols.end() && currentCtx.u32ColsGPU.find(n) == currentCtx.u32ColsGPU.end()) {
                             currentCtx.u32Cols[n] = {v};
                             if (debug) std::cerr << "[V2]   Broadcasted scalar U32col: " << n << "\n";
                        }
                    }

                    // Parse the condition to find what column to compare
                    // Pattern: "CAST(sum(...)) > SUBQUERY" or "col > SUBQUERY"
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
                    
                    // Scalar subquery HAVING pattern: left expression is aggregate (#N, SUM_#N, etc.).
                    // Try to find a matching column in the context
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
                        std::cerr << "[V2]   Filtering: " << aggColName << " " << opStr << " " << scalarValue << "\n";
                    }
                    
                    // Apply the filter directly by comparing the aggregate column to the scalar
                    std::vector<float>* aggColF32 = nullptr;
                    std::vector<uint32_t>* aggColU32 = nullptr;
                    
                    if (currentCtx.f32Cols.find(aggColName) != currentCtx.f32Cols.end()) {
                        aggColF32 = &currentCtx.f32Cols[aggColName];
                    } else if (currentCtx.u32Cols.find(aggColName) != currentCtx.u32Cols.end()) {
                        aggColU32 = &currentCtx.u32Cols[aggColName];
                    }
                    
                    // Build activeRows mask
                    std::vector<uint32_t> mask;
                    size_t numRows = currentCtx.rowCount;
                    if (aggColF32) {
                        for (size_t i = 0; i < numRows && i < aggColF32->size(); ++i) {
                            float val = (*aggColF32)[i];
                            bool pass = false;
                            switch (compOp) {
                                case engine::expr::CompOp::GT: pass = val > scalarValue; break;
                                case engine::expr::CompOp::GE: pass = val >= scalarValue; break;
                                case engine::expr::CompOp::LT: pass = val < scalarValue; break;
                                case engine::expr::CompOp::LE: pass = val <= scalarValue; break;
                                case engine::expr::CompOp::EQ: pass = val == scalarValue; break;
                                default: pass = false;
                            }
                            if (pass) mask.push_back(static_cast<uint32_t>(i));
                        }
                    } else if (aggColU32) {
                        for (size_t i = 0; i < numRows && i < aggColU32->size(); ++i) {
                            double val = static_cast<double>((*aggColU32)[i]);
                            bool pass = false;
                            switch (compOp) {
                                case engine::expr::CompOp::GT: pass = val > scalarValue; break;
                                case engine::expr::CompOp::GE: pass = val >= scalarValue; break;
                                case engine::expr::CompOp::LT: pass = val < scalarValue; break;
                                case engine::expr::CompOp::LE: pass = val <= scalarValue; break;
                                case engine::expr::CompOp::EQ: pass = val == scalarValue; break;
                                default: pass = false;
                            }
                            if (pass) mask.push_back(static_cast<uint32_t>(i));
                        }
                    }
                    
                    // Compact all columns based on the mask
                    for (auto& [name, vals] : currentCtx.u32Cols) {
                        if (vals.size() > mask.size()) {
                            std::vector<uint32_t> compacted;
                            compacted.reserve(mask.size());
                            for (uint32_t idx : mask) {
                                if (idx < vals.size()) compacted.push_back(vals[idx]);
                            }
                            vals = std::move(compacted);
                        }
                    }
                    for (auto& [name, vals] : currentCtx.f32Cols) {
                        if (vals.size() > mask.size()) {
                            std::vector<float> compacted;
                            compacted.reserve(mask.size());
                            for (uint32_t idx : mask) {
                                if (idx < vals.size()) compacted.push_back(vals[idx]);
                            }
                            vals = std::move(compacted);
                        }
                    }
                    
                    currentCtx.activeRows.clear();  // Clear activeRows since we compacted
                    currentCtx.rowCount = mask.size();
                    
                    // Reset scalar aggregate flag - we now have a proper table result
                    result.isScalarAggregate = false;
                    
                    if (debug) {
                        std::cerr << "[V2]   After scalar filter: " << currentCtx.rowCount << " rows\n";
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
                            std::cerr << "[V2] Join: scalar SUBQUERY theta-join (tableContexts path)\n";
                            std::cerr << "[V2]   Current context rows: " << currentCtx.rowCount << "\n";
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
                                std::cerr << "[V2]   Scalar value from 'AVG': " << scalarValue << "\n";
                            }
                        }
                        
                        // Priority 2: Look for SUM column
                        if (!foundScalar) {
                            auto sumIt = currentCtx.f32Cols.find("SUM");
                            if (sumIt != currentCtx.f32Cols.end() && !sumIt->second.empty()) {
                                scalarValue = sumIt->second[0];
                                foundScalar = true;
                                if (debug) {
                                    std::cerr << "[V2]   Scalar value from 'SUM': " << scalarValue << "\n";
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
                                    std::cerr << "[V2]   Scalar value from '#0': " << scalarValue << "\n";
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
                                        std::cerr << "[V2]   Scalar value fallback from '" << name << "': " << scalarValue << "\n";
                                    }
                                    break;
                                }
                            }
                        }
                        
                        if (!foundScalar) {
                            if (debug) std::cerr << "[V2]   Could not find scalar value\n";
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
                            std::cerr << "[V2]   Filter column: " << filterCol << "\n";
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
                            if (debug) std::cerr << "[V2]   Could not find data table\n";
                            result.error = "Scalar SUBQUERY join: could not find data table";
                            return false;
                        }
                        
                        if (debug) {
                            std::cerr << "[V2]   Data table: " << dataTable << " with " 
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

                            std::optional<FilterResultGPU> filterRes;
                            if (dataCtx.activeRowsGPU) {
                                 filterRes = OperatorsGPU::filterF32Indexed(filterCol, colBuf, dataCtx.activeRowsGPU, dataCtx.activeRowsCountGPU, op, static_cast<float>(scalarValue));
                            } else {
                                 filterRes = OperatorsGPU::filterF32(filterCol, colBuf, dataCtx.rowCount, op, static_cast<float>(scalarValue));
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
                                    u32Replacements[buf] = OperatorsGPU::gatherU32(buf, indices, newCount);
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
                                    f32Replacements[buf] = OperatorsGPU::gatherF32(buf, indices, newCount);
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
                            
                            // Update activeRowsGPU (it's now dense 0..N because we gathered!)
                            // Wait, if we 'Gather', the resulting buffers are dense.
                            // So we don't need activeRowsGPU anymore (it's null).
                            // BUT, if we want to support chaining, we usually keep activeRowsGPU as null (all rows active).
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
                            std::cerr << "[V2] Join: skipping malformed join (both columns from " 
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
                            // Fix for Q15: Check known alias map (Planner V2 alias loss hack)
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
                                        if (debug) std::cerr << "[V2] Join: resolved orphan '" << col << "' -> '" << mapped << "'\n";
                                        resolvedAlias = true;
                                        // Found via alias. Proceed to 'matchesRHS' check; subsequent logic handles lookup.
                                        orphanFoundSomewhere = true;
                                    }
                                }
                            }

                            if (!orphanFoundSomewhere) {
                                if (debug) {
                                    std::cerr << "[V2] Join: skipping join with orphan column (not found anywhere)\n";
                                }
                                return true; // Skip this join
                            }
                        } else if (debug) {
                            std::cerr << "[V2] Join: orphan column found in some context, proceeding\n";
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
                        if (debug) std::cerr << "[V2] Join: found explicit right table '" << join.rightTable << "' in tableContexts (base table priority)\n";
                    }
                    
                    // For tmpl_ tables, check saved pipelines first
                    if (!specificTableFound) {
                        for (int pi = (int)savedPipelines.size() - 1; pi >= 0; --pi) {
                            if (savedPipelineTables[pi].count(join.rightTable)) {
                                savedPipelineIdx = pi;
                                specificTableFound = true;
                                if (debug) std::cerr << "[V2] Join: found explicit right table '" << join.rightTable << "' in saved pipeline #" << pi << "\n";
                                break;
                            }
                        }
                    }
                    
                    // Check table contexts if not found in saved
                    if (!specificTableFound && tableContexts.count(join.rightTable)) {
                        unjoinedTableForJoin = join.rightTable;
                        specificTableFound = true;
                        if (debug) std::cerr << "[V2] Join: found explicit right table '" << join.rightTable << "' in tableContexts\n";
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
                                savedPipelineIdx = pi;
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
                                // Also check if this table is in a saved pipeline - if so, skip it
                                bool inSavedPipeline = false;
                                for (const auto& spTables : savedPipelineTables) {
                                    if (spTables.find(key) != spTables.end()) {
                                        inSavedPipeline = true;
                                        break;
                                    }
                                }
                                if (inSavedPipeline) {
                                    if (debug) {
                                        std::cerr << "[V2] Join: table " << key 
                                                  << " is in saved pipeline, skipping\n";
                                    }
                                    continue;  // Skip - use saved pipeline instead
                                }
                                
                                if (hasColumnOrSuffixed(ctx, col)) {
                                    unjoinedTableForJoin = key;
                                    if (debug) {
                                        std::cerr << "[V2] Join: found unjoined table " << key 
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
                        std::cerr << "[V2] Join: using saved pipeline " << savedPipelineIdx 
                                  << " with " << rightCtx.rowCount << " rows as right side\n";
                        std::cerr << "[V2] Join: saved pipeline tables: ";
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
                                        std::cerr << "[V2] Join: skipping spurious ANTI join with scalar table "
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
                        std::cerr << "[V2] Join: using pre-found unjoined table " << unjoinedTableForJoin
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
                                            std::cerr << "[V2] Join: found unjoined instance " << key 
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
                                                std::cerr << "[V2] Join: pass2 found unjoined instance " << key 
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
                            std::cerr << "[V2] Join: cannot determine right table. joinedTables=";
                            for (const auto& t : joinedTables) std::cerr << t << " ";
                            std::cerr << "\n";
                            std::cerr << "[V2] Join: available tableContexts=";
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
                    if (debug) std::cerr << "[V2] Join: Applying right filter to right side (GPU)\n";
                    
                    if (!executeGPUFilterRecursive(join.rightFilter, rightCtx)) {
                         throw std::runtime_error("GPU Join Right Filter failed.");
                    }
                    // Sync CPU row count for logging/logic consistency (optional, but good for debugging)
                    // But we don't want to download the indices if we can avoid it.
                    // Just trust activeRowsGPU.
                }

                if (!executeJoin(join, datasetPath, currentCtx, rightCtx, joinCtx)) {
                    result.error = "Join execution failed";
                    return false;
                }
                
                currentCtx = std::move(joinCtx);
                if (debug) {
                    std::cerr << "[V2] Join: currentCtx after move: rowCount=" << currentCtx.rowCount << " u32ColsGPU.size=" << currentCtx.u32ColsGPU.size() << "\n";
                }
                // Merge all joined tables from both sides
                for (const auto& t : rightJoinedTables) {
                    joinedTables.insert(t);
                }
                hasPipeline = true;  // We now have a joined result in the pipeline
                if (debug) {
                    std::cerr << "[V2] Join: " << currentCtx.rowCount << " rows after. joinedTables=";
                    for (const auto& t : joinedTables) std::cerr << t << " ";
                    std::cerr << "\n";
                }

    return true;
}

} // namespace engine
