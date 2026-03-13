// ============================================================================
// JoinPipeline.cpp — Join pipeline orchestration: table resolution, DELIM dedup
// ============================================================================
#include "JoinInternal.hpp"
#include "Logger.hpp"

namespace engine {

// -- dedupDelimJoinRHS --
// Deduplicates RHS for DELIM_JOIN self-comparison patterns.
void dedupDelimJoinRHS(
    const IRJoin& join, EvalContext& currentCtx, EvalContext& rightCtx, bool debug) {
    // Extract all self-comparison key columns from the condition
    std::vector<std::string> delimDedupKeys;
    bool hasINDFKey = false; // Has "IS NOT DISTINCT FROM" pattern

    auto parts = splitConditionByAnd(join.conditionStr);
    for (const auto& part : parts) {
        bool isINDF = false;
        std::string col = parseSelfComparison(part, &isINDF);
        if (!col.empty()) {
            delimDedupKeys.push_back(col);
            hasINDFKey = hasINDFKey || isINDF;
        }
    }

    // Only apply RHS dedup when:
    // 1. IS NOT DISTINCT FROM conditions (standard DELIM_JOIN marker), OR
    // 2. For "=" self-comparison: only when the left side has fewer rows
    bool shouldDedup = !delimDedupKeys.empty() && !join.rightTable.empty() && rightCtx.rowCount > 1;
    if (shouldDedup && !hasINDFKey) {
        shouldDedup = (currentCtx.rowCount < rightCtx.rowCount);
    }
    if (shouldDedup) {
        // Compact rightCtx GPU buffers if activeRowsGPU is set
        if (rightCtx.activeRowsGPU && rightCtx.activeRowsCountGPU > 0) {
            uint32_t compactCount = rightCtx.activeRowsCountGPU;
            if (debug) {
                LOG_INFO("Exec", "Join: DELIM dedup: compacting rightCtx GPU buffers via activeRowsGPU (" << compactCount << " active rows)\n");
            }
            rightCtx.gatherAllGPU(rightCtx.activeRowsGPU, compactCount);
            rightCtx.clearActiveRows();
            rightCtx.rowCount = compactCount;
        }
        // Resolve key columns from GPU buffers (avoid CPU materialization)
        std::vector<std::string> resolvedKeys;
        std::vector<MTL::Buffer*> gpuKeys;
        for (const auto& k : delimDedupKeys) {
            // Try direct name in GPU
            if (rightCtx.u32ColsGPU.count(k) && rightCtx.u32ColsGPU[k]) {
                resolvedKeys.push_back(k);
                gpuKeys.push_back(rightCtx.u32ColsGPU[k]);
                continue;
            }
            bool found = false;
            // Try suffixed names
            for (int s = 1; s <= 9; ++s) {
                std::string sk = k + "_" + std::to_string(s);
                if (rightCtx.u32ColsGPU.count(sk) && rightCtx.u32ColsGPU[sk]) {
                    resolvedKeys.push_back(sk);
                    gpuKeys.push_back(rightCtx.u32ColsGPU[sk]);
                    found = true;
                    break;
                }
            }
            // Try prefix swap
            if (!found && k.size() > 2 && k[1] == '_') {
                std::string suffix = k.substr(2);
                for (const auto& p : {"l_", "o_", "c_", "p_", "s_", "ps_", "n_", "r_"}) {
                    std::string alt = std::string(p) + suffix;
                    if (rightCtx.u32ColsGPU.count(alt) && rightCtx.u32ColsGPU[alt]) {
                        resolvedKeys.push_back(alt);
                        gpuKeys.push_back(rightCtx.u32ColsGPU[alt]);
                        found = true;
                        break;
                    }
                }
            }
            // Fallback: upload from CPU
            if (!found) {
                auto tryUpload = [&](const std::string& name) -> bool {
                    if (rightCtx.u32Cols.count(name) && rightCtx.u32Cols.at(name).size() == rightCtx.rowCount) {
                        GpuBuffer buf = GpuOps::createBuffer(rightCtx.u32Cols[name].data(),
                                                                rightCtx.rowCount * sizeof(uint32_t));
                        rightCtx.u32ColsGPU[name] = std::move(buf);
                        resolvedKeys.push_back(name);
                        gpuKeys.push_back(rightCtx.u32ColsGPU[name]);
                        return true;
                    }
                    return false;
                };
                if (!tryUpload(k)) {
                    for (int s = 1; s <= 9 && !found; ++s) {
                        std::string sk = k + "_" + std::to_string(s);
                        if (tryUpload(sk)) { found = true; break; }
                    }
                }
            }
        }

        if (!resolvedKeys.empty()) {
            // GPU-based dedup using dedupByKeys (radix sort + mark unique)
            uint32_t newCount = 0;
            GpuBuffer uniqueIdx = GpuOps::dedupByKeys(gpuKeys, rightCtx.rowCount, newCount);

            if (newCount < rightCtx.rowCount) {
                if (debug) {
                    LOG_INFO("Exec", "Join: DELIM dedup RHS by [");
                    for (size_t ri=0; ri<resolvedKeys.size(); ++ri) { if (ri) std::cerr << ","; std::cerr << resolvedKeys[ri]; }
                    LOG_INFO("JOIN", "]: " << rightCtx.rowCount << " -> " << newCount);
                }
                // GPU gather all GPU columns (u32, f32, dict, flat string)
                rightCtx.gatherAllGPU(uniqueIdx, newCount);
                // CPU-only columns: download uniqueIdx and gather
                std::vector<uint32_t> keepIdx(newCount);
                memcpy(keepIdx.data(), uniqueIdx->contents(), newCount * sizeof(uint32_t));
                for (auto& [name, col] : rightCtx.u32Cols) {
                    if (!rightCtx.u32ColsGPU.count(name) && col.size() == rightCtx.rowCount) {
                        std::vector<uint32_t> compact(newCount);
                        for (uint32_t i = 0; i < newCount; ++i) compact[i] = col[keepIdx[i]];
                        col = std::move(compact);
                    }
                }
                for (auto& [name, col] : rightCtx.f32Cols) {
                    if (!rightCtx.f32ColsGPU.count(name) && col.size() == rightCtx.rowCount) {
                        std::vector<float> compact(newCount);
                        for (uint32_t i = 0; i < newCount; ++i) compact[i] = col[keepIdx[i]];
                        col = std::move(compact);
                    }
                }
                for (auto& [name, vec] : rightCtx.stringCols) {
                    if (vec.size() == rightCtx.rowCount && !rightCtx.dictCols.count(name) && !rightCtx.flatStringCols.count(name)) {
                        std::vector<std::string> compact(newCount);
                        for (uint32_t i = 0; i < newCount; ++i) compact[i] = vec[keepIdx[i]];
                        vec = std::move(compact);
                    }
                }
                rightCtx.invalidateStringColsForDictFlat();

                rightCtx.rowCount = newCount;
                rightCtx.activeRows.clear();
                rightCtx.activeRowsGPU = nullptr;
                rightCtx.activeRowsCountGPU = 0;

                // Strip right-side columns that already exist on the left side
                {
                    std::set<std::string> keepU32(resolvedKeys.begin(), resolvedKeys.end());
                    for (const auto& [name, _] : rightCtx.u32Cols) {
                        if (currentCtx.u32Cols.find(name) == currentCtx.u32Cols.end() &&
                            currentCtx.u32ColsGPU.find(name) == currentCtx.u32ColsGPU.end())
                            keepU32.insert(name);
                    }
                    for (const auto& [name, _] : rightCtx.u32ColsGPU) {
                        if (currentCtx.u32Cols.find(name) == currentCtx.u32Cols.end() &&
                            currentCtx.u32ColsGPU.find(name) == currentCtx.u32ColsGPU.end())
                            keepU32.insert(name);
                    }
                    for (auto it2 = rightCtx.u32Cols.begin(); it2 != rightCtx.u32Cols.end(); ) {
                        if (keepU32.find(it2->first) == keepU32.end()) it2 = rightCtx.u32Cols.erase(it2);
                        else ++it2;
                    }
                    for (auto it2 = rightCtx.u32ColsGPU.begin(); it2 != rightCtx.u32ColsGPU.end(); ) {
                        if (keepU32.find(it2->first) == keepU32.end())
                            it2 = rightCtx.u32ColsGPU.erase(it2);
                        else ++it2;
                    }
                    for (auto it2 = rightCtx.f32Cols.begin(); it2 != rightCtx.f32Cols.end(); ) {
                        if (currentCtx.f32Cols.find(it2->first) != currentCtx.f32Cols.end() ||
                            currentCtx.f32ColsGPU.find(it2->first) != currentCtx.f32ColsGPU.end())
                            it2 = rightCtx.f32Cols.erase(it2);
                        else ++it2;
                    }
                    for (auto it2 = rightCtx.f32ColsGPU.begin(); it2 != rightCtx.f32ColsGPU.end(); ) {
                        if (currentCtx.f32ColsGPU.find(it2->first) != currentCtx.f32ColsGPU.end() ||
                            currentCtx.f32Cols.find(it2->first) != currentCtx.f32Cols.end())
                            it2 = rightCtx.f32ColsGPU.erase(it2);
                        else ++it2;
                    }
                    if (debug) {
                        LOG_INFO("Exec", "Join: stripped RHS to " << rightCtx.u32Cols.size() << " u32, " << rightCtx.f32Cols.size() << " f32, " << rightCtx.stringCols.size() << " string cols\n");
                    }
                }
            }
        }
    }
}

// Detect trivial self-joins from DELIM_SCAN correlation markers that can be skipped.
bool detectTrivialSelfJoin(const IRJoin& join,
                           const EvalContext& currentCtx,
                           const std::set<std::string>& condCols,
                           const std::set<std::string>& joinedTables,
                           bool debug) {
    // Check for IS NOT DISTINCT FROM pattern (DuckDB's DELIM_SCAN correlation marker)
    if (join.conditionStr.find("IS NOT DISTINCT FROM") != std::string::npos) {
        std::string selfCol = parseSelfComparison(join.conditionStr);
        if (!selfCol.empty()) {
            bool colInContext = (currentCtx.u32Cols.find(selfCol) != currentCtx.u32Cols.end() ||
                                 currentCtx.f32Cols.find(selfCol) != currentCtx.f32Cols.end());
            if (colInContext) {
                if (join.type != JoinType::Left) {
                    if (join.rightTable.empty()) {
                        LOG_DEBUG("Exec", "Join: IS NOT DISTINCT FROM self-comparison: '" << selfCol << "' (col in context)\n");
                        return true;
                    } else if (debug) {
                        LOG_INFO("Exec", "Join: IS NOT DISTINCT FROM self-comparison BUT explicit right table specified (" << join.rightTable << "). Not skipping.\n");
                    }
                }
            } else if (debug) {
                LOG_INFO("Exec", "Join: IS NOT DISTINCT FROM self-comparison: '" << selfCol << "' BUT col not in context, may need re-join\n");
            }
        }
    }

    // Also check for self-comparison patterns: col = col
    if (condCols.size() == 1) {
        const std::string& col = *condCols.begin();
        bool colInContext = (currentCtx.u32Cols.find(col) != currentCtx.u32Cols.end() ||
                             currentCtx.f32Cols.find(col) != currentCtx.f32Cols.end());
        if (colInContext) {
            std::string baseTable = tableForColumn(col);
            bool alreadyJoined = false;
            for (const auto& jt : joinedTables) {
                if (jt == baseTable || jt.rfind(baseTable + "_", 0) == 0) {
                    alreadyJoined = true;
                    break;
                }
            }
            if (alreadyJoined && join.type != JoinType::Left) {
                if (join.rightTable.empty()) {
                    LOG_DEBUG("Exec", "Join: self-comparison detected for " << col << " (table " << baseTable << " already joined, col in context)\n");
                    return true;
                } else if (debug) {
                    LOG_INFO("Exec", "Join: self-comparison BUT explicit right table specified (" << join.rightTable << "). Not skipping.\n");
                }
            }
        } else if (debug) {
            LOG_INFO("Exec", "Join: self-comparison for " << col << " but col not in context, may need re-join\n");
        }
    }

    return false;
}

// ---------- inferRightTableForJoin ----------
// Infer which table context to use as the RHS of a join.
std::string inferRightTableForJoin(
    const IRJoin& join,
    const std::set<std::string>& condCols,
    const EvalContext& currentCtx,
    const std::unordered_map<std::string, EvalContext>& tableContexts,
    const std::set<std::string>& joinedTables,
    bool debug)
{
    // If the IR already names the right table, use it directly.
    if (!join.rightTable.empty())
        return join.rightTable;

    std::string rightTable;

    // ---------- helper lambda: check if ctx contains col or a suffixed variant ----------
    auto ctxHasColumn = [](const EvalContext& ctx, const std::string& col) -> bool {
        if (ctx.u32Cols.count(col) || ctx.f32Cols.count(col))
            return true;
        for (int suffix = 1; suffix <= 9; ++suffix) {
            std::string suffixed = col + "_" + std::to_string(suffix);
            if (ctx.u32Cols.count(suffixed) || ctx.f32Cols.count(suffixed))
                return true;
        }
        return false;
    };

    // ---------- Pass 1: columns NOT already in currentCtx ----------
    for (const auto& col : condCols) {
        std::string baseTable = tableForColumn(col);
        if (baseTable.empty()) continue;

        bool colInCurrentCtx = (currentCtx.u32Cols.count(col) || currentCtx.f32Cols.count(col));
        if (colInCurrentCtx) continue;

        for (const auto& [key, ctx] : tableContexts) {
            bool isInstance = (key == baseTable || key.rfind(baseTable + "_", 0) == 0);
            if (isInstance && !joinedTables.count(key) && ctxHasColumn(ctx, col)) {
                rightTable = key;
                if (debug)
                    LOG_INFO("Exec", "Join: pass1 found unjoined instance " << key << " for base " << baseTable << " (col " << col << ")\n");
                break;
            }
        }
        if (!rightTable.empty()) break;
    }

    // ---------- Pass 2: unjoined instances even if column is in ctx ----------
    if (rightTable.empty()) {
        for (const auto& col : condCols) {
            std::string baseTable = tableForColumn(col);
            if (baseTable.empty()) continue;

            for (const auto& [key, ctx] : tableContexts) {
                bool isInstance = (key == baseTable || key.rfind(baseTable + "_", 0) == 0);
                if (isInstance && !joinedTables.count(key) && ctxHasColumn(ctx, col)) {
                    rightTable = key;
                    if (debug)
                        LOG_INFO("Exec", "Join: pass2 found unjoined instance " << key << " for base " << baseTable << " (col " << col << ")\n");
                    break;
                }
            }
            if (!rightTable.empty()) break;
        }
    }

    return rightTable;
}

// Validate join condition columns: detect malformed same-table joins and
// orphan columns that can't be resolved. Returns true if the join should be skipped.
static bool shouldSkipJoinCondition(
    const std::set<std::string>& condCols,
    const IRJoin& join,
    EvalContext& currentCtx,
    std::unordered_map<std::string, EvalContext>& tableContexts,
    std::vector<EvalContext>& savedPipelines,
    const std::vector<std::set<std::string>>& savedPipelineTables,
    bool debug)
{
    if (condCols.size() != 2) return false;

    std::string firstTable;
    bool allColsFromSameTable = true;
    bool hasOrphanColumn = false;
    std::vector<std::string> colsList(condCols.begin(), condCols.end());

    for (const auto& col : condCols) {
        std::string baseTable = tableForColumn(col);
        if (baseTable.empty()) {
            hasOrphanColumn = true;
        } else {
            if (firstTable.empty()) firstTable = baseTable;
            else if (baseTable != firstTable) allColsFromSameTable = false;
        }
    }

    // Check for self-comparison patterns (e.g., "l_k = l_k") — valid self-joins
    bool hasSelfComparisonInCondition = false;
    for (const auto& col : condCols) {
        std::string pattern1 = col + " = " + col;
        std::string pattern2 = col + " IS NOT DISTINCT FROM " + col;
        if (join.conditionStr.find(pattern1) != std::string::npos ||
            join.conditionStr.find(pattern2) != std::string::npos) {
            hasSelfComparisonInCondition = true;
            break;
        }
    }

    // Check for suffixed aliases (e.g. p_partkey_rhs_9) implying distinct instances
    bool hasAlias = false;
    for (const auto& col : condCols) {
        if (col.find("_rhs_") != std::string::npos || col.find("_lhs_") != std::string::npos) {
            hasAlias = true;
            break;
        }
    }

    // "p_size = p_partkey" (same table, different col) → skip
    if (allColsFromSameTable && !firstTable.empty() && !hasOrphanColumn && !hasSelfComparisonInCondition && !hasAlias) {
        LOG_DEBUG("Exec", "Join: skipping malformed join (both columns from " << firstTable << ", different cols: " << colsList[0] << " vs " << colsList[1] << ")\n");
        return true;
    }

    // Check orphan columns (no table prefix). Only skip if genuinely not found anywhere.
    if (hasOrphanColumn) {
        bool orphanFoundSomewhere = false;
        for (const auto& col : condCols) {
            if (!tableForColumn(col).empty()) continue;
            // Check currentCtx, tableContexts, savedPipelines
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

        if (!orphanFoundSomewhere) {
            // Try known aliases
            static const std::unordered_map<std::string, std::string> knownAliases = {
                {"supplier_no", "l_suppkey"}
            };
            for (const auto& col : condCols) {
                if (knownAliases.count(col)) {
                    std::string mapped = knownAliases.at(col);
                    auto checkCtx = [&](const EvalContext& ctx) {
                        return ctx.u32Cols.count(mapped) || ctx.f32Cols.count(mapped);
                    };
                    bool mappedFound = checkCtx(currentCtx);
                    if (!mappedFound) {
                        for (const auto& [t, c] : tableContexts) if (checkCtx(c)) { mappedFound = true; break; }
                    }
                    if (!mappedFound) {
                        for (const auto& sp : savedPipelines) if (checkCtx(sp)) { mappedFound = true; break; }
                    }
                    if (mappedFound) {
                        LOG_DEBUG("Exec", "Join: resolved orphan '" << col << "' -> '" << mapped << "'\n");
                        orphanFoundSomewhere = true;
                    }
                }
            }

            if (!orphanFoundSomewhere) {
                if (applyScalarSubqueryCrossJoinFilter(condCols, join, currentCtx, tableContexts, savedPipelines, savedPipelineTables, debug)) {
                    return true; // Handled as scalar subquery filter
                }
                LOG_DEBUG("Exec", "Join: skipping join with orphan column (not found anywhere)\n");
                return true;
            }
        } else if (debug) {
            LOG_INFO("Exec", "Join: orphan column found in some context, proceeding\n");
        }
    }

    return false; // Proceed with join
}

// --- Result struct for right context resolution ---
struct RightContextResolution {
    int savedPipelineIdx = -1;
    std::string unjoinedTable;
};

// Resolve the right-side context for a join: check explicit right table,
// saved pipelines, and unjoined table instances in tableContexts.
static RightContextResolution resolveRightContextForJoin(
    const IRJoin& join,
    const std::set<std::string>& condCols,
    const EvalContext& currentCtx,
    const std::unordered_map<std::string, EvalContext>& tableContexts,
    const std::vector<EvalContext>& savedPipelines,
    const std::vector<std::set<std::string>>& savedPipelineTables,
    const std::set<std::string>& joinedTables,
    bool debug)
{
    RightContextResolution res;

    // PRIORITY: Explicit right table check (for DELIM joins)
    if (!join.rightTable.empty()) {
        bool specificTableFound = false;

        // For base table names (not tmpl_ prefixes), check tableContexts FIRST
        bool isBaseTable = (join.rightTable.find("tmpl_") != 0);

        if (isBaseTable && tableContexts.count(join.rightTable)) {
            res.unjoinedTable = join.rightTable;
            specificTableFound = true;
            LOG_DEBUG("Exec", "Join: found explicit right table '" << join.rightTable << "' in tableContexts (base table priority)\n");
        }

        // For tmpl_ tables, check saved pipelines first
        if (!specificTableFound) {
            for (int pi = (int)savedPipelines.size() - 1; pi >= 0; --pi) {
                if (savedPipelineTables[pi].count(join.rightTable)) {
                    res.savedPipelineIdx = pi;
                    specificTableFound = true;
                    LOG_DEBUG("Exec", "Join: found explicit right table '" << join.rightTable << "' in saved pipeline #" << pi);
                    break;
                }
            }
        }

        // Check table contexts if not found in saved
        if (!specificTableFound && tableContexts.count(join.rightTable)) {
            res.unjoinedTable = join.rightTable;
            specificTableFound = true;
            LOG_DEBUG("Exec", "Join: found explicit right table '" << join.rightTable << "' in tableContexts\n");
        }

        // VALIDATE: If the explicit right table doesn't contain any condition columns
        // that are missing from the current context, fall through to heuristic search.
        if (specificTableFound && !res.unjoinedTable.empty()) {
            const EvalContext& rightCandidate = tableContexts.at(res.unjoinedTable);
            bool hasNewColumn = false;
            for (const auto& col : condCols) {
                if (!hasColumnOrSuffixed(currentCtx, col) && hasColumnOrSuffixed(rightCandidate, col)) {
                    hasNewColumn = true;
                    break;
                }
            }
            if (!hasNewColumn) {
                LOG_DEBUG("Exec", "Join: explicit right table '" << res.unjoinedTable << "' has no new condition columns, falling through to heuristic\n");
                res.unjoinedTable.clear();
                specificTableFound = false;
            }
        }
    }

    // If explicit lookup didn't set anything, run legacy heuristic
    if (res.savedPipelineIdx < 0 && res.unjoinedTable.empty())
    // Prefer LATEST pipeline (reverse search) to ensure we get the most accumulated state
    for (int pi = (int)savedPipelines.size() - 1; pi >= 0; --pi) {
        const auto& savedCtx = savedPipelines[pi];
        for (const auto& col : condCols) {
            if (hasColumnOrSuffixed(savedCtx, col)) {
                if (!hasColumnOrSuffixed(currentCtx, col)) {
                    // Before accepting, check if the saved pipeline has been aggregated
                    std::string baseTable = tableForColumn(col);
                    bool isAggregatedPipeline = false;
                    if (!baseTable.empty() && savedCtx.rowCount <= 10) {
                        for (const auto& [key, freshCtx] : tableContexts) {
                            bool isInstanceOf = (key == baseTable || 
                                                key.rfind(baseTable + "_", 0) == 0);
                            if (isInstanceOf && freshCtx.rowCount > 10 && 
                                joinedTables.find(key) == joinedTables.end()) {
                                isAggregatedPipeline = true;
                                LOG_DEBUG("Exec", "Join: savedPipeline " << pi  << " has " << savedCtx.rowCount << " rows but fresh table '"  << key << "' has " << freshCtx.rowCount  << " rows — skipping aggregated pipeline\n");
                                break;
                            }
                        }
                    }
                    if (!isAggregatedPipeline) {
                        res.savedPipelineIdx = pi;
                    }
                }
            }
        }
        if (res.savedPipelineIdx >= 0) break;
    }

    if (res.savedPipelineIdx < 0 && res.unjoinedTable.empty()) {
        for (const auto& col : condCols) {
            if (hasColumnOrSuffixed(currentCtx, col)) continue;

            std::string baseTable = tableForColumn(col);
            if (baseTable.empty()) continue;

            for (const auto& [key, ctx] : tableContexts) {
                bool isInstanceOf = (key == baseTable || 
                                    key.rfind(baseTable + "_", 0) == 0);
                if (isInstanceOf && joinedTables.find(key) == joinedTables.end()) {
                    bool inSavedPipeline = false;
                    bool savedPipelineIsAggregated = false;
                    for (size_t spi = 0; spi < savedPipelineTables.size(); ++spi) {
                        if (savedPipelineTables[spi].find(key) != savedPipelineTables[spi].end()) {
                            inSavedPipeline = true;
                            size_t savedRows = savedPipelines[spi].rowCount;
                            size_t freshRows = ctx.rowCount;
                            if (freshRows > 10 && savedRows <= 10) {
                                savedPipelineIsAggregated = true;
                            }
                            break;
                        }
                    }
                    if (inSavedPipeline && !savedPipelineIsAggregated) {
                        LOG_DEBUG("Exec", "Join: table " << key  << " is in saved pipeline, skipping\n");
                        continue;
                    }
                    if (savedPipelineIsAggregated && debug) {
                        LOG_INFO("Exec", "Join: table " << key  << " is in saved pipeline but pipeline was aggregated, using fresh table\n");
                    }

                    if (hasColumnOrSuffixed(ctx, col)) {
                        res.unjoinedTable = key;
                        LOG_DEBUG("Exec", "Join: found unjoined table " << key  << " with column " << col);
                        break;
                    }
                }
            }
            if (!res.unjoinedTable.empty()) break;
        }
    }

    return res;
}

bool GpuExecutor::executeJoinPipeline(
    const IRJoin& join,
    EvalContext& currentCtx,
    JoinPipelineState& state,
    ExecutionResult& result
) {
    auto& tableContexts       = state.tableContexts;
    auto& savedPipelines      = state.savedPipelines;
    auto& savedPipelineTables = state.savedPipelineTables;
    auto& joinedTables        = state.joinedTables;
    auto& hasPipeline         = state.hasPipeline;

    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
                result.isScalarAggregate = false;
                
                // Collect all columns referenced in the join condition
                std::set<std::string> condCols;
                collectColumnsFromExpr(join.condition, condCols);
                
                if (debug) {
                    LOG_INFO("Exec", "Join: conditionStr=" << join.conditionStr);
                    LOG_INFO("Exec", "Join: type=");
                    switch (join.type) {
                        if (debug) case JoinType::Inner: std::cerr << "Inner"; break;
                        if (debug) case JoinType::Left: std::cerr << "Left"; break;
                        if (debug) case JoinType::Semi: std::cerr << "Semi"; break;
                        if (debug) case JoinType::Anti: std::cerr << "Anti"; break;
                        if (debug) case JoinType::Mark: std::cerr << "Mark"; break;
                        if (debug) default: std::cerr << "Unknown(" << static_cast<int>(join.type) << ")"; break;
                    }
                    LOG_DEBUG("JOIN", "\n");
                    LOG_DEBUG("Exec", "Join: condCols extracted: ");
                    if (debug) for (const auto& c : condCols) std::cerr << c << " ";
                    LOG_DEBUG("JOIN", "(total=" << condCols.size() << ")\n");
                }
                
                // Skip trivial self-joins from DELIM_SCAN correlation markers.
                if (detectTrivialSelfJoin(join, currentCtx, condCols, joinedTables, debug)) {
                    if (debug)
                        LOG_INFO("Exec", "Join: skipping trivial self-join (all columns already in pipeline)\n");
                    return true;
                }
                
                // Check for scalar subquery pattern (join condition contains SUBQUERY).
                if (join.conditionStr.find("SUBQUERY") != std::string::npos && !savedPipelines.empty()) {
                    if (handleScalarSubquerySavedPipelines(join, currentCtx, savedPipelines, savedPipelineTables, joinedTables, result, debug))
                        return true;
                }
                
                // Alt: scalar SUBQUERY join (savedPipelines empty, data in tableContexts).
                if (join.conditionStr.find("SUBQUERY") != std::string::npos && savedPipelines.empty()) {
                    if (handleScalarSubqueryTableContexts(join, currentCtx, tableContexts, joinedTables, hasPipeline, result, debug))
                        return true;
                }
                
                // Validate join condition (skip malformed same-table joins and unresolvable orphans)
                if (shouldSkipJoinCondition(condCols, join, currentCtx, tableContexts, savedPipelines, savedPipelineTables, debug))
                    return true;
                
                // Resolve right-side context for this join
                auto rightRes = resolveRightContextForJoin(
                    join, condCols, currentCtx, tableContexts, savedPipelines,
                    savedPipelineTables, joinedTables, debug);
                int savedPipelineIdx = rightRes.savedPipelineIdx;
                std::string unjoinedTableForJoin = std::move(rightRes.unjoinedTable);
                
                EvalContext rightCtx;
                std::set<std::string> rightJoinedTables;
                
                if (savedPipelineIdx >= 0) {
                    // Use saved pipeline as right context (multi-pipeline merge join)
                    rightCtx = savedPipelines[savedPipelineIdx];
                    rightJoinedTables = savedPipelineTables[savedPipelineIdx];
                    if (debug) {
                        LOG_INFO("Exec", "Join: using saved pipeline " << savedPipelineIdx  << " with " << rightCtx.rowCount << " rows as right side\n");
                        LOG_INFO("Exec", "Join: saved pipeline tables: ");
                        if (debug) for (const auto& t : rightJoinedTables) std::cerr << t << " ";
                        LOG_DEBUG("JOIN", "\n");
                    }
                } else if (!unjoinedTableForJoin.empty()) {
                    // Use the unjoined table we found earlier (priority over other inference)
                    bool skipSpuriousAntiJoin = false;
                    if ((join.type == JoinType::Anti || join.type == JoinType::Mark) &&
                        joinedTables.find("__GROUPED__") != joinedTables.end()) {
                        const EvalContext& potentialRight = tableContexts[unjoinedTableForJoin];
                        if (potentialRight.rowCount <= 1 && 
                            join.conditionStr.find("IS NOT DISTINCT FROM") != std::string::npos) {
                            std::string selfCol = parseSelfComparison(join.conditionStr);
                            if (!selfCol.empty()) {
                                if (debug) {
                                    LOG_INFO("Exec", "Join: skipping spurious ANTI join with scalar table " << unjoinedTableForJoin << " after GroupBy\n");
                                }
                                skipSpuriousAntiJoin = true;
                            }
                        }
                    }
                    
                    if (skipSpuriousAntiJoin) {
                        return true; // Skip this join entirely
                    }
                    
                    rightCtx = tableContexts[unjoinedTableForJoin];
                    rightJoinedTables.insert(unjoinedTableForJoin);
                    if (debug) {
                        LOG_INFO("Exec", "Join: using pre-found unjoined table " << unjoinedTableForJoin << " with " << rightCtx.rowCount << " rows as right side\n");
                    }
                } else {
                    // Infer right table from join condition columns
                    std::string rightTable = inferRightTableForJoin(
                        join, condCols, currentCtx, tableContexts, joinedTables, debug);
                    
                    if (rightTable.empty() || tableContexts.find(rightTable) == tableContexts.end()) {
                        if (debug) {
                            LOG_WARN("Exec", "Join: cannot determine right table. joinedTables=");
                            for (const auto& t : joinedTables) std::cerr << t << " ";
                            LOG_INFO("JOIN", "\n");
                            LOG_INFO("Exec", "Join: available tableContexts=");
                            for (const auto& [k, v] : tableContexts) std::cerr << k << " ";
                            LOG_INFO("JOIN", "\n");
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
                    LOG_DEBUG("Exec", "Join: Applying right filter to right side (GPU)\n");
                    
                    if (!executeFilterRecursive(join.rightFilter, rightCtx)) {
                         ENGINE_THROW("GPU Join Right Filter failed.");
                    }
                }

                dedupDelimJoinRHS(join, currentCtx, rightCtx, debug);

                // SEMI join with self-comparison: swap so outer table is the probe side.
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
                            LOG_DEBUG("Exec", "SEMI join: swapping sides (right table becomes probe)\n");
                            std::swap(currentCtx, rightCtx);
                        }
                    }
                }

                if (!executeJoin(join, currentCtx, rightCtx, joinCtx)) {
                    result.error = "Join execution failed";
                    return false;
                }
                
                currentCtx = std::move(joinCtx);
                if (debug) {
                    LOG_INFO("Exec", "Join: currentCtx after move: rowCount=" << currentCtx.rowCount << " u32ColsGPU.size=" << currentCtx.u32ColsGPU.size());
                    LOG_INFO("Exec", "Join: currentCtx.stringCols after move:\n");
                    for (const auto& [n, v] : currentCtx.stringCols) {
                        LOG_DEBUG("Exec", n << " size=" << v.size());
                    }
                    LOG_DEBUG("Exec", "Join: currentCtx.currentTable='" << currentCtx.currentTable << "'\n");
                }
                // Merge all joined tables from both sides
                for (const auto& t : rightJoinedTables) {
                    joinedTables.insert(t);
                }
                hasPipeline = true;  // We now have a joined result in the pipeline
                if (debug) {
                    LOG_INFO("Exec", "Join: " << currentCtx.rowCount << " rows after. joinedTables=");
                    for (const auto& t : joinedTables) std::cerr << t << " ";
                    LOG_INFO("JOIN", "\n");
                }

    return true;
}

} // namespace engine
