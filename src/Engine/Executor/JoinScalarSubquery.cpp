// ============================================================================
// JoinScalarSubquery.cpp — Scalar subquery handling for join paths
// ============================================================================
#include "JoinInternal.hpp"
#include "Logger.hpp"

namespace engine {

// -- handleScalarSubquerySavedPipelines --
// Handles scalar SUBQUERY join when savedPipelines contains main data.
// Returns true if handled (caller should return).
bool handleScalarSubquerySavedPipelines(
    const IRJoin& join, EvalContext& currentCtx,
    std::vector<EvalContext>& savedPipelines,
    std::vector<std::set<std::string>>& savedPipelineTables,
    std::set<std::string>& joinedTables,
    GpuExecutor::ExecutionResult& result, bool debug) {
    if (debug) {
        LOG_INFO("Exec", "Join: detected scalar subquery pattern\n");
        LOG_INFO("Exec", "Current context rows: " << currentCtx.rowCount);
        LOG_INFO("Exec", "Saved pipelines: " << savedPipelines.size());
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
                    LOG_INFO("Exec", "Found scalar pipeline via flag (rowCount=" << savedPipelines[pi].rowCount << ")\n");
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
                 LOG_DEBUG("Exec", "Scalar value from f32 col '" << name << "': " << scalarValue);
                 return true;
             }
         }
         // Search u32
         for (const auto& [name, values] : scalarCtx->u32Cols) {
             if (values.empty()) continue;
             bool match = exact ? (name == pattern) : (name.find(pattern) != std::string::npos);
             if (match) {
                 scalarValue = static_cast<double>(values[0]);
                 LOG_DEBUG("Exec", "Scalar value from u32 col '" << name << "': " << scalarValue);
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
            LOG_INFO("Exec", "Restored saved pipeline with " << currentCtx.rowCount << " rows\n");
        }
    } else {
        // Data is already currentCtx. Just remove the scalar pipeline from saved.
        if (scalarPipelineIdx >= 0) {
            savedPipelines.erase(savedPipelines.begin() + scalarPipelineIdx);
            savedPipelineTables.erase(savedPipelineTables.begin() + scalarPipelineIdx);
        }
        if (debug) {
            LOG_INFO("Exec", "Using current context as data table with " << currentCtx.rowCount << " rows\n");
        }
    }

    // Inject broadcasted scalars into the data context
    for(auto& [n, v] : scalarF32s) {
        if (currentCtx.f32Cols.find(n) == currentCtx.f32Cols.end() && currentCtx.f32ColsGPU.find(n) == currentCtx.f32ColsGPU.end()) {
             currentCtx.f32Cols[n] = {v}; // Size 1 vector (scalar broadcast)
             LOG_DEBUG("Exec", "Broadcasted scalar F32col: " << n);
        }
    }
    for(auto& [n, v] : scalarU32s) {
        if (currentCtx.u32Cols.find(n) == currentCtx.u32Cols.end() && currentCtx.u32ColsGPU.find(n) == currentCtx.u32ColsGPU.end()) {
             currentCtx.u32Cols[n] = {v};
             LOG_DEBUG("Exec", "Broadcasted scalar U32col: " << n);
        }
    }

    // Parse condition to extract comparison column and operator
    std::string condStr = join.conditionStr;

    // Find the comparison operator
    size_t opPos = std::string::npos;
    std::string opStr;
    engine::GpuFilterOp compOp = engine::GpuFilterOp::EQ;
    if ((opPos = condStr.find(" > SUBQUERY")) != std::string::npos) {
        opStr = ">";
        compOp = engine::GpuFilterOp::GT;
    } else if ((opPos = condStr.find(" >= SUBQUERY")) != std::string::npos) {
        opStr = ">=";
        compOp = engine::GpuFilterOp::GE;
    } else if ((opPos = condStr.find(" < SUBQUERY")) != std::string::npos) {
        opStr = "<";
        compOp = engine::GpuFilterOp::LT;
    } else if ((opPos = condStr.find(" <= SUBQUERY")) != std::string::npos) {
        opStr = "<=";
        compOp = engine::GpuFilterOp::LE;
    } else if ((opPos = condStr.find(" = SUBQUERY")) != std::string::npos) {
        opStr = "=";
        compOp = engine::GpuFilterOp::EQ;
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
        LOG_INFO("Exec", "Filtering: " << aggColName << " " << opStr << " " << scalarValue);
    }

    // Apply scalar subquery filter on GPU
    // 1. Ensure data columns are uploaded to GPU
    auto device = GpuColumnStore::instance().device();
    for (auto& [name, vec] : currentCtx.f32Cols) {
        if (currentCtx.f32ColsGPU.find(name) == currentCtx.f32ColsGPU.end()) {
            auto buf = device->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
            if (buf) currentCtx.f32ColsGPU[name].reset(buf);
        }
    }
    for (auto& [name, vec] : currentCtx.u32Cols) {
        if (currentCtx.u32ColsGPU.find(name) == currentCtx.u32ColsGPU.end()) {
            auto buf = device->newBuffer(vec.data(), vec.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
            if (buf) currentCtx.u32ColsGPU[name].reset(buf);
        }
    }

    // 2. Build a TypedExpr comparison predicate: aggColName <op> scalarValue
    CompareOp typedOp = CompareOp::Eq;
    switch (compOp) {
        case engine::GpuFilterOp::GT: typedOp = CompareOp::Gt; break;
        case engine::GpuFilterOp::GE: typedOp = CompareOp::Ge; break;
        case engine::GpuFilterOp::LT: typedOp = CompareOp::Lt; break;
        case engine::GpuFilterOp::LE: typedOp = CompareOp::Le; break;
        case engine::GpuFilterOp::EQ: typedOp = CompareOp::Eq; break;
        case engine::GpuFilterOp::NE: typedOp = CompareOp::Ne; break;
        default: break;
    }
    auto filterPred = TypedExpr::compare(
        typedOp,
        TypedExpr::column(aggColName),
        TypedExpr::literal(static_cast<double>(scalarValue))
    );

    // 3. Execute GPU filter
    if (!GpuExecutor::executeFilterRecursive(filterPred, currentCtx)) {
        result.error = "Scalar subquery join: GPU filter failed for " + aggColName;
        return false;
    }

    // 4. Materialize: compact all columns using activeRowsGPU
    if (currentCtx.activeRowsGPU && currentCtx.activeRowsCountGPU > 0) {
        uint32_t count = currentCtx.activeRowsCountGPU;
        uint32_t* indices = static_cast<uint32_t*>(currentCtx.activeRowsGPU->contents());

        // Compact GPU columns
        for (auto& [name, buf] : currentCtx.u32ColsGPU) {
            if (!buf) continue;
            uint32_t bufRows = (uint32_t)(buf->length() / sizeof(uint32_t));
            if (bufRows > count) {
                auto compacted = GpuOps::gatherU32(buf, currentCtx.activeRowsGPU, count, false);
                if (compacted) buf.reset(compacted);
            }
        }
        for (auto& [name, buf] : currentCtx.f32ColsGPU) {
            if (!buf) continue;
            uint32_t bufRows = (uint32_t)(buf->length() / sizeof(float));
            if (bufRows > count) {
                auto compacted = GpuOps::gatherF32(buf, currentCtx.activeRowsGPU, count, false);
                if (compacted) buf.reset(compacted);
            }
        }
        GpuOps::sync();
        // Compact CPU columns: clear when GPU compacted, else CPU gather
        for (auto& [name, vec] : currentCtx.u32Cols) {
            if (vec.size() > count) {
                if (currentCtx.u32ColsGPU.count(name) && currentCtx.u32ColsGPU[name]) {
                    // GPU buffer already compacted — skip CPU download, lazy-fetch later
                    vec.clear();
                } else {
                    std::vector<uint32_t> c;
                    c.reserve(count);
                    for (uint32_t i = 0; i < count; ++i)
                        c.push_back(indices[i] < (uint32_t)vec.size() ? vec[indices[i]] : 0u);
                    vec = std::move(c);
                }
            }
        }
        for (auto& [name, vec] : currentCtx.f32Cols) {
            if (vec.size() > count) {
                if (currentCtx.f32ColsGPU.count(name) && currentCtx.f32ColsGPU[name]) {
                    // GPU buffer already compacted — skip CPU download, lazy-fetch later
                    vec.clear();
                } else {
                    std::vector<float> c;
                    c.reserve(count);
                    for (uint32_t i = 0; i < count; ++i)
                        c.push_back(indices[i] < (uint32_t)vec.size() ? vec[indices[i]] : 0.0f);
                    vec = std::move(c);
                }
            }
        }

        if (currentCtx.activeRowsGPU) { currentCtx.activeRowsGPU = nullptr; }
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
        LOG_INFO("Exec", "After scalar filter: " << currentCtx.rowCount << " rows\n");
    }

    // Don't do the normal join - we've handled this specially
    return true;
}

// ── Extract a scalar float value from a single-row context ──
// Priority order: AVG → SUM → #0 → any non-COUNT f32 column.
static std::pair<double, bool> extractScalarValue(const EvalContext& ctx, bool /*debug*/) {
    // Priority 1: AVG
    auto avgIt = ctx.f32Cols.find("AVG");
    if (avgIt != ctx.f32Cols.end() && !avgIt->second.empty()) {
        LOG_DEBUG("Exec", "Scalar value from 'AVG': " << avgIt->second[0]);
        return {avgIt->second[0], true};
    }
    // Priority 2: SUM
    auto sumIt = ctx.f32Cols.find("SUM");
    if (sumIt != ctx.f32Cols.end() && !sumIt->second.empty()) {
        LOG_DEBUG("Exec", "Scalar value from 'SUM': " << sumIt->second[0]);
        return {sumIt->second[0], true};
    }
    // Priority 3: #0
    auto numIt = ctx.f32Cols.find("#0");
    if (numIt != ctx.f32Cols.end() && !numIt->second.empty()) {
        LOG_DEBUG("Exec", "Scalar value from '#0': " << numIt->second[0]);
        return {numIt->second[0], true};
    }
    // Fallback: any f32 column except COUNT
    for (const auto& [name, values] : ctx.f32Cols) {
        if (!values.empty() && name.find("COUNT") == std::string::npos) {
            LOG_DEBUG("Exec", "Scalar value fallback from '" << name << "': " << values[0]);
            return {values[0], true};
        }
    }
    LOG_DEBUG("Exec", "Could not find scalar value\n");
    return {0.0, false};
}

// -- handleScalarSubqueryTableContexts --
// Handles scalar SUBQUERY join via tableContexts (theta-comparison).
// Returns true if handled (caller should return).
bool handleScalarSubqueryTableContexts(
    const IRJoin& join, EvalContext& currentCtx,
    std::unordered_map<std::string, EvalContext>& tableContexts,
    std::set<std::string>& joinedTables, bool& hasPipeline,
    GpuExecutor::ExecutionResult& result, bool debug) {
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
            LOG_INFO("Exec", "Join: scalar SUBQUERY theta-join (tableContexts path)\n");
            LOG_INFO("Exec", "Current context rows: " << currentCtx.rowCount);
        }

        // Extract scalar value from currentCtx
        auto [scalarValue, foundScalar] = extractScalarValue(currentCtx, debug);

        if (!foundScalar) {
            LOG_DEBUG("Exec", "Could not find scalar value\n");
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
            LOG_INFO("Exec", "Filter column: " << filterCol);
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
            LOG_DEBUG("Exec", "Could not find data table\n");
            result.error = "Scalar SUBQUERY join: could not find data table";
            return false;
        }

        if (debug) {
            LOG_INFO("Exec", "Data table: " << dataTable << " with "  << tableContexts[dataTable].rowCount << " rows\n");
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
            auto& store = GpuColumnStore::instance();

            // Ensure column is on GPU
            MTL::Buffer* colBuf = nullptr;
            if (dataCtx.f32ColsGPU.count(filterCol)) {
                colBuf = dataCtx.f32ColsGPU[filterCol];
            } else {
                // Upload (Lazy)
                const auto& vec = it->second;
                colBuf = store.device()->newBuffer(vec.data(), vec.size() * sizeof(float), MTL::ResourceStorageModeShared);
                dataCtx.f32ColsGPU[filterCol].reset(colBuf);
            }

            // Map Op
            engine::GpuFilterOp op = engine::GpuFilterOp::EQ;
            if (opStr == ">") op = engine::GpuFilterOp::GT;
            else if (opStr == ">=") op = engine::GpuFilterOp::GE;
            else if (opStr == "<") op = engine::GpuFilterOp::LT;
            else if (opStr == "<=") op = engine::GpuFilterOp::LE;
            else if (opStr == "=") op = engine::GpuFilterOp::EQ;

            std::optional<FilterResult> filterRes;
            if (dataCtx.activeRowsGPU) {
                 filterRes = GpuOps::filterF32Indexed(filterCol, colBuf, dataCtx.activeRowsGPU, dataCtx.activeRowsCountGPU, op, static_cast<float>(scalarValue));
            } else {
                 filterRes = GpuOps::filterF32(filterCol, colBuf, dataCtx.rowCount, op, static_cast<float>(scalarValue));
            }

            if (!filterRes) ENGINE_THROW("GPU Scalar Filter failed");

            MTL::Buffer* indices = filterRes->indices;
            uint32_t newCount = filterRes->count;

            // Download indices for CPU String sync
            std::vector<uint32_t> cpuPassingIndices(newCount);
            if (newCount > 0) {
                std::memcpy(cpuPassingIndices.data(), indices->contents(), newCount * sizeof(uint32_t));
            }

            // Safe Gather for U32 (preserving aliases and avoiding double-free)
            std::unordered_map<MTL::Buffer*, MTL::Buffer*> u32Replacements;
            for (auto& [name, buf] : dataCtx.u32ColsGPU) {
                if (buf && u32Replacements.find(buf) == u32Replacements.end()) {
                    u32Replacements[buf] = GpuOps::gatherU32(buf, indices, newCount, false).detach();
                }
            }
            // Update map with new buffers
            for (auto& [name, buf] : dataCtx.u32ColsGPU) {
                if (buf) {
                    MTL::Buffer* newBuf = u32Replacements[buf];
                    newBuf->retain(); 
                    buf.reset(newBuf); 
                }
            }
            // Consume creation refs of new buffers (old buffers already released by GpuBuffer::reset)
            for (auto& [_, newBuf] : u32Replacements) {
                newBuf->release(); 
            }

            // Safe Gather for F32
            std::unordered_map<MTL::Buffer*, MTL::Buffer*> f32Replacements;
            for (auto& [name, buf] : dataCtx.f32ColsGPU) {
                if (buf && f32Replacements.find(buf.get()) == f32Replacements.end()) {
                    f32Replacements[buf.get()] = GpuOps::gatherF32(buf, indices, newCount, false).detach();
                }
            }
            for (auto& [name, buf] : dataCtx.f32ColsGPU) {
                if (buf) {
                    MTL::Buffer* newBuf = f32Replacements[buf.get()];
                    newBuf->retain();
                    buf.reset(newBuf);
                }
            }
            for (auto& [_, newBuf] : f32Replacements) {
                newBuf->release();
            }

            // Handle strings on CPU (fallback when dict/flat not available)
            for (auto& [name, vals] : dataCtx.stringCols) {
                if (dataCtx.dictCols.count(name)) continue; // dict path below
                if (dataCtx.flatStringCols.count(name)) continue; // flat path below
                std::vector<std::string> compacted;
                compacted.reserve(cpuPassingIndices.size());
                for (uint32_t idx : cpuPassingIndices) {
                    if (idx < vals.size()) compacted.push_back(vals[idx]);
                    else compacted.push_back("");
                }
                vals = std::move(compacted);
            }

            // GPU gather for dict and flat string columns
            dataCtx.compactDictCols(indices, newCount);
            dataCtx.compactFlatStringCols(indices, newCount);
            dataCtx.invalidateStringColsForDictFlat();

            // Update Context
            dataCtx.rowCount = newCount;

            dataCtx.clearActiveRows();

            // Clear CPU vectors to enforce GPU usage
            for(auto& [n, v] : dataCtx.u32Cols) v.clear(); 
            for(auto& [n, v] : dataCtx.f32Cols) v.clear();
        }

        // Switch currentCtx to the filtered data table
        currentCtx = dataCtx;
        joinedTables.clear();
        joinedTables.insert(dataTable);
        hasPipeline = true;

        return true;  // Handled this join
    }
    return false;
}

// -- applyScalarSubqueryCrossJoinFilter --
// Apply a scalar-subquery cross-join filter on currentCtx.
// When condCols contains "SUBQUERY" and the right table has 1 row,
// parse the comparison, extract the scalar value, and GPU-filter in place.
// Returns true if the filter was applied (caller can skip normal join).
bool applyScalarSubqueryCrossJoinFilter(
    const std::set<std::string>& condCols,
    const IRJoin& join,
    EvalContext& currentCtx,
    std::unordered_map<std::string, EvalContext>& tableContexts,
    std::vector<EvalContext>& savedPipelines,
    const std::vector<std::set<std::string>>& savedPipelineTables,
    bool debug)
{
    if (!condCols.count("SUBQUERY") || join.rightTable.empty()) return false;

    // Find the scalar context (1-row right table)
    EvalContext* scalarCtx = nullptr;
    if (tableContexts.count(join.rightTable)) {
        scalarCtx = &tableContexts[join.rightTable];
    }
    if (!scalarCtx) {
        for (auto& sp : savedPipelines) {
            if (savedPipelineTables[&sp - &savedPipelines[0]].count(join.rightTable)) {
                scalarCtx = &sp;
                break;
            }
        }
    }
    if (!scalarCtx || scalarCtx->rowCount > 1) return false;

    // Find the scalar value — prefer avg()/first() columns
    float scalarVal = 0.0f;
    bool foundScalar = false;
    for (const auto& [name, vec] : scalarCtx->f32Cols) {
        if (!vec.empty() && (name.find("avg") != std::string::npos ||
                             name.find("first") != std::string::npos)) {
            scalarVal = vec[0];
            foundScalar = true;
            LOG_DEBUG("Exec", "Join: SUBQUERY scalar from '" << name << "' = " << scalarVal);
            break;
        }
    }
    if (!foundScalar) {
        for (const auto& [name, vec] : scalarCtx->f32Cols) {
            if (!vec.empty() && name.find("count") == std::string::npos) {
                scalarVal = vec[0];
                foundScalar = true;
                LOG_DEBUG("Exec", "Join: SUBQUERY scalar from '" << name << "' = " << scalarVal);
                break;
            }
        }
    }
    if (!foundScalar) return false;

    // Parse condition: "CAST(c_acctbal AS DOUBLE) > SUBQUERY" → column + operator
    std::string filterCol;
    std::string filterOp;
    std::string cond = join.conditionStr;

    size_t castPos = cond.find("CAST(");
    if (castPos != std::string::npos) {
        size_t asPos = cond.find(" AS ", castPos);
        if (asPos != std::string::npos)
            filterCol = cond.substr(castPos + 5, asPos - castPos - 5);
    }
    for (const auto& op : {">", "<", ">=", "<=", "="}) {
        size_t opPos2 = cond.find(std::string(" ") + op + " ");
        if (opPos2 != std::string::npos) {
            filterOp = op;
            if (filterCol.empty()) filterCol = base_ident(cond.substr(0, opPos2));
            break;
        }
    }
    if (filterCol.empty() || filterOp.empty()) return false;

    if (debug)
        LOG_INFO("Exec", "Join: SUBQUERY scalar cross-join: " << filterCol << " " << filterOp << " " << scalarVal);

    // Resolve filter column — prefer highest-suffixed version (latest scan instance)
    {
        std::string bestMatch;
        int bestSuffix = -1;
        for (const auto& [n, v] : currentCtx.f32Cols) {
            if (n == filterCol) {
                if (bestSuffix < 0) { bestMatch = n; bestSuffix = 0; }
            } else {
                auto pos = n.rfind('_');
                if (pos != std::string::npos) {
                    std::string sfx = n.substr(pos + 1);
                    if (!sfx.empty() && std::all_of(sfx.begin(), sfx.end(), ::isdigit)
                        && n.substr(0, pos) == filterCol) {
                        int sfxNum = std::stoi(sfx);
                        if (sfxNum > bestSuffix) { bestMatch = n; bestSuffix = sfxNum; }
                    }
                }
            }
        }
        if (bestMatch.empty()) return false;
        filterCol = bestMatch;
        LOG_DEBUG("Exec", "Join: SUBQUERY resolved filterCol to '" << filterCol << "'\n");
    }

    // Map operator string → GpuFilterOp
    engine::GpuFilterOp compOp = engine::GpuFilterOp::EQ;
    if (filterOp == ">")       compOp = engine::GpuFilterOp::GT;
    else if (filterOp == ">=") compOp = engine::GpuFilterOp::GE;
    else if (filterOp == "<")  compOp = engine::GpuFilterOp::LT;
    else if (filterOp == "<=") compOp = engine::GpuFilterOp::LE;

    // Ensure GPU buffer for filter column
    MTL::Buffer* filterColGPU = nullptr;
    if (currentCtx.f32ColsGPU.count(filterCol)) {
        filterColGPU = currentCtx.f32ColsGPU[filterCol];
    } else if (!currentCtx.f32Cols[filterCol].empty()) {
        auto& vec = currentCtx.f32Cols[filterCol];
        currentCtx.f32ColsGPU[filterCol] = GpuOps::createBuffer(vec.data(), vec.size() * sizeof(float));
        filterColGPU = currentCtx.f32ColsGPU[filterCol];
    }
    if (!filterColGPU) return false;

    // Run GPU filter
    std::optional<FilterResult> gpuFilterRes;
    if (currentCtx.activeRowsGPU && currentCtx.activeRowsCountGPU > 0) {
        gpuFilterRes = GpuOps::filterF32Indexed(filterCol, filterColGPU,
                         currentCtx.activeRowsGPU, currentCtx.activeRowsCountGPU, compOp, scalarVal);
    } else if (!currentCtx.activeRows.empty()) {
        GpuBuffer arGPU = GpuOps::createBuffer(currentCtx.activeRows.data(),
                                 currentCtx.activeRows.size() * sizeof(uint32_t));
        uint32_t arCount = (uint32_t)currentCtx.activeRows.size();
        gpuFilterRes = GpuOps::filterF32Indexed(filterCol, filterColGPU,
                         arGPU, arCount, compOp, scalarVal);
    } else {
        uint32_t fullRowCount = (uint32_t)currentCtx.f32Cols[filterCol].size();
        if (fullRowCount == 0) fullRowCount = currentCtx.rowCount;
        gpuFilterRes = GpuOps::filterF32(filterCol, filterColGPU, fullRowCount, compOp, scalarVal);
    }
    if (!gpuFilterRes) return false;

    // Apply filter results — gather all columns to keep only matching rows
    MTL::Buffer* keepIndicesGPU = gpuFilterRes->indices;
    uint32_t keepCount = gpuFilterRes->count;

    LOG_DEBUG("Exec", "Join: SUBQUERY GPU scalar filter: " << keepCount << " rows after\n");

    // GPU gather for u32/f32 columns
    for (auto& [name, buf] : currentCtx.u32ColsGPU) {
        if (buf) { buf = GpuOps::gatherU32(buf, keepIndicesGPU, keepCount, false); }
    }
    for (auto& [name, buf] : currentCtx.f32ColsGPU) {
        if (buf) { buf = GpuOps::gatherF32(buf, keepIndicesGPU, keepCount, false); }
    }
    GpuOps::sync();

    // CPU gather for string columns without dict/flat representation
    std::vector<uint32_t> keepIdx(keepCount);
    if (keepCount > 0) memcpy(keepIdx.data(), keepIndicesGPU->contents(), keepCount * sizeof(uint32_t));
    for (auto& [name, vec] : currentCtx.stringCols) {
        if (!vec.empty() && !currentCtx.dictCols.count(name) && !currentCtx.flatStringCols.count(name)) {
            std::vector<std::string> compact(keepCount);
            for (uint32_t i = 0; i < keepCount; ++i) compact[i] = vec[keepIdx[i]];
            vec = std::move(compact);
        }
    }

    // GPU gather for dict and flat string columns
    currentCtx.compactDictCols(keepIndicesGPU, keepCount);
    currentCtx.compactFlatStringCols(keepIndicesGPU, keepCount);
    currentCtx.invalidateStringColsForDictFlat();

    // Skip CPU download when GPU buffer exists — lazy-fetch later
    for (auto& [name, vec] : currentCtx.u32Cols) {
        if (currentCtx.u32ColsGPU.count(name) && currentCtx.u32ColsGPU[name]) {
            vec.clear();
        } else if (!vec.empty()) {
            std::vector<uint32_t> compact(keepCount);
            for (uint32_t i = 0; i < keepCount; ++i) compact[i] = vec[keepIdx[i]];
            vec = std::move(compact);
        }
    }
    for (auto& [name, vec] : currentCtx.f32Cols) {
        if (currentCtx.f32ColsGPU.count(name) && currentCtx.f32ColsGPU[name]) {
            vec.clear();
        } else if (!vec.empty()) {
            std::vector<float> compact(keepCount);
            for (uint32_t i = 0; i < keepCount; ++i) compact[i] = vec[keepIdx[i]];
            vec = std::move(compact);
        }
    }

    currentCtx.activeRows.clear();
    currentCtx.activeRowsGPU = nullptr;
    currentCtx.rowCount = keepCount;
    return true;
}

} // namespace engine
