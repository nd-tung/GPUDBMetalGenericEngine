#include "GpuExecutorDetail.hpp"
#include "Operators.hpp"
#include "GpuColumnStore.hpp"
#include <iostream>
#include <map>
#include <set>
#include <algorithm>
#include <vector>
#include <chrono>
#include "Logger.hpp"

namespace engine {

// Helper: flatten a vector<string> into pre-computed Metal GPU buffers
// and store in ctx.flatStringCols for zero-copy filter dispatch.
void flattenStringCol(EvalContext& ctx, const std::string& colName) {
    auto it = ctx.stringCols.find(colName);
    if (it == ctx.stringCols.end() || it->second.empty()) return;

    auto& store = GpuColumnStore::instance();
    if (!store.device()) return;

    const auto& data = it->second;
    uint32_t rowCount = static_cast<uint32_t>(data.size());

    std::vector<uint32_t> offsets(rowCount);
    std::vector<uint32_t> lengths(rowCount);
    size_t totalChars = 0;
    for (const auto& s : data) totalChars += s.size();

    // Guard: refuse to flatten if total chars would overflow uint32_t offsets
    if (totalChars > (size_t)UINT32_MAX) return;

    std::vector<char> chars;
    chars.reserve(totalChars);

    size_t currentOffset = 0;
    for (size_t i = 0; i < rowCount; ++i) {
        offsets[i] = static_cast<uint32_t>(currentOffset);
        lengths[i] = static_cast<uint32_t>(data[i].size());
        chars.insert(chars.end(), data[i].begin(), data[i].end());
        currentOffset += data[i].size();
    }

    FlatStringCol flat;
    flat.rowCount   = rowCount;
    flat.totalBytes = static_cast<uint32_t>(totalChars);

    if (!chars.empty())
        flat.chars.reset(store.device()->newBuffer(chars.data(), chars.size(), MTL::ResourceStorageModeShared));
    else
        flat.chars.reset(store.device()->newBuffer(1, MTL::ResourceStorageModeShared));

    flat.offsets.reset(store.device()->newBuffer(offsets.data(), offsets.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    flat.lengths.reset(store.device()->newBuffer(lengths.data(), lengths.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared));

    ctx.flatStringCols[colName] = flat;
}

// Build FlatStringCol directly from DictEncoded on GPU — no CPU string materialization.
// Flattens the dictionary (K unique strings) into GPU buffers, then gathers by IDs.
static bool buildFlatFromDict(EvalContext& ctx, const std::string& colName) {
    auto dit = ctx.dictCols.find(colName);
    if (dit == ctx.dictCols.end() || !dit->second.valid() || !dit->second.idsGPU)
        return false;

    auto& store = GpuColumnStore::instance();
    if (!store.device()) return false;

    const auto& dict = dit->second;
    const auto& strings = dict.dictionary;
    uint32_t K = static_cast<uint32_t>(strings.size());
    if (K == 0) return false;

    // Flatten dictionary strings (K unique) into GPU buffers — small CPU work
    std::vector<uint32_t> dOffsets(K);
    std::vector<uint32_t> dLengths(K);
    size_t totalChars = 0;
    for (size_t i = 0; i < K; ++i) totalChars += strings[i].size();

    // Estimate output size and refuse if it would be too large (>2GB)
    // Average string length * N rows → estimated total chars in expanded output
    double avgLen = (K > 0) ? (double)totalChars / K : 0;
    size_t estimatedOutputBytes = (size_t)(avgLen * dict.rowCount);
    if (estimatedOutputBytes > (size_t)2u * 1024 * 1024 * 1024) return false;

    std::vector<char> dChars;
    dChars.reserve(totalChars);
    size_t cur = 0;
    for (size_t i = 0; i < K; ++i) {
        dOffsets[i] = static_cast<uint32_t>(cur);
        dLengths[i] = static_cast<uint32_t>(strings[i].size());
        dChars.insert(dChars.end(), strings[i].begin(), strings[i].end());
        cur += strings[i].size();
    }

    // Upload dictionary flat buffers to GPU
    auto* dev = store.device();
    GpuBuffer dictCharsBuf(dev->newBuffer(
        dChars.empty() ? (const void*)"\0" : dChars.data(),
        std::max(dChars.size(), (size_t)1), MTL::ResourceStorageModeShared));
    GpuBuffer dictOffBuf(dev->newBuffer(
        dOffsets.data(), K * sizeof(uint32_t), MTL::ResourceStorageModeShared));
    GpuBuffer dictLenBuf(dev->newBuffer(
        dLengths.data(), K * sizeof(uint32_t), MTL::ResourceStorageModeShared));

    // GPU gather: expand dictionary flat by per-row IDs → per-row FlatStringCol
    auto result = GpuOps::gatherFlatString(
        dictCharsBuf.get(), dictOffBuf.get(), dictLenBuf.get(),
        dict.idsGPU.get(), dict.rowCount, true);
    if (!result.chars) return false;

    FlatStringCol flat;
    flat.takeFrom(result.chars, result.offsets, result.lengths,
                  result.rowCount, result.totalBytes);
    ctx.flatStringCols[colName] = std::move(flat);
    return true;
}

// EvalContext::ensureFlatStringCol — implemented here because it needs flattenStringCol.
void EvalContext::ensureFlatStringCol(const std::string& colName) {
    if (flatStringCols.count(colName) && flatStringCols[colName].chars) return;
    // Fast path: build directly from DictEncoded on GPU (no CPU string materialization)
    if (buildFlatFromDict(*this, colName)) return;
    // Fallback: materialize strings to CPU, then flatten to GPU
    ensureStringCol(colName);
    if (stringCols.count(colName) && !stringCols[colName].empty()) {
        flattenStringCol(*this, colName);
    }
}

// Helper: compute FNV1a-32 hashes on GPU from flat string buffers.
// Replaces CPU per-row hashing (loadStringHashU32) with GPU batch computation.
static void computeGpuHash(EvalContext& ctx, const std::string& colName) {
    auto it = ctx.flatStringCols.find(colName);
    if (it == ctx.flatStringCols.end()) return;
    auto& flat = it->second;
    if (flat.rowCount == 0) return;

    auto hashBuf = GpuOps::stringFnv1aU32(flat.chars, flat.offsets, flat.lengths, flat.rowCount);
    if (!hashBuf) return;

    ctx.u32ColsGPU[colName] = std::move(hashBuf);
    // Keep empty entry for column name discovery; consumers lazy-fetch from GPU
    ctx.u32Cols[colName] = {};
}

// Helper: build dictionary encoding for a string column.
// Creates sorted unique dictionary + per-row IDs, uploads IDs to GPU.
void buildDictCol(EvalContext& ctx, const std::string& colName) {
    auto it = ctx.stringCols.find(colName);
    if (it == ctx.stringCols.end() || it->second.empty()) return;

    auto& store = GpuColumnStore::instance();
    if (!store.device()) return;

    const auto& data = it->second;
    uint32_t rowCount = static_cast<uint32_t>(data.size());

    // Build sorted unique dictionary — O(N) dedup + O(K log K) sort (K << N)
    std::unordered_map<std::string, uint32_t> fwd;
    fwd.reserve(std::min<size_t>(data.size(), 1u << 20));
    std::vector<std::string> uniq;
    for (const auto& s : data) {
        auto [jt, inserted] = fwd.try_emplace(s, 0u);
        if (inserted) uniq.push_back(s);
    }
    std::sort(uniq.begin(), uniq.end());
    for (uint32_t i = 0; i < static_cast<uint32_t>(uniq.size()); ++i) {
        fwd[uniq[i]] = i;
    }

    // Assign per-row dictionary IDs
    std::vector<uint32_t> ids(rowCount);
    for (uint32_t i = 0; i < rowCount; ++i) {
        ids[i] = fwd[data[i]];
    }

    DictEncoded dict;
    dict.dictionary = std::move(uniq);
    dict.ids = ids;
    dict.rowCount = rowCount;
    dict.idsGPU.reset(store.device()->newBuffer(ids.data(), ids.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared));

    ctx.dictCols[colName] = std::move(dict);
}

std::map<size_t, ScanInstance> buildScanInstanceMap(const Plan& plan) {
    std::map<size_t, ScanInstance> result;
    std::map<std::string, int> tableCounts;
    std::map<std::string, int> tableCurrentInstance;
    
    for (size_t i = 0; i < plan.nodes.size(); ++i) {
        if (plan.nodes[i].type == IRNode::Type::Scan) {
            const auto& scan = plan.nodes[i].asScan();
            if (!scan.table.empty()) {
                tableCounts[scan.table]++;
            }
        }
    }
    
    for (size_t i = 0; i < plan.nodes.size(); ++i) {
        if (plan.nodes[i].type == IRNode::Type::Scan) {
            const auto& scan = plan.nodes[i].asScan();
            if (!scan.table.empty() && tableCounts[scan.table] > 1) {
                int& instNum = tableCurrentInstance[scan.table];
                instNum++;
                
                ScanInstance inst;
                inst.baseTable = scan.table;
                inst.instanceKey = scan.table + "_" + std::to_string(instNum);
                inst.instanceNum = instNum;
                inst.nodeIndex = i;
                result[i] = inst;
            }
        }
    }
    
    return result;
}

std::unordered_map<std::string, std::set<std::string>> collectNeededColumns(const Plan& plan) {
    std::unordered_map<std::string, std::set<std::string>> tableCols;

    auto add = [&](const std::string& col) {
        const std::string c = base_ident(col);
        const std::string t = tableForColumn(c);
        if (!t.empty() && !c.empty()) tableCols[t].insert(c);
    };

    for (const auto& node : plan.nodes) {
        switch (node.type) {
            case IRNode::Type::Scan: {
                const auto& scan = node.asScan();
                for (const auto& col : scan.columns) add(col);
                if (scan.filter) {
                    std::set<std::string> tmp;
                    collectColumnsFromExpr(scan.filter, tmp);
                    for (const auto& c : tmp) add(c);
                }
                break;
            }
            case IRNode::Type::Project: {
                for (const auto& e : node.asProject().exprs) {
                    std::set<std::string> tmp;
                    collectColumnsFromExpr(e, tmp);
                    for (const auto& c : tmp) add(c);
                }
                break;
            }
            case IRNode::Type::Filter: {
                std::set<std::string> tmp;
                collectColumnsFromExpr(node.asFilter().predicate, tmp);
                for (const auto& c : tmp) add(c);
                break;
            }
            case IRNode::Type::Join: {
                std::set<std::string> tmp;
                collectColumnsFromExpr(node.asJoin().condition, tmp);
                for (const auto& k : node.asJoin().leftKeys) collectColumnsFromExpr(k, tmp);
                for (const auto& k : node.asJoin().rightKeys) collectColumnsFromExpr(k, tmp);
                collectColumnsFromExpr(node.asJoin().rightFilter, tmp);
                
                if (env_truthy("GPUDB_DEBUG_OPS")) {
                     LOG_INFO("Exec", "DEBUG: Join collected cols:");
                     for(const auto& c : tmp) std::cerr << " " << c;
                     LOG_INFO("SCAN", "\n");
                }

                for (const auto& c : tmp) add(c);
                break;
            }
            case IRNode::Type::GroupBy: {
                const auto& gb = node.asGroupBy();
                for (const auto& k : gb.keys) {
                    std::set<std::string> tmp;
                    collectColumnsFromExpr(k, tmp);
                    for (const auto& c : tmp) add(c);
                }
                for (const auto& agg : gb.aggregates) {
                    std::set<std::string> tmp;
                    collectColumnsFromExpr(agg, tmp);
                    for (const auto& c : tmp) add(c);
                }
                break;
            }
            case IRNode::Type::Aggregate: {
                std::set<std::string> tmp;
                collectColumnsFromExpr(node.asAggregate().expr, tmp);
                for (const auto& c : tmp) add(c);
                break;
            }
            case IRNode::Type::OrderBy: {
                for (const auto& spec : node.asOrderBy().specs) {
                    std::set<std::string> tmp;
                    collectColumnsFromExpr(spec.expr, tmp);
                    for (const auto& c : tmp) add(c);
                }
                break;
            }
            default:
                break;
        }
    }
    return tableCols;
}

void IRGpuLoader::loadTables(
    const std::unordered_map<std::string, std::set<std::string>>& tableColsMap,
    const std::map<size_t, ScanInstance>& scanInstanceMap,
    const std::string& datasetPath,
    std::unordered_map<std::string, EvalContext>& tableContexts,
    GpuExecutor::ExecutionResult& result,
    bool debug
) {
    auto start_load = std::chrono::high_resolution_clock::now();
    std::set<std::string> multiInstanceTables;
    for (const auto& [nodeIdx, inst] : scanInstanceMap) {
        multiInstanceTables.insert(inst.baseTable);
    }

    for (const auto& [tableName, cols] : tableColsMap) {
        if (debug) {
            LOG_INFO("Exec", "DEBUG: Loading table " << tableName << " with cols:");
            for(const auto& c : cols) std::cerr << " " << c;
            LOG_INFO("SCAN", "\n");
        }
        std::vector<std::string> colVec(cols.begin(), cols.end());
        
        if (multiInstanceTables.count(tableName)) {
            for (const auto& [nodeIdx, inst] : scanInstanceMap) {
                if (inst.baseTable == tableName) {
                    GpuRelation rel = GpuOps::scanTable(datasetPath, tableName, colVec);

                    EvalContext ctx;
                    ctx.currentTable = inst.instanceKey;
                    ctx.rowCount = rel.rowCount;

                    for (const auto& [name, buf] : rel.u32cols) {
                        if (inst.instanceNum == 1) {
                            ctx.u32Cols[name] = {};  // empty sentinel for column name discovery
                            ctx.u32ColsGPU[name] = buf;  // GpuBuffer copy auto-retains
                        }
                        ctx.u32Cols[name + "_" + std::to_string(inst.instanceNum)] = {};  // empty sentinel
                        ctx.u32ColsGPU[name + "_" + std::to_string(inst.instanceNum)] = buf;  // GpuBuffer copy auto-retains
                    }
                    for (const auto& [name, buf] : rel.f32cols) {
                        if (inst.instanceNum == 1) {
                            ctx.f32Cols[name] = {};  // empty sentinel for column name discovery
                            ctx.f32ColsGPU[name] = buf;  // GpuBuffer copy auto-retains
                        }
                        ctx.f32Cols[name + "_" + std::to_string(inst.instanceNum)] = {};  // empty sentinel
                        ctx.f32ColsGPU[name + "_" + std::to_string(inst.instanceNum)] = buf;  // GpuBuffer copy auto-retains
                    }
                    
                    // Always load raw strings + flatten for ALL StringHash columns
                    {
                        const auto& schema = SchemaRegistry::instance();
                        const auto* tblSchema = schema.getTable(tableName);
                        if (tblSchema) {
                            for (const auto& colName : cols) {
                                if (tblSchema->getColumnType(colName) == ColumnType::StringHash) {
                                    auto rawStrings = GpuOps::loadStringColumnRaw(datasetPath, tableName, colName);
                                    if (!rawStrings.empty()) {
                                        // Determine rowCount from string data when no numeric cols loaded
                                        if (!ctx.rowCount) ctx.rowCount = rawStrings.size();
                                        if (inst.instanceNum == 1) {
                                            ctx.stringCols[colName] = rawStrings;
                                            flattenStringCol(ctx, colName);
                                            computeGpuHash(ctx, colName);
                                            buildDictCol(ctx, colName);
                                        }
                                        std::string suffixed = colName + "_" + std::to_string(inst.instanceNum);
                                        ctx.stringCols[suffixed] = std::move(rawStrings);
                                        flattenStringCol(ctx, suffixed);
                                        computeGpuHash(ctx, suffixed);
                                        buildDictCol(ctx, suffixed);
                                    }
                                }
                            }
                        }
                    }

                    if (!ctx.rowCount) {
                        result.error = "Failed to load table: " + tableName;
                        return;
                    }

                    tableContexts[inst.instanceKey] = std::move(ctx);
                    
                    if (debug) {
                        LOG_INFO("Exec", "Loaded instance " << inst.instanceKey  << " (" << ctx.rowCount << " rows)\n");
                    }
                }
            }
        } else {
            GpuRelation rel = GpuOps::scanTable(datasetPath, tableName, colVec);

            EvalContext ctx;
            ctx.currentTable = tableName;
            ctx.rowCount = rel.rowCount;

            for (const auto& [name, buf] : rel.u32cols) {
                ctx.u32Cols[name] = {};  // empty sentinel for column name discovery
                ctx.u32ColsGPU[name] = buf;  // GpuBuffer copy auto-retains
            }
            for (const auto& [name, buf] : rel.f32cols) {
                ctx.f32Cols[name] = {};  // empty sentinel for column name discovery
                ctx.f32ColsGPU[name] = buf;  // GpuBuffer copy auto-retains
            }
            
            // Always load raw strings + flatten for ALL StringHash columns
            {
                const auto& schema = SchemaRegistry::instance();
                const auto* tblSchema = schema.getTable(tableName);
                if (tblSchema) {
                    for (const auto& colName : cols) {
                        if (tblSchema->getColumnType(colName) == ColumnType::StringHash) {
                            auto rawStrings = GpuOps::loadStringColumnRaw(datasetPath, tableName, colName);
                            if (!rawStrings.empty()) {
                                // Determine rowCount from string data when no numeric cols loaded
                                if (!ctx.rowCount) ctx.rowCount = rawStrings.size();
                                ctx.stringCols[colName] = std::move(rawStrings);
                                flattenStringCol(ctx, colName);
                                computeGpuHash(ctx, colName);
                                buildDictCol(ctx, colName);
                                if (debug) {
                                    LOG_INFO("Exec", "Loaded raw strings for " << tableName << "." << colName  << " (" << ctx.stringCols[colName].size() << " rows, flat=" << ctx.flatStringCols.count(colName) << ")\n");
                                }
                            }
                        }
                    }
                }
            }

            if (!ctx.rowCount) {
                result.error = "Failed to load table: " + tableName;
                return;
            }

            tableContexts[tableName] = std::move(ctx);
        }
    }
    
    auto end_load = std::chrono::high_resolution_clock::now();
    result.table.uploadMs = std::chrono::duration<double, std::milli>(end_load - start_load).count();
}

} // namespace engine
