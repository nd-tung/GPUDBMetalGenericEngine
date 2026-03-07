#include "GpuExecutorDetail.hpp"
#include "Operators.hpp"
#include "GpuColumnStore.hpp"
#include <iostream>
#include <map>
#include <set>
#include <algorithm>
#include <vector>
#include <chrono>

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

// EvalContext::ensureFlatStringCol — implemented here because it needs flattenStringCol.
void EvalContext::ensureFlatStringCol(const std::string& colName) {
    if (flatStringCols.count(colName) && flatStringCols[colName].chars) return;
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

    ctx.u32ColsGPU[colName].reset(hashBuf);
    // Download to CPU for compatibility with CPU-side consumers
    std::vector<uint32_t> hashCPU(flat.rowCount);
    memcpy(hashCPU.data(), hashBuf->contents(), flat.rowCount * sizeof(uint32_t));
    ctx.u32Cols[colName] = std::move(hashCPU);
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

    // Build sorted unique dictionary
    std::vector<std::string> uniq(data.begin(), data.end());
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

    // Build forward map: string -> dict ID (0-based)
    std::unordered_map<std::string, uint32_t> fwd;
    fwd.reserve(uniq.size());
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
                     std::cerr << "[Exec] DEBUG: Join collected cols:";
                     for(const auto& c : tmp) std::cerr << " " << c;
                     std::cerr << "\n";
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
            std::cerr << "[Exec] DEBUG: Loading table " << tableName << " with cols:";
            for(const auto& c : cols) std::cerr << " " << c;
            std::cerr << "\n";
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
                        std::vector<uint32_t> data(rel.rowCount);
                        const uint32_t* ptr = static_cast<const uint32_t*>(buf->contents());
                        std::copy(ptr, ptr + rel.rowCount, data.begin());
                        if (inst.instanceNum == 1) {
                            ctx.u32Cols[name] = data;
                            ctx.u32ColsGPU[name] = buf;  // GpuBuffer copy auto-retains
                        }
                        ctx.u32Cols[name + "_" + std::to_string(inst.instanceNum)] = std::move(data);
                        ctx.u32ColsGPU[name + "_" + std::to_string(inst.instanceNum)] = buf;  // GpuBuffer copy auto-retains
                    }
                    for (const auto& [name, buf] : rel.f32cols) {
                        std::vector<float> data(rel.rowCount);
                        const float* ptr = static_cast<const float*>(buf->contents());
                        std::copy(ptr, ptr + rel.rowCount, data.begin());
                        if (inst.instanceNum == 1) {
                            ctx.f32Cols[name] = data;
                            ctx.f32ColsGPU[name] = buf;  // GpuBuffer copy auto-retains
                        }
                        ctx.f32Cols[name + "_" + std::to_string(inst.instanceNum)] = std::move(data);
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
                        std::cerr << "[Exec] Loaded instance " << inst.instanceKey 
                                  << " (" << ctx.rowCount << " rows)\n";
                    }
                }
            }
        } else {
            GpuRelation rel = GpuOps::scanTable(datasetPath, tableName, colVec);

            EvalContext ctx;
            ctx.currentTable = tableName;
            ctx.rowCount = rel.rowCount;

            for (const auto& [name, buf] : rel.u32cols) {
                std::vector<uint32_t> data(rel.rowCount);
                const uint32_t* ptr = static_cast<const uint32_t*>(buf->contents());
                std::copy(ptr, ptr + rel.rowCount, data.begin());
                ctx.u32Cols[name] = std::move(data);
                ctx.u32ColsGPU[name] = buf;  // GpuBuffer copy auto-retains
            }
            for (const auto& [name, buf] : rel.f32cols) {
                std::vector<float> data(rel.rowCount);
                const float* ptr = static_cast<const float*>(buf->contents());
                std::copy(ptr, ptr + rel.rowCount, data.begin());
                ctx.f32Cols[name] = std::move(data);
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
                                    std::cerr << "[Exec] Loaded raw strings for " << tableName << "." << colName 
                                              << " (" << ctx.stringCols[colName].size() << " rows, flat="
                                              << ctx.flatStringCols.count(colName) << ")\n";
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
