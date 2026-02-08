#include "GpuExecutorPriv.hpp"
#include "Operators.hpp"
#include "ColumnStoreGPU.hpp"
#include <iostream>
#include <map>
#include <set>
#include <algorithm>
#include <vector>
#include <chrono>

namespace engine {

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
                
                if (std::getenv("GPUDB_DEBUG_OPS")) {
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

static void collectPatternMatchColumns(const TypedExprPtr& expr, 
                                       std::unordered_map<std::string, std::set<std::string>>& tableCols,
                                       const std::unordered_map<std::string, std::string>& aliasMap) {
    if (!expr) return;
    
    if (expr->kind == TypedExpr::Kind::Function) {
        const auto& func = expr->asFunction();
        if ((func.name == "LIKE" || func.name == "NOTLIKE" || func.name == "CONTAINS" ||
             func.name == "PREFIX" || func.name == "SUFFIX" ||
             func.name == "SUBSTRING" || func.name == "SUBSTR" ||
             func.name == "substring" || func.name == "substr") && func.args.size() >= 1) {
            if (func.args[0] && func.args[0]->kind == TypedExpr::Kind::Column) {
                std::string colName = func.args[0]->asColumn().column;
                // Resolve alias
                if (aliasMap.count(colName)) colName = aliasMap.at(colName);
                
                std::string table = tableForColumn(colName);
                if (!table.empty()) {
                    tableCols[table].insert(colName);
                }
            }
        }
        for (const auto& arg : func.args) {
            collectPatternMatchColumns(arg, tableCols, aliasMap);
        }
    } else if (expr->kind == TypedExpr::Kind::Binary) {
        collectPatternMatchColumns(expr->asBinary().left, tableCols, aliasMap);
        collectPatternMatchColumns(expr->asBinary().right, tableCols, aliasMap);
    } else if (expr->kind == TypedExpr::Kind::Unary) {
        collectPatternMatchColumns(expr->asUnary().operand, tableCols, aliasMap);
    } else if (expr->kind == TypedExpr::Kind::Compare) {
        const auto& cmp = expr->asCompare();
        
        // Detect String Comparison (Column = StringLiteral)
        bool isStringComp = false;
        if ((cmp.right && cmp.right->kind == TypedExpr::Kind::Literal && 
             std::holds_alternative<std::string>(cmp.right->asLiteral().value)) ||
            (cmp.left && cmp.left->kind == TypedExpr::Kind::Literal && 
             std::holds_alternative<std::string>(cmp.left->asLiteral().value))) {
            isStringComp = true;
        }

        // Detect IN list with Strings
        if (cmp.op == CompareOp::In && !cmp.inList.empty()) {
            for (const auto& item : cmp.inList) {
                if (item && item->kind == TypedExpr::Kind::Literal && 
                    std::holds_alternative<std::string>(item->asLiteral().value)) {
                    isStringComp = true;
                    break;
                }
            }
        }

        if (isStringComp) {
            // If string comparison, mark columns for raw string loading
            auto registerCol = [&](const TypedExprPtr& e) {
                if (e && e->kind == TypedExpr::Kind::Column) {
                    std::string colName = e->asColumn().column;
                    // Resolve alias
                    if (aliasMap.count(colName)) colName = aliasMap.at(colName);

                    std::string table = tableForColumn(colName);
                    if (std::getenv("GPUDB_DEBUG_OPS")) {
                        std::cerr << "[Exec] DEBUG: Found string comparison col " << colName << " table=" << table << "\n";
                    }
                    if (!table.empty()) {
                        tableCols[table].insert(colName);
                    }
                }
            };
            registerCol(cmp.left);
            registerCol(cmp.right);
        }

        collectPatternMatchColumns(cmp.left, tableCols, aliasMap);
        collectPatternMatchColumns(cmp.right, tableCols, aliasMap);
    } else if (expr->kind == TypedExpr::Kind::Case) {
        const auto& caseExpr = expr->asCase();
        for (const auto& whenClause : caseExpr.cases) {
            collectPatternMatchColumns(whenClause.when, tableCols, aliasMap);
            collectPatternMatchColumns(whenClause.then, tableCols, aliasMap);
        }
        collectPatternMatchColumns(caseExpr.elseExpr, tableCols, aliasMap);
    } else if (expr->kind == TypedExpr::Kind::Aggregate) {
        collectPatternMatchColumns(expr->asAggregate().arg, tableCols, aliasMap);
    } else if (expr->kind == TypedExpr::Kind::Cast) {
        collectPatternMatchColumns(expr->asCast().expr, tableCols, aliasMap);
    } else if (expr->kind == TypedExpr::Kind::Alias) {
        collectPatternMatchColumns(expr->asAlias().expr, tableCols, aliasMap);
    } else if (expr->kind == TypedExpr::Kind::Column) {
        std::string colName = expr->asColumn().column;
        if (aliasMap.count(colName)) colName = aliasMap.at(colName);

        // Handle mangled function call in column name
        if (colName.rfind("NOT prefix(", 0) == 0) {
            // "NOT prefix(p_type, ...)"
            size_t comma = colName.find(',');
            if (comma != std::string::npos && comma > 11) {
                std::string c = colName.substr(11, comma - 11);
                // trim
                c.erase(0, c.find_first_not_of(" "));
                c.erase(c.find_last_not_of(" ") + 1);
                
                if (aliasMap.count(c)) c = aliasMap.at(c);
                std::string table = tableForColumn(c);
                if (!table.empty()) {
                    tableCols[table].insert(c);
                    if (std::getenv("GPUDB_DEBUG_OPS")) {
                        std::cerr << "[Exec] DEBUG: Found mangled prefix comparison col " << c << " table=" << table << "\n";
                    }
                }
            }
        }
    }
}

std::unordered_map<std::string, std::set<std::string>> collectPatternMatchColumns(const Plan& plan) {
    std::unordered_map<std::string, std::set<std::string>> tableCols;
    std::unordered_map<std::string, std::string> aliasMap;
    
    for (const auto& node : plan.nodes) {
        switch (node.type) {
            case IRNode::Type::Scan:
                if (node.asScan().filter) {
                    collectPatternMatchColumns(node.asScan().filter, tableCols, aliasMap);
                }
                break;
            case IRNode::Type::Filter:
                collectPatternMatchColumns(node.asFilter().predicate, tableCols, aliasMap);
                break;
            case IRNode::Type::Project:
                for (size_t i = 0; i < node.asProject().exprs.size(); ++i) {
                    const auto& expr = node.asProject().exprs[i];
                    std::string outName = i < node.asProject().outputNames.size() ? node.asProject().outputNames[i] : "";
                    
                    collectPatternMatchColumns(expr, tableCols, aliasMap);
                    
                    // Track aliases: outName -> underlying column
                    if (!outName.empty() && expr->kind == TypedExpr::Kind::Column) {
                        std::string src = expr->asColumn().column;
                        if (aliasMap.count(src)) src = aliasMap.at(src);
                        aliasMap[outName] = src;
                        if (std::getenv("GPUDB_DEBUG_OPS")) {
                            std::cerr << "[Exec] DEBUG: Tracked pattern alias " << outName << " -> " << src << "\n";
                        }
                    }
                }
                break;
            case IRNode::Type::Join: {
                const auto& join = node.asJoin();
                // Check condition
                collectPatternMatchColumns(join.condition, tableCols, aliasMap);
                // Check right filter (pushed down predicates)
                if (join.rightFilter) {
                    collectPatternMatchColumns(join.rightFilter, tableCols, aliasMap);
                }
                // Check keys
                for (const auto& k : join.leftKeys) collectPatternMatchColumns(k, tableCols, aliasMap);
                for (const auto& k : join.rightKeys) collectPatternMatchColumns(k, tableCols, aliasMap);
                break;
            }
            case IRNode::Type::Aggregate: {
                const auto& agg = node.asAggregate();
                collectPatternMatchColumns(agg.expr, tableCols, aliasMap);
                break;
            }
            case IRNode::Type::GroupBy: {
                const auto& gb = node.asGroupBy();
                for (const auto& k : gb.keys) {
                    collectPatternMatchColumns(k, tableCols, aliasMap);
                    
                    // Check if GROUP BY key is a string column - need raw strings for output
                    if (k && k->kind == TypedExpr::Kind::Column) {
                        std::string colName = k->asColumn().column;
                        // Resolve alias
                        if (aliasMap.count(colName)) colName = aliasMap.at(colName);
                        
                        std::string table = tableForColumn(colName);
                        if (!table.empty()) {
                            // Check if this column is a StringHash type in schema
                            const auto& schema = SchemaRegistry::instance();
                            const auto* tbl = schema.getTable(table);
                            if (tbl) {
                                auto colType = tbl->getColumnType(colName);
                                if (colType == ColumnType::StringHash) {
                                    // Need raw strings for this GROUP BY key
                                    tableCols[table].insert(colName);
                                    if (std::getenv("GPUDB_DEBUG_OPS")) {
                                        std::cerr << "[Exec] DEBUG: GroupBy key " << colName << " needs raw strings for output\n";
                                    }
                                }
                            }
                        }
                    }
                }
                for (const auto& agg : gb.aggregates) collectPatternMatchColumns(agg, tableCols, aliasMap);
                break;
            }
            case IRNode::Type::OrderBy: {
                const auto& ob = node.asOrderBy();
                for (const auto& spec : ob.specs) {
                    collectPatternMatchColumns(spec.expr, tableCols, aliasMap);
                    // ORDER BY on a StringHash column needs raw strings for lexicographic sort
                    if (spec.expr && spec.expr->kind == TypedExpr::Kind::Column) {
                        std::string colName = spec.expr->asColumn().column;
                        if (aliasMap.count(colName)) colName = aliasMap.at(colName);
                        std::string table = tableForColumn(colName);
                        if (!table.empty()) {
                            const auto& schema = SchemaRegistry::instance();
                            const auto* tbl = schema.getTable(table);
                            if (tbl) {
                                auto colType = tbl->getColumnType(colName);
                                if (colType == ColumnType::StringHash) {
                                    tableCols[table].insert(colName);
                                }
                            }
                        }
                    }
                }
                // Also check simple column names list
                for (const auto& colName : ob.columns) {
                    std::string resolved = colName;
                    if (aliasMap.count(resolved)) resolved = aliasMap.at(resolved);
                    std::string table = tableForColumn(resolved);
                    if (!table.empty()) {
                        const auto& schema = SchemaRegistry::instance();
                        const auto* tbl = schema.getTable(table);
                        if (tbl) {
                            auto colType = tbl->getColumnType(resolved);
                            if (colType == ColumnType::StringHash) {
                                tableCols[table].insert(resolved);
                            }
                        }
                    }
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
    const std::unordered_map<std::string, std::set<std::string>>& patternMatchCols,
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
                    RelationGPU rel = GpuOps::scanTable(datasetPath, tableName, colVec);
                    if (!rel.rowCount) {
                        result.error = "Failed to load table: " + tableName;
                        return;
                    }

                    EvalContext ctx;
                    ctx.currentTable = inst.instanceKey;
                    ctx.rowCount = rel.rowCount;

                    for (const auto& [name, buf] : rel.u32cols) {
                        std::vector<uint32_t> data(rel.rowCount);
                        const uint32_t* ptr = static_cast<const uint32_t*>(buf->contents());
                        std::copy(ptr, ptr + rel.rowCount, data.begin());
                        if (inst.instanceNum == 1) {
                            ctx.u32Cols[name] = data;
                            ctx.u32ColsGPU[name] = buf;
                            buf->retain();
                        }
                        ctx.u32Cols[name + "_" + std::to_string(inst.instanceNum)] = std::move(data);
                        ctx.u32ColsGPU[name + "_" + std::to_string(inst.instanceNum)] = buf;
                        buf->retain(); 
                    }
                    for (const auto& [name, buf] : rel.f32cols) {
                        std::vector<float> data(rel.rowCount);
                        const float* ptr = static_cast<const float*>(buf->contents());
                        std::copy(ptr, ptr + rel.rowCount, data.begin());
                        if (inst.instanceNum == 1) {
                            ctx.f32Cols[name] = data;
                            ctx.f32ColsGPU[name] = buf;
                            buf->retain();
                        }
                        ctx.f32Cols[name + "_" + std::to_string(inst.instanceNum)] = std::move(data);
                        ctx.f32ColsGPU[name + "_" + std::to_string(inst.instanceNum)] = buf;
                        buf->retain();
                    }
                    
                    auto pmIt = patternMatchCols.find(tableName);
                    if (pmIt != patternMatchCols.end()) {
                        for (const auto& colName : pmIt->second) {
                            std::string actualCol = colName;
                            if (tableName == "nation" && colName == "nation") actualCol = "n_name";
                            
                            auto rawStrings = GpuOps::loadStringColumnRaw(datasetPath, tableName, actualCol);
                            if (!rawStrings.empty()) {
                                // Build flat Arrow-style column for GPU use
                                auto& cstore = ColumnStoreGPU::instance();
                                if (inst.instanceNum == 1) {
                                    ctx.stringCols[colName] = rawStrings;
                                    ctx.flatStringColsGPU[colName] = makeFlatStringColumn(cstore.device(), rawStrings);
                                }
                                std::string suffixed = colName + "_" + std::to_string(inst.instanceNum);
                                ctx.stringCols[suffixed] = std::move(rawStrings);
                                ctx.flatStringColsGPU[suffixed] = makeFlatStringColumn(cstore.device(), ctx.stringCols[suffixed]);
                            }
                        }
                    }

                    tableContexts[inst.instanceKey] = std::move(ctx);
                    
                    if (debug) {
                        std::cerr << "[Exec] Loaded instance " << inst.instanceKey 
                                  << " (" << rel.rowCount << " rows)\n";
                    }
                }
            }
        } else {
            RelationGPU rel = GpuOps::scanTable(datasetPath, tableName, colVec);
            if (!rel.rowCount) {
                result.error = "Failed to load table: " + tableName;
                return;
            }

            EvalContext ctx;
            ctx.currentTable = tableName;
            ctx.rowCount = rel.rowCount;

            for (const auto& [name, buf] : rel.u32cols) {
                std::vector<uint32_t> data(rel.rowCount);
                const uint32_t* ptr = static_cast<const uint32_t*>(buf->contents());
                std::copy(ptr, ptr + rel.rowCount, data.begin());
                ctx.u32Cols[name] = std::move(data);
                ctx.u32ColsGPU[name] = buf;
                buf->retain();
            }
            for (const auto& [name, buf] : rel.f32cols) {
                std::vector<float> data(rel.rowCount);
                const float* ptr = static_cast<const float*>(buf->contents());
                std::copy(ptr, ptr + rel.rowCount, data.begin());
                ctx.f32Cols[name] = std::move(data);
                ctx.f32ColsGPU[name] = buf;
                buf->retain();
            }
            
            auto pmIt = patternMatchCols.find(tableName);
            if (pmIt != patternMatchCols.end()) {
                for (const auto& colName : pmIt->second) {
                    std::string actualCol = colName;
                    if (tableName == "nation" && colName == "nation") actualCol = "n_name";

                    auto rawStrings = GpuOps::loadStringColumnRaw(datasetPath, tableName, actualCol);
                    if (!rawStrings.empty()) {
                        ctx.stringCols[colName] = std::move(rawStrings);
                        // Build flat Arrow-style column for GPU use
                        auto& cstore = ColumnStoreGPU::instance();
                        ctx.flatStringColsGPU[colName] = makeFlatStringColumn(cstore.device(), ctx.stringCols[colName]);
                        if (debug) {
                            std::cerr << "[Exec] Loaded raw strings + flat GPU column for " << tableName << "." << colName 
                                      << " (" << ctx.stringCols[colName].size() << " rows, "
                                      << ctx.flatStringColsGPU[colName].totalChars << " chars)\n";
                        }
                    }
                }
            }

            tableContexts[tableName] = std::move(ctx);
        }
    }
    
    auto end_load = std::chrono::high_resolution_clock::now();
    result.table.upload_ms = std::chrono::duration<double, std::milli>(end_load - start_load).count();
}

} // namespace engine
