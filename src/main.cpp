#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <sstream>
#include "IR.hpp"
#include "Planner.hpp"
#include "GpuExecutor.hpp"
#include "Schema.hpp"
#include "DuckDBAdapter.hpp"
#include "KernelTimer.hpp"
#include "EnvUtil.hpp"

static int executeQuery(const std::string& sql, const std::string& datasetPath) {
    using namespace engine;
    std::cout << "--- Running (Engine Host) ---" << std::endl;

    // Initialize schema registry for TPC-H
    SchemaRegistry::instance().initTPCH();

    // V2 Planner: Full SQL support (Q1-Q22)
    auto t_plan_start = std::chrono::high_resolution_clock::now();
    Plan plan = Planner::fromSQL(sql);
    auto t_plan_end = std::chrono::high_resolution_clock::now();
    double plan_ms = std::chrono::duration<double, std::milli>(t_plan_end - t_plan_start).count();
    
    if (env_truthy("GPUDB_DEBUG_PLAN")) {
        std::cerr << "[Exec] Plan nodes: " << plan.nodes.size() << "\n";
        for (size_t i = 0; i < plan.nodes.size(); ++i) {
            const auto& n = plan.nodes[i];
            std::cerr << "  [" << i << "] ";
            switch (n.type) {
                case IRNode::Type::Scan: 
                    std::cerr << "Scan table=" << n.asScan().table; 
                    if (n.asScan().filter) std::cerr << " [HAS_FILTER]";
                    break;
                case IRNode::Type::Filter: std::cerr << "Filter pred=" << n.asFilter().predicateStr; break;
                case IRNode::Type::Join: std::cerr << "Join type=" << joinTypeName(n.asJoin().type) << " cond=" << n.asJoin().conditionStr; break;
                case IRNode::Type::GroupBy: std::cerr << "GroupBy keys=" << n.asGroupBy().keys.size() 
                                                       << " aggs=" << n.asGroupBy().aggSpecs.size(); break;
                case IRNode::Type::Aggregate: std::cerr << "Aggregate " << n.asAggregate().exprStr; break;
                case IRNode::Type::OrderBy: std::cerr << "OrderBy cols=" << n.asOrderBy().columns.size(); break;
                case IRNode::Type::Limit: std::cerr << "Limit " << n.asLimit().count; break;
                case IRNode::Type::Project: std::cerr << "Project cols=" << n.asProject().exprs.size(); break;
                default: std::cerr << "Unknown"; break;
            }
            std::cerr << "\n";
        }
    }
    
    if (!plan.isValid()) {
        std::cerr << "[Exec] Plan parse error: " << plan.parseError << std::endl;
        return 1;
    }
    
    // Execute with V2 executor (uses GPU Native Executor)
    std::cout << "[Main] Using GpuExecutor generic executor.\n";
    auto t_exec_start = std::chrono::high_resolution_clock::now();
    auto result = GpuExecutor::execute(plan, datasetPath);
    auto t_exec_end = std::chrono::high_resolution_clock::now();
    double exec_ms = std::chrono::duration<double, std::milli>(t_exec_end - t_exec_start).count();
    
    if (!result.success) {
        std::cerr << "[Native] Execution failed: " << result.error << std::endl;
        return 1;
    }
    
    // Print results
    std::cout << "---------------------------------------------------" << std::endl;
    std::cout << "======================RESULT=======================" << std::endl;
    std::cout << "---------------------------------------------------" << std::endl;
    
    if (result.isScalarAggregate) {
        std::cout << "Scalar " << result.scalarName << ": " 
                  << std::fixed << std::setprecision(2) << result.scalarValue << std::endl;
    } else {
        const auto& t = result.table;

        // --- Value rendering helpers ---
        auto printU32 = [&](const std::string& name, uint32_t v) {
            if (t.singleCharCols.count(name)) {
                std::cout << static_cast<char>(v) << "|";
            } else if (name.find("date") != std::string::npos) {
                char dateBuf[16];
                std::snprintf(dateBuf, sizeof(dateBuf), "%04u-%02u-%02u",
                              v / 10000, (v / 100) % 100, v % 100);
                std::cout << dateBuf << "|";
            } else {
                std::cout << v << "|";
            }
        };
        auto printF32 = [](float v) {
            std::cout << std::fixed << std::setprecision(2) << v << "|";
        };
        auto printStr = [](const std::string& v) {
            std::cout << v << "|";
        };
        
        // Print header
        if (!t.order.empty()) {
            for (const auto& c : t.order) std::cout << c.name << "|";
        } else {
            for (const auto& n : t.u32Names) std::cout << n << "|";
            for (const auto& n : t.f32Names) std::cout << n << "|";
            for (const auto& n : t.stringNames) std::cout << n << "|";
        }
        std::cout << std::endl;
        
        // Print rows
        static const std::string emptyStr;
        for (size_t i = 0; i < t.rowCount; ++i) {
            if (!t.order.empty()) {
                for (const auto& ref : t.order) {
                    if (ref.kind == TableResult::ColRef::Kind::U32) {
                        printU32(ref.name, (i < t.u32Cols[ref.index].size()) ? t.u32Cols[ref.index][i] : 0);
                    } else if (ref.kind == TableResult::ColRef::Kind::F32) {
                        printF32((i < t.f32Cols[ref.index].size()) ? t.f32Cols[ref.index][i] : 0.0f);
                    } else if (ref.kind == TableResult::ColRef::Kind::String) {
                        printStr((i < t.stringCols[ref.index].size()) ? t.stringCols[ref.index][i] : emptyStr);
                    }
                }
            } else {
                for (size_t c = 0; c < t.u32Names.size(); ++c)
                    printU32(t.u32Names[c], (i < t.u32Cols[c].size()) ? t.u32Cols[c][i] : 0);
                for (size_t c = 0; c < t.f32Names.size(); ++c)
                    printF32((i < t.f32Cols[c].size()) ? t.f32Cols[c][i] : 0.0f);
                for (size_t c = 0; c < t.stringNames.size(); ++c)
                    printStr((i < t.stringCols[c].size()) ? t.stringCols[c][i] : emptyStr);
            }
            std::cout << "\n";
        }
    }
    
    std::cout << "---------------------------------------------------" << std::endl;
    printf("Planning time: %.2f ms\n", plan_ms);
    printf("Data Load Time (Disk+Upload): %.2f ms\n", result.table.uploadMs);
    printf("GPU kernels time: %.2f ms\n", result.table.gpuMs);
    printf("CPU postprocess time: %.2f ms\n", result.table.cpuPostMs);
    printf("Total Internal Pipeline time: %.2f ms\n", result.table.uploadMs + result.table.gpuMs + result.table.cpuPostMs);
    printf("Total Host Execution time: %.2f ms\n", exec_ms);
    printf("Total Wall time (Plan+Exec): %.2f ms\n", plan_ms + exec_ms);
    std::cout << "---------------------------------------------------" << std::endl;
    
    // Print detailed kernel timing summary if any kernels were recorded
    if (engine::KernelTimer::instance().totalGpuMs() > 0) {
        std::cout << engine::KernelTimer::instance().summary();
        // Show detailed breakdown if GPUDB_KERNEL_DETAIL is set
        const char* detail = std::getenv("GPUDB_KERNEL_DETAIL");
        if (detail && std::string(detail) == "1") {
            std::cout << engine::KernelTimer::instance().detailed();
        }
    }
    
    return 0;
}
int main(int argc, const char* argv[]) {
    std::string datasetPath = "data/SF-1/";
    std::string sql =
        "SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue\n"
        "FROM lineitem\n"
        "WHERE l_shipdate >= DATE '1994-01-01'\n"
        "  AND l_shipdate <  DATE '1995-01-01'\n"
        "  AND l_discount >= 0.05 AND l_discount <= 0.07\n"
        "  AND l_quantity < 24";

    auto read_file_text = [](const std::string& path) -> std::string {
        std::ifstream file(path);
        if (!file.is_open()) return {};
        std::ostringstream oss;
        oss << file.rdbuf();
        return oss.str();
    };

    // Args: sf1|sf10|v1 and optional --sql "..." or .sql file or inline SQL
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "sf1") datasetPath = "data/SF-1/";
        else if (arg == "sf10") datasetPath = "data/SF-10/";
        // "v1" flag removed — GPUDB_V1 was never read anywhere
        else if (arg == "--sql" && i+1 < argc) { sql = argv[++i]; }
        else if (arg == "help" || arg == "--help" || arg == "-h") {
            std::cout << "MetalGenericDBEngine" << std::endl;
            std::cout << "Usage: MetalGenericDBEngine [v1] [sf1|sf10] [--sql 'QUERY' | QUERY.sql | 'QUERY']" << std::endl;
            return 0;
        }
        else if ((arg.size() >= 4 && arg.substr(arg.size() - 4) == ".sql") && std::filesystem::exists(arg)) {
            // Arg is a SQL file path.
            std::string fileSql = read_file_text(arg);
            if (!fileSql.empty()) sql = fileSql;
            else std::cerr << "Warning: failed to read SQL file: " << arg << std::endl;
        }
        else if (arg.find("SELECT") != std::string::npos || arg.find("select") != std::string::npos) {
            // Arg is a SQL query if it contains SELECT (case-insensitive)
            sql = arg;
        }
    }

    // Make sure DuckDB EXPLAIN reads the same dataset directory as execution.
    setenv("GPUDB_DATASET_PATH", datasetPath.c_str(), 1);

    // Initialise embedded DuckDB once (persistent DB or in-memory views)
    engine::DuckDBAdapter::init(datasetPath);

    int rc = executeQuery(sql, datasetPath);

    engine::DuckDBAdapter::shutdown();
    return rc;
}
