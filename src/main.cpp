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
#include "GpuExecutorDetail.hpp"  // for env_truthy

static std::string s_datasetPath = "data/SF-1/";

static int executeQuery(const std::string& sql) {
    using namespace engine;
    std::cout << "--- Running (Engine Host) ---" << std::endl;

    // Initialize schema registry for TPC-H (redundant with constructor, but explicit)
    // SchemaRegistry::instance().initTPCH();  // Already called in singleton constructor

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
    auto result = GpuExecutor::execute(plan, s_datasetPath);
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
        for (size_t i = 0; i < t.rowCount; ++i) {
            if (!t.order.empty()) {
                for (const auto& ref : t.order) {
                    if (ref.kind == TableResult::ColRef::Kind::U32) {
                        const auto& col = t.u32Cols[ref.index];
                        const uint32_t v = (i < col.size()) ? col[i] : 0;
                        if (t.singleCharCols.count(ref.name)) {
                            std::cout << static_cast<char>(v) << "|";
                        } else if (ref.name.find("date") != std::string::npos) {
                            uint32_t y = v / 10000;
                            uint32_t m = (v / 100) % 100;
                            uint32_t day = v % 100;
                            char dateBuf[16];
                            std::snprintf(dateBuf, sizeof(dateBuf), "%04u-%02u-%02u", y, m, day);
                            std::cout << dateBuf << "|";
                        } else {
                            std::cout << v << "|";
                        }
                    } else if (ref.kind == TableResult::ColRef::Kind::F32) {
                        const auto& col = t.f32Cols[ref.index];
                        const float v = (i < col.size()) ? col[i] : 0.0f;
                        std::cout << std::fixed << std::setprecision(2) << v << "|";
                    } else if (ref.kind == TableResult::ColRef::Kind::String) {
                        const auto& col = t.stringCols[ref.index];
                        const std::string& v = (i < col.size()) ? col[i] : "";
                        std::cout << v << "|";
                    }
                }
            } else {
                for (size_t c = 0; c < t.u32Names.size(); ++c) {
                    const auto& name = t.u32Names[c];
                    const auto& col = t.u32Cols[c];
                    const uint32_t v = (i < col.size()) ? col[i] : 0;
                    if (t.singleCharCols.count(name)) {
                        std::cout << static_cast<char>(v) << "|";
                    } else if (name.find("date") != std::string::npos) {
                        uint32_t y = v / 10000;
                        uint32_t m = (v / 100) % 100;
                        uint32_t day = v % 100;
                        char dateBuf[16];
                        std::snprintf(dateBuf, sizeof(dateBuf), "%04u-%02u-%02u", y, m, day);
                        std::cout << dateBuf << "|";
                    } else {
                        std::cout << v << "|";
                    }
                }
                for (size_t c = 0; c < t.f32Names.size(); ++c) {
                    const auto& col = t.f32Cols[c];
                    const float v = (i < col.size()) ? col[i] : 0.0f;
                    std::cout << std::fixed << std::setprecision(2) << v << "|";
                }
                for (size_t c = 0; c < t.stringNames.size(); ++c) {
                    const auto& col = t.stringCols[c];
                    const std::string& v = (i < col.size()) ? col[i] : "";
                    std::cout << v << "|";
                }
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
        if (arg == "sf1") s_datasetPath = "data/SF-1/";
        else if (arg == "sf10") s_datasetPath = "data/SF-10/";
        else if (arg == "v1") setenv("GPUDB_V1", "1", 1);
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
    setenv("GPUDB_DATASET_PATH", s_datasetPath.c_str(), 1);
    return executeQuery(sql);
}
