#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <iomanip>
#include <set>
#include <filesystem>
#include <sstream>
#include <cmath>
#include <variant>
#include "IR.hpp"
#include "Planner.hpp"
#include "GpuExecutor.hpp"
#include "Schema.hpp"
#include "ExprEval.hpp"
#include "DuckDBAdapter.hpp"
#include "KernelTimer.hpp"

static std::string g_dataset_path = "data/SF-1/";

static bool env_truthy(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return (s == "1" || s == "true" || s == "on" || s == "yes");
}

static int runEngineSQL(const std::string& sql) {
    using namespace engine;
    std::cout << "--- Running (Engine Host) ---" << std::endl;

    std::string query_lower = sql;
    std::transform(query_lower.begin(), query_lower.end(), query_lower.begin(), ::tolower);

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
    // auto result = GPUNativeExecutor::execute(plan, g_dataset_path);
    std::cout << "[Main] Using GpuExecutor generic executor.\n";
    auto t_exec_start = std::chrono::high_resolution_clock::now();
    auto result = GpuExecutor::execute(plan, g_dataset_path);
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
            for (const auto& n : t.u32_names) std::cout << n << "|";
            for (const auto& n : t.f32_names) std::cout << n << "|";
            for (const auto& n : t.string_names) std::cout << n << "|";
        }
        std::cout << std::endl;
        
        // Print rows
        const size_t rows_to_print = t.rowCount;
        for (size_t i = 0; i < rows_to_print; ++i) {
            if (!t.order.empty()) {
                for (const auto& ref : t.order) {
                    if (ref.kind == TableResult::ColRef::Kind::U32) {
                        const auto& col = t.u32_cols[ref.index];
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
                        const auto& col = t.f32_cols[ref.index];
                        const float v = (i < col.size()) ? col[i] : 0.0f;
                        std::cout << std::fixed << std::setprecision(2) << v << "|";
                    } else if (ref.kind == TableResult::ColRef::Kind::String) {
                        const auto& col = t.string_cols[ref.index];
                        const std::string& v = (i < col.size()) ? col[i] : "";
                        std::cout << v << "|";
                    }
                }
            } else {
                for (size_t c = 0; c < t.u32_names.size(); ++c) {
                    const auto& name = t.u32_names[c];
                    const auto& col = t.u32_cols[c];
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
                for (size_t c = 0; c < t.f32_names.size(); ++c) {
                    const auto& col = t.f32_cols[c];
                    const float v = (i < col.size()) ? col[i] : 0.0f;
                    std::cout << std::fixed << std::setprecision(2) << v << "|";
                }
                for (size_t c = 0; c < t.string_names.size(); ++c) {
                    const auto& col = t.string_cols[c];
                    const std::string& v = (i < col.size()) ? col[i] : "";
                    std::cout << v << "|";
                }
            }
            std::cout << "\n";
        }
        
        if (t.rowCount > rows_to_print) {
            std::cout << "... (" << (t.rowCount - rows_to_print) << " more rows)" << std::endl;
        }
    }
    
    std::cout << "---------------------------------------------------" << std::endl;
    printf("Planning time: %.2f ms\n", plan_ms);
    printf("Data Load Time (Disk+Upload): %.2f ms\n", result.table.upload_ms);
    printf("GPU kernels time: %.2f ms\n", result.table.gpu_ms);
    printf("CPU postprocess time: %.2f ms\n", result.table.cpu_post_ms);
    printf("Total Internal Pipeline time: %.2f ms\n", result.table.upload_ms + result.table.gpu_ms + result.table.cpu_post_ms);
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
        if (arg == "sf1") g_dataset_path = "data/SF-1/";
        else if (arg == "sf10") g_dataset_path = "data/SF-10/";
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
    setenv("GPUDB_DATASET_PATH", g_dataset_path.c_str(), 1);
    return runEngineSQL(sql);
}
