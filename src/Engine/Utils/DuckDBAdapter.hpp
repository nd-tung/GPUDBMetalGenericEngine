#pragma once
#include <string>
#include <memory>

// Forward-declare DuckDB types to avoid pulling the full header into every TU.
namespace duckdb { class DuckDB; class Connection; }

namespace engine {

class DuckDBAdapter {
public:
    /// Returns EXPLAIN (FORMAT JSON) output as a JSON array string, or empty on failure.
    static std::string explainJSON(const std::string& sql);

    /// Call once at startup (optional – lazy-initialised on first use).
    static void init(const std::string& datasetPath);

    /// Tear down the embedded instance (optional – called automatically at exit).
    static void shutdown();

private:
    /// Ensure the singleton DuckDB instance is ready.
    static void ensureReady();

    static std::unique_ptr<duckdb::DuckDB>      s_db;
    static std::unique_ptr<duckdb::Connection>   s_con;
    static std::string                           s_datasetPath;
    static bool                                  s_ready;
};

} // namespace engine
