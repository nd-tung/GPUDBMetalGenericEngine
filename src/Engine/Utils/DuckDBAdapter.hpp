#pragma once
#include <string>

namespace engine {

class DuckDBAdapter {
public:
    // Returns EXPLAIN (FORMAT JSON) output as a JSON array string, or empty on failure.
    static std::string explainJSON(const std::string& sql);
    // Runs the query and returns the first numeric cell as double, NaN on failure.
    static double runScalarDouble(const std::string& sql);

    // Runs the query in DuckDB and returns pipe-delimited rows (no header).
    // Intended for correctness gating against engine outputs.
    static std::string runQueryPipe(const std::string& sql);
};

} // namespace engine
