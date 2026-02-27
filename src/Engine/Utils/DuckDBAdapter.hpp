#pragma once
#include <string>

namespace engine {

class DuckDBAdapter {
public:
    // Returns EXPLAIN (FORMAT JSON) output as a JSON array string, or empty on failure.
    static std::string explainJSON(const std::string& sql);
};

} // namespace engine
