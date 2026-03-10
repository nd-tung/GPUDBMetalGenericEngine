// ============================================================================
// Planner.cpp — Entry point: Planner::fromSQL
// ============================================================================
#include "PlannerInternal.hpp"
#include "DuckDBAdapter.hpp"
#include "EnvUtil.hpp"
#include <iostream>
#include <regex>

namespace engine {

// --- Main parsing function ---

Plan Planner::fromSQL(const std::string& sql) {
    Plan plan;
    
    // Get DuckDB EXPLAIN JSON
    std::string raw = DuckDBAdapter::explainJSON(sql);
    
    // Extract JSON array
    std::string jsonStr;
    auto start = raw.find('[');
    if (start != std::string::npos) {
        int depth = 0;
        size_t end = start;
        for (size_t i = start; i < raw.size(); i++) {
            if (raw[i] == '[') depth++;
            else if (raw[i] == ']') {
                depth--;
                if (depth == 0) { end = i + 1; break; }
            }
        }
        jsonStr = raw.substr(start, end - start);
        while (!jsonStr.empty() && (jsonStr.back() == '%' || jsonStr.back() == '\n' || jsonStr.back() == '\r')) {
            jsonStr.pop_back();
        }
    } else {
        if (env_truthy("GPUDB_DEBUG_PLANNER")) std::cerr << "DuckDB Raw Output:\n" << raw << "\n";
        plan.parseError = "Could not find JSON array in DuckDB output";
        return plan;
    }
    
    // Parse JSON
    try {
        json j = json::parse(jsonStr);
        if (!j.is_array() || j.size() == 0) {
            plan.parseError = "DuckDB JSON is not a non-empty array";
            return plan;
        }
        
        auto aliases = parseSelectAliases(sql);
        
        debug_log("Parsed aliases:");
        for (const auto& [k, v] : aliases) {
            debug_log("  '" + k + "' -> '" + v + "'");
        }
        
        TraverseContext ctx{plan, aliases, {}, {}, {}, false, {}, {}, {}, {}};
        collectGlobalColumns(j[0], ctx.forceKeepColumns);
        traverseNode(j[0], ctx);
        
        // Save final output column names parsed from SQL SELECT clause
        plan.outputColumns = parseSelectColumnNames(sql);
        debug_log("plan.outputColumns from SQL:");
        for (const auto& c : plan.outputColumns) debug_log("  '" + c + "'");
        
        // Recover LIMIT from SQL if not in plan
        bool hasLimit = false;
        for (const auto& n : plan.nodes) {
            if (n.type == IRNode::Type::Limit) { hasLimit = true; break; }
        }
        if (!hasLimit) {
            std::regex re_limit(R"(limit\s+(\d+))", std::regex::icase);
            std::smatch m;
            if (std::regex_search(sql, m, re_limit) && m.size() > 1) {
                plan.nodes.push_back(IRNode::limit(std::stoll(m[1].str())));
            }
        }
        
    } catch (const std::exception& e) {
        plan.parseError = std::string("JSON parse error: ") + e.what();
    }
    
    return plan;
}

} // namespace engine
