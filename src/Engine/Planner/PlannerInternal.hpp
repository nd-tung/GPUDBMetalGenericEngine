#pragma once
// ============================================================================
// PlannerInternal.hpp — Shared declarations for the Planner subsystem
// ============================================================================
#include "Planner.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>

namespace engine {

using nlohmann::json;

// ─── TraverseContext: mutable state threaded through the DuckDB JSON walk ───
struct TraverseContext {
    Plan& plan;
    const std::unordered_map<std::string, std::string>& aliases;
    std::unordered_map<std::string, std::string> localAliases;
    std::vector<std::string> projections;
    std::unordered_set<std::string> seenTables;
    bool pastGroupBy = false;
    std::vector<std::pair<std::string, std::vector<std::string>>> delimStack;
    std::map<int64_t, std::string> cteMap;
    std::unordered_set<std::string> forceKeepColumns;
    std::unordered_map<std::string, std::string> qualifiedColumnMapping;
};

// ─── JoinCapture: bundle of state captured during join RHS pre-processing ───
struct JoinCapture {
    std::string capturedRightTable;
    TypedExprPtr capturedRightFilter;
    bool capturedRHS = false;
    std::unordered_set<std::string> rhsTables;
    std::vector<std::string> lhsProjections;
    std::vector<std::string> rhsProjections;
};

// ─── PlannerUtils ───
void debug_log(const std::string& msg);
std::string tolower_str(std::string s);
std::string trim_str(std::string s);
std::string normalizeNumericLiterals(const std::string& s);
std::string normalizeOperators(const std::string& s);
std::string strip_parens(std::string s);
std::string stripTableQualifier(const std::string& s);
std::vector<std::string> renameDuplicateColumns(
    const std::vector<std::string>& lhsProjs,
    const std::vector<std::string>& rhsProjs,
    std::unordered_map<std::string, std::string>& renameMap);

// ─── PlannerSQLParser ───
std::string resolveColRef(const std::string& ref, const std::vector<std::string>& projections);
std::vector<std::string> parseSelectColumnNames(const std::string& sql);
std::unordered_map<std::string, std::string> parseSelectAliases(const std::string& sql);

// ─── PlannerTraversal ───
void collectGlobalColumns(const json& j, std::unordered_set<std::string>& cols);
void traverseNode(const json& node, TraverseContext& ctx);

// ─── PlannerNodeHandlers ───
bool handleScan(const json& node, const std::string& name, const std::string& nameLower,
                const json& extraInfo, std::vector<std::string>& myProjs, TraverseContext& ctx);
void handleFilter(const json& node, const std::string& name,
                  const json& extraInfo, const std::string& extraStr,
                  std::vector<std::string>& childProjs, TraverseContext& ctx);
void handleGroupBy(const json& node, const std::string& name,
                   const json& extraInfo, const std::vector<std::string>& childProjs,
                   std::vector<std::string>& myProjs, TraverseContext& ctx);
void handleUngroupedAggregate(const json& node, const std::string& name,
                              const json& extraInfo,
                              std::vector<std::string>& childProjs, TraverseContext& ctx);
void handleProjection(const json& node, const std::string& name,
                      const std::vector<std::string>& myProjs,
                      const std::vector<std::string>& childProjs, TraverseContext& ctx);
void handleOrderBy(const json& node, const std::string& name, const std::string& nameLower,
                   const json& extraInfo, const std::vector<std::string>& childProjs,
                   TraverseContext& ctx);
void handleJoinEmit(const json& node, const std::string& name, const std::string& nameLower,
                    const json& extraInfo, const std::vector<std::string>& childProjs,
                    const JoinCapture& jc, TraverseContext& ctx);

} // namespace engine
