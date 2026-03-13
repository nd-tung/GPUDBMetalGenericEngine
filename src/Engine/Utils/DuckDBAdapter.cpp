#include "DuckDBAdapter.hpp"
#include "EnvUtil.hpp"

// DuckDB C++ API – linked via -lduckdb
#include <duckdb.hpp>
#include <duckdb/main/database.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include "Logger.hpp"

// ── DuckDB version contract ────────────────────────────────────────────
// Tested with DuckDB 1.x (Homebrew). The EXPLAIN JSON format and plan
// node names are not part of a stable API, so upgrading DuckDB may break
// the planner or require updating workarounds. A runtime check below
// logs a warning when the linked library version differs from the last
// validated version.
//
// Key DuckDB plan-level workarounds maintained by this engine:
//
// 1. Truncated IN-list:  DuckDB EXPLAIN JSON truncates long IN-value
//    lists to "IN (...)".  Evaluator.cpp silently treats this as a
//    pass-through filter.  (Evaluator.cpp ~L1764)
//
// 2. Scalar-subquery CASE wrapper:  DuckDB wraps scalar subquery results
//    in CASE WHEN count>1 THEN error(...) ELSE first(val) END.  The
//    engine strips the error() guard and FIRST() aggregate.
//    (Evaluator.cpp ~L1345, ~L1452; GpuExecutor.cpp ~L1469;
//     PlannerNodeHandlers.cpp ~L669)
//
// 3. DELIM_SCAN / COLUMN_DATA_SCAN:  DuckDB decorrelates subqueries
//    via these nodes.  The planner recognises them and the executor
//    deduplicates correlation keys.  (PlannerTraversal.cpp ~L210;
//    PlannerNodeHandlers.cpp ~L35; GpuExecutor.cpp ~L286)
//
// 4. IS NOT DISTINCT FROM:  DuckDB uses NULL-safe equality in
//    DELIM_SCAN correlation conditions; treated as regular '='.
//    (PlannerExprParser.cpp ~L539)
//
// 5. __internal_compress / decompress:  DuckDB's storage optimizer wraps
//    columns in internal functions in EXPLAIN output; stripped to get the
//    real column name.  (PlannerTraversal.cpp ~L97; DetailHelpers.hpp
//    ~L113; Project.cpp ~L1352)
//
// 6. disabled_optimizers='deliminator':  Forces DuckDB to emit
//    DELIM_SCAN / COLUMN_DATA_SCAN nodes instead of optimizing them away.
//    (DuckDBAdapter.cpp, this file)
// ────────────────────────────────────────────────────────────────────────

// Last validated DuckDB library version.  Update after verifying all
// 47 tests pass on the new version.
static constexpr const char* kExpectedDuckDBVersion = "v1.4.4";

namespace engine {

// ── static member definitions ──
std::unique_ptr<duckdb::DuckDB>     DuckDBAdapter::s_db;
std::unique_ptr<duckdb::Connection> DuckDBAdapter::s_con;
std::string                         DuckDBAdapter::s_datasetPath;
bool                                DuckDBAdapter::s_ready = false;

// ── helpers ──

static std::string stripSqlComments(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (i + 1 < s.size() && s[i] == '-' && s[i + 1] == '-') {
            while (i < s.size() && s[i] != '\n' && s[i] != '\r') ++i;
            out += ' ';
        } else {
            out += s[i];
            ++i;
        }
    }
    return out;
}

static bool fileExists(const std::string& path) {
    struct stat buf;
    return stat(path.c_str(), &buf) == 0;
}

// ── public API ──

void DuckDBAdapter::init(const std::string& datasetPath) {
    s_datasetPath = datasetPath;
    if (!s_datasetPath.empty() && s_datasetPath.back() != '/')
        s_datasetPath.push_back('/');
    ensureReady();
}

void DuckDBAdapter::shutdown() {
    s_con.reset();
    s_db.reset();
    s_ready = false;
}

void DuckDBAdapter::ensureReady() {
    if (s_ready) return;

    // Resolve dataset path from env if not set via init()
    if (s_datasetPath.empty()) {
        if (const char* p = std::getenv("GPUDB_DATASET_PATH")) {
            s_datasetPath = p;
        } else {
            s_datasetPath = "data/SF-1/";
        }
        if (!s_datasetPath.empty() && s_datasetPath.back() != '/')
            s_datasetPath.push_back('/');
    }

    const std::string dbPath = s_datasetPath + "data.duckdb";

    if (fileExists(dbPath)) {
        // Open persistent database (read-only to avoid lock contention)
        duckdb::DBConfig config;
        config.options.access_mode = duckdb::AccessMode::READ_ONLY;
        s_db  = std::make_unique<duckdb::DuckDB>(dbPath, &config);
        s_con = std::make_unique<duckdb::Connection>(*s_db);
    } else {
        // In-memory with views over the .tbl files
        s_db  = std::make_unique<duckdb::DuckDB>(nullptr);   // :memory:
        s_con = std::make_unique<duckdb::Connection>(*s_db);

        // TPC-H table definitions  (null_padding=true for trailing '|')
        struct TblDef { const char* name; const char* cols; };
        static const TblDef defs[] = {
            {"lineitem",
             "'l_orderkey':'INTEGER','l_partkey':'INTEGER','l_suppkey':'INTEGER',"
             "'l_linenumber':'INTEGER','l_quantity':'DECIMAL(15,2)','l_extendedprice':'DECIMAL(15,2)',"
             "'l_discount':'DECIMAL(15,2)','l_tax':'DECIMAL(15,2)','l_returnflag':'VARCHAR',"
             "'l_linestatus':'VARCHAR','l_shipdate':'DATE','l_commitdate':'DATE',"
             "'l_receiptdate':'DATE','l_shipinstruct':'VARCHAR','l_shipmode':'VARCHAR',"
             "'l_comment':'VARCHAR'"},
            {"orders",
             "'o_orderkey':'INTEGER','o_custkey':'INTEGER','o_orderstatus':'VARCHAR',"
             "'o_totalprice':'DECIMAL(15,2)','o_orderdate':'DATE','o_orderpriority':'VARCHAR',"
             "'o_clerk':'VARCHAR','o_shippriority':'INTEGER','o_comment':'VARCHAR'"},
            {"customer",
             "'c_custkey':'INTEGER','c_name':'VARCHAR','c_address':'VARCHAR',"
             "'c_nationkey':'INTEGER','c_phone':'VARCHAR','c_acctbal':'DECIMAL(15,2)',"
             "'c_mktsegment':'VARCHAR','c_comment':'VARCHAR'"},
            {"supplier",
             "'s_suppkey':'INTEGER','s_name':'VARCHAR','s_address':'VARCHAR',"
             "'s_nationkey':'INTEGER','s_phone':'VARCHAR','s_acctbal':'DECIMAL(15,2)',"
             "'s_comment':'VARCHAR'"},
            {"nation",
             "'n_nationkey':'INTEGER','n_name':'VARCHAR','n_regionkey':'INTEGER',"
             "'n_comment':'VARCHAR'"},
            {"region",
             "'r_regionkey':'INTEGER','r_name':'VARCHAR','r_comment':'VARCHAR'"},
            {"part",
             "'p_partkey':'INTEGER','p_name':'VARCHAR','p_mfgr':'VARCHAR',"
             "'p_brand':'VARCHAR','p_type':'VARCHAR','p_size':'INTEGER',"
             "'p_container':'VARCHAR','p_retailprice':'DECIMAL(15,2)','p_comment':'VARCHAR'"},
            {"partsupp",
             "'ps_partkey':'INTEGER','ps_suppkey':'INTEGER','ps_availqty':'INTEGER',"
             "'ps_supplycost':'DECIMAL(15,2)','ps_comment':'VARCHAR'"},
        };

        for (auto& d : defs) {
            std::string tblFile = s_datasetPath + d.name + ".tbl";
            if (!fileExists(tblFile)) continue;

            std::ostringstream sql;
            sql << "CREATE OR REPLACE VIEW " << d.name
                << " AS SELECT * FROM read_csv('" << tblFile
                << "', delim='|', header=false, null_padding=true, columns={"
                << d.cols << "})";

            auto res = s_con->Query(sql.str());
            if (res->HasError()) {
                LOG_ERROR("DuckDBAdapter", "view creation error for " << d.name << ": " << res->GetError());
            }
        }
    }

    // Disable deliminator optimizer to match the planner's expectations
    s_con->Query("PRAGMA disabled_optimizers='deliminator'");

    // Runtime version check — warn if DuckDB library differs from the
    // last validated version so developers know to re-verify workarounds.
    const char* linkedVer = duckdb::DuckDB::LibraryVersion();
    if (linkedVer && std::string(linkedVer) != kExpectedDuckDBVersion) {
        LOG_WARN("DuckDB", "Linked DuckDB " << linkedVer
                 << " differs from validated " << kExpectedDuckDBVersion
                 << ". Plan workarounds may need updating.");
    }

    s_ready = true;
}

std::string DuckDBAdapter::explainJSON(const std::string& sql) {
    ensureReady();

    const std::string cleaned = stripSqlComments(sql);
    const std::string stmt = "EXPLAIN (FORMAT JSON) " + cleaned;

    if (env_truthy("GPUDB_DEBUG_DUCKDB_CMD")) {
        LOG_INFO("DuckDBAdapter", "embedded query: " << stmt);
    }

    auto result = s_con->Query(stmt);
    if (result->HasError()) {
        LOG_ERROR("DuckDBAdapter", "EXPLAIN error: " << result->GetError());
        return {};
    }

    // The result is a single-column, single-row table containing the JSON string.
    if (result->RowCount() == 0 || result->ColumnCount() == 0) {
        LOG_INFO("DuckDBAdapter", "EXPLAIN returned empty result\n");
        return {};
    }

    // EXPLAIN (FORMAT JSON) returns a table with columns: explain_key | explain_value
    // The physical plan JSON is in the "explain_value" column (index 1) of the
    // row whose explain_key is "physical_plan".
    // If only one column exists, fall back to column 0.
    idx_t jsonCol = (result->ColumnCount() >= 2) ? 1 : 0;

    for (idx_t r = 0; r < result->RowCount(); ++r) {
        auto key = result->GetValue(0, r).ToString();
        if (key == "physical_plan") {
            return result->GetValue(jsonCol, r).ToString();
        }
    }

    // Fallback: return the first cell
    return result->GetValue(jsonCol, 0).ToString();
}

} // namespace engine
