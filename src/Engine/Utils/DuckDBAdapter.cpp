#include "DuckDBAdapter.hpp"
#include "EnvUtil.hpp"

// DuckDB C++ API – linked via -lduckdb
#include <duckdb.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>

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
                std::cerr << "[DuckDBAdapter] view creation error for "
                          << d.name << ": " << res->GetError() << "\n";
            }
        }
    }

    // Disable deliminator optimizer to match the planner's expectations
    s_con->Query("PRAGMA disabled_optimizers='deliminator'");

    s_ready = true;
}

std::string DuckDBAdapter::explainJSON(const std::string& sql) {
    ensureReady();

    const std::string cleaned = stripSqlComments(sql);
    const std::string stmt = "EXPLAIN (FORMAT JSON) " + cleaned;

    if (env_truthy("GPUDB_DEBUG_DUCKDB_CMD")) {
        std::cerr << "[DuckDBAdapter] embedded query: " << stmt << "\n";
    }

    auto result = s_con->Query(stmt);
    if (result->HasError()) {
        std::cerr << "[DuckDBAdapter] EXPLAIN error: " << result->GetError() << "\n";
        return {};
    }

    // The result is a single-column, single-row table containing the JSON string.
    if (result->RowCount() == 0 || result->ColumnCount() == 0) {
        std::cerr << "[DuckDBAdapter] EXPLAIN returned empty result\n";
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
