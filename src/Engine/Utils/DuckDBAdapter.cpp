#include "DuckDBAdapter.hpp"
#include <cstdio>
#include <array>
#include <memory>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <sys/stat.h>

namespace engine {

static std::string run_cmd_capture(const std::string& cmd) {
    std::array<char, 4096> buf{};
    std::string out;
#if defined(__APPLE__)
    FILE* pipe = popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) return out;
    while (fgets(buf.data(), buf.size(), pipe)) { out.append(buf.data()); }
    pclose(pipe);
    return out;
}

static std::string escape_for_double_quoted_shell_arg(std::string s) {
    // This string will be embedded inside a shell double-quoted argument.
    // Escape backslashes and double-quotes to preserve the SQL.
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '$') out += "\\$"; // avoid accidental env expansion
        else if (c == '\n' || c == '\r' || c == '\t') out += ' ';
        else out += c;
    }
    return out;
}

static std::string strip_sql_comments(std::string s) {
    // Remove SQL single-line comments (-- ...)
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        // Check for -- comment
        if (i + 1 < s.size() && s[i] == '-' && s[i+1] == '-') {
            // Skip until end of line
            while (i < s.size() && s[i] != '\n' && s[i] != '\r') i++;
            // Add a space to preserve separation
            out += ' ';
        } else {
            out += s[i];
            i++;
        }
    }
    return out;
}

static std::string strip_trailing_semicolon(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r' || s.back() == '\t')) {
        s.pop_back();
    }
    if (!s.empty() && s.back() == ';') s.pop_back();
    return s;
}

std::string DuckDBAdapter::explainJSON(const std::string& sql) {
    // Use the persistent DuckDB database file if it exists.
    // This is CRITICAL: using :memory: with views causes DuckDB to incorrectly
    // transform ">= X AND < Y" into "BETWEEN X AND Y" (which is >= X AND <= Y).
    // The persistent database preserves the correct predicate semantics.
    const std::string q = escape_for_double_quoted_shell_arg(strip_sql_comments(sql));

    std::string datasetPath = "data/SF-1/";
    if (const char* p = std::getenv("GPUDB_DATASET_PATH")) {
        datasetPath = p;
    }
    if (!datasetPath.empty() && datasetPath.back() != '/') datasetPath.push_back('/');
    
    // Check if persistent database exists
    std::string dbPath = datasetPath + "data.duckdb";
    struct stat buffer;
    bool dbExists = (stat(dbPath.c_str(), &buffer) == 0);
    
    if (dbExists) {
        // Use persistent database - correct predicate semantics
        std::ostringstream oss;
        // PRAGMA disable_optimizer to avoid Delim Joins which we don't support well yet
        // Also enable verify_parallelism to force parallelism logic if needed
        oss << "duckdb \"" << dbPath << "\" -json "
            << "-c \"PRAGMA disabled_optimizers='deliminator'; EXPLAIN (FORMAT JSON) " << q << ";\" 2>&1";
        
        if (std::getenv("GPUDB_DEBUG_DUCKDB_CMD")) {
            std::cerr << "[DuckDBAdapter] cmd: " << oss.str() << "\n";
        }
        
        return run_cmd_capture(oss.str());
    }
    
    // Fallback to :memory: with views (may have BETWEEN transformation issue)
    // Note: datasetPath already defined above

    // TPC-H table paths
    const std::string lineitemPath = datasetPath + "lineitem.tbl";
    const std::string ordersPath = datasetPath + "orders.tbl";
    const std::string customerPath = datasetPath + "customer.tbl";
    const std::string supplierPath = datasetPath + "supplier.tbl";
    const std::string nationPath = datasetPath + "nation.tbl";
    const std::string regionPath = datasetPath + "region.tbl";
    const std::string partPath = datasetPath + "part.tbl";
    const std::string partsuppPath = datasetPath + "partsupp.tbl";

    // Note: null_padding=true is important because TPC-H .tbl has a trailing '|'.
    std::string view_lineitem =
        "CREATE OR REPLACE VIEW lineitem AS "
        "SELECT * FROM read_csv('" + lineitemPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'l_orderkey':'INTEGER',"
            "'l_partkey':'INTEGER',"
            "'l_suppkey':'INTEGER',"
            "'l_linenumber':'INTEGER',"
            "'l_quantity':'DECIMAL(15,2)',"
            "'l_extendedprice':'DECIMAL(15,2)',"
            "'l_discount':'DECIMAL(15,2)',"
            "'l_tax':'DECIMAL(15,2)',"
            "'l_returnflag':'VARCHAR',"
            "'l_linestatus':'VARCHAR',"
            "'l_shipdate':'DATE',"
            "'l_commitdate':'DATE',"
            "'l_receiptdate':'DATE',"
            "'l_shipinstruct':'VARCHAR',"
            "'l_shipmode':'VARCHAR',"
            "'l_comment':'VARCHAR'"
        "});";

    std::string view_orders =
        "CREATE OR REPLACE VIEW orders AS "
        "SELECT * FROM read_csv('" + ordersPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'o_orderkey':'INTEGER',"
            "'o_custkey':'INTEGER',"
            "'o_orderstatus':'VARCHAR',"
            "'o_totalprice':'DECIMAL(15,2)',"
            "'o_orderdate':'DATE',"
            "'o_orderpriority':'VARCHAR',"
            "'o_clerk':'VARCHAR',"
            "'o_shippriority':'INTEGER',"
            "'o_comment':'VARCHAR'"
        "});";

    std::string view_customer =
        "CREATE OR REPLACE VIEW customer AS "
        "SELECT * FROM read_csv('" + customerPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'c_custkey':'INTEGER',"
            "'c_name':'VARCHAR',"
            "'c_address':'VARCHAR',"
            "'c_nationkey':'INTEGER',"
            "'c_phone':'VARCHAR',"
            "'c_acctbal':'DECIMAL(15,2)',"
            "'c_mktsegment':'VARCHAR',"
            "'c_comment':'VARCHAR'"
        "});";

    std::string view_supplier =
        "CREATE OR REPLACE VIEW supplier AS "
        "SELECT * FROM read_csv('" + supplierPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'s_suppkey':'INTEGER',"
            "'s_name':'VARCHAR',"
            "'s_address':'VARCHAR',"
            "'s_nationkey':'INTEGER',"
            "'s_phone':'VARCHAR',"
            "'s_acctbal':'DECIMAL(15,2)',"
            "'s_comment':'VARCHAR'"
        "});";

    std::string view_nation =
        "CREATE OR REPLACE VIEW nation AS "
        "SELECT * FROM read_csv('" + nationPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'n_nationkey':'INTEGER',"
            "'n_name':'VARCHAR',"
            "'n_regionkey':'INTEGER',"
            "'n_comment':'VARCHAR'"
        "});";

    std::string view_region =
        "CREATE OR REPLACE VIEW region AS "
        "SELECT * FROM read_csv('" + regionPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'r_regionkey':'INTEGER',"
            "'r_name':'VARCHAR',"
            "'r_comment':'VARCHAR'"
        "});";

    std::string view_part =
        "CREATE OR REPLACE VIEW part AS "
        "SELECT * FROM read_csv('" + partPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'p_partkey':'INTEGER',"
            "'p_name':'VARCHAR',"
            "'p_mfgr':'VARCHAR',"
            "'p_brand':'VARCHAR',"
            "'p_type':'VARCHAR',"
            "'p_size':'INTEGER',"
            "'p_container':'VARCHAR',"
            "'p_retailprice':'DECIMAL(15,2)',"
            "'p_comment':'VARCHAR'"
        "});";

    std::string view_partsupp =
        "CREATE OR REPLACE VIEW partsupp AS "
        "SELECT * FROM read_csv('" + partsuppPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'ps_partkey':'INTEGER',"
            "'ps_suppkey':'INTEGER',"
            "'ps_availqty':'INTEGER',"
            "'ps_supplycost':'DECIMAL(15,2)',"
            "'ps_comment':'VARCHAR'"
        "});";

    std::ostringstream oss;
    oss << "duckdb :memory: "
        << "-c \"" << view_lineitem << "\" "
        << "-c \"" << view_orders << "\" "
        << "-c \"" << view_customer << "\" "
        << "-c \"" << view_supplier << "\" "
        << "-c \"" << view_nation << "\" "
        << "-c \"" << view_region << "\" "
        << "-c \"" << view_part << "\" "
        << "-c \"" << view_partsupp << "\" "
        << "-c \"EXPLAIN (FORMAT JSON) " << q << ";\" 2>&1";

    if (std::getenv("GPUDB_DEBUG_DUCKDB_CMD")) {
        std::cerr << "[DuckDBAdapter] cmd: " << oss.str() << "\n";
    }

    return run_cmd_capture(oss.str());
}

double DuckDBAdapter::runScalarDouble(const std::string& sql) {
    std::ostringstream oss;
    // Note: running against :memory: without data will not produce actual values.
    // This is kept for future use when populating :memory: with COPY ...
    oss << "duckdb :memory: -c \".read schema.sql\" -c \"" << sql << ";\" 2>/dev/null";
    std::string out = run_cmd_capture(oss.str());
    // Try to find the last numeric token in the output
    // DuckDB prints a header row by default; we keep it simple for MVP by scanning tokens.
    double val = std::nan("");
    std::istringstream iss(out);
    std::string tok;
    while (iss >> tok) {
        char* end = nullptr;
        const char* c = tok.c_str();
        double v = std::strtod(c, &end);
        if (end && *end == '\0') { val = v; }
    }
    return val;
}

std::string DuckDBAdapter::runQueryPipe(const std::string& sql) {
    const std::string sql_no_sc = strip_trailing_semicolon(strip_sql_comments(sql));
    const std::string q = escape_for_double_quoted_shell_arg(sql_no_sc);

    std::string datasetPath = "data/SF-1/";
    if (const char* p = std::getenv("GPUDB_DATASET_PATH")) {
        datasetPath = p;
    }
    if (!datasetPath.empty() && datasetPath.back() != '/') datasetPath.push_back('/');

    const std::string lineitemPath = datasetPath + "lineitem.tbl";
    const std::string ordersPath = datasetPath + "orders.tbl";
    const std::string customerPath = datasetPath + "customer.tbl";
    const std::string supplierPath = datasetPath + "supplier.tbl";
    const std::string nationPath = datasetPath + "nation.tbl";
    const std::string regionPath = datasetPath + "region.tbl";
    const std::string partPath = datasetPath + "part.tbl";
    const std::string partsuppPath = datasetPath + "partsupp.tbl";

    // Keep these views in sync with explainJSON() so the query runs on real data.
    std::string view_lineitem =
        "CREATE OR REPLACE VIEW lineitem AS "
        "SELECT * FROM read_csv('" + lineitemPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'l_orderkey':'INTEGER',"
            "'l_partkey':'INTEGER',"
            "'l_suppkey':'INTEGER',"
            "'l_linenumber':'INTEGER',"
            "'l_quantity':'DECIMAL(15,2)',"
            "'l_extendedprice':'DECIMAL(15,2)',"
            "'l_discount':'DECIMAL(15,2)',"
            "'l_tax':'DECIMAL(15,2)',"
            "'l_returnflag':'VARCHAR',"
            "'l_linestatus':'VARCHAR',"
            "'l_shipdate':'DATE',"
            "'l_commitdate':'DATE',"
            "'l_receiptdate':'DATE',"
            "'l_shipinstruct':'VARCHAR',"
            "'l_shipmode':'VARCHAR',"
            "'l_comment':'VARCHAR'"
        "});";

    std::string view_orders =
        "CREATE OR REPLACE VIEW orders AS "
        "SELECT * FROM read_csv('" + ordersPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'o_orderkey':'INTEGER',"
            "'o_custkey':'INTEGER',"
            "'o_orderstatus':'VARCHAR',"
            "'o_totalprice':'DECIMAL(15,2)',"
            "'o_orderdate':'DATE',"
            "'o_orderpriority':'VARCHAR',"
            "'o_clerk':'VARCHAR',"
            "'o_shippriority':'INTEGER',"
            "'o_comment':'VARCHAR'"
        "});";

    std::string view_customer =
        "CREATE OR REPLACE VIEW customer AS "
        "SELECT * FROM read_csv('" + customerPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'c_custkey':'INTEGER',"
            "'c_name':'VARCHAR',"
            "'c_address':'VARCHAR',"
            "'c_nationkey':'INTEGER',"
            "'c_phone':'VARCHAR',"
            "'c_acctbal':'DECIMAL(15,2)',"
            "'c_mktsegment':'VARCHAR',"
            "'c_comment':'VARCHAR'"
        "});";

    std::string view_supplier =
        "CREATE OR REPLACE VIEW supplier AS "
        "SELECT * FROM read_csv('" + supplierPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'s_suppkey':'INTEGER',"
            "'s_name':'VARCHAR',"
            "'s_address':'VARCHAR',"
            "'s_nationkey':'INTEGER',"
            "'s_phone':'VARCHAR',"
            "'s_acctbal':'DECIMAL(15,2)',"
            "'s_comment':'VARCHAR'"
        "});";

    std::string view_nation =
        "CREATE OR REPLACE VIEW nation AS "
        "SELECT * FROM read_csv('" + nationPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'n_nationkey':'INTEGER',"
            "'n_name':'VARCHAR',"
            "'n_regionkey':'INTEGER',"
            "'n_comment':'VARCHAR'"
        "});";

    std::string view_region =
        "CREATE OR REPLACE VIEW region AS "
        "SELECT * FROM read_csv('" + regionPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'r_regionkey':'INTEGER',"
            "'r_name':'VARCHAR',"
            "'r_comment':'VARCHAR'"
        "});";

    std::string view_part =
        "CREATE OR REPLACE VIEW part AS "
        "SELECT * FROM read_csv('" + partPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'p_partkey':'INTEGER',"
            "'p_name':'VARCHAR',"
            "'p_mfgr':'VARCHAR',"
            "'p_brand':'VARCHAR',"
            "'p_type':'VARCHAR',"
            "'p_size':'INTEGER',"
            "'p_container':'VARCHAR',"
            "'p_retailprice':'DECIMAL(15,2)',"
            "'p_comment':'VARCHAR'"
        "});";

    std::string view_partsupp =
        "CREATE OR REPLACE VIEW partsupp AS "
        "SELECT * FROM read_csv('" + partsuppPath + "', delim='|', header=false, null_padding=true, "
        "columns={"
            "'ps_partkey':'INTEGER',"
            "'ps_suppkey':'INTEGER',"
            "'ps_availqty':'INTEGER',"
            "'ps_supplycost':'DECIMAL(15,2)',"
            "'ps_comment':'VARCHAR'"
        "});";

    std::ostringstream oss;
    oss << "duckdb :memory: "
        << "-c \"" << view_lineitem << "\" "
        << "-c \"" << view_orders << "\" "
        << "-c \"" << view_customer << "\" "
        << "-c \"" << view_supplier << "\" "
        << "-c \"" << view_nation << "\" "
        << "-c \"" << view_region << "\" "
        << "-c \"" << view_part << "\" "
        << "-c \"" << view_partsupp << "\" "
        << "-c \"COPY (" << q << ") TO STDOUT (DELIMITER '|', HEADER false);\" 2>/dev/null";

    if (std::getenv("GPUDB_DEBUG_DUCKDB_CMD")) {
        std::cerr << "[DuckDBAdapter] cmd: " << oss.str() << "\n";
    }

    return run_cmd_capture(oss.str());
}

} // namespace engine
