#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace engine {

// ============================================================================
// Generic column schema system
// ============================================================================

enum class ColumnType {
    Int32,
    Float32,
    Date,           // YYYYMMDD as integer
    StringHash,     // FNV1a hash of string
    StringChar,     // Single character stored as char code
};

struct ColumnSchema {
    std::string name;
    int index;              // 0-based column index in .tbl file
    ColumnType type;
};

struct TableSchema {
    std::string name;
    std::vector<ColumnSchema> columns;
    
    // Lookup by name
    const ColumnSchema* getColumn(const std::string& colName) const {
        for (const auto& c : columns) {
            if (c.name == colName) return &c;
        }
        return nullptr;
    }
    
    ColumnType getColumnType(const std::string& colName) const {
        const auto* c = getColumn(colName);
        return c ? c->type : ColumnType::StringHash;
    }
};

// ============================================================================
// SchemaRegistry: Global schema registry
// ============================================================================

class SchemaRegistry {
public:
    static SchemaRegistry& instance() {
        static SchemaRegistry inst;
        return inst;
    }
    
    // Register a table schema
    void registerTable(TableSchema schema) {
        tables_[schema.name] = std::move(schema);
    }
    
    // Get table schema
    const TableSchema* getTable(const std::string& name) const {
        auto it = tables_.find(name);
        return it != tables_.end() ? &it->second : nullptr;
    }
    
    // Get column type for a table.column
    ColumnType getColumnType(const std::string& table, const std::string& column) const {
        const auto* t = getTable(table);
        if (!t) return ColumnType::StringHash;
        return t->getColumnType(column);
    }
    
    // Check if column stores single char (reversible)
    bool isSingleCharColumn(const std::string& table, const std::string& column) const {
        return getColumnType(table, column) == ColumnType::StringChar;
    }
    
    // Initialize with TPC-H schema
    void initTPCH() {
        // lineitem
        registerTable({"lineitem", {
            {"l_orderkey", 0, ColumnType::Int32},
            {"l_partkey", 1, ColumnType::Int32},
            {"l_suppkey", 2, ColumnType::Int32},
            {"l_linenumber", 3, ColumnType::Int32},
            {"l_quantity", 4, ColumnType::Float32},
            {"l_extendedprice", 5, ColumnType::Float32},
            {"l_discount", 6, ColumnType::Float32},
            {"l_tax", 7, ColumnType::Float32},
            {"l_returnflag", 8, ColumnType::StringChar},   // A/N/R
            {"l_linestatus", 9, ColumnType::StringChar},   // F/O
            {"l_shipdate", 10, ColumnType::Date},
            {"l_commitdate", 11, ColumnType::Date},
            {"l_receiptdate", 12, ColumnType::Date},
            {"l_shipinstruct", 13, ColumnType::StringHash},
            {"l_shipmode", 14, ColumnType::StringHash},
            {"l_comment", 15, ColumnType::StringHash},
        }});
        
        // orders
        registerTable({"orders", {
            {"o_orderkey", 0, ColumnType::Int32},
            {"o_custkey", 1, ColumnType::Int32},
            {"o_orderstatus", 2, ColumnType::StringChar},  // F/O/P
            {"o_totalprice", 3, ColumnType::Float32},
            {"o_orderdate", 4, ColumnType::Date},
            {"o_orderpriority", 5, ColumnType::StringHash},
            {"o_clerk", 6, ColumnType::StringHash},
            {"o_shippriority", 7, ColumnType::Int32},
            {"o_comment", 8, ColumnType::StringHash},
        }});
        
        // customer
        registerTable({"customer", {
            {"c_custkey", 0, ColumnType::Int32},
            {"c_name", 1, ColumnType::StringHash},
            {"c_address", 2, ColumnType::StringHash},
            {"c_nationkey", 3, ColumnType::Int32},
            {"c_phone", 4, ColumnType::StringHash},
            {"c_acctbal", 5, ColumnType::Float32},
            {"c_mktsegment", 6, ColumnType::StringHash},
            {"c_comment", 7, ColumnType::StringHash},
        }});
        
        // part
        registerTable({"part", {
            {"p_partkey", 0, ColumnType::Int32},
            {"p_name", 1, ColumnType::StringHash},
            {"p_mfgr", 2, ColumnType::StringHash},
            {"p_brand", 3, ColumnType::StringHash},
            {"p_type", 4, ColumnType::StringHash},
            {"p_size", 5, ColumnType::Int32},
            {"p_container", 6, ColumnType::StringHash},
            {"p_retailprice", 7, ColumnType::Float32},
            {"p_comment", 8, ColumnType::StringHash},
        }});
        
        // supplier
        registerTable({"supplier", {
            {"s_suppkey", 0, ColumnType::Int32},
            {"s_name", 1, ColumnType::StringHash},
            {"s_address", 2, ColumnType::StringHash},
            {"s_nationkey", 3, ColumnType::Int32},
            {"s_phone", 4, ColumnType::StringHash},
            {"s_acctbal", 5, ColumnType::Float32},
            {"s_comment", 6, ColumnType::StringHash},
        }});
        
        // partsupp
        registerTable({"partsupp", {
            {"ps_partkey", 0, ColumnType::Int32},
            {"ps_suppkey", 1, ColumnType::Int32},
            {"ps_availqty", 2, ColumnType::Int32},
            {"ps_supplycost", 3, ColumnType::Float32},
            {"ps_comment", 4, ColumnType::StringHash},
        }});
        
        // nation
        registerTable({"nation", {
            {"n_nationkey", 0, ColumnType::Int32},
            {"n_name", 1, ColumnType::StringHash},
            {"n_regionkey", 2, ColumnType::Int32},
            {"n_comment", 3, ColumnType::StringHash},
        }});
        
        // region
        registerTable({"region", {
            {"r_regionkey", 0, ColumnType::Int32},
            {"r_name", 1, ColumnType::StringHash},
            {"r_comment", 2, ColumnType::StringHash},
        }});
    }
    
private:
    SchemaRegistry() {
        initTPCH();  // Initialize with TPC-H by default
    }
    
    std::unordered_map<std::string, TableSchema> tables_;
};

} // namespace engine
