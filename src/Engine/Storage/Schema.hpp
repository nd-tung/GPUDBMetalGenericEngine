#pragma once
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

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
        if (s_override) return *s_override;
        static SchemaRegistry inst;
        return inst;
    }
    
    // For testing: inject a custom/mock instance; call resetTestInstance() to restore.
    static void setTestInstance(SchemaRegistry* mock) { s_override = mock; }
    static void resetTestInstance() { s_override = nullptr; }
    
    // Register a table schema (thread-safe)
    void registerTable(TableSchema schema) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_tables[schema.name] = std::move(schema);
    }
    
    // Get table schema (thread-safe)
    const TableSchema* getTable(const std::string& name) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_tables.find(name);
        return it != m_tables.end() ? &it->second : nullptr;
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

    // Determine which table a column belongs to using column-name prefix matching.
    // Returns empty string if the column prefix is not recognized.
    std::string tableForColumn(const std::string& col) const {
        // Strip trailing suffix like "_1", "_2" for multi-instance columns
        std::string c = col;
        if (c.size() > 2 && c[c.size()-2] == '_' && c.back() >= '0' && c.back() <= '9')
            c = c.substr(0, c.size()-2);

        for (const auto& [tblName, tblSchema] : m_tables) {
            for (const auto& cs : tblSchema.columns) {
                if (cs.name == c) return tblName;
            }
        }
        return "";
    }

    // GPU scan helpers: column type classification for numeric/date/char columns.
    enum class GpuColKind { U32, F32, DateU32, StrCharU32 };
    struct GpuColInfo { int index; GpuColKind kind; };

    // Get GPU-compatible column info. Returns nullopt for StringHash columns
    // (those are handled separately via raw-string loading).
    std::optional<GpuColInfo> getGpuColInfo(const std::string& table,
                                             const std::string& column) const {
        const auto* t = getTable(table);
        if (!t) return std::nullopt;
        const auto* cs = t->getColumn(column);
        if (!cs) return std::nullopt;
        switch (cs->type) {
            case ColumnType::Int32:      return GpuColInfo{cs->index, GpuColKind::U32};
            case ColumnType::Float32:    return GpuColInfo{cs->index, GpuColKind::F32};
            case ColumnType::Date:       return GpuColInfo{cs->index, GpuColKind::DateU32};
            case ColumnType::StringChar: return GpuColInfo{cs->index, GpuColKind::StrCharU32};
            case ColumnType::StringHash: return std::nullopt;
        }
        return std::nullopt;
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
    
    SchemaRegistry() = default;  // Starts empty; call initTPCH() to register TPC-H schema
    
private:
    static inline SchemaRegistry* s_override = nullptr;
    mutable std::mutex m_mutex;
    std::unordered_map<std::string, TableSchema> m_tables;
};

} // namespace engine
