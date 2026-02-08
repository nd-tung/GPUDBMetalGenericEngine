#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace engine {

// ============================================================================
// Generic column schema system
// ============================================================================

enum class ColumnType {
    Int32,
    Int64,
    Float32,
    Float64,
    Date,           // YYYYMMDD as integer
    StringHash,     // FNV1a hash of string
    StringChar,     // Single character stored as char code
    StringRaw       // Raw string (CPU only)
};

inline const char* columnTypeName(ColumnType t) {
    switch (t) {
        case ColumnType::Int32: return "int32";
        case ColumnType::Int64: return "int64";
        case ColumnType::Float32: return "float32";
        case ColumnType::Float64: return "float64";
        case ColumnType::Date: return "date";
        case ColumnType::StringHash: return "string_hash";
        case ColumnType::StringChar: return "string_char";
        case ColumnType::StringRaw: return "string_raw";
    }
    return "?";
}

struct ColumnSchema {
    std::string name;
    int index;              // 0-based column index in .tbl file
    ColumnType type;
    bool nullable = true;
    bool isPrimaryKey = false;
    bool isForeignKey = false;
    std::string references;  // table.column for foreign keys
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
    
    int getColumnIndex(const std::string& colName) const {
        for (const auto& c : columns) {
            if (c.name == colName) return c.index;
        }
        return -1;
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
    
    // Load schema from a SQL file (CREATE TABLE statements)
    bool loadFromSQL(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return false;
        
        std::stringstream buf;
        buf << file.rdbuf();
        std::string content = buf.str();
        
        // Parse CREATE TABLE statements
        // Format: CREATE TABLE name (col1 type1, col2 type2, ...);
        
        size_t pos = 0;
        while ((pos = content.find("CREATE TABLE", pos)) != std::string::npos) {
            size_t nameStart = pos + 12;
            while (nameStart < content.size() && std::isspace(content[nameStart])) nameStart++;
            
            size_t nameEnd = nameStart;
            while (nameEnd < content.size() && !std::isspace(content[nameEnd]) && content[nameEnd] != '(') nameEnd++;
            
            std::string tableName = content.substr(nameStart, nameEnd - nameStart);
            
            size_t parenStart = content.find('(', nameEnd);
            size_t parenEnd = content.find(')', parenStart);
            if (parenStart == std::string::npos || parenEnd == std::string::npos) {
                pos = nameEnd;
                continue;
            }
            
            std::string colDefs = content.substr(parenStart + 1, parenEnd - parenStart - 1);
            
            TableSchema schema;
            schema.name = tableName;
            
            // Parse column definitions
            int colIndex = 0;
            size_t start = 0;
            while (start < colDefs.size()) {
                size_t comma = colDefs.find(',', start);
                if (comma == std::string::npos) comma = colDefs.size();
                
                std::string def = colDefs.substr(start, comma - start);
                // Trim
                while (!def.empty() && std::isspace(def.front())) def.erase(def.begin());
                while (!def.empty() && std::isspace(def.back())) def.pop_back();
                
                // Skip constraints like PRIMARY KEY, FOREIGN KEY
                std::string defLower = def;
                std::transform(defLower.begin(), defLower.end(), defLower.begin(), ::tolower);
                if (defLower.find("primary key") != std::string::npos ||
                    defLower.find("foreign key") != std::string::npos ||
                    defLower.find("constraint") != std::string::npos) {
                    start = comma + 1;
                    continue;
                }
                
                // Parse "colname type"
                size_t space = def.find(' ');
                if (space != std::string::npos) {
                    std::string colName = def.substr(0, space);
                    std::string typePart = def.substr(space + 1);
                    
                    // Determine type
                    std::string typeLower = typePart;
                    std::transform(typeLower.begin(), typeLower.end(), typeLower.begin(), ::tolower);
                    
                    ColumnType type = ColumnType::StringHash;
                    if (typeLower.find("int") != std::string::npos) {
                        type = ColumnType::Int32;
                    } else if (typeLower.find("decimal") != std::string::npos ||
                               typeLower.find("numeric") != std::string::npos ||
                               typeLower.find("float") != std::string::npos ||
                               typeLower.find("double") != std::string::npos ||
                               typeLower.find("real") != std::string::npos) {
                        type = ColumnType::Float32;
                    } else if (typeLower.find("date") != std::string::npos) {
                        type = ColumnType::Date;
                    } else if (typeLower.find("char(1)") != std::string::npos) {
                        type = ColumnType::StringChar;
                    } else if (typeLower.find("varchar") != std::string::npos ||
                               typeLower.find("char") != std::string::npos ||
                               typeLower.find("text") != std::string::npos) {
                        type = ColumnType::StringHash;
                    }
                    
                    schema.columns.push_back({colName, colIndex, type});
                    colIndex++;
                }
                
                start = comma + 1;
            }
            
            if (!schema.columns.empty()) {
                registerTable(std::move(schema));
            }
            
            pos = parenEnd;
        }
        
        return true;
    }
    
    // Get all table names
    std::vector<std::string> getTableNames() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : tables_) {
            names.push_back(name);
        }
        return names;
    }
    
private:
    SchemaRegistry() {
        initTPCH();  // Initialize with TPC-H by default
    }
    
    std::unordered_map<std::string, TableSchema> tables_;
};

// ============================================================================
// Helper to infer table from column name prefix
// ============================================================================

inline std::string inferTableFromColumn(const std::string& colName) {
    if (colName.rfind("l_", 0) == 0) return "lineitem";
    if (colName.rfind("o_", 0) == 0) return "orders";
    if (colName.rfind("c_", 0) == 0) return "customer";
    if (colName.rfind("p_", 0) == 0) return "part";
    if (colName.rfind("s_", 0) == 0) return "supplier";
    if (colName.rfind("ps_", 0) == 0) return "partsupp";
    if (colName.rfind("n_", 0) == 0) return "nation";
    if (colName.rfind("r_", 0) == 0) return "region";
    return "";
}

} // namespace engine
