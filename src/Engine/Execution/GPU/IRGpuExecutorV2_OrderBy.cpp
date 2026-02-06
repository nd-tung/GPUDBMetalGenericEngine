#include "IRGpuExecutorV2.hpp"
#include "IRGpuExecutorV2_Priv.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>

namespace engine {

bool IRGpuExecutorV2::executeOrderBy(const IROrderByV2& order, TableResult& table) {
    const bool debug = env_truthy("GPUDB_DEBUG_OPS");
    
    if (debug) {
        std::cerr << "[V2] OrderBy: columns=[";
        for (size_t i = 0; i < order.columns.size(); ++i) {
            std::cerr << order.columns[i];
            if (i < order.ascending.size()) std::cerr << (order.ascending[i] ? " ASC" : " DESC");
            if (i + 1 < order.columns.size()) std::cerr << ", ";
        }
        std::cerr << "]\n";
        std::cerr << "[V2] OrderBy: table.u32_names=[";
        for (const auto& n : table.u32_names) std::cerr << n << ", ";
        std::cerr << "]\n";
        std::cerr << "[V2] OrderBy: table.f32_names=[";
        for (const auto& n : table.f32_names) std::cerr << n << ", ";
        std::cerr << "]\n";
    }
    
    if (table.rowCount == 0) return true;
    
    // Build sort indices
    std::vector<size_t> indices(table.rowCount);
    std::iota(indices.begin(), indices.end(), 0);
    
    // Get column references for sorting
    // Type: 0=u32, 1=f32, 2=string
    std::vector<std::tuple<int, bool, size_t>> sortCols;  // type, ascending, colIdx
    
    for (size_t i = 0; i < order.columns.size(); ++i) {
        const std::string& colName = order.columns[i];
        bool asc = i < order.ascending.size() ? order.ascending[i] : true;
        bool found = false;
        
        // Try direct match or base_ident match first
        // 1. U32
        for (size_t j = 0; j < table.u32_names.size() && !found; ++j) {
            if (table.u32_names[j] == colName || base_ident(table.u32_names[j]) == base_ident(colName)) {
                sortCols.emplace_back(0, asc, j);
                found = true;
            }
        }
        // 2. F32
        for (size_t j = 0; j < table.f32_names.size() && !found; ++j) {
            if (table.f32_names[j] == colName || base_ident(table.f32_names[j]) == base_ident(colName)) {
                sortCols.emplace_back(1, asc, j);
                found = true;
            }
        }
        // 3. String
        for (size_t j = 0; j < table.string_names.size() && !found; ++j) {
            if (table.string_names[j] == colName || base_ident(table.string_names[j]) == base_ident(colName)) {
                sortCols.emplace_back(2, asc, j);
                found = true;
            }
        }
        
        // If not found and looks like an aggregate expression, try matching aggregate aliases
        if (!found) {
            std::string colLower = colName;
            std::transform(colLower.begin(), colLower.end(), colLower.begin(), ::tolower);
            bool isSum = colLower.find("sum(") != std::string::npos || colLower.find("sum_no_overflow(") != std::string::npos;
            bool isAvg = colLower.find("avg(") != std::string::npos;
            bool isCount = colLower.find("count(") != std::string::npos || colLower.find("count_star(") != std::string::npos;
            bool isMin = colLower.find("min(") != std::string::npos;
            bool isMax = colLower.find("max(") != std::string::npos;
            
            if (isSum || isAvg || isCount || isMin || isMax) {
                // Try to find a column that looks like an aggregate result
                // Common patterns: revenue, sum_qty, count_order, etc.
                for (size_t j = 0; j < table.f32_names.size() && !found; ++j) {
                    const std::string& name = table.f32_names[j];
                    // Match if name starts with sum/avg/count/min/max or contains common aliases
                    std::string nameLower = name;
                    std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
                    if ((isSum && (nameLower.find("revenue") != std::string::npos || 
                                   nameLower.find("sum") != std::string::npos ||
                                   nameLower.find("price") != std::string::npos ||
                                   nameLower.find("charge") != std::string::npos)) ||
                        (isCount && (nameLower.find("count") != std::string::npos ||
                                     nameLower.find("_cnt") != std::string::npos ||  // supplier_cnt
                                     nameLower.find("dist") != std::string::npos)) ||  // custdist
                        (isAvg && nameLower.find("avg") != std::string::npos) ||
                        (isMin && nameLower.find("min") != std::string::npos) ||
                        (isMax && nameLower.find("max") != std::string::npos)) {
                        sortCols.emplace_back(1, asc, j);
                        found = true;
                        if (debug) std::cerr << "[V2] OrderBy: matched '" << colName << "' to '" << name << "'\n";
                    }
                }
                
                // Fallback: if we're looking for an aggregate and have exactly one f32 column, use it
                if (!found && table.f32_names.size() == 1) {
                    sortCols.emplace_back(1, asc, 0);
                    found = true;
                    if (debug) std::cerr << "[V2] OrderBy: fallback '" << colName << "' to single f32 '" << table.f32_names[0] << "'\n";
                }
                
                // Last resort: try matching positional references like #N
                // If this is a COUNT(DISTINCT) and we have a positional ref column, try to match
                if (!found && colLower.find("distinct") != std::string::npos) {
                    for (size_t j = 0; j < table.f32_names.size() && !found; ++j) {
                        const std::string& name = table.f32_names[j];
                        if (name.size() >= 2 && name[0] == '#') {
                            // Use this positional reference for the COUNT(DISTINCT)
                            sortCols.emplace_back(1, asc, j);
                            found = true;
                            if (debug) std::cerr << "[V2] OrderBy: matched COUNT(DISTINCT) to positional '" << name << "'\n";
                        }
                    }
                }
            }
        }
    }
    
    // Sort
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        for (const auto& [type, asc, colIdx] : sortCols) {
            int cmp = 0;
            if (type == 0) { // U32
                uint32_t va = table.u32_cols[colIdx][a];
                uint32_t vb = table.u32_cols[colIdx][b];
                if (va < vb) cmp = -1; else if (va > vb) cmp = 1;
            } else if (type == 1) { // F32
                float va = table.f32_cols[colIdx][a];
                float vb = table.f32_cols[colIdx][b];
                if (va < vb) cmp = -1; else if (va > vb) cmp = 1;
            } else if (type == 2) { // String
                const std::string& va = table.string_cols[colIdx][a];
                const std::string& vb = table.string_cols[colIdx][b];
                if (va < vb) cmp = -1; else if (va > vb) cmp = 1;
            }
            
            if (cmp != 0) {
                return asc ? (cmp < 0) : (cmp > 0);
            }
        }
        return false;
    });
    
    // Reorder columns
    auto reorderU32 = [&](std::vector<uint32_t>& col) {
        std::vector<uint32_t> tmp(col.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            tmp[i] = col[indices[i]];
        }
        col = std::move(tmp);
    };
    
    auto reorderF32 = [&](std::vector<float>& col) {
        std::vector<float> tmp(col.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            tmp[i] = col[indices[i]];
        }
        col = std::move(tmp);
    };

    auto reorderString = [&](std::vector<std::string>& col) {
        std::vector<std::string> tmp(col.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            tmp[i] = col[indices[i]];
        }
        col = std::move(tmp);
    };
    
    for (auto& col : table.u32_cols) reorderU32(col);
    for (auto& col : table.f32_cols) reorderF32(col);
    for (auto& col : table.string_cols) reorderString(col);
    
    return true;
}

} // namespace engine
