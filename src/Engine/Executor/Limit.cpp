#include "GpuExecutor.hpp"
#include <vector>
#include <algorithm>

namespace engine {

bool GpuExecutor::executeLimit(const IRLimit& limit, TableResult& table) {
    if (limit.count < 0) return true;
    
    size_t offset = static_cast<size_t>(std::max(limit.offset, int64_t(0)));
    size_t count = static_cast<size_t>(limit.count);
    
    if (offset >= table.rowCount) {
        table.rowCount = 0;
        for (auto& col : table.u32_cols) col.clear();
        for (auto& col : table.f32_cols) col.clear();
        return true;
    }
    
    size_t end = std::min(offset + count, table.rowCount);
    
    for (auto& col : table.u32_cols) {
        col = std::vector<uint32_t>(col.begin() + offset, col.begin() + end);
    }
    for (auto& col : table.f32_cols) {
        col = std::vector<float>(col.begin() + offset, col.begin() + end);
    }
    for (auto& col : table.string_cols) {
        col = std::vector<std::string>(col.begin() + offset, col.begin() + end);
    }
    
    table.rowCount = end - offset;
    return true;
}

} // namespace engine
