#include "GpuExecutor.hpp"
#include "Operators.hpp"
#include <vector>
#include <algorithm>
#include <cstring>

namespace engine {

bool GpuExecutor::executeLimit(const IRLimit& limit, TableResult& table) {
    if (limit.count < 0) return true;
    
    size_t offset = static_cast<size_t>(std::max(limit.offset, int64_t(0)));
    size_t count = static_cast<size_t>(limit.count);
    
    if (offset >= table.rowCount) {
        table.rowCount = 0;
        for (auto& col : table.u32Cols) col.clear();
        for (auto& col : table.f32Cols) col.clear();
        for (auto& col : table.stringCols) col.clear();
        table.u32ColsGPU.clear();
        table.f32ColsGPU.clear();
        return true;
    }
    
    size_t end = std::min(offset + count, table.rowCount);
    uint32_t newCount = static_cast<uint32_t>(end - offset);

    // Build GPU index buffer [offset, offset+1, ..., end-1]
    GpuBuffer indices = GpuOps::iotaU32(newCount);
    if (indices && offset > 0) {
        GpuBuffer shifted = GpuOps::arithAddConstU32(indices, static_cast<uint32_t>(offset), newCount);
        indices = std::move(shifted);
    }

    bool hasGPU = (indices != nullptr);

    // GPU gather u32 columns
    for (size_t i = 0; i < table.u32Cols.size(); ++i) {
        MTL::Buffer* gpuBuf = (i < table.u32ColsGPU.size()) ? table.u32ColsGPU[i] : nullptr;
        if (hasGPU && gpuBuf) {
            GpuBuffer gathered = GpuOps::gatherU32(gpuBuf, indices, newCount);
            if (gathered) {
                // Skip CPU download — GPU buffer is authoritative; lazy-fetch at output
                table.u32Cols[i].clear();
                if (i < table.u32ColsGPU.size()) {
                    table.u32ColsGPU[i] = std::move(gathered);
                }
                continue;
            }
        }
        // CPU fallback
        table.u32Cols[i] = std::vector<uint32_t>(table.u32Cols[i].begin() + offset,
                                                  table.u32Cols[i].begin() + end);
    }

    // GPU gather f32 columns
    for (size_t i = 0; i < table.f32Cols.size(); ++i) {
        MTL::Buffer* gpuBuf = (i < table.f32ColsGPU.size()) ? table.f32ColsGPU[i] : nullptr;
        if (hasGPU && gpuBuf) {
            GpuBuffer gathered = GpuOps::gatherF32(gpuBuf, indices, newCount);
            if (gathered) {
                // Skip CPU download — GPU buffer is authoritative; lazy-fetch at output
                table.f32Cols[i].clear();
                if (i < table.f32ColsGPU.size()) {
                    table.f32ColsGPU[i] = std::move(gathered);
                }
                continue;
            }
        }
        // CPU fallback
        table.f32Cols[i] = std::vector<float>(table.f32Cols[i].begin() + offset,
                                               table.f32Cols[i].begin() + end);
    }

    // String columns: CPU slice (no GPU representation in TableResult)
    for (auto& col : table.stringCols) {
        col = std::vector<std::string>(col.begin() + offset, col.begin() + end);
    }
    
    table.rowCount = newCount;
    return true;
}

} // namespace engine
