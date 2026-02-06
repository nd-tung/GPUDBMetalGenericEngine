// Minimal implementation of GPU column staging
#include "ColumnStoreGPU.hpp"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <chrono>
#include <iostream>

namespace engine {

ColumnStoreGPU& ColumnStoreGPU::instance() { static ColumnStoreGPU inst; return inst; }

void ColumnStoreGPU::initialize() {
    if (m_device) return; // already

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    m_device = MTL::CreateSystemDefaultDevice();
    if (!m_device) {
        std::cerr << "[GPU] No Metal device available" << std::endl;
        pool->release();
        return;
    }
    m_device->setShouldMaximizeConcurrentCompilation(true);
    m_queue = m_device->newCommandQueue();
    // Load metallib from file to get latest kernels
    NS::Error* error = nullptr;
    auto path = NS::String::string("build/kernels.metallib", NS::UTF8StringEncoding);
    m_library = m_device->newLibrary(path, &error);
    if (!m_library) {
        std::cerr << "[GPU] Failed to load Metal library from " << path->utf8String() << std::endl;
        if (error) std::cerr << "  Reason: " << error->localizedDescription()->utf8String() << std::endl;
        // Try default library as fallback
        m_library = m_device->newDefaultLibrary();
    }

    pool->release();
}

GPUColumn* ColumnStoreGPU::stageFloatColumn(const std::string& name,
                                            const std::vector<float>& data) {
    initialize();
    if (!m_device || !m_library) return nullptr;
    auto it = m_columns.find(name);
    if (it != m_columns.end()) {
        // Reuse if counts match; else recreate
        if (it->second.count == data.size()) return &it->second;
        // Release old buffer and recreate
        if (it->second.buffer) it->second.buffer->release();
        m_columns.erase(it);
    }
    GPUColumn col; col.name = name; col.count = data.size();
    const unsigned long bytes = data.size() * sizeof(float);
    col.buffer = m_device->newBuffer(data.data(), bytes, MTL::ResourceStorageModeShared);
    auto [insertIt, _] = m_columns.emplace(name, col);
    return &insertIt->second;
}

GPUColumn* ColumnStoreGPU::stageU32Column(const std::string& name,
                                          const std::vector<uint32_t>& data) {
    initialize();
    if (!m_device || !m_library) return nullptr;
    auto it = m_columns.find(name);
    if (it != m_columns.end()) {
        if (it->second.count == data.size()) return &it->second;
        if (it->second.buffer) it->second.buffer->release();
        m_columns.erase(it);
    }
    GPUColumn col;
    col.name = name;
    col.count = data.size();
    const unsigned long bytes = data.size() * sizeof(uint32_t);
    col.buffer = m_device->newBuffer(data.data(), bytes, MTL::ResourceStorageModeShared);
    auto [insertIt, _] = m_columns.emplace(name, col);
    return &insertIt->second;
}

GPUColumn* ColumnStoreGPU::getColumn(const std::string& name) {
    initialize();
    auto it = m_columns.find(name);
    if (it == m_columns.end()) return nullptr;
    return &it->second;
}

} // namespace engine
