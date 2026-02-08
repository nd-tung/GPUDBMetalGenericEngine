// GPU column staging implementation
#include "ColumnStoreGPU.hpp"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>

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
    std::cerr << "[GPU] Device: " << m_device->name()->utf8String() << std::endl;
    m_device->setShouldMaximizeConcurrentCompilation(true);
    m_queue = m_device->newCommandQueue();

    // Strategy 1: Load pre-compiled metallib (fastest startup, requires Xcode metal CLI)
    NS::Error* error = nullptr;
    auto path = NS::String::string("build/kernels.metallib", NS::UTF8StringEncoding);
    m_library = m_device->newLibrary(path, &error);
    if (m_library) {
        std::cerr << "[GPU] Loaded pre-compiled Metal library (build/kernels.metallib)" << std::endl;
        pool->release();
        return;
    }

    // Strategy 2: Runtime compilation from .metal source (works with CommandLineTools only)
    std::cerr << "[GPU] Pre-compiled metallib not found, compiling shaders at runtime..." << std::endl;
    std::ifstream metalFile("kernels/Operators.metal");
    if (metalFile.is_open()) {
        std::ostringstream oss;
        oss << metalFile.rdbuf();
        std::string src = oss.str();
        auto srcStr = NS::String::string(src.c_str(), NS::UTF8StringEncoding);
        auto opts = MTL::CompileOptions::alloc()->init();
        NS::Error* compileError = nullptr;
        m_library = m_device->newLibrary(srcStr, opts, &compileError);
        opts->release();
        if (m_library) {
            std::cerr << "[GPU] Runtime Metal shader compilation succeeded (" 
                      << src.size() / 1024 << " KB source)." << std::endl;
        } else {
            std::cerr << "[GPU] Runtime compilation FAILED." << std::endl;
            if (compileError) std::cerr << "  Error: " << compileError->localizedDescription()->utf8String() << std::endl;
        }
    } else {
        std::cerr << "[GPU] Could not open kernels/Operators.metal for runtime compilation." << std::endl;
    }

    // Strategy 3: Default library (embedded in binary — unlikely to have our kernels)
    if (!m_library) {
        std::cerr << "[GPU] Falling back to default Metal library." << std::endl;
        m_library = m_device->newDefaultLibrary();
    }

    if (!m_library) {
        std::cerr << "[GPU] FATAL: No Metal shader library available. GPU operations will fail." << std::endl;
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
