// GPU column staging implementation
#include "GpuColumnStore.hpp"
#include "Logger.hpp"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <filesystem>

namespace engine {

// Resolve #include "..." directives in Metal source for runtime compilation
// (the runtime newLibraryWithSource API does not support #include).
static std::string preprocessMetalIncludes(const std::string& source,
                                           const std::string& baseDir) {
    std::istringstream stream(source);
    std::ostringstream out;
    std::regex includeRe(R"raw(^\s*#include\s+"([^"]+)")raw");    std::string line;
    while (std::getline(stream, line)) {
        std::smatch m;
        if (std::regex_search(line, m, includeRe)) {
            std::string incPath = baseDir + "/" + m[1].str();
            std::ifstream incFile(incPath);
            if (incFile.is_open()) {
                std::ostringstream inc;
                inc << incFile.rdbuf();
                out << inc.str() << "\n";
            } else {
                out << "// WARNING: could not resolve " << line << "\n";
            }
        } else {
            out << line << "\n";
        }
    }
    return out.str();
}

GpuColumnStore& GpuColumnStore::instance() {
    if (s_override) return *s_override;
    static GpuColumnStore inst;
    return inst;
}

void GpuColumnStore::setTestInstance(GpuColumnStore* mock) { s_override = mock; }
void GpuColumnStore::resetTestInstance() { s_override = nullptr; }

GpuColumnStore::~GpuColumnStore() {
    // Release all staged column buffers
    for (auto& [name, col] : m_columns) {
        if (col.buffer) { col.buffer->release(); col.buffer = nullptr; }
    }
    m_columns.clear();
    // Release Metal infrastructure (order: queue, library, device)
    if (m_queue)   { m_queue->release();   m_queue = nullptr; }
    if (m_library) { m_library->release(); m_library = nullptr; }
    if (m_device)  { m_device->release();  m_device = nullptr; }
}

// Internal: must be called with m_mutex already held.
void GpuColumnStore::initializeImpl() {
    if (m_device) return; // already

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    m_device = MTL::CreateSystemDefaultDevice();
    if (!m_device) {
        std::cerr << "[GPU] No Metal device available" << std::endl;
        pool->release();
        return;
    }
    LOG_INFO("GPU", "Device: " << m_device->name()->utf8String());
    m_device->setShouldMaximizeConcurrentCompilation(true);
    m_queue = m_device->newCommandQueue();

    // Strategy 1: Load pre-compiled metallib (fastest startup, requires Xcode metal CLI)
    NS::Error* error = nullptr;
    auto path = NS::String::string("build/kernels.metallib", NS::UTF8StringEncoding);
    m_library = m_device->newLibrary(path, &error);
    if (m_library) {
        LOG_INFO("GPU", "Loaded pre-compiled Metal library (build/kernels.metallib)");
        pool->release();
        return;
    }

    // Strategy 2: Runtime compilation from .metal source (works with CommandLineTools only)
    LOG_INFO("GPU", "Pre-compiled metallib not found, compiling shaders at runtime...");
    std::ifstream metalFile("kernels/Operators.metal");
    if (metalFile.is_open()) {
        std::ostringstream oss;
        oss << metalFile.rdbuf();
        std::string raw = oss.str();
        // Resolve #include "..." directives (newLibraryWithSource doesn't support them)
        std::string src = preprocessMetalIncludes(raw, "kernels");
        auto srcStr = NS::String::string(src.c_str(), NS::UTF8StringEncoding);
        auto opts = MTL::CompileOptions::alloc()->init();
        NS::Error* compileError = nullptr;
        m_library = m_device->newLibrary(srcStr, opts, &compileError);
        opts->release();
        if (m_library) {
            LOG_INFO("GPU", "Runtime Metal shader compilation succeeded (" 
                      << src.size() / 1024 << " KB source)");
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

void GpuColumnStore::initialize() {
    std::lock_guard<std::mutex> lock(m_mutex);
    initializeImpl();
}

GpuColumn* GpuColumnStore::stageFloatColumn(const std::string& name,
                                            const std::vector<float>& data) {
    std::lock_guard<std::mutex> lock(m_mutex);
    initializeImpl();
    if (!m_device || !m_library) return nullptr;
    auto it = m_columns.find(name);
    if (it != m_columns.end()) {
        // Reuse if counts match; else recreate
        if (it->second.count == data.size()) return &it->second;
        // Release old buffer and recreate
        if (it->second.buffer) it->second.buffer->release();
        m_columns.erase(it);
    }
    GpuColumn col; col.name = name; col.count = data.size();
    const unsigned long bytes = data.size() * sizeof(float);
    col.buffer = m_device->newBuffer(data.data(), bytes, MTL::ResourceStorageModeShared);
    if (!col.buffer) {
        LOG_ERROR("GPU", "newBuffer failed for float column '" << name << "' (" << bytes << " bytes)");
        return nullptr;
    }
    auto [insertIt, _] = m_columns.emplace(name, col);
    return &insertIt->second;
}

GpuColumn* GpuColumnStore::stageU32Column(const std::string& name,
                                          const std::vector<uint32_t>& data) {
    std::lock_guard<std::mutex> lock(m_mutex);
    initializeImpl();
    if (!m_device || !m_library) return nullptr;
    auto it = m_columns.find(name);
    if (it != m_columns.end()) {
        if (it->second.count == data.size()) return &it->second;
        if (it->second.buffer) it->second.buffer->release();
        m_columns.erase(it);
    }
    GpuColumn col;
    col.name = name;
    col.count = data.size();
    const unsigned long bytes = data.size() * sizeof(uint32_t);
    col.buffer = m_device->newBuffer(data.data(), bytes, MTL::ResourceStorageModeShared);
    if (!col.buffer) {
        LOG_ERROR("GPU", "newBuffer failed for u32 column '" << name << "' (" << bytes << " bytes)");
        return nullptr;
    }
    auto [insertIt, _] = m_columns.emplace(name, col);
    return &insertIt->second;
}

GpuColumn* GpuColumnStore::getColumn(const std::string& name) {
    std::lock_guard<std::mutex> lock(m_mutex);
    initializeImpl();
    auto it = m_columns.find(name);
    if (it == m_columns.end()) return nullptr;
    return &it->second;
}

} // namespace engine
