// GPU column staging
#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>

// Forward declare Metal types (included in .cpp)
namespace MTL { class Device; class Buffer; class Library; class CommandQueue; }

namespace engine {

struct FlatStringColumn; // defined in GpuExecutorPriv.hpp

struct GPUColumn {
    std::string name;
    std::size_t count = 0;
    MTL::Buffer* buffer = nullptr; // Shared memory buffer
};

// Cached flat-string column pair (offsets + chars).
struct GPUFlatStringColumn {
    std::string name;
    std::size_t rowCount   = 0;
    std::size_t totalChars = 0;
    MTL::Buffer* offsets = nullptr; // uint32_t[rowCount + 1]  Arrow-style
    MTL::Buffer* chars   = nullptr; // uint8_t[totalChars]
};

// Singleton staging cache for GPU buffers.
class ColumnStoreGPU {
public:
    static ColumnStoreGPU& instance();

    void initialize(); // lazy Metal device/library acquisition

    // Upload (or reuse) a float column. Returns GPUColumn* (owned by store).
    GPUColumn* stageFloatColumn(const std::string& name,
                                const std::vector<float>& data);

    // Upload (or reuse) a u32 column. Returns GPUColumn* (owned by store).
    GPUColumn* stageU32Column(const std::string& name,
                              const std::vector<uint32_t>& data);

    // Upload (or reuse) a flat Arrow-style string column (offsets + chars).
    // Returns FlatStringColumn with retained buffers the caller can use directly.
    FlatStringColumn stageFlatStringColumn(const std::string& name,
                                           const std::vector<std::string>& data);

    // Return an already-staged column, or nullptr if not present.
    GPUColumn* getColumn(const std::string& name);

    MTL::Device* device() const { return m_device; }
    MTL::Library* library() const { return m_library; }
    MTL::CommandQueue* queue() const { return m_queue; }

private:
    ColumnStoreGPU() = default;
    ColumnStoreGPU(const ColumnStoreGPU&) = delete;
    ColumnStoreGPU& operator=(const ColumnStoreGPU&) = delete;

    MTL::Device* m_device = nullptr;
    MTL::Library* m_library = nullptr;
    MTL::CommandQueue* m_queue = nullptr;
    std::map<std::string, GPUColumn> m_columns; // name → column
    std::map<std::string, GPUFlatStringColumn> m_flatStrings; // name → flat string
};

} // namespace engine
