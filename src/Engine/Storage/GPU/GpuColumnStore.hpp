// GPU column staging
#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>

// Forward declare Metal types (included in .cpp)
namespace MTL { class Device; class Buffer; class Library; class CommandQueue; }

namespace engine {

struct GpuColumn {
    std::string name;
    std::size_t count = 0;
    MTL::Buffer* buffer = nullptr; // Shared memory buffer
};

// Singleton staging cache for GPU buffers.
// Owns Metal device, library, command queue, and all staged column buffers.
// Thread-safe: all public methods are guarded by an internal mutex.
class GpuColumnStore {
public:
    static GpuColumnStore& instance();
    ~GpuColumnStore();

    // For testing: inject a custom/mock instance; call resetTestInstance() to restore.
    static void setTestInstance(GpuColumnStore* mock);
    static void resetTestInstance();

    void initialize(); // lazy Metal device/library acquisition

    // Upload (or reuse) a float column. Returns GpuColumn* (owned by store).
    GpuColumn* stageFloatColumn(const std::string& name,
                                const std::vector<float>& data);

    // Upload (or reuse) a u32 column. Returns GpuColumn* (owned by store).
    GpuColumn* stageU32Column(const std::string& name,
                              const std::vector<uint32_t>& data);

    // Return an already-staged column, or nullptr if not present.
    GpuColumn* getColumn(const std::string& name);

    MTL::Device* device() const { return m_device; }
    MTL::Library* library() const { return m_library; }
    MTL::CommandQueue* queue() const { return m_queue; }

private:
    GpuColumnStore() = default;
    GpuColumnStore(const GpuColumnStore&) = delete;
    GpuColumnStore& operator=(const GpuColumnStore&) = delete;

    void initializeImpl(); // must be called with m_mutex held

    static inline GpuColumnStore* s_override = nullptr;
    mutable std::mutex m_mutex;
    MTL::Device* m_device = nullptr;
    MTL::Library* m_library = nullptr;
    MTL::CommandQueue* m_queue = nullptr;
    std::map<std::string, GpuColumn> m_columns; // name → column
};

} // namespace engine
