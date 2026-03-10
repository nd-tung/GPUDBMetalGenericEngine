#pragma once
#include <Metal/Metal.hpp>

namespace engine {

// ============================================================================
// GpuBuffer — RAII wrapper for MTL::Buffer* with shared ownership.
//
// Semantics:
//   - Default-constructs to nullptr.
//   - Wrapping ctor takes ownership (no extra retain).
//   - Copy retains, destruction releases, move steals.
//   - Multiple GpuBuffer instances may safely alias the same Metal buffer;
//     each independently holds +1 refcount.
//   - Implicit conversion to MTL::Buffer* for seamless use with existing APIs.
//
// Migration note (E5):
//   GpuOps static methods currently return raw MTL::Buffer*. Callers are
//   responsible for wrapping the result in a GpuBuffer or calling release().
//   Changing GpuOps return types to GpuBuffer is desirable but UNSAFE to do
//   blindly: if a caller stores the result as MTL::Buffer* (via implicit
//   conversion), the temporary GpuBuffer is destroyed immediately and the
//   raw pointer becomes dangling.
//   Safe migration path: change GpuOps returns one-at-a-time, grep every
//   call site, and ensure each stores the result as GpuBuffer (not raw ptr).
// ============================================================================
class GpuBuffer {
    MTL::Buffer* buf_ = nullptr;
public:
    // -- Constructors / destructor --
    GpuBuffer() noexcept = default;
    explicit GpuBuffer(MTL::Buffer* b) noexcept : buf_(b) {}          // takes ownership
    ~GpuBuffer() { if (buf_) buf_->release(); }

    // -- Copy (shared ownership via retain) --
    GpuBuffer(const GpuBuffer& o) noexcept : buf_(o.buf_) { if (buf_) buf_->retain(); }
    GpuBuffer& operator=(const GpuBuffer& o) noexcept {
        if (this != &o) {
            if (buf_) buf_->release();
            buf_ = o.buf_;
            if (buf_) buf_->retain();
        }
        return *this;
    }

    // -- Move (steal ownership) --
    GpuBuffer(GpuBuffer&& o) noexcept : buf_(o.buf_) { o.buf_ = nullptr; }
    GpuBuffer& operator=(GpuBuffer&& o) noexcept {
        if (this != &o) {
            if (buf_) buf_->release();
            buf_ = o.buf_;
            o.buf_ = nullptr;
        }
        return *this;
    }

    // -- Accessors --
    MTL::Buffer* get()  const noexcept { return buf_; }
    MTL::Buffer* operator->() const noexcept { return buf_; }   // enables buf->contents() etc.
    operator MTL::Buffer*()   const noexcept { return buf_; }   // implicit for API compat
    explicit operator bool()  const noexcept { return buf_ != nullptr; }

    // -- Mutators --
    /// Replace the managed pointer. Releases the old buffer (if any).
    /// The new pointer is taken with NO extra retain (caller transfers ownership).
    void reset(MTL::Buffer* b = nullptr) noexcept {
        if (buf_ != b) { if (buf_) buf_->release(); buf_ = b; }
    }

    /// Convenience: assign nullptr to release.
    GpuBuffer& operator=(std::nullptr_t) noexcept { reset(); return *this; }

    /// Release ownership and return the raw pointer WITHOUT decrementing refcount.
    [[nodiscard]] MTL::Buffer* detach() noexcept { auto* p = buf_; buf_ = nullptr; return p; }

    // -- Comparison (for use in containers / dedup sets) --
    bool operator==(const GpuBuffer& o) const noexcept { return buf_ == o.buf_; }
    bool operator!=(const GpuBuffer& o) const noexcept { return buf_ != o.buf_; }
    bool operator==(std::nullptr_t)     const noexcept { return buf_ == nullptr; }
    bool operator!=(std::nullptr_t)     const noexcept { return buf_ != nullptr; }
};

} // namespace engine
