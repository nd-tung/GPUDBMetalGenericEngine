// KernelTimer.hpp - Comprehensive GPU kernel timing for Metal operations
#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <iomanip>
#include <sstream>

namespace engine {

// ============================================================================
// KernelTiming - Stores timing for a single kernel invocation
// ============================================================================
struct KernelTiming {
    std::string kernelName;
    std::string operation;      // e.g., "filter", "gather", "groupby"
    double durationMs = 0.0;
    uint32_t elementCount = 0;  // Number of elements processed
    double throughputME = 0.0;  // Million elements per second
};

// ============================================================================
// KernelTimer - Singleton for tracking all GPU kernel invocations
// ============================================================================
class KernelTimer {
public:
    static KernelTimer& instance() {
        static KernelTimer timer;
        return timer;
    }
    
    // Reset all timings for a new query
    void reset() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_timings.clear();
        m_totalGpuMs = 0.0;
    }
    
    // Record a kernel timing
    void record(const std::string& kernelName, const std::string& operation, 
                double durationMs, uint32_t elementCount = 0) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        KernelTiming timing;
        timing.kernelName = kernelName;
        timing.operation = operation;
        timing.durationMs = durationMs;
        timing.elementCount = elementCount;
        timing.throughputME = (durationMs > 0.0 && elementCount > 0) 
            ? (static_cast<double>(elementCount) / 1000000.0) / (durationMs / 1000.0)
            : 0.0;
        
        m_timings.push_back(timing);
        m_totalGpuMs += durationMs;
    }
    
    // Get total GPU time in milliseconds
    double totalGpuMs() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_totalGpuMs;
    }
    
    // Get all timings
    std::vector<KernelTiming> timings() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_timings;
    }
    
    // Get summary statistics
    std::string summary() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3);
        
        if (m_timings.empty()) {
            ss << "No GPU kernels recorded\n";
            return ss.str();
        }
        
        // Group by operation type
        std::map<std::string, double> opTotals;
        std::map<std::string, int> opCounts;
        
        for (const auto& t : m_timings) {
            opTotals[t.operation] += t.durationMs;
            opCounts[t.operation]++;
        }
        
        ss << "GPU Kernel Timing Summary:\n";
        ss << "─────────────────────────────────────────\n";
        
        for (const auto& [op, totalMs] : opTotals) {
            double pct = (m_totalGpuMs > 0) ? (totalMs / m_totalGpuMs * 100.0) : 0.0;
            ss << "  " << std::left << std::setw(15) << op 
               << ": " << std::right << std::setw(8) << totalMs << " ms"
               << " (" << std::setw(5) << pct << "%) "
               << "[" << opCounts[op] << " calls]\n";
        }
        
        ss << "─────────────────────────────────────────\n";
        ss << "  TOTAL GPU TIME: " << m_totalGpuMs << " ms\n";
        ss << "  Kernel invocations: " << m_timings.size() << "\n";
        
        return ss.str();
    }
    
    // Get detailed per-kernel breakdown
    std::string detailed() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3);
        
        if (m_timings.empty()) {
            return "No GPU kernels recorded\n";
        }
        
        ss << "Detailed Kernel Timings:\n";
        ss << "─────────────────────────────────────────────────────────────────\n";
        ss << std::left << std::setw(30) << "Kernel" 
           << std::right << std::setw(12) << "Time (ms)"
           << std::setw(15) << "Elements"
           << std::setw(15) << "ME/s"
           << "\n";
        ss << "─────────────────────────────────────────────────────────────────\n";
        
        for (const auto& t : m_timings) {
            ss << std::left << std::setw(30) << t.kernelName.substr(0, 29)
               << std::right << std::setw(12) << t.durationMs
               << std::setw(15) << t.elementCount
               << std::setw(15) << t.throughputME
               << "\n";
        }
        
        ss << "─────────────────────────────────────────────────────────────────\n";
        ss << "TOTAL: " << m_totalGpuMs << " ms\n";
        
        return ss.str();
    }

private:
    KernelTimer() = default;
    
    mutable std::mutex m_mutex;
    std::vector<KernelTiming> m_timings;
    double m_totalGpuMs = 0.0;
};

// ============================================================================
// ScopedKernelTimer - RAII timer for automatic kernel timing
// ============================================================================
class ScopedKernelTimer {
public:
    ScopedKernelTimer(const std::string& kernelName, const std::string& operation, 
                      uint32_t elementCount = 0)
        : m_kernelName(kernelName)
        , m_operation(operation)
        , m_elementCount(elementCount)
        , m_start(std::chrono::high_resolution_clock::now())
    {}
    
    ~ScopedKernelTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - m_start).count();
        KernelTimer::instance().record(m_kernelName, m_operation, ms, m_elementCount);
    }
    
    // Update element count if not known at construction time
    void setElementCount(uint32_t count) { m_elementCount = count; }

private:
    std::string m_kernelName;
    std::string m_operation;
    uint32_t m_elementCount;
    std::chrono::high_resolution_clock::time_point m_start;
};

} // namespace engine
