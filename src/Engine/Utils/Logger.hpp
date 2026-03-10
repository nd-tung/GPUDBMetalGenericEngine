#pragma once
// ============================================================================
// Logger.hpp — Lightweight structured logging for the GPU DB engine.
//
// Replaces raw `std::cerr` with level-gated, tag-prefixed output.
// Usage:
//   LOG_DEBUG("Filter", "applied predicate on " << rowCount << " rows");
//   LOG_INFO("Scan", "loaded table " << tableName);
//   LOG_WARN("Join", "fallback to CPU path");
//   LOG_ERROR("GPU", "kernel dispatch failed");
//
// Levels are controlled by environment variable GPUDB_LOG_LEVEL:
//   0 = ERROR only, 1 = +WARN, 2 = +INFO, 3 = +DEBUG (default: 0)
//
// The GPUDB_DEBUG_OPS env var (existing) enables DEBUG level for backward compat.
// ============================================================================

#include "EnvUtil.hpp"
#include <iostream>
#include <string>

namespace engine {

enum class LogLevel : int { Error = 0, Warn = 1, Info = 2, Debug = 3 };

class Logger {
public:
    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    LogLevel level() const { return level_; }

    void setLevel(LogLevel l) { level_ = l; }

    /// Check if a given level should produce output.
    bool enabled(LogLevel l) const { return static_cast<int>(l) <= static_cast<int>(level_); }

    /// Raw output stream (std::cerr). Callers should check enabled() first.
    std::ostream& stream() { return std::cerr; }

private:
    Logger() {
        // Honour existing GPUDB_DEBUG_OPS for backward compatibility
        if (env_truthy("GPUDB_DEBUG_OPS")) {
            level_ = LogLevel::Debug;
        } else {
            // GPUDB_LOG_LEVEL: 0-3
            const char* envVal = std::getenv("GPUDB_LOG_LEVEL");
            if (envVal) {
                int v = std::atoi(envVal);
                if (v >= 0 && v <= 3) level_ = static_cast<LogLevel>(v);
            }
        }
    }

    LogLevel level_ = LogLevel::Error;
};

} // namespace engine

// ============================================================================
// Convenience macros — zero overhead when level is disabled.
// Each macro checks the log level before evaluating the stream expression,
// so expensive string formatting is skipped when logging is off.
// ============================================================================

#define LOG_AT_LEVEL(lvl, tag, expr) \
    do { \
        if (::engine::Logger::instance().enabled(lvl)) { \
            ::engine::Logger::instance().stream() \
                << "[" << tag << "] " << expr << "\n"; \
        } \
    } while (0)

#define LOG_ERROR(tag, expr)  LOG_AT_LEVEL(::engine::LogLevel::Error, tag, expr)
#define LOG_WARN(tag, expr)   LOG_AT_LEVEL(::engine::LogLevel::Warn,  tag, expr)
#define LOG_INFO(tag, expr)   LOG_AT_LEVEL(::engine::LogLevel::Info,  tag, expr)
#define LOG_DEBUG(tag, expr)  LOG_AT_LEVEL(::engine::LogLevel::Debug, tag, expr)
