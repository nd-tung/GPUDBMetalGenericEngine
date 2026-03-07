#pragma once
/// Lightweight error helpers for the GPU DB engine.
/// Provides macros that embed file:line info in exception messages.

#include <stdexcept>
#include <string>

/// Throw std::runtime_error with file:line prefix.
/// Usage: ENGINE_THROW("message " + detail);
#define ENGINE_THROW(msg) \
    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + (msg))

/// Assert a condition; throw on failure with file:line + expression text.
/// Usage: ENGINE_ASSERT(ptr != nullptr, "buffer must not be null");
#define ENGINE_ASSERT(cond, msg) \
    do { if (!(cond)) ENGINE_THROW(std::string("Assertion failed: ") + #cond + " — " + (msg)); } while (0)
