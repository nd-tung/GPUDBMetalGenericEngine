#pragma once
#include <cstdlib>
#include <string>
#include <algorithm>

/// Return true if the environment variable @p name is set to
/// "1", "true", "on", or "yes" (case-insensitive).
inline bool env_truthy(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return false;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return (s == "1" || s == "true" || s == "on" || s == "yes");
}
