#pragma once
#include <cstdlib>
#include <string>
#include <algorithm>
#include <unordered_map>

/// Return true if the environment variable @p name is set to
/// "1", "true", "on", or "yes" (case-insensitive).
/// Results are cached on first lookup per variable name.
inline bool env_truthy(const char* name) {
    static std::unordered_map<std::string, bool> cache;
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    const char* v = std::getenv(name);
    bool result = false;
    if (v) {
        std::string s(v);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        result = (s == "1" || s == "true" || s == "on" || s == "yes");
    }
    cache[name] = result;
    return result;
}
