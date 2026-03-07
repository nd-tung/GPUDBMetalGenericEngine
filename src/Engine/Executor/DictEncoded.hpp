#pragma once
// Dictionary-encoded string column: sorted unique strings with per-row IDs.
// Uses GpuBuffer RAII — compiler-generated copy/move/dtor handle retain/release.

#include "GpuBuffer.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <cstdint>

namespace engine {

struct DictEncoded {
    std::vector<std::string> dictionary;  // sorted unique strings
    std::vector<uint32_t> ids;            // per-row dictionary ID (CPU mirror, may be lazy)
    GpuBuffer idsGPU;                     // per-row dictionary ID (GPU) — primary representation
    uint32_t rowCount = 0;

    // Lookup: given a string value, return its dictionary ID (or UINT32_MAX if not found)
    uint32_t lookupId(const std::string& value) const {
        auto it = std::lower_bound(dictionary.begin(), dictionary.end(), value);
        if (it != dictionary.end() && *it == value)
            return static_cast<uint32_t>(it - dictionary.begin());
        return UINT32_MAX;
    }

    // Reverse lookup: given a dictionary ID, return the string (or "" if out of range)
    const std::string& lookupString(uint32_t id) const {
        static const std::string empty;
        return (id < dictionary.size()) ? dictionary[id] : empty;
    }

    // Sync CPU mirror from GPU buffer (lazy — call when CPU ids needed)
    void ensureIdsCPU() {
        if (idsGPU && ids.size() != rowCount) {
            ids.resize(rowCount);
            if (rowCount > 0) {
                std::memcpy(ids.data(), idsGPU->contents(), rowCount * sizeof(uint32_t));
            }
        }
    }

    // Materialize full string column from dict IDs (for output or legacy consumers)
    std::vector<std::string> materialize() const {
        std::vector<std::string> result(rowCount);
        const uint32_t* idPtr = nullptr;
        if (idsGPU) idPtr = static_cast<const uint32_t*>(idsGPU->contents());
        else if (!ids.empty()) idPtr = ids.data();
        if (!idPtr) return result;
        for (uint32_t i = 0; i < rowCount; ++i) {
            uint32_t id = idPtr[i];
            if (id < dictionary.size()) result[i] = dictionary[id];
        }
        return result;
    }

    // Check if this dict encoding is valid and has data
    bool valid() const { return !dictionary.empty() && (idsGPU || !ids.empty()) && rowCount > 0; }

    // Release all resources
    void release() {
        idsGPU = nullptr;
        ids.clear();
        dictionary.clear();
        rowCount = 0;
    }
};

} // namespace engine
