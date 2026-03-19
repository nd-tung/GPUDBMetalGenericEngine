#pragma once
// StringMaterializer: consolidated string materialization utilities.
// Eliminates duplicated dict→CPU and flat→CPU conversion logic that was
// previously scattered across EvalContext::ensureStringCol(),
// ResultTable::materializeDeferredStrings(), and inline handlers.

#include "FlatStringCol.hpp"
#include "DictEncoded.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace engine {

struct StringMaterializer {
    // Materialize a FlatStringCol (GPU Arrow-style) into a CPU string vector.
    static std::vector<std::string> fromFlat(const FlatStringCol& flat) {
        if (!flat.chars || !flat.offsets || !flat.lengths || flat.rowCount == 0)
            return {};
        const uint32_t* offs = static_cast<const uint32_t*>(flat.offsets->contents());
        const uint32_t* lens = static_cast<const uint32_t*>(flat.lengths->contents());
        const char* ch = static_cast<const char*>(flat.chars->contents());
        std::vector<std::string> result(flat.rowCount);
        for (uint32_t i = 0; i < flat.rowCount; ++i)
            result[i].assign(ch + offs[i], lens[i]);
        return result;
    }

    // Materialize a DictEncoded column into a CPU string vector.
    static std::vector<std::string> fromDict(const DictEncoded& dict) {
        if (!dict.valid()) return {};
        return dict.materialize();
    }

    // Materialize whichever string source is available (flat preferred, then dict).
    // Returns empty vector if neither is valid.
    static std::vector<std::string> materialize(const FlatStringCol* flat,
                                                 const DictEncoded* dict) {
        if (flat && flat->chars && flat->rowCount > 0)
            return fromFlat(*flat);
        if (dict && dict->valid())
            return fromDict(*dict);
        return {};
    }
};

} // namespace engine
