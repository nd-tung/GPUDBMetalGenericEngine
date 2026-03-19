#pragma once
// ColumnResolver: consolidated column name resolution for EvalContext.
// Replaces duplicated resolution logic with a single template that works
// with any set of column maps (GPU-only, all maps, etc.).

#include "EngineConfig.hpp"
#include <string>
#include <utility>

namespace engine {

struct ColumnResolver {
    // Resolve a column name against a set of maps using the standard search order:
    //   1) Exact match
    //   2) Suffixed _1 through _kMaxColumnSuffixSearch
    //   3) _rhs_ prefix match
    //   4) Column alias lookup
    //
    // `existsFn(name) -> bool`: returns true if the name exists in the relevant maps.
    // `prefixScanFn(prefix) -> string`: scans maps for a key starting with `prefix`, returns it or "".
    // `aliases`: the alias map (alias → canonical name).
    template<typename ExistsFn, typename PrefixScanFn>
    static std::string resolve(
        const std::string& name,
        ExistsFn&& existsFn,
        PrefixScanFn&& prefixScanFn,
        const std::unordered_map<std::string, std::string>& aliases)
    {
        // 1. Exact match
        if (existsFn(name)) return name;

        // 2. Suffixed _1 through _9
        for (int i = 1; i <= config::kMaxColumnSuffixSearch; ++i) {
            std::string s = name + "_" + std::to_string(i);
            if (existsFn(s)) return s;
        }

        // 3. _rhs_ prefix match
        std::string rhsPfx = name + "_rhs_";
        std::string rhsMatch = prefixScanFn(rhsPfx);
        if (!rhsMatch.empty()) return rhsMatch;

        // 4. Column alias
        auto aliasIt = aliases.find(name);
        if (aliasIt != aliases.end() && existsFn(aliasIt->second))
            return aliasIt->second;

        return "";
    }
};

} // namespace engine
