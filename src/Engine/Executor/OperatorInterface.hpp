#pragma once
// ============================================================================
// OperatorInterface.hpp — Abstract base class for relational operators.
//
// Design rationale (E6):
//   The current execution dispatch uses a switch on IRNode::Type with ~10 cases
//   calling static methods. This header defines a polymorphic Operator interface
//   for future incremental migration to virtual dispatch.
//
//   Migration path:
//     1. Wrap each existing static executeXxx() in a concrete Operator subclass.
//     2. Build an Operator pipeline from the IR plan.
//     3. Execute via virtual dispatch: for (auto& op : pipeline) op->execute(ctx);
//     4. Remove the switch-case dispatch once all operators are migrated.
//
//   This file is intentionally kept minimal — no implementation, just the
//   interface contract that future operators must satisfy.
// ============================================================================

#include <string>

namespace engine {

struct EvalContext;
struct TableResult;

// Abstract base for relational operators (scan, filter, join, groupby, etc.).
class Operator {
public:
    virtual ~Operator() = default;

    // Execute this operator, mutating the context in place.
    // Returns true on success, false on failure (sets errorMsg).
    virtual bool execute(EvalContext& ctx) = 0;

    // Human-readable name for diagnostics (e.g., "Filter", "HashJoin").
    virtual const char* name() const = 0;

    // Last error message (valid only after execute() returns false).
    const std::string& errorMsg() const { return errorMsg_; }

protected:
    std::string errorMsg_;
};

} // namespace engine
