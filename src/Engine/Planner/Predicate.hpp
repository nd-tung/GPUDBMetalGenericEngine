// Lightweight comparison operator enum used across the engine
#pragma once

namespace engine::expr {

enum class CompOp { LT, LE, GT, GE, EQ, NE, LIKE_PATTERN = 999 };

} // namespace engine::expr
