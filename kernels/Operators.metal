// Operators.metal — thin wrapper that includes all domain kernel headers
#include <metal_stdlib>
using namespace metal;

namespace ops {

#include "OpCommon.h"
#include "FilterKernels.h"
#include "JoinKernels.h"
#include "SortKernels.h"
#include "AggregateKernels.h"
#include "StringKernels.h"
#include "ArithUtilKernels.h"

} // namespace ops
