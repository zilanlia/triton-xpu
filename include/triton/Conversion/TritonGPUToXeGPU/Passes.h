#ifndef XEGPU_CONVERSION_PASSES_H
#define XEGPU_CONVERSION_PASSES_H

#include "triton/Conversion/TritonGPUToXeGPU/TritonGPUToXeGPUPass.h"
#include "triton/Target/PTX/TmaMetadata.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonGPUToXeGPU/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
