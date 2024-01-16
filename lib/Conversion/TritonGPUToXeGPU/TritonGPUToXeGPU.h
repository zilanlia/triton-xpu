#ifndef TRITON_TRITONGPUTOXEGPU_H
#define TRITON_TRITONGPUTOXEGPU_H

#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Transforms/DialectConversion.h>

#include "TypeConverter.h"

namespace mlir {
class SPIRVTypeConverter;
class RewritePatternSet;
class Pass;
} // namespace mlir

// TritonGPU to XeGPU Intrinsics pattern
void populateTritonGPUToXeGPUPatterns(
    TritonGPUToXeGPUTypeConverter &typeConverter, mlir::RewritePatternSet &patterns);

#endif // TRITON_CONVERSION_TRITONGPUTOXEGPU_H