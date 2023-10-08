//===----------------------------------------------------------------------===//
//
// Defines utilities to use while converting to the XeGPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_XEGPU_TRANSFORMS_XEGPUCONVERSION_H_
#define TRITON_DIALECT_XEGPU_TRANSFORMS_XEGPUCONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class XeGPUTypeConverter : public TypeConverter {
public:
  XeGPUTypeConverter(MLIRContext *context, int numWarps, int threadsPerWarp);
  int getNumWarps() const { return numWarps; }
  int getThreadsPerWarp() const { return threadsPerWarp; }

private:
  MLIRContext *context;
  int numWarps;
  int threadsPerWarp;
};

class XeGPUConversionTarget : public ConversionTarget {

public:
  explicit XeGPUConversionTarget(MLIRContext &ctx,
                                     XeGPUTypeConverter &typeConverter);
};

} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_TRITONGPUCONVERSION_H_
