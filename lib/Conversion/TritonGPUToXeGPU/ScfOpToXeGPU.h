#ifndef TRITON_CONVERSION_TRITONGPU_TO_XEGPU_SCF_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_XEGPU_SCF_OP_H

#include "TypeConverter.h"
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace mlir::triton;

void populateScfOpToXeGPUPatterns(
    TritonGPUToXeGPUTypeConverter &typeConverter, RewritePatternSet &patterns);

#endif