#ifndef TRITON_CONVERSION_TRITONGPU_TO_XEGPU_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONGPU_TO_XEGPU_TYPECONVERTER_H

#include "triton/Conversion/MLIRTypes.h"

using namespace mlir;
using namespace mlir::triton;

class TritonGPUToXeGPUTypeConverter : public mlir::TypeConverter {
public:
  using TypeConverter::convertType;

  TritonGPUToXeGPUTypeConverter(mlir::MLIRContext &context);

  Type convertTritonPointerType(triton::PointerType type);

  Value packLLElements(Location loc, ValueRange resultVals,
                       ConversionPatternRewriter &rewriter, Type type);

  SmallVector<Value> unpackLLElements(Location loc, Value llvmStruct,
                                      ConversionPatternRewriter &rewriter,
                                      Type type);

  llvm::Optional<Type> convertTritonTensorType(RankedTensorType type);


private:
  mlir::MLIRContext &context;
};

#endif
