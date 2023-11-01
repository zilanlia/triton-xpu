#ifndef TRITON_CONVERSION_TRITONGPU_TO_XEGPU_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONGPU_TO_XEGPU_TYPECONVERTER_H

#include "triton/Conversion/MLIRTypes.h"
#include <mlir/Transforms/OneToNTypeConversion.h>

using namespace mlir;
using namespace mlir::triton;

Value packLLElements(Location loc, ValueRange resultVals,
                      ConversionPatternRewriter &rewriter, Type type);

SmallVector<Value> unpackLLElements(Location loc, Value llvmStruct,
                                    ConversionPatternRewriter &rewriter);

class TritonGPUToXeGPUTypeConverter : public mlir::OneToNTypeConverter {
public:
  using OneToNTypeConverter::convertType;

  TritonGPUToXeGPUTypeConverter(mlir::MLIRContext &context);

  std::optional<mlir::LogicalResult> 
    convertTritonPointerType(triton::PointerType type, llvm::SmallVectorImpl<mlir::Type>& resultTypes);

  std::optional<mlir::LogicalResult>
    convertTritonTensorType(RankedTensorType type, llvm::SmallVectorImpl<mlir::Type>& resultTypes);


private:
  mlir::MLIRContext &context;
};

#endif



// Value packLLElements(Location loc, ValueRange resultVals,
//                       ConversionPatternRewriter &rewriter, Type type);

// SmallVector<Value> unpackLLElements(Location loc, Value llvmStruct,
//                                     ConversionPatternRewriter &rewriter);

// class TritonGPUToXeGPUTypeConverter : public mlir::TypeConverter {
// public:
//   using TypeConverter::convertType;

//   TritonGPUToXeGPUTypeConverter(mlir::MLIRContext &context);

//   Type convertTritonPointerType(triton::PointerType type);

//   llvm::Optional<Type> convertTritonTensorType(RankedTensorType type);


// private:
//   mlir::MLIRContext &context;
// };

// #endif
