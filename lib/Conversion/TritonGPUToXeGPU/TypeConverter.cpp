#include "TypeConverter.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "triton/Conversion/MLIRTypes.h"
//#include "../TritonGPUToLLVM/Utility.h"
#include "Utility.h"
//#include "../TritonGPUToLLVM/TypeConverter.h"
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::GenericEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

TritonGPUToXeGPUTypeConverter::TritonGPUToXeGPUTypeConverter(
        mlir::MLIRContext &context): context(context) {
  addConversion([&](triton::PointerType type) -> llvm::Optional<Type> {
    return convertTritonPointerType(type);
  });
  addConversion([&](mlir::MemRefType type) -> llvm::Optional<Type> {
    return type; 
  });
  addConversion([&](VectorType type) -> llvm::Optional<Type> {
    return type;
  });
  addConversion([&](RankedTensorType type) -> llvm::Optional<Type> {
    return convertTritonTensorType(type);
  });
  // Internally store float8 as int8
  addConversion([&](mlir::Float8E4M3B11FNUZType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E4M3FNType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E4M3FNUZType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E5M2Type type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  // Internally store bfloat16 as int16
  addConversion([&](BFloat16Type type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 16);
  });

  addConversion([&](IndexType type) -> llvm::Optional<Type> {
    return IntegerType::get(type.getContext(), 32);
  });

  addConversion([&](IntegerType type) -> llvm::Optional<Type> {
    return type;
  });

  addConversion([&](FloatType type) -> llvm::Optional<Type> {
    return type;
  });

  // Add generic source and target materializations to handle cases where
  // non-SPIRV types persist after an SPIRV conversion.
  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
  });
  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> Optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
  });
}

Type TritonGPUToXeGPUTypeConverter::convertTritonPointerType(
        triton::PointerType type)  {
  //return ::mlir::MemRefType::get({::mlir::ShapedType::kDynamic}, type.getPointeeType());
  return ::mlir::MemRefType::get({32}, type.getPointeeType());
}

llvm::Optional<Type>
TritonGPUToXeGPUTypeConverter::convertTritonTensorType(RankedTensorType type) {
  auto context = type.getContext();
  Attribute layout = type.getEncoding();
  SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());

  if (layout &&
      (layout.isa<BlockedEncodingAttr>() || layout.isa<SliceEncodingAttr>() ||
       layout.isa<MmaEncodingAttr>())) {
    unsigned numElementsPerThread = getTotalElemsPerThread(type);
    SmallVector<Type, 4> types(numElementsPerThread,
                               convertType(type.getElementType()));
    return spirv::StructType::get(types);
  } else if(layout.isa<GenericEncodingAttr>()){
    auto genericLayout = llvm::dyn_cast<GenericEncodingAttr>(layout);
    Type elemTy = type.getElementType();
    auto isMma = genericLayout.getIsLayoutUpdated();

    if(elemTy.isa<triton::PointerType>()){
      elemTy = elemTy.cast<triton::PointerType>().getPointeeType();
    }

    auto threadShape = genericLayout.getThreadShape();
    unsigned simd = product_interval<unsigned>(threadShape, 0, threadShape.size() / 2);
    return mlir::VectorType::get(simd, elemTy);
  } else if (auto shared_layout =
                 layout.dyn_cast_or_null<SharedEncodingAttr>()) {
    return type;
  } else if (auto dotOpLayout =
                 layout.dyn_cast_or_null<DotOperandEncodingAttr>()) {
    if (dotOpLayout.getParent().isa<BlockedEncodingAttr>()) { // for parent is blocked layout
      return type;
    } else if (dotOpLayout.getParent().isa<GenericEncodingAttr>()) { // for parent is generic layout
      return type;
    }else { // for parent is MMA layout
      return type;
    }

    llvm::errs() << "Unexpected dot operand layout detected in "
                    "TritonToLLVMTypeConverter";
    return std::nullopt;
  }

  return std::nullopt;
}