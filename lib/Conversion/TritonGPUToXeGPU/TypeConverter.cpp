#include "TypeConverter.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/XeGPU/IR/XeGPUOps.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::xegpu;

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::GenericEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

TritonGPUToXeGPUTypeConverter::TritonGPUToXeGPUTypeConverter(
        mlir::MLIRContext &context): context(context) {
  addConversion([&](triton::PointerType type, llvm::SmallVectorImpl<mlir::Type>& resultTypes) -> std::optional<mlir::LogicalResult> {
    return convertTritonPointerType(type, resultTypes);
  });
  addConversion([&](mlir::MemRefType type) -> llvm::Optional<Type> {
    return type; 
  });
  addConversion([&](VectorType type) -> llvm::Optional<Type> {
    return type;
  });
  addConversion([&](RankedTensorType type, llvm::SmallVectorImpl<mlir::Type>& resultTypes) -> std::optional<mlir::LogicalResult> {
    return convertTritonTensorType(type, resultTypes);
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

  addConversion([&](IndexType type) -> std::optional<Type> { 
    return IntegerType::get(type.getContext(), 64);
  });

  addConversion([&](IntegerType type) -> llvm::Optional<Type> {
    return type;
  });

  addConversion([&](FloatType type) -> llvm::Optional<Type> {
    return type;
  });

  // Add generic source and target materializations to handle cases where
  // non-SPIRV types persist after an SPIRV conversion.
  addArgumentMaterialization([&](mlir::OpBuilder &builder, mlir::Type resultType,
                             mlir::ValueRange inputs,
                             mlir::Location loc) -> std::optional<mlir::Value> {
    return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
  });
    
  addSourceMaterialization([&](mlir::OpBuilder &builder, mlir::Type resultType,
                             mlir::ValueRange inputs,
                             mlir::Location loc) -> std::optional<mlir::Value> {
    return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
            .getResult(0);
  });
}

std::optional<mlir::LogicalResult>
TritonGPUToXeGPUTypeConverter::convertTritonPointerType(
        triton::PointerType type, llvm::SmallVectorImpl<mlir::Type>& resultTypes)  {
  auto pointeeType = type.getPointeeType();
  if(isa<RankedTensorType>(pointeeType)){
    auto tensorType = pointeeType.cast<RankedTensorType>();
    Attribute layout = tensorType.getEncoding();
    SmallVector<int64_t> shape(tensorType.getShape().begin(), tensorType.getShape().end());
    Type elemTy = tensorType.getElementType();

    if(layout.isa<GenericEncodingAttr>()){
      auto genericLayout = llvm::dyn_cast<GenericEncodingAttr>(layout);
      auto mmaFlag = genericLayout.getMmaFlag();
      auto threadShape = genericLayout.getThreadShape();
      auto elemShape = genericLayout.getElemPerThread();
      auto elemStride = genericLayout.getElemStride();
      int dim0 = threadShape[0] * elemShape[0];
      int dim1 = threadShape[1] * elemShape[1];

      TensorDescType tdescTy = TensorDescType::get({dim0, dim1}, 
                    elemTy, MemoryScopeAttr::get(type.getContext(), MemoryScope::GLOBAL));

      auto numElements = elemShape[2] * elemShape[3];

      //Combined access
      if(mmaFlag == 0){
        int loadM = shape[0] / threadShape[2];
        loadM = std::min(std::max(loadM, dim0), 32);
        tdescTy = TensorDescType::get({loadM, dim1}, 
                    elemTy, MemoryScopeAttr::get(type.getContext(), MemoryScope::GLOBAL));
        int blockShapeM = elemShape[2] * elemStride[2];
        numElements = (blockShapeM / loadM) * elemShape[3];
      } else if(mmaFlag == 1){
        int loadK = shape[0];
        loadK = std::min(std::max(loadK, dim0), 32);
        tdescTy = TensorDescType::get({loadK, dim1}, 
                    elemTy, MemoryScopeAttr::get(type.getContext(), MemoryScope::GLOBAL));
        int blockShapeK = elemShape[2] * elemStride[2];
        numElements = (blockShapeK / loadK) * elemShape[3];
      } else{

      } 

      resultTypes.assign(numElements, tdescTy);
    }
  } else {
    auto newType = MemRefType::get({ShapedType::kDynamic}, type.getPointeeType());
    resultTypes.assign(1, newType);
  }

  return success();
}

std::optional<mlir::LogicalResult>
TritonGPUToXeGPUTypeConverter::convertTritonTensorType(
  RankedTensorType type, llvm::SmallVectorImpl<mlir::Type>& resultTypes) {
  auto context = type.getContext();
  Attribute layout = type.getEncoding();
  SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());

  if(layout.isa<GenericEncodingAttr>()){
    auto genericLayout = llvm::dyn_cast<GenericEncodingAttr>(layout);
    Type elemTy = type.getElementType();
    auto MmaFlag = genericLayout.getMmaFlag();

    if(elemTy.isa<triton::PointerType>()){
      elemTy = elemTy.cast<triton::PointerType>().getPointeeType();
      std::vector<int64_t> storeShape{32};
      auto newType = ::mlir::triton::xegpu::TensorDescType::get(context, storeShape, elemTy, 
                                              ScatteredAttr::get(context));
      resultTypes.assign(1, newType);
      return success();
    }

    auto threadShape = genericLayout.getThreadShape();
    if(MmaFlag==2){
      auto newType = mlir::VectorType::get(ArrayRef<int64_t>{8, 16}, elemTy);
      resultTypes.assign(16, newType);
    }
    else if(MmaFlag==0){
      auto newType = mlir::VectorType::get(ArrayRef<int64_t>{8, 8, 2}, elemTy);
      resultTypes.assign(8, newType);
    } else if(MmaFlag==1 && elemTy == f16Type){
      auto newType = mlir::VectorType::get(ArrayRef<int64_t>{8, 16, 2}, elemTy);
      resultTypes.assign(8, newType);
    } else if(MmaFlag==1 && elemTy == bf16Type){
      auto newType = mlir::VectorType::get(ArrayRef<int64_t>{8, 16, 2}, bf16Type);
      resultTypes.assign(8, newType);
    } else{
      unsigned simd = product_interval<unsigned>(threadShape, 0, threadShape.size() / 2);
      auto newType = mlir::VectorType::get(simd, elemTy);
      resultTypes.assign(1, newType);
    }
  } else if (auto dotOpLayout =
                 layout.dyn_cast_or_null<DotOperandEncodingAttr>()) {
    if (dotOpLayout.getParent().isa<GenericEncodingAttr>()) { // for parent is generic layout
      auto genericLayout = llvm::dyn_cast<GenericEncodingAttr>(dotOpLayout.getParent());
      auto MmaFlag = genericLayout.getMmaFlag();
      Type elemTy = type.getElementType();

      if(MmaFlag == 0){
        auto newType = mlir::VectorType::get(ArrayRef<int64_t>{8, 8, 2}, elemTy);
        resultTypes.assign(16, newType);
      } else if(MmaFlag == 1){
        auto newType = mlir::VectorType::get(ArrayRef<int64_t>{8, 16, 2}, elemTy);
        resultTypes.assign(8, newType);
      } 
    }
  } else{
      llvm::errs() << "Unexpected dot operand layout detected in "
                    "TritonGPUToXeGPUTypeConverter";
    return std::nullopt;
  }

  return success();
}