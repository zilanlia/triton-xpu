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
    //return type; 
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

    xegpu::TensorDescType tdescTy;
    if(shape[1]==32){
      tdescTy = xegpu::TensorDescType::get({8, 16}, elemTy, MemoryScopeAttr::get(type.getContext(), MemoryScope::GLOBAL));
    } else{
      tdescTy = xegpu::TensorDescType::get({16, 16}, elemTy, MemoryScopeAttr::get(type.getContext(), MemoryScope::GLOBAL));
    }
    auto numElements = 8;
    resultTypes.assign(numElements, tdescTy);
    //return resultTypes;
  }else{
    auto newType = ::mlir::MemRefType::get({32}, type.getPointeeType());
    resultTypes.assign(1, newType);
  }
  //return ::mlir::MemRefType::get({::mlir::ShapedType::kDynamic}, type.getPointeeType());
  
  return success();
}

Value packLLElements(
        Location loc, ValueRange resultVals, ConversionPatternRewriter &rewriter,
        Type type) {
  auto structType = type;
  if (!structType.isa<spirv::StructType>()) {
    return *resultVals.begin();
  }

  Value spirvStruct = rewriter.create<spirv::UndefOp>(loc, structType);

  for (const auto &v : llvm::enumerate(resultVals)) {
    assert(v.value() && "can not insert null values");
    spirvStruct = insert_val(structType, v.value(), spirvStruct, rewriter.getI32ArrayAttr(v.index()));
  }
  return spirvStruct;
}

SmallVector<Value> unpackLLElements(
        Location loc, Value spirvStruct, ConversionPatternRewriter &rewriter) {
  assert(bool(spirvStruct) && "can not unpack null values");
  if (spirvStruct.getType().isIntOrIndexOrFloat() ||
          spirvStruct.getType().isa<triton::PointerType>() ||
          spirvStruct.getType().isa<spirv::PointerType>())
    return {spirvStruct};
  auto types =
          spirvStruct.getType().cast<spirv::StructType>().getElementTypes();
  SmallVector<Value> results(types.size());
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    results[i] = extract_val(type, spirvStruct, rewriter.getI32ArrayAttr(i));
  }
  return results;
}


std::optional<mlir::LogicalResult>
TritonGPUToXeGPUTypeConverter::convertTritonTensorType(
  RankedTensorType type, llvm::SmallVectorImpl<mlir::Type>& resultTypes) {
  auto context = type.getContext();
  Attribute layout = type.getEncoding();
  SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());

  if (layout &&
      (layout.isa<BlockedEncodingAttr>() || layout.isa<SliceEncodingAttr>() ||
       layout.isa<MmaEncodingAttr>())) {
    unsigned numElementsPerThread = getTotalElemsPerThread(type);
    SmallVector<Type, 4> types(numElementsPerThread,
                               convertType(type.getElementType()));
    auto newType =  spirv::StructType::get(types);
    resultTypes.assign(1, newType);
    return success();
  } else if(layout.isa<GenericEncodingAttr>()){
    auto genericLayout = llvm::dyn_cast<GenericEncodingAttr>(layout);
    Type elemTy = type.getElementType();
    auto MmaFlag = genericLayout.getIsLayoutUpdated();

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
      return success();
    }
    else if(shape.size()==2 && shape[1]==32){
      auto newType = mlir::VectorType::get(ArrayRef<int64_t>{8, 8, 2}, elemTy);
      resultTypes.assign(16, newType);
      llvm::outs()<<"\n\nresultTypes.size(): "<<resultTypes.size()<<"\n";
      llvm::outs()<<"\n\nresultTypes[0]: "<<resultTypes[0]<<"\n";
      return success();
    } else if(shape.size()==2 && shape[1]==256 && elemTy == f16Type){
      auto newType = mlir::VectorType::get(ArrayRef<int64_t>{8, 16, 2}, elemTy);
      resultTypes.assign(8, newType);
      return success();
    } else if(shape.size()==2 && shape[1]==256 && elemTy == BFloat16Type::get(type.getContext())){
      auto newType = mlir::VectorType::get(ArrayRef<int64_t>{8, 16, 2}, Float16Type::get(type.getContext()));
      resultTypes.assign(8, newType);
      return success();
    } else if(shape.size()==2 && shape[1]==256 && elemTy == f32Type){
      auto newType = mlir::VectorType::get(ArrayRef<int64_t>{8, 16}, elemTy);
      resultTypes.assign(16, newType);
      return success();
    } else{
      unsigned simd = product_interval<unsigned>(threadShape, 0, threadShape.size() / 2);
      auto newType = mlir::VectorType::get(simd, elemTy);
      resultTypes.assign(1, newType);
      llvm::outs()<<"\n\nresultTypes.size(): "<<resultTypes.size()<<"\n";
      llvm::outs()<<"\n\nresultTypes[0]: "<<resultTypes[0]<<"\n";
      return success();
      // return mlir::VectorType::get(simd, elemTy);
    }
  } else if (auto shared_layout =
                 layout.dyn_cast_or_null<SharedEncodingAttr>()) {
    resultTypes.assign(1, type);
    return success();
    // return type;
  } else if (auto dotOpLayout =
                 layout.dyn_cast_or_null<DotOperandEncodingAttr>()) {
    if (dotOpLayout.getParent().isa<BlockedEncodingAttr>()) { // for parent is blocked layout
      resultTypes.assign(1, type);
      return success();
    } else if (dotOpLayout.getParent().isa<GenericEncodingAttr>()) { // for parent is generic layout
      Type elemTy = type.getElementType();
      if(shape.size()==2 && shape[1]==32){
        auto newType = mlir::VectorType::get(ArrayRef<int64_t>{8, 8, 2}, elemTy);
        resultTypes.assign(16, newType);
        return success();
      } else if(shape.size()==2 && shape[1]==256){
        auto newType = mlir::VectorType::get(ArrayRef<int64_t>{8, 16, 2}, elemTy);
        resultTypes.assign(8, newType);
        return success();
      } 
    }else { // for parent is MMA layout
      resultTypes.assign(1, type);
      return success();
      //return type;
    }

    llvm::errs() << "Unexpected dot operand layout detected in "
                    "TritonToLLVMTypeConverter";
    return std::nullopt;
  }

  return std::nullopt;
}