#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <algorithm>
#include <numeric>

using namespace mlir;
using namespace mlir::triton::gpu;

//
// TypeConverter
//
TritonGPUTypeConverter::TritonGPUTypeConverter(MLIRContext *context,
                                               int numWarps, int threadsPerWarp,
                                               int numCTAs)
    : context(context), numWarps(numWarps), threadsPerWarp(threadsPerWarp),
      numCTAs(numCTAs) {
  addConversion([](Type type) { return type; });

  // Add encoding for tensor
  addConversion([this](RankedTensorType tensorType) -> RankedTensorType {
    // types with encoding are already in the right format
    // TODO: check for layout encodings more specifically
    if (tensorType.getEncoding())
      return tensorType;
    // pessimistic values for attributes:
    //   - 1 element per thread
    //   - order = arange(rank)
    ArrayRef<int64_t> shape = tensorType.getShape();
    // int rank = shape.size();
    // llvm::SmallVector<unsigned> order(rank);
    // std::iota(order.begin(), order.end(), 0);
    // llvm::SmallVector<unsigned> sizePerThread(rank, 1);
    // Attribute encoding = triton::gpu::BlockedEncodingAttr::get(
    //     this->context, shape, sizePerThread, order, this->numWarps,
    //     this->threadsPerWarp, this->numCTAs);
    // return RankedTensorType::get(shape, tensorType.getElementType(), encoding);

    //unsigned int elems = 32 / this->threadsPerWarp;
    const std::vector<unsigned int> threadShapeVec{1, 16, 8, 4};
    const std::vector<unsigned int> threadStrideVec{0, 1, 32, 64};
    const std::vector<unsigned int> elemPerThreadVec{8, 1, 4, 4};
    const std::vector<unsigned int> elemStrideVec{1, 0, 8, 16};
    const std::vector<unsigned int> subGroupShapeVec{4, 4, 0, 0};
    const std::vector<unsigned int> orderVec{1, 0, 1, 0};
    // const std::vector<unsigned int> threadShapeVec{32, 32};
    // const std::vector<unsigned int> threadStrideVec{1, 32};
    // const std::vector<unsigned int> elemPerThreadVec{1, 0};
    // const std::vector<unsigned int> elemStrideVec{0, 0};
    // const std::vector<unsigned int> subGroupShapeVec{1, 1};
    // const std::vector<unsigned int> orderVec{1, 0};

    ArrayRef<unsigned int> threadShape(threadShapeVec);
    ArrayRef<unsigned int> threadstride(threadStrideVec);
    ArrayRef<unsigned int> elemPerThread(elemPerThreadVec);
    ArrayRef<unsigned int> elemStride(elemStrideVec);
    ArrayRef<unsigned int> subGroupShape(subGroupShapeVec);
    ArrayRef<unsigned int> order(orderVec);
    Attribute encoding = triton::gpu::GenericEncodingAttr::get(
        this->context, threadShape, threadstride, elemPerThread, elemStride, subGroupShape, order, -1);
    return RankedTensorType::get(shape, tensorType.getElementType(), encoding);
  });

  // Add encoding for tensor pointer
  addConversion([this](triton::PointerType ptrType) -> triton::PointerType {
    // Check whether tensor pointer `tt.ptr<tensor<>>`
    auto pointeeTensorType =
        ptrType.getPointeeType().dyn_cast<RankedTensorType>();
    if (pointeeTensorType == nullptr)
      return ptrType;

    // Add layout into the tensor
    auto convertedTensorType = convertType(pointeeTensorType);
    return triton::PointerType::get(convertedTensorType,
                                    ptrType.getAddressSpace());
  });

  //
  // Materializations
  //
  // This will be called when (newArgType != origArgType)
  // This will create newArg, and map(origArg, newArg)
  addArgumentMaterialization([&](OpBuilder &builder,
                                 RankedTensorType tensorType, ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
    llvm_unreachable("Argument rematerialization should not happen in Triton "
                     "-> TritonGPU conversion");
    return std::nullopt;
  });

  // If the origValue still has live user(s), use this to
  // convert origValue to newValue
  addSourceMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    llvm_unreachable("Source rematerialization should not happen in Triton -> "
                     "TritonGPU Conversion");
    return std::nullopt;
  });

  // This will be called when (desiredType != newOperandType)
  // where, desiredType = typeConverter->convertType(origType)
  // NOTE: only for remapped values.
  addTargetMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                               ValueRange inputs, Location loc) {
    auto cast =
        builder.create<triton::gpu::ConvertLayoutOp>(loc, tensorType, inputs);
    return std::optional<Value>(cast.getResult());
  });
}

//
// TritonGPUConversion
//
TritonGPUConversionTarget::TritonGPUConversionTarget(
    MLIRContext &context, TritonGPUTypeConverter &typeConverter)
    : ConversionTarget(context) {
  // TODO: we should also verify ops of TritonGPUDialect
  addLegalDialect<triton::gpu::TritonGPUDialect>();

  // Some ops from SCF are illegal
  addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, scf::ReduceOp,
               scf::ReduceReturnOp>();
  // We have custom versions of some arith operators
  addIllegalOp<arith::CmpIOp, arith::CmpFOp>();

  addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect,
                             triton::TritonDialect, cf::ControlFlowDialect,
                             scf::SCFDialect>([&](Operation *op) {
    bool hasLegalRegions = true;
    for (auto &region : op->getRegions()) {
      hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
    }
    if (hasLegalRegions && typeConverter.isLegal(op)) {
      return true;
    }
    return false;
  });

  // We have requirements for the data layouts
  addDynamicallyLegalOp<triton::DotOp>([](triton::DotOp dotOp) -> bool {
    Attribute aEncoding =
        dotOp.getA().getType().cast<RankedTensorType>().getEncoding();
    Attribute bEncoding =
        dotOp.getB().getType().cast<RankedTensorType>().getEncoding();
    if (aEncoding && aEncoding.isa<triton::gpu::DotOperandEncodingAttr>() &&
        bEncoding && bEncoding.isa<triton::gpu::DotOperandEncodingAttr>())
      return true;
    return false;
  });
}
