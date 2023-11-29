#include "mlir/Dialect/Arith/IR/Arith.h"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Pass/Pass.h>
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/Passes.h"
#include <mlir/Dialect/Vector/IR/VectorOps.h>

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

#include "ReduceOpToXeGPU.h"
#include "triton/Dialect/XeGPU/IR/XeGPUOps.h"
//#include "../TritonGPUToLLVM/Utility.h"
#include "Utility.h"


using namespace mlir;
using namespace mlir::triton;
using namespace mlir::spirv;
using namespace mlir::triton::xegpu;
using mlir::triton::gpu::GenericEncodingAttr;

class ReduceOpToXeGPUPattern : public OpConversionPattern<triton::ReduceOp> {
public:
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto src = adaptor.getOperands()[0];
    Type type = op.getResult()[0].getType();

    //todo combine the two func
    if(isa<RankedTensorType>(type)){
      return VectorNDReduceOp(rewriter, adaptor, op);
    } else{
      return Vector1DReduceOp(rewriter, adaptor, op);
    }
  }

private:
  void accumulate(ConversionPatternRewriter &rewriter, Location loc,
                  Operation* reduceOp, Value &acc, Value cur) const {
    if (isa<arith::AddIOp>(reduceOp))
      acc = add(acc, cur);
    else if (isa<arith::AddFOp>(reduceOp))
      acc = fadd(acc, cur);
    else if (isa<arith::MinSIOp>(reduceOp))
      acc = smin(acc, cur);
    else if (isa<arith::MaxSIOp>(reduceOp))
      acc = smax(acc, cur);
    else if (isa<arith::MinUIOp>(reduceOp))
      acc = umin(acc, cur);
    else if (isa<arith::MaxUIOp>(reduceOp))
      acc = umax(acc, cur);
    else if (isa<arith::MinFOp>(reduceOp))
      acc = fmin(acc, cur);
    else if (isa<arith::MaxFOp>(reduceOp))
      acc = fmax(acc, cur);
    else if (isa<arith::XOrIOp>(reduceOp))
      acc = xor_(acc, cur);
    else
      llvm::report_fatal_error("Unsupported reduce op");
  }

  LogicalResult Vector1DReduceOp(ConversionPatternRewriter &rewriter, 
                        OpAdaptor adaptor, triton::ReduceOp op) const {
    Location loc = op->getLoc();
    auto context = rewriter.getContext();
    auto src = adaptor.getOperands()[0];
    auto elemType = op.getResult().getType()[0];

    Block *block = &(*op.getCombineOp().begin());
    Operation *yield = block->getTerminator();
    Operation *reduceOp = yield->getOperand(0).getDefiningOp();

    auto loadL1Hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto loadL2Hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto loadL3Hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto storeL1Hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);
    auto storeL2Hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);
    auto storeL3Hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);

    //get subgroupId
    Value subgroubId = rewriter.create<::mlir::gpu::SubgroupIdOp>(loc, rewriter.getIndexType());
    Value sgId = rewriter.create<UnrealizedConversionCastOp>(loc, i64_ty, subgroubId).getResult(0);
    sgId = rewriter.create<spirv::UConvertOp>(loc, i32_ty, sgId);

    //get subgroup size and workgroup size
    auto module = op.getOperation()->getParentOfType<mlir::ModuleOp>();
    int sgSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(module);
    int wgSize = triton::gpu::TritonGPUDialect::getNumWarps(module);

    Value acc = src;
    Value cur;
    Type curVecType;

    // reduce within subgroup
    for (unsigned N = sgSize / 2; N > 1; N >>= 1) {
      curVecType = mlir::VectorType::get(N, elemType);
      SmallVector<int32_t, 2> indices(N);
      uint64_t offset = N;
      std::iota(indices.begin(), indices.end(), offset);
      cur = rewriter.create<spirv::VectorShuffleOp>(loc, curVecType, acc, acc, rewriter.getI32ArrayAttr(indices));

      offset = 0;
      std::iota(indices.begin(), indices.end(), offset);
      acc = rewriter.create<spirv::VectorShuffleOp>(loc, curVecType, acc, acc, rewriter.getI32ArrayAttr(indices));

      accumulate(rewriter, loc, reduceOp, acc, cur);
    }

    auto memory_scope = MemoryScopeAttr::get(context, triton::xegpu::MemoryScope::SLM);
    auto vert_size = IntegerAttr::get(i32_ty, 1);

    const int elemsPerSg = 2;
    //4: sizeof(f32)
    sgId = rewriter.create<spirv::UModOp>(loc, i32_ty, sgId,
                  rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(32)));
    auto slmStoreAddr = rewriter.create<arith::MulIOp>(loc, i32_ty, sgId, 
                  rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(elemsPerSg * 4)));

    std::vector<int64_t> storeShape{elemsPerSg};
    auto tensorDescType = ::mlir::triton::xegpu::TensorDescType::get(context, storeShape, elemType, 
                                                                  MemoryScopeAttr::get(context, MemoryScope::SLM));

    llvm::outs()<<"\n\ntensorDescType: "<<tensorDescType<<"\n";

    //nd store
    Value start = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(0));
    mlir::OpFoldResult tmp = start;
    SmallVector<mlir::OpFoldResult> NdOffset{tmp};

    xegpu::CreateNdDescOp descOp = rewriter.create<xegpu::CreateNdDescOp>(loc, tensorDescType, slmStoreAddr,
                NdOffset, ArrayRef<int>{1, 0}, triton::xegpu::MemoryScope::SLM, true);

    Value desc = descOp.getODSResults(0)[0];
    rewriter.create<xegpu::StoreNDOp>(loc, desc, acc, storeL1Hint, storeL2Hint, storeL3Hint);

    //fencen + barrier
    rewriter.create<xegpu::MfenceOp>(loc, 
                                    ::mlir::StringAttr::get(context, ::llvm::StringRef("slm")),
                                    ::mlir::StringAttr::get(context, ::llvm::StringRef("none")),
                                    ::mlir::StringAttr::get(context, ::llvm::StringRef("group")));

    rewriter.create<xegpu::AllocNbarrierOp>(loc, 
                          rewriter.create<arith::ConstantOp>(loc, i8_ty, rewriter.getI8IntegerAttr(16)));
    auto payload = rewriter.create<xegpu::CreateNbarrierOp>(loc, v8i32Type, i8_val(1), i8_val(0), 
                                                          ::mlir::IntegerAttr::get(mlir::IntegerType::get(context, 8), 32), 
                                                          ::mlir::IntegerAttr::get(mlir::IntegerType::get(context, 8), 32));
    rewriter.create<xegpu::NbarrierArriveOp>(loc, payload);
    rewriter.create<xegpu::NbarrierWaitOp>(loc, payload);

    //Since spirv::VectorShuffleOp does not support vector1, each subgroup writes back two values
    wgSize *= elemsPerSg;
    //todo
    wgSize = 32;
    //load from slm
    curVecType = mlir::VectorType::get(wgSize, elemType);
    auto loadOffsetsType = mlir::VectorType::get(wgSize, i32Type);
    std::vector<int> loadOffsetsValues(wgSize, 0);
    for(int i = 0; i < wgSize; i++){
      loadOffsetsValues[i] = i;
    }
    auto offsetRef = ArrayRef<int>(loadOffsetsValues);
    DenseElementsAttr constData = DenseElementsAttr::get(loadOffsetsType, offsetRef);
    Value offsets = rewriter.create<arith::ConstantOp>(loc, loadOffsetsType, constData);

    llvm::SmallVector<bool> loadMaskValues(wgSize, 0);
    for(int i = 0; i < wgSize; i++){
      loadMaskValues[i] = 1;
    }
    auto loadMaskType = mlir::VectorType::get(wgSize, i1Type);
    auto maskRef = ArrayRef<bool>(loadMaskValues);
    constData = DenseElementsAttr::get(loadMaskType, maskRef);
    Value mask = rewriter.create<arith::ConstantOp>(loc, loadMaskType, constData);

    //todo slm load max v32f32
    auto slmLoadAddr0 = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(0 * 4));
    auto slmLoadAddr1 = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(32 * 4));
    std::vector<int64_t> loadShape{32};
    tensorDescType = xegpu::TensorDescType::get(context, loadShape, elemType, ScatteredAttr::get(context));
    auto desc0 = rewriter.create<xegpu::CreateDescOp>(loc, tensorDescType, slmLoadAddr0, offsets, memory_scope, vert_size);
    auto desc1 = rewriter.create<xegpu::CreateDescOp>(loc, tensorDescType, slmLoadAddr1, offsets, memory_scope, vert_size);
    
    auto load0 = rewriter.create<xegpu::LoadGatherOp>(loc, curVecType, desc0, mask, IntegerAttr{}, DenseI64ArrayAttr{}, loadL1Hint, loadL2Hint, loadL3Hint);
    auto load1 = rewriter.create<xegpu::LoadGatherOp>(loc, curVecType, desc1, mask, IntegerAttr{}, DenseI64ArrayAttr{}, loadL1Hint, loadL2Hint, loadL3Hint);

    acc = load0;
    accumulate(rewriter, loc, reduceOp, acc, load1);

    // llvm::outs()<<"\n\nafter reduce whthin subgroup acc: "<<acc<<"\n";
    // reduce with workgroup
    for (unsigned N = 16; N > 1; N >>= 1) {
      curVecType = mlir::VectorType::get(N, elemType);
      SmallVector<int32_t, 2> indices(N);
      uint64_t offset = N;
      std::iota(indices.begin(), indices.end(), offset);
      cur = rewriter.create<spirv::VectorShuffleOp>(loc, curVecType, acc, acc, rewriter.getI32ArrayAttr(indices));

      offset = 0;
      std::iota(indices.begin(), indices.end(), offset);
      acc = rewriter.create<spirv::VectorShuffleOp>(loc, curVecType, acc, acc, rewriter.getI32ArrayAttr(indices));

      accumulate(rewriter, loc, reduceOp, acc, cur);
    }

    cur = extract_val(elemType, acc, rewriter.getI32ArrayAttr(1));
    acc = extract_val(elemType, acc, rewriter.getI32ArrayAttr(0));
    accumulate(rewriter, loc, reduceOp, acc, cur);

    llvm::outs()<<"\n\nacc: "<<acc<<"\n";
    rewriter.replaceOp(op, acc);
    return success();
  }

  LogicalResult VectorNDReduceOp(ConversionPatternRewriter &rewriter, 
                        OpAdaptor adaptor, triton::ReduceOp op) const {
    Location loc = op->getLoc();
    auto context = rewriter.getContext();
    uint32_t axis = op.getAxis();

    auto layout =  op.getOperands()[0].getType()
                      .cast<RankedTensorType>().getEncoding()
                      .cast<GenericEncodingAttr>();
    auto threadShape = layout.getThreadShape();
    auto threadStride = layout.getThreadStride();
    auto elemShape = layout.getElemPerThread();
    auto elemStride = layout.getElemStride();

    auto operands = adaptor.getOperands();
    ValueRange src(operands);

    if(auto *parentOp = src[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        auto cast = (&castOp)->getInputs();
        src = ValueRange(cast);
      }
    }

    VectorType vectorType = src[0].getType().cast<VectorType>();
    Type elemType = vectorType.getElementType();
    auto shape = vectorType.getShape();

    Block *block = &(*op.getCombineOp().begin());
    Operation *yield = block->getTerminator();
    Operation *reduceOp = yield->getOperand(0).getDefiningOp();

    auto loadL1Hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto loadL2Hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto loadL3Hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto storeL1Hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);
    auto storeL2Hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);
    auto storeL3Hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);

    //get subgroupId
    Value subgroubId = rewriter.create<::mlir::gpu::SubgroupIdOp>(loc, rewriter.getIndexType());
    Value sgId = rewriter.create<UnrealizedConversionCastOp>(loc, i64_ty, subgroubId).getResult(0);
    sgId = rewriter.create<spirv::UConvertOp>(loc, i32_ty, sgId);

    //get subgroup size and workgroup size
    auto module = op.getOperation()->getParentOfType<mlir::ModuleOp>();
    int sgSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(module);
    int wgSize = triton::gpu::TritonGPUDialect::getNumWarps(module);

    int accNums;
    int accBlockNums;
    int accLens;
    if(axis == 1){
      accNums = elemShape[2];
      accBlockNums = elemShape[3];
      accLens = shape[1];
    } else {
      //todo
    }
    SmallVector<Value> acc(accNums);

    Value cur;
    Type curVecType = src[0].getType();

    // reduce within different blocks
    for(int i = 0; i < accNums; i++){
      acc[i] = src[i * accBlockNums];
      //curVecType = mlir::VectorType::get(ArrayRef<int64_t>{shape[0], shape[1], 1}, elemType);
      //acc[i] = rewriter.create<vector::ShapeCastOp>(loc, curVecType, acc[i]);

      for(int j = 1; j < accBlockNums; j++){
        cur = src[i * accBlockNums + j];
        //cur = rewriter.create<vector::ShapeCastOp>(loc, curVecType, cur);
        accumulate(rewriter, loc, reduceOp, acc[i], cur);
      }

      for (unsigned N = accLens / 2; N > 0; N >>= 1) {
        curVecType = mlir::VectorType::get(ArrayRef<int64_t>{shape[0], N}, elemType);

        SmallVector<int32_t, 2> indices(shape[0] * N);
        for(int d0 = 0;d0 < shape[0];d0++){
          for(int d1 = 0; d1 < N; d1++){
            indices[d0 * N + d1] = d0 * (N * 2) + N + d1;
          }
        }

        cur = rewriter.create<spirv::VectorShuffleOp>(loc, curVecType, acc[i], acc[i], rewriter.getI32ArrayAttr(indices));

        for(int d0 = 0;d0 < shape[0];d0++){
          for(int d1 = 0; d1 < N; d1++){
            indices[d0 * N + d1] = d0 * (N * 2) + d1;
          }
        }

        acc[i] = rewriter.create<spirv::VectorShuffleOp>(loc, curVecType, acc[i], acc[i], rewriter.getI32ArrayAttr(indices));

        accumulate(rewriter, loc, reduceOp, acc[i], cur);
      }
    }

    for(int i = 1; i < accNums; i++){
      auto nElem = (i + 1) * shape[0]; 
      curVecType = mlir::VectorType::get(nElem, elemType);
      SmallVector<int32_t, 2> indices(nElem);
      uint64_t offset = 0;
      std::iota(indices.begin(), indices.end(), offset);
      acc[0] = rewriter.create<spirv::VectorShuffleOp>(loc, curVecType, acc[0], acc[i], rewriter.getI32ArrayAttr(indices));
    }


    llvm::outs()<<"\n\nacc: "<<acc[0]<<"\n";
    rewriter.replaceOp(op, acc[0]);
    return success();
  }
};

void populateReduceOpToXeGPUPatterns(
    TritonGPUToXeGPUTypeConverter &typeConverter, RewritePatternSet &patterns) {
  llvm::outs()<<"\n\npopulateReduceOpToXeGPUPatterns\n";
  auto context = patterns.getContext();
  patterns.add<ReduceOpToXeGPUPattern>(typeConverter, context);
}