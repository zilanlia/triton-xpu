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

class ReduceOpToXeGPUPattern : public OpConversionPattern<triton::ReduceOp> {
public:
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto src = adaptor.getOperands()[0];
    auto type = op.getResult().getType();

    return createSLMReduceOp(rewriter, adaptor, op);
  }

private:
  void accumulate(ConversionPatternRewriter &rewriter, Location loc,
                  Operation* reduceOp, Value &acc, Value cur) const {
    if (isa<arith::AddIOp>(reduceOp))
      acc = add(acc, cur);
    else if (isa<arith::AddFOp>(reduceOp))
      acc = fadd(acc.getType(), acc, cur);
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

  LogicalResult createSLMReduceOp(ConversionPatternRewriter &rewriter, 
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
    Value sgId = rewriter.create<UnrealizedConversionCastOp>(loc, i32_ty, subgroubId).getResult(0);
    //Value sgId = zext(i32_ty, subgroubId);

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

    // cur = extract_val(elemType, acc, rewriter.getI32ArrayAttr(1));
    // acc = extract_val(elemType, acc, rewriter.getI32ArrayAttr(0));
    // accumulate(rewriter, loc, reduceOp, acc, cur);
    // curVecType = mlir::VectorType::get(1, elemType);
    // acc = rewriter.create<vector::SplatOp>(loc, curVecType, acc);

    // store into slm
    // todo get the base of slm
    // auto memrefType = ::mlir::MemRefType::get({32}, elemType);
    // auto slmBase = rewriter.create<memref::AllocaOp>(loc, memrefType);
    // IntegerAttr::get(getIntegerType(64, false), APInt(64, (uint64_t)value, false));
    // auto slmBase = rewriter.create<arith::ConstantOp>(loc, ui64_ty, 
    //                         IntegerAttr::get(ui64_ty, APInt(64, (uint64_t)(0), false)));

    auto memory_scope = MemoryScopeAttr::get(context, triton::xegpu::MemoryScope::SLM);
    auto vert_size = IntegerAttr::get(i32_ty, 1);

    const int elemsPerSg = 2;
    //4: sizeof(f32)
    auto slmStoreAddr = rewriter.create<arith::MulIOp>(loc, i32_ty, sgId, 
                  rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(elemsPerSg * 4)));

    auto sgAddrsType = mlir::VectorType::get(elemsPerSg, i32Type);
    auto offsetType = mlir::VectorType::get(elemsPerSg, i32Type);
    auto sgAddr = rewriter.create<arith::MulIOp>(loc, i32_ty, sgId, 
        rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(elemsPerSg)));
    Value sgAddrs = rewriter.create<vector::SplatOp>(loc, sgAddrsType, sgAddr);

    DenseElementsAttr constData = DenseElementsAttr::get(sgAddrsType, ArrayRef<int>(std::vector<int>(elemsPerSg, 0)));
    // Value sgAddrs = rewriter.create<spirv::ConstantOp>(loc, sgAddrsType, constData);
    // for(int i = 0; i < elemsPerSg; i++){
    //   Value idx = rewriter.create<spirv::ConstantOp>(loc, i32Type, IntegerAttr::get(i32Type, i));
    //   sgAddrs = rewriter.create<spirv::VectorInsertDynamicOp>(loc, sgAddrsType, sgAddrs, sgAddr, idx);
    // }

    llvm::SmallVector<int> offsetValues(elemsPerSg, 0);
    for(int i = 0 ; i < elemsPerSg ; i++){
      offsetValues[i] = i;
    }
    ArrayRef<int> offsetRef(offsetValues);
    constData = DenseElementsAttr::get(offsetType, offsetRef);
    Value offsets = rewriter.create<arith::ConstantOp>(loc, offsetType, constData);
    offsets =  rewriter.create<arith::AddIOp>(loc, offsetType, sgAddrs, offsets);

    llvm::SmallVector<bool> maskValues(elemsPerSg, true);
    auto maskType = mlir::VectorType::get(elemsPerSg, i1Type);
    ArrayRef<bool> maskRef(maskValues);
    constData = DenseElementsAttr::get(maskType, maskRef);
    Value mask = rewriter.create<arith::ConstantOp>(loc, maskType, constData);

    std::vector<int64_t> storeShape{elemsPerSg};

    // using scatter store
    //auto tensorDescType = ::mlir::triton::xegpu::TensorDescType::get(context, storeShape, elemType, ScatteredAttr::get(context));
    // Value desc = rewriter.create<xegpu::CreateDescOp>(loc, tensorDescType, slmStoreAddr, offsets, memory_scope, vert_size);
    //rewriter.create<xegpu::StoreScatterOp>(loc, acc, desc, mask, storeL1Hint, storeL2Hint, storeL3Hint);

    //using nd store
    auto tensorDescType = ::mlir::triton::xegpu::TensorDescType::get(context, storeShape, elemType, 
                                                                  MemoryScopeAttr::get(context, MemoryScope::SLM));

    llvm::outs()<<"\n\ntensorDescType: "<<tensorDescType<<"\n";
    // Value start = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(0));
    // ::llvm::ArrayRef<mlir::OpFoldResult> NdOffset = ::llvm::ArrayRef<mlir::OpFoldResult>{start};

    // nd
    // Value NdOffset = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(0)); 
    // Value NdShape = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(2));
    // Value NdStride = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(1));
    // xegpu::CreateNdDescOp descOp = rewriter.create<xegpu::CreateNdDescOp>(loc, tensorDescType, slmStoreAddr,
    //             ::mlir::ValueRange{NdOffset}, ::mlir::ValueRange{NdShape}, ::mlir::ValueRange{NdStride}, 
    //             ::llvm::ArrayRef<int64_t>{0}, triton::xegpu::MemoryScope::SLM, true);

    //1d
    Value start = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(0));
    mlir::OpFoldResult tmp = start;
    llvm::outs()<<"\n\ntmp: "<<tmp<<"\n";
    SmallVector<mlir::OpFoldResult> NdOffset{tmp};

    llvm::outs()<<"\n\nNdOffset[0]: "<<NdOffset[0]<<"\n";
    xegpu::CreateNdDescOp descOp = rewriter.create<xegpu::CreateNdDescOp>(loc, tensorDescType, slmStoreAddr,
                NdOffset, triton::xegpu::MemoryScope::SLM, true);

    llvm::outs()<<"\n\ndescOp: "<<descOp<<"\n";
    Value desc = descOp.getODSResults(0)[0];
    llvm::outs()<<"\n\ndesc: "<<desc<<"\n";
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
    offsetRef = ArrayRef<int>(loadOffsetsValues);
    constData = DenseElementsAttr::get(loadOffsetsType, offsetRef);
    offsets = rewriter.create<arith::ConstantOp>(loc, loadOffsetsType, constData);

    llvm::SmallVector<bool> loadMaskValues(wgSize, 0);
    for(int i = 0; i < wgSize; i++){
      loadMaskValues[i] = 1;
    }
    auto loadMaskType = mlir::VectorType::get(wgSize, i1Type);
    maskRef = ArrayRef<bool>(loadMaskValues);
    constData = DenseElementsAttr::get(loadMaskType, maskRef);
    mask = rewriter.create<arith::ConstantOp>(loc, loadMaskType, constData);

    //todo slm load max v32f32
    auto slmLoadAddr0 = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(0 * 4));
    auto slmLoadAddr1 = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(32 * 4));
    std::vector<int64_t> loadShape{32};
    tensorDescType = xegpu::TensorDescType::get(context, loadShape, elemType, ScatteredAttr::get(context));
    auto desc0 = rewriter.create<xegpu::CreateDescOp>(loc, tensorDescType, slmLoadAddr0, offsets, memory_scope, vert_size);
    auto desc1 = rewriter.create<xegpu::CreateDescOp>(loc, tensorDescType, slmLoadAddr1, offsets, memory_scope, vert_size);
    
    auto load0 = rewriter.create<xegpu::LoadGatherOp>(loc, curVecType, desc0, mask, IntegerAttr{}, DenseI64ArrayAttr{}, loadL1Hint, loadL2Hint, loadL3Hint);
    auto load1 = rewriter.create<xegpu::LoadGatherOp>(loc, curVecType, desc1, mask, IntegerAttr{}, DenseI64ArrayAttr{}, loadL1Hint, loadL2Hint, loadL3Hint);

    curVecType = mlir::VectorType::get(16, elemType);
    SmallVector<int32_t, 2> indices(16);
    uint64_t offset = 16;
    std::iota(indices.begin(), indices.end(), offset);
    auto load0_16_31 = rewriter.create<spirv::VectorShuffleOp>(loc, curVecType, load0, load0, rewriter.getI32ArrayAttr(indices));
    auto load1_16_31 = rewriter.create<spirv::VectorShuffleOp>(loc, curVecType, load1, load1, rewriter.getI32ArrayAttr(indices));
    
    offset = 0;
    std::iota(indices.begin(), indices.end(), offset);
    auto load0_0_16 = rewriter.create<spirv::VectorShuffleOp>(loc, curVecType, load0, load0, rewriter.getI32ArrayAttr(indices));
    auto load1_0_16 = rewriter.create<spirv::VectorShuffleOp>(loc, curVecType, load1, load1, rewriter.getI32ArrayAttr(indices));

    acc = load0_16_31;
    accumulate(rewriter, loc, reduceOp, acc, load1_16_31);
    accumulate(rewriter, loc, reduceOp, acc, load0_0_16);
    accumulate(rewriter, loc, reduceOp, acc, load1_0_16);

    llvm::outs()<<"\n\nafter reduce whthin subgroup acc: "<<acc<<"\n";

    //reduce with workgroup
    for (unsigned N = 32 / 4; N > 1; N >>= 1) {
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



  // Use shared memory for reduction within warps and across warps
  LogicalResult
  matchAndRewriteBasic(triton::ReduceOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();


    return success();
  }

};


void populateReduceOpToXeGPUPatterns(
    TritonGPUToXeGPUTypeConverter &typeConverter, RewritePatternSet &patterns) {
  llvm::outs()<<"\n\npopulateReduceOpToXeGPUPatterns\n";
  auto context = patterns.getContext();
  patterns.add<ReduceOpToXeGPUPattern>(typeConverter, context);
}