#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::scf;
using ::mlir::triton::gpu::GenericEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;

class Prefetcher {
  scf::ForOp forOp;
  scf::YieldOp yieldOp;

  SetVector<Value> dots;
  SetVector<Value> dotas;
  SetVector<Value> dotbs;

  DenseMap<Value, triton::LoadOp> dot2aLoad;
  DenseMap<Value, triton::LoadOp> dot2bLoad;
  DenseMap<Value, Value> dot2aLoopArg;
  DenseMap<Value, Value> dot2bLoopArg;
  DenseMap<Value, Value> dot2aPtr;
  DenseMap<Value, Value> dot2bPtr;
  DenseMap<Value, Value> dot2aYield;
  DenseMap<Value, Value> dot2bYield;
  DenseMap<Value, Value> operand2headPrefetch;

  Value generatePrefetch(OpBuilder &builder, Value ptr, LoadOp loadOp);
public:
  Prefetcher() = delete;

  Prefetcher(scf::ForOp forOp) : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LogicalResult initialize();

  void emitPrologue();

  scf::ForOp createNewForOp();
};

LogicalResult Prefetcher::initialize() {
  Block *loop = forOp.getBody();
  SmallVector<triton::DotOp> dotsInFor;
  for (Operation &op : *loop)
    if (auto dotOp = dyn_cast<triton::DotOp>(op))
      dotsInFor.push_back(dotOp);

  if (dotsInFor.empty()){
    dbgInfo("[Prefetch to cache][No dotOp in For Loop]");
    return failure();
  }

  // todo when used in flash attention that has 2 dots in the loop
  // if (dotsInFor.size() > 1){
  //   dbgInfo("[Prefetch to cache][dotsInFor.size() > 1]");
  //   return failure();
  // }

  // returns source of cvt
  auto getCvtLayoutSrc = [](Value v) -> Value {
    if (auto cvt = v.getDefiningOp<triton::gpu::ConvertLayoutOp>()){
      auto src = cvt.getOperand();
      auto encoding = src.getType().dyn_cast<RankedTensorType>().getEncoding();
      if (encoding.isa<triton::gpu::GenericEncodingAttr>()){
        return cvt.getSrc();
      }
    }
    return Value();
  };

  auto getLoadPtr = [](Value v) -> Value {
    if (auto loadOp = v.getDefiningOp<triton::LoadOp>()){
      auto ptr = loadOp.getPtr();
      auto encoding = ptr.getType().dyn_cast<RankedTensorType>().getEncoding();
      if (encoding.isa<triton::gpu::GenericEncodingAttr>()){
        return loadOp.getPtr();
      }
    }
    return Value();
  };

  auto getIncomingOp = [this](Value v) -> Value {
    if (auto arg = v.dyn_cast<BlockArgument>())
      if (arg.getOwner()->getParentOp() == forOp.getOperation()){
        Value ptr = forOp.getOpOperandForRegionIterArg(arg).get();
        if(auto loadOp = ptr.getDefiningOp<triton::MakeTensorPtrOp>()){
          return ptr;
        }else if(auto advanceOp = ptr.getDefiningOp<triton::AdvanceOp>()){
          ptr = advanceOp.getPtr();
          while(auto *parentOp = ptr.getDefiningOp()){
            if(auto parentAdvanceOp = dyn_cast<AdvanceOp>(parentOp)){
              ptr = parentAdvanceOp.getPtr();
            }else{
              break;
            }
          }
          return ptr;
        }
      }
    return Value();
  };

  auto getYieldOp = [this](Value v) -> Value {
    auto arg = v.cast<BlockArgument>();
    unsigned yieldIdx = arg.getArgNumber() - forOp.getNumInductionVars();
    return yieldOp.getOperand(yieldIdx);
  };

  for (triton::DotOp dot : dotsInFor) {
    Value aCvtSrc = getCvtLayoutSrc(dot.getA());
    Value bCvtSrc = getCvtLayoutSrc(dot.getB());
    dbgInfo("[Prefetch to cache]aCvt", aCvtSrc);
    dbgInfo("[Prefetch to cache]bCvt", bCvtSrc);

    if (aCvtSrc && bCvtSrc) {
      if(auto *parentOp = aCvtSrc.getDefiningOp()){
        if(auto aLoadOp = dyn_cast<triton::LoadOp>(parentOp)){
          Value aPtr = getIncomingOp(aLoadOp.getPtr());
          dbgInfo("[Prefetch to cache]aPtr", aPtr);
          if(aPtr){
            dots.insert(dot);
            dotas.insert(dot.getA());
            dot2aPtr[dot] = aPtr;
            dot2aLoopArg[dot] = aLoadOp.getPtr();
            dot2aYield[dot] = getYieldOp(aLoadOp.getPtr());
            dot2aLoad[dot] = aLoadOp;
          }
        }
      }

      if(auto *parentOp = bCvtSrc.getDefiningOp()){
        if(auto bLoadOp = dyn_cast<triton::LoadOp>(parentOp)){
          Value bPtr = getIncomingOp(bLoadOp.getPtr());
          dbgInfo("[Prefetch to cache]bPtr", bPtr);
          if(bPtr){
            if(dots.count(dot)==0)
              dots.insert(dot);
            dotbs.insert(dot.getB());
            dot2bPtr[dot] = bPtr;
            dot2bLoopArg[dot] = bLoadOp.getPtr();
            dot2bYield[dot] = getYieldOp(bLoadOp.getPtr());
            dot2bLoad[dot] = bLoadOp;
          }
        }
      }
    }
  }

  return success();
}

#define udiv(...) builder.create<arith::DivUIOp>(loc, __VA_ARGS__)
#define urem(...) builder.create<arith::RemUIOp>(loc, __VA_ARGS__)
#define add(...) builder.create<arith::AddIOp>(loc, __VA_ARGS__)
#define mul(...) builder.create<arith::MulIOp>(loc, __VA_ARGS__)
#define i32_ty mlir::IntegerType::get(context, 32)
#define i64_ty mlir::IntegerType::get(context, 64)
#define i32_val(v)  builder.create<arith::ConstantOp>(loc, i32_ty, IntegerAttr::get(i32_ty, v))

Value Prefetcher::generatePrefetch(OpBuilder &builder, Value ptr, LoadOp loadOp) {
  Location loc = ptr.getLoc();
  auto context = ptr.getContext();
  dbgInfo("[Prefetch to cahce]ptr: ", ptr);
  auto ptrType = ptr.getType().cast<triton::PointerType>();
  auto tensorType = ptrType.getPointeeType().cast<RankedTensorType>();
  auto encoding = tensorType.getEncoding();
  auto layout = encoding.cast<GenericEncodingAttr>();
  auto threadShape = layout.getThreadShape();
  auto threadStride = layout.getThreadStride();
  auto elemShape = layout.getElemPerThread();
  auto elemStride = layout.getElemStride();
  auto mmaFlag = layout.getMmaFlag();

  //get subGroup Id
  Value subgroubId = builder.create<::mlir::gpu::SubgroupIdOp>(loc, builder.getIndexType());
  Value sgId = builder.create<UnrealizedConversionCastOp>(loc, i64_ty, subgroubId).getResult(0);
  sgId = builder.create<arith::TruncIOp>(loc, i32_ty, sgId);
  sgId = urem(sgId, i32_val(8)); //subGroupNUm 32 for gemm, 8 for attention

  int sgBlockDim0 = elemShape[2] * elemStride[2];
  int sgBlockDim1 = elemShape[3] * elemStride[3];
  int blockDim0 = elemShape[0] * threadShape[0];
  int blockDim1 = elemShape[1] * threadShape[1];
  int sgSizeM = threadShape[2];
  int sgSizeN = threadShape[3];
  auto sgIdM = udiv(sgId, i32_val(sgSizeN));
  auto sgIdN = urem(sgId, i32_val(sgSizeN));

  //dbgInfo("[Prefetch to cahce]tensorType", tensorType);

  auto elemType = tensorType.getElementType();
  SmallVector<int32_t> blockShape{
    tensorType.getShape().begin(), tensorType.getShape().end()};
  SmallVector<int32_t> newBlockShape{
    tensorType.getShape().begin(), tensorType.getShape().end()};

  auto makePtrOp = ptr.getDefiningOp<triton::MakeTensorPtrOp>();
  Value base = makePtrOp.getODSOperands(0)[0];
  llvm::ArrayRef<int> order = makePtrOp.getOrder();
  SmallVector<Value> tensorShape = makePtrOp.getShape();
  SmallVector<Value> tensorStride = makePtrOp.getStrides();
  SmallVector<Value> tensorOffsets = makePtrOp.getOffsets();

  Value prefetchPtr;

  //todo
  int prefetchStage = 2;

  auto newLayout = layout.updatemmaFlag(3);

  if(mmaFlag == 0){ //prefetch matrix A
    //to do: need to add some inference for newBlockShape
    //https://gfxspecs.intel.com/Predator/Home/Index/55490
    // newBlockShape[0] = 8;
    // newBlockShape[1] = 32;

    //for attention 1024*64
    newBlockShape[0] = 8;
    newBlockShape[1] = 32;

    int prefetchInDim1 = sgBlockDim1 / newBlockShape[1];
    int prefetchInDim0 = sgBlockDim0 / newBlockShape[0];

    tensorOffsets[0] = add(tensorOffsets[0], mul(sgIdM, i32_val(sgBlockDim0)));
    tensorOffsets[1] = add(tensorOffsets[1], i32_val(0));
    Value sgOffsetM = mul(udiv(sgIdN, i32_val(prefetchInDim1)), i32_val(newBlockShape[0]));
    Value sgOffsetN = mul(urem(sgIdN, i32_val(prefetchInDim1)), i32_val(newBlockShape[1]));
    tensorOffsets[0] = add(tensorOffsets[0], sgOffsetM);
    tensorOffsets[1] = add(tensorOffsets[1], sgOffsetN);

    for(int i = 0; i < prefetchStage; i++){
      prefetchPtr = builder.create<triton::MakeTensorPtrOp>(
            loc, base, tensorShape, tensorStride, 
            tensorOffsets, newBlockShape, order);

      tensorType = prefetchPtr.getType().cast<triton::PointerType>()
                      .getPointeeType().cast<RankedTensorType>();
      auto newType = RankedTensorType::get(tensorType.getShape(),
                    elemType, newLayout);
      ptrType = PointerType::get(newType, 1);
      prefetchPtr.setType(ptrType);

      tensorOffsets[1] = add(tensorOffsets[1], i32_val(blockShape[1]));

      auto prefetchA = builder.create<triton::LoadOp>(loc, prefetchPtr, 
              loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    }

    prefetchPtr = builder.create<triton::MakeTensorPtrOp>(
          loc, base, tensorShape, tensorStride, 
          tensorOffsets, newBlockShape, order);

  }else{ //prefetch matrix B
    //todo need to add some inference for newBlockShape
    newBlockShape[0] = 16;
    newBlockShape[1] = 32;
    int prefetchInDim1 = sgBlockDim1 / newBlockShape[1];
    int prefetchInDim0 = sgBlockDim0 / newBlockShape[0];

    tensorOffsets[0] = add(tensorOffsets[0], i32_val(0));
    tensorOffsets[1] = add(tensorOffsets[1], mul(sgIdN, i32_val(sgBlockDim1)));
    Value sgOffsetM = mul(udiv(sgIdM, i32_val(prefetchInDim1)), i32_val(newBlockShape[0]));
    Value sgOffsetN = mul(urem(sgIdM, i32_val(prefetchInDim1)), i32_val(newBlockShape[1]));
    tensorOffsets[0] = add(tensorOffsets[0], sgOffsetM);
    tensorOffsets[1] = add(tensorOffsets[1], sgOffsetN);

    for(int i = 0; i < prefetchStage; i++){
      prefetchPtr = builder.create<triton::MakeTensorPtrOp>(
            loc, base, tensorShape, tensorStride, 
            tensorOffsets, newBlockShape, order);

      tensorType = prefetchPtr.getType().cast<triton::PointerType>()
                      .getPointeeType().cast<RankedTensorType>();
      auto newType = RankedTensorType::get(tensorType.getShape(),
                    elemType, newLayout);
      ptrType = PointerType::get(newType, 1);
      prefetchPtr.setType(ptrType);

      tensorOffsets[0] = add(tensorOffsets[0], i32_val(blockShape[0]));

      auto prefetchB = builder.create<triton::LoadOp>(loc, prefetchPtr, 
              loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    }

    prefetchPtr = builder.create<triton::MakeTensorPtrOp>(
          loc, base, tensorShape, tensorStride, 
          tensorOffsets, newBlockShape, order);
  }

  tensorType = prefetchPtr.getType().cast<triton::PointerType>()
                      .getPointeeType().cast<RankedTensorType>();
  auto newType = RankedTensorType::get(tensorType.getShape(),
                elemType, newLayout);
  ptrType = PointerType::get(newType, 1);
  prefetchPtr.setType(ptrType);

  //dbgInfo("[Prefetch to cahce]prefetchPtr", prefetchPtr);

  return prefetchPtr;
}

scf::ForOp Prefetcher::createNewForOp() {
  dbgInfo("[Prefetch to cahce]createNewForOp");
  OpBuilder builder(forOp);

  SmallVector<Value> loopArgs;
  for (auto v : forOp.getIterOperands()){
    loopArgs.push_back(v);
  }
  for (Value dot : dots) {
    if(dot2aPtr.count(dot) == 1){
      loopArgs.push_back(
          operand2headPrefetch[dot.getDefiningOp<triton::DotOp>().getA()]);
    }
    if(dot2bPtr.count(dot) == 1){
      loopArgs.push_back(
          operand2headPrefetch[dot.getDefiningOp<triton::DotOp>().getB()]);
    }
  }

  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), loopArgs);

  builder.setInsertionPointToStart(newForOp.getBody());
  IRMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  std::vector<Value> aAdvancedPtr;
  std::vector<Value> bAdvancedPtr;

  for (Operation &op : forOp.getBody()->without_terminator()) {
    Operation *newOp;
    // Operation *prevDot;
    auto dotOp = dyn_cast<triton::DotOp>(&op);
    auto loadOp = dyn_cast<triton::LoadOp>(&op);

    int dotIdx = -1;
    if(loadOp){
      for(int i = 0;i < dots.size();i++){
        if(dot2bLoad[dots[i]] == loadOp){
          dotIdx = i;
          break;
        }
      }
    }

    if(dotIdx >= 0){
      DotOp dotOp = dots[dotIdx].getDefiningOp<DotOp>();
      Operation *newLoadOp = builder.clone(*loadOp, mapping);
      dbgInfo("[Prefetch to cache]newLoadOp");
      //newLoadOp->print(llvm::outs());
      auto insertionPoint = builder.saveInsertionPoint();
      builder.setInsertionPointAfter(newLoadOp);
      Location loc = newLoadOp->getLoc();

      if(dot2aLoad.count(dotOp) > 0){
        LoadOp loadOp = dot2aLoad[dotOp];
        Value aPrefetchPtr = operand2headPrefetch.lookup(dotOp.getA());
        aPrefetchPtr = newForOp.getRegionIterArgForOpOperand(*aPrefetchPtr.use_begin());
        auto prefetchA = builder.create<triton::LoadOp>(loc, aPrefetchPtr, 
                loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
        
        auto retType = aPrefetchPtr.getType();
        auto dot = dotOp.getODSResults(0)[0];
        auto advanceOp = dot2aYield[dot].getDefiningOp<triton::AdvanceOp>();
        SmallVector<Value> aOffset = advanceOp.getOffsets();
        auto adaOp = builder.create<triton::AdvanceOp>(loc, retType, aPrefetchPtr, aOffset);
        aAdvancedPtr.push_back(adaOp);
      }

      if(dot2bLoad.count(dotOp) > 0){
        LoadOp loadBOp = dot2bLoad[dotOp];
        Value bPrefetchPtr = operand2headPrefetch.lookup(dotOp.getB());
        bPrefetchPtr = newForOp.getRegionIterArgForOpOperand(*bPrefetchPtr.use_begin());
        auto prefetchB = builder.create<triton::LoadOp>(loc, bPrefetchPtr, 
                loadOp.getCache(),loadOp.getEvict(), loadOp.getIsVolatile());

        auto retType = bPrefetchPtr.getType();
        auto dot = dotOp.getODSResults(0)[0];
        auto advanceOp = dot2bYield[dot].getDefiningOp<triton::AdvanceOp>();
        SmallVector<Value> bOffset = advanceOp.getOffsets();
        auto adaOp = builder.create<triton::AdvanceOp>(loc, retType, bPrefetchPtr, bOffset);
        bAdvancedPtr.push_back(adaOp);
        builder.restoreInsertionPoint(insertionPoint);
      }

      newOp = newLoadOp;
    } else{
      newOp = builder.clone(op, mapping);
    }
    // update mapping of results
    for (unsigned dstIdx : llvm::seq(unsigned(0), op.getNumResults()))
      mapping.map(op.getResult(dstIdx), newOp->getResult(dstIdx));
  }

  // prefetch next iteration
  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands()){
    yieldValues.push_back(mapping.lookup(v));
  }
  for(int i = 0;i < dots.size();i++) {
    if(dot2aPtr.count(dots[i]) == 1){
      yieldValues.push_back(aAdvancedPtr[i]);
    }
    if(dot2bPtr.count(dots[i]) == 1){
      yieldValues.push_back(bAdvancedPtr[i]);
    }
  }
  // Update ops of yield
  builder.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);

  return newForOp;
}

void Prefetcher::emitPrologue() {
  dbgInfo("[Prefetch to cache]emitPrologue");
  OpBuilder builder(forOp);

  for (int i = 0; i < dots.size(); i++) {
    auto dot = dots[i];
    Attribute dotEncoding =
        dot.getType().cast<RankedTensorType>().getEncoding();
    if(dot2aPtr.count(dot) == 1){
      Value aPrefetched = generatePrefetch(builder, dot2aPtr[dot], dot2aLoad[dot]);
      operand2headPrefetch[dot.getDefiningOp<triton::DotOp>().getA()] =
          aPrefetched;
    }
    if(dot2bPtr.count(dot) == 1){
      Value bPrefetched = generatePrefetch(builder, dot2bPtr[dot], dot2bLoad[dot]);
      operand2headPrefetch[dot.getDefiningOp<triton::DotOp>().getB()] =
          bPrefetched;
    }
  }
}

class TritonGPUPrefetchToCachePass
    : public TritonGPUPrefetchToCacheBase<TritonGPUPrefetchToCachePass> {
public:
  TritonGPUPrefetchToCachePass() = default;

  void runOnOperation() override {
    getOperation()->walk([&](scf::ForOp forOp) {
      dbgInfo("[Prefetch to cache]");
      Prefetcher prefetcher(forOp);

      if (prefetcher.initialize().failed())
        return;

      prefetcher.emitPrologue();

      scf::ForOp newForOp = prefetcher.createNewForOp();

      for (unsigned i = 0; i < forOp->getNumResults(); ++i){
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      }

      forOp->erase();

      // dbgInfo("[Prefetch to cache] After prefetch");
      // auto module = newForOp.getOperation()->getParentOfType<ModuleOp>();
      // module.print(llvm::outs());
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUPrefetchToCachePass() {
  return std::make_unique<TritonGPUPrefetchToCachePass>();
}