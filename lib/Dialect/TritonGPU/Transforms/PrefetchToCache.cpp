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

  Value generatePrefetch(Value ptr, OpBuilder &builder);
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
    llvm::outs() << "\n\n[Prefetch to cache][No dotOp in For Loop]\n";
    return failure();
  }

  // todo when used in flash attention that has 2 dots in the loop
  if (dotsInFor.size() > 1){
    llvm::outs()<<"\n\n[Prefetch to cache][dotsInFor.size() > 1]";
    return failure();
  }

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
      if (arg.getOwner()->getParentOp() == forOp.getOperation())
        return forOp.getOpOperandForRegionIterArg(arg).get();
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
    llvm::outs()<<"\n\n[Prefetch to cache]aCvt: "<<aCvtSrc<<"\n";
    llvm::outs()<<"\n\n[Prefetch to cache]bCvt: "<<bCvtSrc<<"\n";

    if (aCvtSrc && bCvtSrc) {
      auto aLoadOp = aCvtSrc.getDefiningOp<triton::LoadOp>();
      auto bLoadOp = bCvtSrc.getDefiningOp<triton::LoadOp>();

      Value aPtr = getIncomingOp(aLoadOp.getPtr());
      Value bPtr = getIncomingOp(bLoadOp.getPtr());

      llvm::outs()<<"\n\n[Prefetch to cache]aPtr: "<<aPtr<<"\n";
      llvm::outs()<<"\n\n[Prefetch to cache]bPtr: "<<bPtr<<"\n";

      // Only prefetch loop arg
      if (aPtr && bPtr) {
        dots.insert(dot);
        dotas.insert(dot.getA());
        dotbs.insert(dot.getB());
        dot2aPtr[dot] = aPtr;
        dot2bPtr[dot] = bPtr;
        dot2aLoopArg[dot] = aLoadOp.getPtr();
        dot2bLoopArg[dot] = bLoadOp.getPtr();
        dot2aYield[dot] = getYieldOp(aLoadOp.getPtr());
        dot2bYield[dot] = getYieldOp(bLoadOp.getPtr());
        dot2aLoad[dot] = aLoadOp;
        dot2bLoad[dot] = bLoadOp;

        // llvm::outs()<<"\n\n[Prefetch to cache]dot2aYield[dot]: "<<dot2aYield[dot]<<"\n";
        // llvm::outs()<<"\n\n[Prefetch to cache]dot2bYield[dot]: "<<dot2bYield[dot]<<"\n";
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

Value Prefetcher::generatePrefetch(Value ptr, OpBuilder &builder) {
  Location loc = ptr.getLoc();
  auto context = ptr.getContext();
  llvm::outs()<<"\n\n[Prefetch to cahce]ptr: "<<ptr<<"\n";
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
  sgId = urem(sgId, i32_val(32)); //subGroupNUm = 32

  int sgBlockDim0 = elemShape[2] * elemStride[2];
  int sgBlockDim1 = elemShape[3] * elemStride[3];
  int blockDim0 = elemShape[0] * threadShape[0];
  int blockDim1 = elemShape[1] * threadShape[1];
  int sgSizeM = threadShape[2];
  int sgSizeN = threadShape[3];
  auto sgIdM = udiv(sgId, i32_val(sgSizeN));
  auto sgIdN = urem(sgId, i32_val(sgSizeN));

  //llvm::outs()<<"\n\n[Prefetch to cahce]tensorType: "<<tensorType<<"\n";

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

  // llvm::outs()<<"\n\n[Prefetch to cahce]tensorOffsets[0]: "<<tensorOffsets[0]<<"\n";
  // llvm::outs()<<"\n\n[Prefetch to cahce]mmaFlag: "<<mmaFlag<<"\n";
  Value prefetchPtr;
  if(mmaFlag == 0){ //prefetch matrix A
    //todo currently, by default, the number of subGroups 
    //in the N direction is greater than the number of blocks.
    //llvm::outs()<<"\n\n[Prefetch to cahce]blockDim1: "<<blockDim1<<"\n";
    int prefetchInDim1 = sgBlockDim1 / blockDim1;
    int prefetchInDim0 = sgSizeN / prefetchInDim1;
    newBlockShape[0] = sgBlockDim0 / prefetchInDim0;
    newBlockShape[1] = blockDim1;

    tensorOffsets[0] = add(tensorOffsets[0], mul(sgIdM, i32_val(sgBlockDim0)));
    tensorOffsets[1] = add(tensorOffsets[1], i32_val(blockShape[1] * 2));
    Value sgOffsetM = mul(udiv(sgIdN, i32_val(prefetchInDim1)), i32_val(newBlockShape[0]));
    Value sgOffsetN = mul(urem(sgIdN, i32_val(prefetchInDim1)), i32_val(newBlockShape[1]));
    tensorOffsets[0] = add(tensorOffsets[0], sgOffsetM);
    tensorOffsets[1] = add(tensorOffsets[1], sgOffsetN);

    prefetchPtr = builder.create<triton::MakeTensorPtrOp>(
          loc, base, tensorShape, tensorStride, 
          tensorOffsets, newBlockShape, order);
  }else{ //prefetch matrix B
    int prefetchInDim1 = sgBlockDim1 / blockDim1;
    int prefetchInDim0 = sgSizeM / prefetchInDim1;
    newBlockShape[0] = sgBlockDim0 / prefetchInDim0;
    newBlockShape[1] = blockDim1;

    tensorOffsets[0] = add(tensorOffsets[0], i32_val(blockShape[0] * 2));
    tensorOffsets[1] = add(tensorOffsets[1], mul(sgIdN, i32_val(sgBlockDim1)));
    Value sgOffsetM = mul(udiv(sgIdM, i32_val(prefetchInDim1)), i32_val(newBlockShape[0]));
    Value sgOffsetN = mul(urem(sgIdM, i32_val(prefetchInDim1)), i32_val(newBlockShape[1]));
    tensorOffsets[0] = add(tensorOffsets[0], sgOffsetM);
    tensorOffsets[1] = add(tensorOffsets[1], sgOffsetN);

    prefetchPtr = builder.create<triton::MakeTensorPtrOp>(
          loc, base, tensorShape, tensorStride, 
          tensorOffsets, newBlockShape, order);
  }

  tensorType = prefetchPtr.getType().cast<triton::PointerType>()
                      .getPointeeType().cast<RankedTensorType>();
  //llvm::outs()<<"\n\n[Prefetch to cahce] tensorType: "<<tensorType<<"\n";
  //layout = tensorType.getEncoding().cast<GenericEncodingAttr>();
  auto newLayout = layout.updatemmaFlag(3);
  auto newType = RankedTensorType::get(tensorType.getShape(),
                elemType, newLayout);
  ptrType = PointerType::get(newType, 1);
  prefetchPtr.setType(ptrType);

  //llvm::outs()<<"\n\n[Prefetch to cahce]prefetchPtr: "<<prefetchPtr<<"\n";

  return prefetchPtr;
}

scf::ForOp Prefetcher::createNewForOp() {
  OpBuilder builder(forOp);

  //llvm::outs()<<"\n\n[Prefetch to cache] set loopArgs\n";
  SmallVector<Value> loopArgs;
  for (auto v : forOp.getIterOperands()){
    loopArgs.push_back(v);
  }
  for (Value dot : dots) {
    loopArgs.push_back(
        operand2headPrefetch[dot.getDefiningOp<triton::DotOp>().getA()]);
    loopArgs.push_back(
        operand2headPrefetch[dot.getDefiningOp<triton::DotOp>().getB()]);
  }

  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), loopArgs);

  builder.setInsertionPointToStart(newForOp.getBody());
  IRMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  Value aAdvancedPtr;
  Value bAdvancedPtr;

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
      llvm::outs()<<"\n\n[Prefetch to cache] newLoadOp: "<<*newLoadOp<<"\n";
      auto insertionPoint = builder.saveInsertionPoint();
      builder.setInsertionPointAfter(newLoadOp);
      Location loc = newLoadOp->getLoc();

      //llvm::outs()<<"\n\n[Prefetch to cache] prefetch A\n";
      LoadOp loadOp = dot2aLoad[dotOp];
      Value aPrefetchPtr = operand2headPrefetch.lookup(dotOp.getA());
      aPrefetchPtr = newForOp.getRegionIterArgForOpOperand(*aPrefetchPtr.use_begin());
      auto prefetchA = builder.create<triton::LoadOp>(loc, aPrefetchPtr, 
              loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());

      //llvm::outs()<<"\n\n[Prefetch to cache] prefetch B\n";
      LoadOp loadBOp = dot2bLoad[dotOp];
      Value bPrefetchPtr = operand2headPrefetch.lookup(dotOp.getB());
      bPrefetchPtr = newForOp.getRegionIterArgForOpOperand(*bPrefetchPtr.use_begin());
      auto prefetchB = builder.create<triton::LoadOp>(loc, bPrefetchPtr, 
              loadOp.getCache(),loadOp.getEvict(), loadOp.getIsVolatile());
      // builder.restoreInsertionPoint(insertionPoint);

      newOp = newLoadOp;

      //llvm::outs()<<"\n\n[Prefetch to cache] advance A ptr\n";
      // insertionPoint = builder.saveInsertionPoint();
      // builder.setInsertionPointAfter(newOp);
      auto retType = aPrefetchPtr.getType();
      auto dot = dotOp.getODSResults(0)[0];
      auto advanceOp = dot2aYield[dot].getDefiningOp<triton::AdvanceOp>();
      SmallVector<Value> aOffset = advanceOp.getOffsets();
      aAdvancedPtr = builder.create<triton::AdvanceOp>(loc, retType, aPrefetchPtr, aOffset);

      //llvm::outs()<<"\n\n[Prefetch to cache] advance A ptr\n";
      retType = bPrefetchPtr.getType();
      dot = dotOp.getODSResults(0)[0];
      advanceOp = dot2bYield[dot].getDefiningOp<triton::AdvanceOp>();
      SmallVector<Value> bOffset = advanceOp.getOffsets();
      bAdvancedPtr = builder.create<triton::AdvanceOp>(loc, retType, bPrefetchPtr, bOffset);
      builder.restoreInsertionPoint(insertionPoint);
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
  for (Value dot : dots) {
    yieldValues.push_back(aAdvancedPtr);
    yieldValues.push_back(bAdvancedPtr);
  }
  // Update ops of yield
  builder.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);

  return newForOp;
}

void Prefetcher::emitPrologue() {
  OpBuilder builder(forOp);

  for (int i = 0; i < dots.size(); i++) {
    auto dot = dots[i];
    Attribute dotEncoding =
        dot.getType().cast<RankedTensorType>().getEncoding();
    llvm::outs()<<"\n\n[Prefetch to cache]dotEncoding:"<<dotEncoding<<"\n";
    Attribute dotaEncoding = dotas[i].getType().cast<RankedTensorType>().getEncoding()
                            .dyn_cast<DotOperandEncodingAttr>().getParent();
    Attribute dotbEncoding = dotbs[i].getType().cast<RankedTensorType>().getEncoding()
                            .dyn_cast<DotOperandEncodingAttr>().getParent();
    Value aPrefetched = generatePrefetch(dot2aPtr[dot], builder);
    operand2headPrefetch[dot.getDefiningOp<triton::DotOp>().getA()] =
        aPrefetched;
    Value bPrefetched = generatePrefetch(dot2bPtr[dot], builder);
    operand2headPrefetch[dot.getDefiningOp<triton::DotOp>().getB()] =
        bPrefetched;
  }
}

class TritonGPUPrefetchToCachePass
    : public TritonGPUPrefetchToCacheBase<TritonGPUPrefetchToCachePass> {
public:
  TritonGPUPrefetchToCachePass() = default;

  void runOnOperation() override {
    getOperation()->walk([&](scf::ForOp forOp) {
      llvm::outs()<<"\n\n[Prefetch to cache]\n";
      Prefetcher prefetcher(forOp);

      if (prefetcher.initialize().failed())
        return;

      prefetcher.emitPrologue();

      scf::ForOp newForOp = prefetcher.createNewForOp();

      for (unsigned i = 0; i < forOp->getNumResults(); ++i){
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      }

      forOp->erase();
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUPrefetchToCachePass() {
  return std::make_unique<TritonGPUPrefetchToCachePass>();
}