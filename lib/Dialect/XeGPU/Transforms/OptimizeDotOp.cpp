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
#include "triton/Dialect/XeGPU/IR/XeGPUOps.h"
#include "triton/Dialect/XeGPU/Transforms/Passes.h"
#include "triton/Dialect/XeGPU/Transforms/XeGPUConversion.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/XeGPU/Transforms/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::xegpu;

#define i8_ty builder.getIntegerType(8)
#define i8_val(value) builder.create<arith::ConstantOp>(loc, i8_ty, builder.getI8IntegerAttr(value))

LogicalResult checkDotOp(scf::ForOp forOp){
  Block *loop = forOp.getBody();
  SmallVector<xegpu::DpasOp> dotsInFor;
  for (Operation &op : *loop)
    if (auto dotOp = dyn_cast<xegpu::DpasOp>(op))
      dotsInFor.push_back(dotOp);

  if (dotsInFor.empty()){
    llvm::outs() << "\n\n[OptimizeDotOp][No dpasOp in For Loop]\n";
    return failure();
  }

  return success();
}

void addSyncBarrierForDot(scf::ForOp forOp){
  Location loc = forOp.getLoc();
  auto context = forOp.getContext();
  OpBuilder builder(forOp);

  Block *loop = forOp.getBody();

  auto op = (*loop).begin();
  llvm::outs()<<"\n\n[OptimizeDotOp]op: "<<*op<<"\n";

  builder.setInsertionPoint(forOp);
  auto i32Type =  mlir::IntegerType::get(context, 32);
  auto v8i32Type = mlir::VectorType::get(8, i32Type);
  auto payload = builder.create<xegpu::CreateNbarrierOp>(loc, v8i32Type, i8_val(1), i8_val(0), 
                                                      ::mlir::IntegerAttr::get(mlir::IntegerType::get(context, 8), 32), 
                                                      ::mlir::IntegerAttr::get(mlir::IntegerType::get(context, 8), 32));

  builder.setInsertionPointToStart(forOp.getBody());
  builder.create<xegpu::NbarrierArriveOp>(loc, payload);

  auto lastOp = (*loop).rbegin();
  scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(*lastOp);
  builder.setInsertionPoint(yieldOp);
  builder.create<xegpu::NbarrierWaitOp>(loc, payload);
}

void addCompilerHintForDot(scf::ForOp forOp){
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();
  Block *loop = forOp.getBody();

  for(auto it = (*loop).begin();it != (*loop).end();){
    if(isa<xegpu::LoadNDOp>(*it) || isa<xegpu::PrefetchNDOp>(*it)){
      it++;
      if(isa<xegpu::UpdateNDOffsetOp>(*(it))){
        xegpu::UpdateNDOffsetOp op = dyn_cast<UpdateNDOffsetOp>(*it);
        builder.setInsertionPoint(op);
        builder.create<xegpu::CompilerHintOp>(loc);
        it++;
      }
    }else{
      it++;
    }
  }
}

class XeGPUOptimizeDotOpPass
    : public XeGPUOptimizeDotOpBase<XeGPUOptimizeDotOpPass> {
public:
  XeGPUOptimizeDotOpPass() = default;

  void runOnOperation() override {
    getOperation()->walk([&](scf::ForOp forOp) {
      if(checkDotOp(forOp).failed()){
        return;
      }

      addSyncBarrierForDot(forOp);

      addCompilerHintForDot(forOp);
    });
  }
};


std::unique_ptr<Pass> mlir::createXeGPUOptimizeDotOpPass() {
  return std::make_unique<XeGPUOptimizeDotOpPass>();
}