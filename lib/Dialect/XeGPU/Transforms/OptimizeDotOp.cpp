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

#include<vector>
#include<map>
#include<set>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::xegpu;

#define i8_ty builder.getIntegerType(8)
#define i8_val(value) builder.create<arith::ConstantOp>(loc, i8_ty, builder.getI8IntegerAttr(value))

//only for attention now
class Tiling {
  scf::ForOp forOp;
  scf::YieldOp yieldOp;

  std::vector<std::vector<Operation *>> dpasOps;
  std::map<xegpu::DpasOp, int> dpasRound;
  std::map<xegpu::LoadNDOp, int> loadRound;
  std::map<Operation *, Operation *> dpas2ALoad;
  std::map<Operation *, Operation *> dpas2BLoad;
  std::set<Operation *> scheduledOps;
  std::vector<Operation *> uselessCastOp;
public:
  Tiling() = delete;

  Tiling(scf::ForOp forOp) : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LogicalResult initialize();

  void scheduleOps();
};

LogicalResult Tiling::initialize() {
  Block *loop = forOp.getBody();
  SmallVector<xegpu::DpasOp> dpasInFor;

  for (Operation &op : *loop){
    if (auto castOp = dyn_cast<mlir::UnrealizedConversionCastOp>(op)){
      Value result = castOp.getODSResults(0)[0];
      auto user = result.getUses();
      llvm::outs() << "[OptimizedDotOp]castOp result: " << result << "\n";
      if(user.begin() == user.end()){
        Operation *cast = castOp.getOperation();
        uselessCastOp.push_back(cast);
      }
    }
  }

  for(auto castOp : uselessCastOp){
    castOp->erase();
  }


  for (Operation &op : *loop)
    if (auto dpasOp = dyn_cast<xegpu::DpasOp>(op))
      dpasInFor.push_back(dpasOp);

  if (dpasInFor.empty()){
    llvm::outs() << "[OptimizedDotOp][No dpasOp in For Loop]" << "\n";
    return failure();
  }

  for (xegpu::DpasOp dpasOp : dpasInFor) {
    Value dpasA = dpasOp.getLhs();
    Value dpasB = dpasOp.getRhs();
    Value dpasC = dpasOp.getAcc();
    Operation *dpas = dpasOp.getOperation();

    int round = 0;
    while(dpasC.getDefiningOp()){
      if(auto parentDpasOp = dpasC.getDefiningOp<xegpu::DpasOp>()){
        dpasC = parentDpasOp.getAcc();
        round++;
      }else{
        dpasRound[dpasOp] = round;
        if(dpasOps.size() <= round){
          dpasOps.push_back({});
        }
        dpasOps[round].push_back(dpas);
        break;
      }
    }

    if(auto loadNDOp = dpasA.getDefiningOp<xegpu::LoadNDOp>()){
      dpas2ALoad[dpas] = loadNDOp.getOperation();
      loadRound[loadNDOp] = dpasRound[dpasOp];
    }

    if(auto loadNDOp = dpasB.getDefiningOp<xegpu::LoadNDOp>()){
      dpas2BLoad[dpas] = loadNDOp.getOperation();
      loadRound[loadNDOp] = dpasRound[dpasOp];
    }
  }

  return success();
}

void Tiling::scheduleOps() {
  int dpasGroupNum = dpasOps[0].size();
  //for attetion
  dpasGroupNum /= 2;

  //schedule dpas Ops
  for(int i = 0; i < dpasOps.size(); i++){
    for(int j = 1; j < dpasGroupNum; j++){
      auto dpas0 = dpasOps[i][j];
      auto dpas1 = dpasOps[i][j - 1];
      dpas0->moveAfter(dpas1);
    }
    for(int j = dpasGroupNum + 1; j < dpasOps[0].size(); j++){
      auto dpas0 = dpasOps[i][j];
      auto dpas1 = dpasOps[i][j - 1];
      dpas0->moveAfter(dpas1);
    }
  }

  llvm::outs() << "[OptimizedDotOp][schedule loadNd Ops]" << "\n";
  //schedule loadNd Ops
  for(int i = 1; i < dpasOps.size(); i++){
    Operation* dpas = dpasOps[0][0];
    bool scheduleA = dpas2ALoad.count(dpas) != 0;
    bool scheduleB = dpas2BLoad.count(dpas) != 0;

    llvm::outs() << "[OptimizedDotOp]scheduleA: " << scheduleA << "\n";
    if(scheduleA){
      dpas = dpasOps[i][0];
      Operation* load = dpas2ALoad[dpas];
      load->moveAfter(dpasOps[i - 1][dpasGroupNum - 1]);
      scheduledOps.insert(load);

      for(int j = 1; j < dpasGroupNum; j++){
        auto load0 = dpas2ALoad[dpasOps[i][j]];
        auto load1 = dpas2ALoad[dpasOps[i][j - 1]];
        if(scheduledOps.count(load0) == 0){
          load0->moveAfter(load1);
          scheduledOps.insert(load0);
        }
      }
    }

    llvm::outs() << "[OptimizedDotOp]scheduleB: " << scheduleB << "\n";
    if(scheduleB){
      dpas = dpasOps[i][0];
      Operation* load = dpas2BLoad[dpas];
      // llvm::outs() << "[OptimizedDotOp]dpas2BLoad[dpas]: " << *dpas2BLoad[dpas] << "\n";
      // llvm::outs() << "[OptimizedDotOp]dpasOps[i - 1]: " << i - 1 << " dpasGroupNum - 1: " << dpasGroupNum - 1 << "\n";
      // llvm::outs() << "[OptimizedDotOp]dpasOps[i - 1][dpasGroupNum - 1]: " << *dpasOps[i - 1][dpasGroupNum - 1] << "\n";
      load->moveAfter(dpasOps[i - 1][dpasGroupNum - 1]);
      scheduledOps.insert(load);

      for(int j = 1; j < dpasGroupNum; j++){
        auto load0 = dpas2BLoad[dpasOps[i][j]];
        auto load1 = dpas2BLoad[dpasOps[i][j - 1]];
        if(scheduledOps.count(load0) == 0){
          load0->moveAfter(load1);
          scheduledOps.insert(load0);
        }
      }
    }

    dpas = dpasOps[0][dpasGroupNum];
    scheduleA = dpas2ALoad.count(dpas) != 0;
    scheduleB = dpas2BLoad.count(dpas) != 0;

    llvm::outs() << "[OptimizedDotOp]scheduleA: " << scheduleA << "\n";
    if(scheduleA){
      dpas = dpasOps[i][dpasGroupNum];
      Operation* load = dpas2ALoad[dpas];
      load->moveAfter(dpasOps[i - 1][dpasOps[0].size() - 1]);
      scheduledOps.insert(load);
      for(int j = dpasGroupNum + 1; j < dpasOps[0].size(); j++){
        auto load0 = dpas2ALoad[dpasOps[i][j]];
        auto load1 = dpas2ALoad[dpasOps[i][j - 1]];
        if(scheduledOps.count(load0) != 1){
          load0->moveAfter(load1);
          scheduledOps.insert(load0);
        }
      }
    }

    llvm::outs() << "[OptimizedDotOp]scheduleB: " << scheduleB << "\n";
    if(scheduleB){
      dpas = dpasOps[i][dpasGroupNum];
      Operation* load = dpas2BLoad[dpas];
      load->moveAfter(dpasOps[i - 1][dpasOps[0].size() - 1]);
      scheduledOps.insert(load);
      for(int j = dpasGroupNum + 1; j < dpasOps[0].size(); j++){
        auto load0 = dpas2BLoad[dpasOps[i][j]];
        auto load1 = dpas2BLoad[dpasOps[i][j - 1]];
        if(scheduledOps.count(load0) != 1){
          load0->moveAfter(load1);
          scheduledOps.insert(load0);
        }
      }
    }
  }
  llvm::outs() << "[OptimizedDotOp]forOp: " << forOp << "\n";
}


LogicalResult checkDotOp(scf::ForOp forOp){
  Block *loop = forOp.getBody();
  SmallVector<xegpu::DpasOp> dpasInFor;
  for (Operation &op : *loop)
    if (auto dpasOp = dyn_cast<xegpu::DpasOp>(op))
      dpasInFor.push_back(dpasOp);

  if (dpasInFor.size() > 1){
    llvm::outs() << "\n\n[OptimizeDotOp][More than 1 dpasOp in For Loop]\n";
    return failure();
  }

  if (dpasInFor.empty()){
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
      // if(checkDotOp(forOp).failed()){
      //   return;
      // }

      // addSyncBarrierForDot(forOp);

      // addCompilerHintForDot(forOp);

      Tiling tiling(forOp);

      if (tiling.initialize().failed())
        return;

      tiling.scheduleOps();
    });
  }
};


std::unique_ptr<Pass> mlir::createXeGPUOptimizeDotOpPass() {
  return std::make_unique<XeGPUOptimizeDotOpPass>();
}