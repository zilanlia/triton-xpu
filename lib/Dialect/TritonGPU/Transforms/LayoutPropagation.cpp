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

#include <queue>
#include <vector>
#include <set>
#include <map>

using namespace mlir;
using namespace mlir::scf;
using ::mlir::triton::gpu::GenericEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;

using opsVectorTy = std::vector<mlir::Operation *>;
using opsQueueTy = std::queue<mlir::Operation *>;
using opsSetTy = std::set<mlir::Operation *>;
using opsGraphTy = std::map<mlir::Operation *, std::set<mlir::Operation *>>;

bool checkType(Type type){
  if(isa<RankedTensorType>(type)){
    return true;
  } else if(isa<triton::PointerType>(type)){
    auto ptrType = type.cast<triton::PointerType>();
    auto pointeeType = ptrType.getPointeeType();
    if(isa<RankedTensorType>(pointeeType)){
      return true;
    }
  }

  return false;
}

void createGraph(opsVectorTy ops, opsGraphTy& preOpsGraph, opsGraphTy& sucOpsGraph){
  int opsNum = ops.size();
  for(int i = 0; i < opsNum; i++){
    auto op = ops[i];
    if(preOpsGraph.count(op) == 0){
      preOpsGraph[op] = std::set<mlir::Operation *>();
      sucOpsGraph[op] = std::set<mlir::Operation *>();
    }

    for(auto operand : op->getOperands()){
      auto type = operand.getType();
      //llvm::outs() << "\n\n[createGraph]type: "<<type<<"\n";
      if(!checkType(type)){
        continue;
      }

      if(Operation *parentOp = operand.getDefiningOp()){
        //use the result of forOp
        if(auto forOp = llvm::dyn_cast<ForOp>(parentOp)){
          auto results = forOp->getResults(); 
          for(unsigned i = 0; i < results.size(); ++i){
            if(operand == results[i]){
              auto operandInForOp = forOp.getOperands()[i + 3];
              auto *forLoopParentOp = operandInForOp.getDefiningOp();
              preOpsGraph[op].insert(forLoopParentOp);
              sucOpsGraph[forLoopParentOp].insert(op);
              break;
            }
          }
        } else {
          preOpsGraph[op].insert(parentOp);

          if(sucOpsGraph.count(op) == 0){
            sucOpsGraph[parentOp] = {};
          }
          sucOpsGraph[parentOp].insert(op);
        }
      }

      //use the input of forOp;
      if(auto forOp = llvm::dyn_cast<ForOp>(op->getParentOp())) 
      {
        //llvm::outs()<<"[forOp related Ops] op: "<<*op<<"\n";
        for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
          if (operand == forOp.getRegionIterArgs()[i]) {
            auto operandInForOp = forOp.getOperands()[i + 3];
            auto *forLoopParentOp = operandInForOp.getDefiningOp();
            preOpsGraph[op].insert(forLoopParentOp);
            sucOpsGraph[forLoopParentOp].insert(op);
            //llvm::outs()<<"[forOp related Ops] i: "<<i<<"\n";
            break;
          }
        }
      }
    }
  }
}

void propagateLayout(MLIRContext *context, opsQueueTy &opsQueue, Attribute &encoding, 
                        opsGraphTy &preOpsGraph, opsGraphTy &sucOpsGraph, opsSetTy& opsSet){
  while(!opsQueue.empty()){
    auto op = opsQueue.front();
    //llvm::outs() << "\n\n[propagateLayout]op: "<<*op<<"\n";
    opsSet.erase(op);
    opsQueue.pop();

    if(auto ConstantOp = dyn_cast<arith::ConstantOp>(op)){
      auto type = ConstantOp.getValue().getType();
      //llvm::outs() << "\n\n[propagateLayout ConstantOp]type: "<<type<<"\n";
      if (type.isa<mlir::RankedTensorType>()){
        auto value = ConstantOp.getValue().dyn_cast<DenseElementsAttr>();
        auto tensorType = type.cast<RankedTensorType>();
        auto layout =  tensorType.getEncoding();
        auto result = ConstantOp.getResult();
        if(layout.dyn_cast<GenericEncodingAttr>().getMmaFlag() == -1){
          auto newType = RankedTensorType::get(tensorType.getShape(), 
                  tensorType.getElementType(), encoding);
          value = value.reshape(newType);
          ConstantOp.setValueAttr(value);
          result.setType(newType);
        }
      }
    } else {
      for(auto operand : op->getOperands()){
        Type type = operand.getType();
        bool isPointerType = 0;
        int addr;
        //llvm::outs() << "\n\n[propagateLayout]type: "<<type<<"\n";
        if(isa<triton::PointerType>(type)){
          addr = type.cast<triton::PointerType>().getAddressSpace();
          type = type.cast<triton::PointerType>().getPointeeType();
          isPointerType = 1;
        }
        //llvm::outs() << "\n\n[propagateLayout]after convert pointType : "<<type<<"\n";
        if(type.isa<mlir::RankedTensorType>()){
          auto tensorType = type.cast<RankedTensorType>();
          auto layout = tensorType.getEncoding();
          if(auto genericLayout = layout.dyn_cast<GenericEncodingAttr>()){
            if(genericLayout.getMmaFlag() == -1){
              auto newType = RankedTensorType::get(tensorType.getShape(), 
                    tensorType.getElementType(), encoding);
              if(isPointerType){
                auto pointerType = triton::PointerType::get(newType, addr);
                operand.setType(pointerType);
              } else {
                operand.setType(newType);
              }
            }
          }
        }
      }

      for(auto result : op->getResults()){
        Type type = result.getType();
        bool isPointerType = 0;
        int addr;
        //llvm::outs() << "\n\n[propagateLayout]type: "<<type<<"\n";
        if(isa<triton::PointerType>(type)){
          addr = type.cast<triton::PointerType>().getAddressSpace();
          type = type.cast<triton::PointerType>().getPointeeType();
          isPointerType = 1;
        }
        //llvm::outs() << "\n\n[propagateLayout]after convert pointType : "<<type<<"\n";
        if(type.isa<mlir::RankedTensorType>()){
          auto tensorType = type.cast<RankedTensorType>();
          auto layout = tensorType.getEncoding();
          if(auto genericLayout = layout.dyn_cast<GenericEncodingAttr>()){
            if(genericLayout.getMmaFlag() == -1){
              auto newType = RankedTensorType::get(tensorType.getShape(), 
                    tensorType.getElementType(), encoding);
              if(isPointerType){
                auto pointerType = triton::PointerType::get(newType, addr);
                result.setType(pointerType);
              } else {
                result.setType(newType);
              }
            }
          }
        }
      }
    }

    //llvm::outs() << "\n\n[propagateLayout]after update layout: "<<*op<<"\n";

    for(auto preOp : preOpsGraph[op]){
      if(opsSet.count(preOp) == 1){
        opsQueue.push(preOp);
      }
    }
    for(auto sucOp : sucOpsGraph[op]){
      if(opsSet.count(sucOp) == 1){
        opsQueue.push(sucOp);
      }
    }
  }
}

void setDotOpLayout(MLIRContext *context, Operation *curr){
  if (auto dotOp = dyn_cast<triton::DotOp>(curr)) {
    //Need to be inferred from hardware info
    const std::vector<unsigned int> aThreadShapeVec{2, 8, 8, 4};
    const std::vector<unsigned int> aThreadStrideVec{1, 2, 32, 0};
    const std::vector<unsigned int> aElemPerThreadVec{4, 2, 4, 2};
    const std::vector<unsigned int> aElemStrideVec{2, 1, 8, 16};
    const std::vector<unsigned int> aSubGroupShapeVec{4, 4};
    const std::vector<unsigned int> aOrderVec{1, 0, 1, 0};
    ArrayRef<unsigned int> aThreadShape(aThreadShapeVec);
    ArrayRef<unsigned int> aThreadstride(aThreadStrideVec);
    ArrayRef<unsigned int> aElemPerThread(aElemPerThreadVec);
    ArrayRef<unsigned int> aElemStride(aElemStrideVec);
    ArrayRef<unsigned int> aSubGroupShape(aSubGroupShapeVec);
    ArrayRef<unsigned int> aOrder(aOrderVec);
    auto encoding = triton::gpu::GenericEncodingAttr::get(
        dotOp.getContext(), aThreadShape, aThreadstride, aElemPerThread, aElemStride, aSubGroupShape, aOrder, 0);
    auto dotLayout = triton::gpu::DotOperandEncodingAttr::get(context, 0, encoding, 0);

    auto A = dotOp.getA();
    auto aTensorType = A.getType().dyn_cast<RankedTensorType>();
    auto newType = RankedTensorType::get(aTensorType.getShape(), aTensorType.getElementType(),
                dotLayout);
    llvm::outs()<<"\n\n[DotOp]matA newType: "<<newType<<"\n";
    A.setType(newType);

    const std::vector<unsigned int> bThreadShapeVec{1, 16, 8, 4};
    const std::vector<unsigned int> bThreadStrideVec{0, 1, 0, 64};
    const std::vector<unsigned int> bElemPerThreadVec{16, 1, 2, 4};
    const std::vector<unsigned int> bElemStrideVec{1, 0, 16, 16};
    const std::vector<unsigned int> bSubGroupShapeVec{4, 4};
    const std::vector<unsigned int> bOrderVec{1, 0, 1, 0};
    ArrayRef<unsigned int> bThreadShape(bThreadShapeVec);
    ArrayRef<unsigned int> bThreadstride(bThreadStrideVec);
    ArrayRef<unsigned int> bElemPerThread(bElemPerThreadVec);
    ArrayRef<unsigned int> bElemStride(bElemStrideVec);
    ArrayRef<unsigned int> bSubGroupShape(bSubGroupShapeVec);
    ArrayRef<unsigned int> bOrder(bOrderVec);
    encoding = triton::gpu::GenericEncodingAttr::get(
        dotOp.getContext(), bThreadShape, bThreadstride, bElemPerThread, bElemStride, bSubGroupShape, bOrder, 1);
    dotLayout = triton::gpu::DotOperandEncodingAttr::get(context, 1, encoding, 0);

    auto B = dotOp.getB();
    auto bTensorType = B.getType().dyn_cast<RankedTensorType>();
    newType = RankedTensorType::get(bTensorType.getShape(), bTensorType.getElementType(),
                dotLayout);
    llvm::outs()<<"\n\n[DotOp]matB newType: "<<newType<<"\n";
    B.setType(newType);

    const std::vector<unsigned int> cThreadShapeVec{1, 16, 8, 4};
    const std::vector<unsigned int> cThreadStrideVec{0, 1, 32, 64};
    const std::vector<unsigned int> cElemPerThreadVec{8, 1, 4, 4};
    const std::vector<unsigned int> cElemStrideVec{1, 0, 8, 16};
    const std::vector<unsigned int> cSubGroupShapeVec{4, 4};
    const std::vector<unsigned int> cOrderVec{1, 0, 1, 0};
    ArrayRef<unsigned int> cThreadShape(cThreadShapeVec);
    ArrayRef<unsigned int> cThreadstride(cThreadStrideVec);
    ArrayRef<unsigned int> cElemPerThread(cElemPerThreadVec);
    ArrayRef<unsigned int> cElemStride(cElemStrideVec);
    ArrayRef<unsigned int> cSubGroupShape(cSubGroupShapeVec);
    ArrayRef<unsigned int> cOrder(cOrderVec);
    encoding = triton::gpu::GenericEncodingAttr::get(
        dotOp.getContext(), cThreadShape, cThreadstride, cElemPerThread, cElemStride, cSubGroupShape, cOrder, 2);
    // dotLayout = triton::gpu::DotOperandEncodingAttr::get(context, 1, encoding, 0);

    auto C = dotOp.getC();
    auto cTensorType = C.getType().dyn_cast<RankedTensorType>();
    newType = RankedTensorType::get(cTensorType.getShape(), cTensorType.getElementType(),
                encoding);
    llvm::outs()<<"\n\n[DotOp]matC/matD newType: "<<newType<<"\n";
    C.setType(newType);

    auto D = dotOp.getD();
    D.setType(newType);
  }
}

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPULayoutPropagationPass
    : public TritonGPULayoutPropagationBase<TritonGPULayoutPropagationPass> {
public:
  TritonGPULayoutPropagationPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *op = getOperation();

    opsVectorTy opsVector;
    opsSetTy opsSet;
    opsGraphTy  preOpsGraph;
    opsGraphTy sucOpsGraph;

    //update layout for dotOp
    op->walk([&](Operation *curr) {
      setDotOpLayout(context, curr);
    });

    //Some special ops do not participate in the process
    op->walk([&](mlir::Operation *op) {
      if (!(mlir::isa<func::FuncOp>(op) || mlir::isa<mlir::ModuleOp>(op) || mlir::isa<scf::YieldOp>(op)
            || mlir::isa<triton::FuncOp>(op) || mlir::isa<scf::ForOp>(op))){
        opsVector.push_back(op);
        opsSet.insert(op);
      }
    });

    // Find the parent and child operations of each op
    createGraph(opsVector, preOpsGraph, sucOpsGraph);

    if(0){
      for(auto op : opsVector){
        llvm::outs()<<"\n\n[tritonGPUIR] op: \n"<<*op<<"\n";
        llvm::outs()<<"[tirtonGPUIR] preOps: \n";
        for(auto preOp : preOpsGraph[op]){
          llvm::outs()<<"[tritonGPUIR] preOp: \n"<<*preOp<<"\n";
        }

        llvm::outs()<<"[tirtonGPUIR] sucOps: \n";
        for(auto sucOp : sucOpsGraph[op]){
          llvm::outs()<<"[tritonGPUIR] sucOp: \n"<<*sucOp<<"\n";
        }
      }
    }

    // propagate Layout, start from DotOp
    op->walk([&](Operation *curr) {
      if (auto dotOp = dyn_cast<triton::DotOp>(curr)){
        opsSet.erase(dotOp);
        auto matA = dotOp.getA();
        auto matB = dotOp.getB();
        auto matC = dotOp.getC();

        auto aEncoding = matA.getType().dyn_cast<RankedTensorType>().getEncoding();
        auto bEncoding = matB.getType().dyn_cast<RankedTensorType>().getEncoding();
        auto cEncoding = matC.getType().dyn_cast<RankedTensorType>().getEncoding();

        aEncoding = aEncoding.dyn_cast<DotOperandEncodingAttr>()
                                      .getParent().cast<GenericEncodingAttr>();
        bEncoding = bEncoding.dyn_cast<DotOperandEncodingAttr>()
                                      .getParent().cast<GenericEncodingAttr>();

        llvm::outs() << "\n\naEncoding: " << aEncoding << "\n";
        llvm::outs() << "\n\nbEncoding: " << bEncoding << "\n";
        opsQueueTy aRelatedOpsQueue, bRelatedOpsQueue, cRelatedOpsQueue;

        for(auto preOp : preOpsGraph[dotOp]){
          if(preOp->getResult(0) == matA){
            aRelatedOpsQueue.push(preOp);
          } else if(preOp->getResult(0) == matB){
            bRelatedOpsQueue.push(preOp);
          } else{
            cRelatedOpsQueue.push(preOp);
          }
        }

        for(auto sucOp : sucOpsGraph[dotOp]){
          cRelatedOpsQueue.push(sucOp);
        }

        propagateLayout(context, aRelatedOpsQueue, aEncoding, preOpsGraph, sucOpsGraph, opsSet);
        propagateLayout(context, bRelatedOpsQueue, bEncoding, preOpsGraph, sucOpsGraph, opsSet);
        propagateLayout(context, cRelatedOpsQueue, cEncoding, preOpsGraph, sucOpsGraph, opsSet);
      }
    });

    // process forOp
    op->walk([&](Operation *curr) {
      if (auto forOp = dyn_cast<scf::ForOp>(curr)){
        for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
            auto operand = forOp.getOperands()[i + 3];
            auto result = forOp.getODSResults(0)[i];
            auto type = operand.getType();
            result.setType(type);
        }
      }
    });

    llvm::outs()<<"\n\n[After propagateLayout]tritonGPU IR: "<<"\n";
    op->print(llvm::outs());

    return;
  }
};

std::unique_ptr<Pass> mlir::createTritonGPULayoutPropagationPass() {
  return std::make_unique<TritonGPULayoutPropagationPass>();
}