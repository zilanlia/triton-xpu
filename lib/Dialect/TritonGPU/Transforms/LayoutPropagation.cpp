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
using ::mlir::triton::gpu::SliceEncodingAttr;
using ::mlir::triton::gpu::GenericEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;

void generateDenGraph(std::vector<mlir::Operation *> ops, std::map<mlir::Operation *, std::vector<mlir::Operation *>>& predecessorOps,
                                                          std::map<mlir::Operation *, std::vector<mlir::Operation *>>& successorOps){
  int opsNum = ops.size();
  for(int i = 0; i < opsNum; i++){
    auto op = ops[i];
    if(predecessorOps.count(op) == 0){
      predecessorOps[op] = {};
    }

    mlir::scf::ForOp forOp = llvm::dyn_cast<mlir::scf::ForOp>(op->getParentOp());
    for(auto operand : op->getOperands()){
      if(mlir::Operation *parentOp = operand.getDefiningOp()){
        predecessorOps[op].push_back(parentOp);

        if(successorOps.count(op) == 0){
          successorOps[parentOp] = {};
        }
        successorOps[parentOp].push_back(op);
      } 
      else if(forOp) 
      {
        predecessorOps[op].push_back(forOp);

        for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
          if (operand == forOp.getRegionIterArgs()[i]) {
            auto operandInForOp = forOp.getOperands()[i + 3];

            auto *forLoopParentOp = operandInForOp.getDefiningOp();
            predecessorOps[op].push_back(forLoopParentOp);
            break;
          }
        }
      }
    }
  }
}

void propagationLayout(MLIRContext *context, mlir::Operation *op, Attribute &encoding, 
                        std::map<mlir::Operation *, std::vector<mlir::Operation *>> &predecessorOps,
                        std::set<mlir::Operation *>& opsSet){
  opsSet.erase(op);
  if(predecessorOps[op].size() == 0){
    return;
  }

  for(auto preOp : predecessorOps[op]){
    //process forloop result type
    if(auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(preOp)){
      Type newType;
      for(auto operand : op->getOperands()){
        for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
          if (operand == forOp.getRegionIterArgs()[i]){
            auto operandInForOp = forOp.getOperands()[i + 3];
            auto resultInForOp = forOp.getODSResults(0)[i];
            auto curTensorType = operandInForOp.getType().cast<RankedTensorType>();
            auto newType = RankedTensorType::get(curTensorType.getShape(), curTensorType.getElementType(),
                          encoding);
            operandInForOp.setType(newType);
            resultInForOp.setType(newType);
            break;
          }
        }
      }
      //update op in for loop(eg: Loadop)
      for(auto operand : op->getOperands()){
        auto curTensorType = operand.getType().cast<RankedTensorType>();
        auto curLayout = curTensorType.getEncoding();

        if (auto genericLayout = curLayout.dyn_cast<GenericEncodingAttr>()){
          auto newType = RankedTensorType::get(curTensorType.getShape(), curTensorType.getElementType(),
                      encoding);
          operand.setType(newType);
        }
      }
    } else if(auto ConstantOp = dyn_cast<arith::ConstantOp>(preOp)){
      auto curType = ConstantOp.getValue().getType();

      if (curType.isa<mlir::RankedTensorType>()){
        auto value = ConstantOp.getValue().dyn_cast<DenseElementsAttr>();
        auto curTensorType = curType.cast<RankedTensorType>();
        auto currlayout =  curTensorType.getEncoding();
        if(currlayout.dyn_cast<GenericEncodingAttr>().getIsLayoutUpdated() == 1){
          auto newType = RankedTensorType::get(curTensorType.getShape(), curTensorType.getElementType(),
            encoding);
          value = value.reshape(newType);
          ConstantOp.setValueAttr(value);
        }
      }
    } else if(auto cvtOp = llvm::dyn_cast<triton::gpu::ConvertLayoutOp>(preOp)){
      //break;
    } else {
      for (auto preOperand : preOp->getResults()){
        auto curType = preOperand.getType();
        if (curType.isa<mlir::RankedTensorType>()){
          auto curTensorType = curType.cast<RankedTensorType>();
          auto currlayout = curTensorType.getEncoding();
          if(auto currDotLayout = currlayout.dyn_cast<DotOperandEncodingAttr>()){

          }
          else if(auto curSliceLayout = currlayout.dyn_cast<SliceEncodingAttr>()){
            unsigned dim = curSliceLayout.getDim();
            auto curGenericLayout = curSliceLayout.getParent().dyn_cast<GenericEncodingAttr>();
            if(curGenericLayout.getIsLayoutUpdated() == 1){
              auto newGenericLayout = curGenericLayout.updateIsLayoutUpdated(2);
              auto newSliceLayout = SliceEncodingAttr::get(context, dim, newGenericLayout);
              auto newType = RankedTensorType::get(curTensorType.getShape(), curTensorType.getElementType(),
                newSliceLayout);
              preOperand.setType(newType);
            }
          }
          else if(currlayout.dyn_cast<GenericEncodingAttr>().getIsLayoutUpdated() == 1){
            auto newType = RankedTensorType::get(curTensorType.getShape(), curTensorType.getElementType(),
              encoding);
            preOperand.setType(newType);
          }
        }
      }
    }

    propagationLayout(context, preOp, encoding, predecessorOps, opsSet);
  }
}

bool updateLayout(mlir::Operation *op, mlir::Value* operand,
                                  std::map<mlir::Operation *,std::vector<mlir::Operation *>> &Ops){
  for(auto preOp : Ops[op]){
    for (auto preOperand : preOp->getResults()){
      auto curType = operand->getType();
      auto type = preOperand.getType();
      if (curType.isa<mlir::RankedTensorType>()){
        auto tensorType = type.cast<RankedTensorType>();
        if(auto curTensorType = curType.cast<RankedTensorType>()){
          auto layout = tensorType.getEncoding();
          auto generic = layout.dyn_cast<GenericEncodingAttr>();
          if(generic.getIsLayoutUpdated() == 2){
            auto newType = RankedTensorType::get(curTensorType.getShape(), curTensorType.getElementType(),
              layout);

            operand->setType(newType);
            return true;
          }
        } else {
          auto layout = tensorType.getEncoding();
          auto generic = layout.dyn_cast<GenericEncodingAttr>();
          if(generic.getIsLayoutUpdated() == 2){
            auto newType = RankedTensorType::get(curTensorType.getShape(), curType,
              layout);

            operand->setType(newType);
            return true;
          }
        }
      }
    }
  }

  return false;
}

bool updateLayout(mlir::Operation *op, DenseElementsAttr value,
                                  std::map<mlir::Operation *,std::vector<mlir::Operation *>> &Ops){
  for(auto preOp : Ops[op]){
    for (auto preOperand : preOp->getResults()){
      auto type = preOperand.getType();
      
      if (type.isa<mlir::RankedTensorType>()){
        auto tensorType = type.cast<RankedTensorType>();
        auto layout = tensorType.getEncoding();
        auto generic = layout.dyn_cast<GenericEncodingAttr>();
        if(generic.getIsLayoutUpdated() == 2){

          if(auto ConstantOp = dyn_cast<arith::ConstantOp>(op)){
            auto curType = ConstantOp.getValue().getType();
            Type newType;
            if (curType.isa<mlir::RankedTensorType>()){
              auto curTensorType = curType.cast<RankedTensorType>();
              newType = RankedTensorType::get(curTensorType.getShape(), curTensorType.getElementType(),
                        layout);
            }else{
              newType = RankedTensorType::get(tensorType.getShape(), curType,
                        layout);
            }
            value = value.reshape(newType.cast<ShapedType>());
            ConstantOp.setValueAttr(value);
          }
          return true;
        }
      }
    }
  }

  return false;
}


#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPULayoutPropagationPass
    : public TritonGPULayoutPropagationBase<
          TritonGPULayoutPropagationPass> {
public:
  TritonGPULayoutPropagationPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *op = getOperation();

    RankedTensorType newType;
    GenericEncodingAttr newGenericLayout;
    Attribute encoding;
    RankedTensorType tensorType;

    std::vector<mlir::Operation *> allOpsInModule;
    std::set<mlir::Operation *> opsSet;
    std::map<mlir::Operation *, std::vector<mlir::Operation *>> predecessorOps;
    std::map<mlir::Operation *, std::vector<mlir::Operation *>> successorOps;

    //update layout for dotOp
    op->walk([&](Operation *curr) {
      if (auto dotOp = dyn_cast<triton::DotOp>(curr)) {
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
            dotOp.getContext(), aThreadShape, aThreadstride, aElemPerThread, aElemStride, aSubGroupShape, aOrderVec, 0);
        auto dotLayout = triton::gpu::DotOperandEncodingAttr::get(context, 0, encoding, 0);

        auto A = dotOp.getA();
        auto aType = A.getType();
        llvm::outs()<<"\n\naType: "<<aType<<"\n";
        auto aTensorType = aType.dyn_cast<RankedTensorType>();
        newType = RankedTensorType::get(aTensorType.getShape(), aTensorType.getElementType(),
                    dotLayout);
        llvm::outs()<<"\n\nnewType: "<<newType<<"\n";
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
        ArrayRef<unsigned int> bOorder(bOrderVec);
        encoding = triton::gpu::GenericEncodingAttr::get(
            dotOp.getContext(), bThreadShape, bThreadstride, bElemPerThread, bElemStride, bSubGroupShape, bOrderVec, 1);
        dotLayout = triton::gpu::DotOperandEncodingAttr::get(context, 1, encoding, 0);

        auto B = dotOp.getB();
        auto bType = B.getType();
        llvm::outs()<<"\n\nbType: "<<bType<<"\n";
        auto bTensorType = bType.dyn_cast<RankedTensorType>();
        newType = RankedTensorType::get(bTensorType.getShape(), bTensorType.getElementType(),
                    dotLayout);
        llvm::outs()<<"\n\nnewType: "<<newType<<"\n";
        B.setType(newType);
      }
      // else if (auto StoreOp = dyn_cast<triton::StoreOp>(curr)) {
      //   const std::vector<unsigned int> threadShapeVec{1, 16, 8, 4};
      //   const std::vector<unsigned int> threadStrideVec{0, 1, 32, 64};
      //   const std::vector<unsigned int> elemPerThreadVec{8, 1, 4, 4};
      //   const std::vector<unsigned int> elemStrideVec{1, 0, 8, 16};
      //   const std::vector<unsigned int> subGroupShapeVec{4, 4, 0, 0};
      //   const std::vector<unsigned int> orderVec{1, 0, 1, 0};
      //   ArrayRef<unsigned int> threadShape(threadShapeVec);
      //   ArrayRef<unsigned int> threadstride(threadStrideVec);
      //   ArrayRef<unsigned int> elemPerThread(elemPerThreadVec);
      //   ArrayRef<unsigned int> elemStride(elemStrideVec);
      //   ArrayRef<unsigned int> subGroupShape(subGroupShapeVec);
      //   ArrayRef<unsigned int> order(orderVec);
      //   encoding = triton::gpu::GenericEncodingAttr::get(
      //       StoreOp.getContext(), threadShape, threadstride, elemPerThread, elemStride, subGroupShape, orderVec, 2);

      //   // auto ptr = StoreOp.getPtr();
      //   // auto type = ptr.getType();
      //   // tensorType = type.cast<RankedTensorType>();
      //   // auto layout = tensorType.getEncoding();
      //   // newType = RankedTensorType::get(tensorType.getShape(), tensorType.getElementType(),
      //   //             encoding);
      //   // ptr.setType(newType);

      //   auto value = StoreOp.getValue();
      //   auto type = value.getType();
      //   tensorType = type.cast<RankedTensorType>();
      //   auto layout = tensorType.getEncoding();
      //   newType = RankedTensorType::get(tensorType.getShape(), tensorType.getElementType(),
      //               encoding);
      //   value.setType(newType);

      //   llvm::outs()<<"\n\nnewType: "<<newType<<"\n";

      //   // auto mask = StoreOp.getMask();
      //   // type = mask.getType();
      //   // tensorType = type.cast<RankedTensorType>();
      //   // layout = tensorType.getEncoding();
      //   // newType = RankedTensorType::get(tensorType.getShape(), tensorType.getElementType(),
      //   //             encoding);
      //   // mask.setType(newType);
      // }else{
      // }
    });

    // llvm::outs()<<"\n\nmodule: \n";
    // op->print(llvm::outs());
    //todo
    return;

    op->walk([&](mlir::Operation *opInModule) {
      if (!(mlir::isa<func::FuncOp>(opInModule) 
                || mlir::isa<mlir::ModuleOp>(opInModule) ||  mlir::isa<mlir::scf::ForOp>(opInModule))){
        allOpsInModule.push_back(opInModule);
        opsSet.insert(opInModule);
      }
    });

    generateDenGraph(allOpsInModule, predecessorOps, successorOps);

    op->walk([&](Operation *curr) {
      for (auto operand : curr->getOperands()) {
        auto type = operand.getType();

        if (!type.isa<mlir::RankedTensorType>()){
          return;
        }

        auto tensorType = type.cast<RankedTensorType>();
        auto layout = tensorType.getEncoding();
        if (auto DotOp = dyn_cast<triton::DotOp>(curr)){
          //start from dotOp
          if (auto dotOpLayout = layout.dyn_cast<DotOperandEncodingAttr>()){
            opsSet.erase(curr);
            auto dotEncoding = dotOpLayout;
          
            auto genericLayout = dotOpLayout.getParent().cast<GenericEncodingAttr>();
            unsigned isLayoutUpdated = genericLayout.getIsLayoutUpdated();

            if (isLayoutUpdated == 2) {
              encoding = genericLayout;
              propagationLayout(context, curr, encoding, predecessorOps, opsSet);
            }
          } else if(auto genericLayout = layout.dyn_cast<GenericEncodingAttr>()){

          }
        }
        else 
        if (auto StoreOp = dyn_cast<triton::StoreOp>(curr)){
          opsSet.erase(curr);
          auto generic = layout.dyn_cast<GenericEncodingAttr>();
          unsigned isLayoutUpdated = generic.getIsLayoutUpdated();

          if (isLayoutUpdated == 2) {
            encoding = generic;
            propagationLayout(context, curr, encoding, predecessorOps, opsSet);

          }
        }
      }
    });

    bool changed = 1;
    while(changed){
      changed = 0;
      for (auto curr = opsSet.cbegin(); curr != opsSet.cend(); curr++)
      {
        if(auto ConstantOp = dyn_cast<arith::ConstantOp>(*curr)){
          auto curType = ConstantOp.getValue().getType();

          if (curType.isa<mlir::RankedTensorType>()){
            auto value = ConstantOp.getValue().dyn_cast<DenseElementsAttr>();
            auto curTensorType = curType.cast<RankedTensorType>();
            auto currlayout =  curTensorType.getEncoding();
            if(currlayout.dyn_cast<GenericEncodingAttr>().getIsLayoutUpdated() != 2){
              if(updateLayout(*curr, value, predecessorOps) || updateLayout(*curr, value, successorOps)){
                changed = 1;
              }
            }
          }
        }
        for (auto operand : (*curr)->getOperands()) {
          auto type = operand.getType();

          if (type.isa<mlir::RankedTensorType>()){
            auto tensorType = type.cast<RankedTensorType>();
            auto layout = tensorType.getEncoding();
            if(auto generic = layout.dyn_cast<GenericEncodingAttr>()){
              if(generic.getIsLayoutUpdated() != 2){
                if(updateLayout(*curr, &operand, predecessorOps) || updateLayout(*curr, &operand, successorOps)){
                  changed = 1;
                }
              }
            }
          }
        }
      }
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonGPULayoutPropagationPass() {
  return std::make_unique<TritonGPULayoutPropagationPass>();
}