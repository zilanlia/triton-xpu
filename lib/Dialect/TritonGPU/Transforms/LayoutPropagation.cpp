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

#include <queue>
#include <vector>
#include <set>
#include <map>

using namespace mlir;
using namespace mlir::scf;
using namespace mlir::triton;
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
      //dbgInfo("[createGraph]type", type);
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
        for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
          if (operand == forOp.getRegionIterArgs()[i]) {
            auto operandInForOp = forOp.getOperands()[i + 3];
            auto *forLoopParentOp = operandInForOp.getDefiningOp();
            preOpsGraph[op].insert(forLoopParentOp);
            sucOpsGraph[forLoopParentOp].insert(op);
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
    opsSet.erase(op);
    opsQueue.pop();

    if(auto ConstantOp = dyn_cast<arith::ConstantOp>(op)){
      auto type = ConstantOp.getValue().getType();
      //dbgInfo("[propagateLayout ConstantOp]type", type);
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
        if(isa<triton::PointerType>(type)){
          addr = type.cast<triton::PointerType>().getAddressSpace();
          type = type.cast<triton::PointerType>().getPointeeType();
          isPointerType = 1;
        }

        if(type.isa<mlir::RankedTensorType>()){
          auto tensorType = type.cast<RankedTensorType>();
          auto layout = tensorType.getEncoding();
          if(isa<DotOperandEncodingAttr>(layout)){
            // dbgInfo("[LayoutPropagation]op");
            // if(std::getenv("ENABLE_TRITON_DEBUG"))
            //   op->print(llvm::outs());
            // dbgInfo("[LayoutPropagation]layout", layout);
            auto opIdx = layout.cast<DotOperandEncodingAttr>().getOpIdx();
            auto newEncoding = DotOperandEncodingAttr::get(context, opIdx, encoding, 0);
            auto newType = RankedTensorType::get(tensorType.getShape(), 
                    tensorType.getElementType(), newEncoding);
            if(isPointerType){
              auto pointerType = triton::PointerType::get(newType, addr);
              operand.setType(pointerType);
            } else {
              operand.setType(newType);
            }
          }
          else if(auto genericLayout = layout.dyn_cast<GenericEncodingAttr>()){
            // if(genericLayout.getMmaFlag() == -1)
            {
              auto newType = RankedTensorType::get(tensorType.getShape(), 
                    tensorType.getElementType(), encoding);
              if(isPointerType){
                auto pointerType = triton::PointerType::get(newType, addr);
                operand.setType(pointerType);
              } else {
                operand.setType(newType);
              }
            }
          }else{

          }
        }
      }

      for(auto result : op->getResults()){
        Type type = result.getType();
        bool isPointerType = 0;
        int addr;
        // dbgInfo("[propagateLayout]type", type);
        if(isa<triton::PointerType>(type)){
          addr = type.cast<triton::PointerType>().getAddressSpace();
          type = type.cast<triton::PointerType>().getPointeeType();
          isPointerType = 1;
        }
        //dbgInfo("[propagateLayout]after convert pointType", type);
        if(type.isa<mlir::RankedTensorType>()){
          auto tensorType = type.cast<RankedTensorType>();
          auto layout = tensorType.getEncoding();
          if(isa<DotOperandEncodingAttr>(layout)){
            // dbgInfo("[LayoutPropagation]op");
            // if(std::getenv("ENABLE_TRITON_DEBUG"))
            //   op->print(llvm::outs());
            // dbgInfo("[LayoutPropagation]layout: ", layout);
            auto opIdx = layout.cast<DotOperandEncodingAttr>().getOpIdx();
            auto newEncoding = DotOperandEncodingAttr::get(context, opIdx, encoding, 0);
            auto newType = RankedTensorType::get(tensorType.getShape(), 
                    tensorType.getElementType(), newEncoding);
            if(isPointerType){
              auto pointerType = triton::PointerType::get(newType, addr);
              result.setType(pointerType);
            } else {
              result.setType(newType);
            }
          }
          else if(auto genericLayout = layout.dyn_cast<GenericEncodingAttr>()){
            // if(genericLayout.getMmaFlag() == -1)
            {
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
          else{

          }
        }
      }
    }

    for(auto preOp : preOpsGraph[op]){
      if(opsSet.count(preOp) == 1){
        opsQueue.push(preOp);
      }
    }
    for(auto sucOp : sucOpsGraph[op]){
      if(opsSet.count(sucOp) == 1){
        //Handling multiple dotOp
        if(auto dotOp = dyn_cast<triton::DotOp>(sucOp)){

        }else{
          opsQueue.push(sucOp);
        }
      }
    }
  }
}

void setDotOpLayout(MLIRContext *context, Operation *curr){
  if (auto dotOp = dyn_cast<triton::DotOp>(curr)) {
    //Need to be inferred from hardware info
    // dotOp
    // const std::vector<unsigned int> aThreadShapeVec{2, 8, 8, 4};
    // const std::vector<unsigned int> aThreadStrideVec{1, 2, 32, 0};
    // const std::vector<unsigned int> aElemPerThreadVec{4, 2, 4, 2};
    // const std::vector<unsigned int> aElemStrideVec{2, 1, 8, 16};
    // const std::vector<unsigned int> aSubGroupShapeVec{4, 4};
    // const std::vector<unsigned int> aOrderVec{1, 0, 1, 0};

    //attention
    const std::vector<unsigned int> aThreadShapeVec{2, 8, 8, 1};
    const std::vector<unsigned int> aThreadStrideVec{1, 2, 16, 0};
    const std::vector<unsigned int> aElemPerThreadVec{4, 2, 2, 4};
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
    dbgInfo("[DotOp]matA newType", newType);
    A.setType(newType);

    // dotOp
    // const std::vector<unsigned int> bThreadShapeVec{1, 16, 8, 4};
    // const std::vector<unsigned int> bThreadStrideVec{0, 1, 0, 64};
    // const std::vector<unsigned int> bElemPerThreadVec{16, 1, 2, 4};
    // const std::vector<unsigned int> bElemStrideVec{1, 0, 16, 16};
    // const std::vector<unsigned int> bSubGroupShapeVec{4, 4};
    // const std::vector<unsigned int> bOrderVec{1, 0, 1, 0};

    //fused attention
    const std::vector<unsigned int> bThreadShapeVec{1, 16, 8, 1};
    const std::vector<unsigned int> bThreadStrideVec{0, 1, 0, 64};
    const std::vector<unsigned int> bElemPerThreadVec{16, 1, 4, 4};
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
    dbgInfo("[DotOp]matB newType", newType);
    B.setType(newType);

    //dot
    // const std::vector<unsigned int> cThreadShapeVec{1, 16, 8, 4};
    // const std::vector<unsigned int> cThreadStrideVec{0, 1, 32, 64};
    // const std::vector<unsigned int> cElemPerThreadVec{8, 1, 4, 4};
    // const std::vector<unsigned int> cElemStrideVec{1, 0, 8, 16};
    // const std::vector<unsigned int> cSubGroupShapeVec{4, 4};
    // const std::vector<unsigned int> cOrderVec{1, 0, 1, 0};

    //attention
    const std::vector<unsigned int> cThreadShapeVec{1, 16, 8, 1};
    const std::vector<unsigned int> cThreadStrideVec{0, 1, 16, 64};
    const std::vector<unsigned int> cElemPerThreadVec{8, 1, 2, 4};
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
    dbgInfo("[DotOp]matC/matD newType", newType);
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
    opsSetTy opsSet1;
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
        opsSet1.insert(op);
      }
    });

    // Find the parent and child operations of each op
    createGraph(opsVector, preOpsGraph, sucOpsGraph);

    if(0 && std::getenv("ENABLE_TRITON_DEBUG")){
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
        dbgInfo("[LayoutPropagation] dotOp in module", dotOp);
        dbgInfo("[LayoutPropagation] opsSet.count(dotOp)", opsSet.count(dotOp));
        if(opsSet.count(dotOp) == 1){
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

          dbgInfo("[LayoutPropagation]aEncoding" , aEncoding);
          dbgInfo("[LayoutPropagation]bEncoding" , bEncoding);
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

    op->walk([&](Operation *curr) {
      if (auto makeTensorPtrOp = dyn_cast<MakeTensorPtrOp>(curr)){
        auto order = makeTensorPtrOp.getOrder();
        dbgInfo("[LayoutPropagation]order[0]: " , order[0]);
        if(order[0] == 0){
          auto ptr = makeTensorPtrOp.getODSResults(0)[0];
          dbgInfo("[LayoutPropagation]ptr: " , ptr);
          auto encoding = ptr.getType().cast<triton::PointerType>()
                             .getPointeeType().dyn_cast<RankedTensorType>()
                             .getEncoding();
          dbgInfo("[LayoutPropagation]encoding: " , encoding);
          auto layout = encoding.cast<GenericEncodingAttr>();
          const std::vector<unsigned int> newOrderVec{0, 1, 0, 1};
          ArrayRef<unsigned int> newOrder(newOrderVec);

          dbgInfo("[LayoutPropagation]newOrder.size(): " , newOrder.size());
          auto newEncoding = layout.updateOrder(newOrder);
          dbgInfo("[LayoutPropagation]newEncoding" , newEncoding);


          opsQueueTy queue;
          queue.push(makeTensorPtrOp);

          propagateLayout(context, queue, newEncoding, preOpsGraph, sucOpsGraph, opsSet1);
        }
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

    // dbgInfo("[After propagateLayout]tritonGPU IR");
    // op->print(llvm::outs());

    return;
  }
};

std::unique_ptr<Pass> mlir::createTritonGPULayoutPropagationPass() {
  return std::make_unique<TritonGPULayoutPropagationPass>();
}