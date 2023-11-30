#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
#include <mlir/Transforms/OneToNTypeConversion.h>

#include "TritonGPUToXeGPU.h"
#include "triton/Dialect/XeGPU/IR/XeGPUOps.h"
#include "TritonGPUToXeGPUBase.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::xegpu;

using ::mlir::triton::gpu::GenericEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;

static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

template <class Op, class dstOp = Op> 
class GenericOpPattern : public ConvertTritonGPUToXeGPUPattern<Op> {
public:
  using ConvertTritonGPUToXeGPUPattern<Op>::ConvertTritonGPUToXeGPUPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("[GenericOpPattern]");
    dbgInfo("[GenericOpPattern] op", op);
    Location loc = op.getLoc();
    Type type = op.getType();
    auto context = op.getContext();
    
    llvm::SmallVector<mlir::Type> resultTypes;
    auto result = this->getTypeConverter()->convertType(op.getType(), resultTypes);
    auto retType = resultTypes[0];
    int srcNum = 1;
    auto operands = adaptor.getOperands();
    Value operand0 = operands[0];
    ValueRange src0(operand0);

    // dbgInfo("[GenericOpPattern]retType", retType);
    // dbgInfo("[GenericOpPattern]operand size", operands.size());
    // dbgInfo("[GenericOpPattern]src0[0]", src0[0]);
    if(auto *parentOp = src0[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        auto inputs =  (&castOp)->getInputs();
        src0 = ValueRange(inputs);
      }
    }

    ValueRange src1;

    if(operands.size() > 1){
      Value operand1 = operands[1];
      src1 = ValueRange(operand1);
      srcNum = 2;
    }

    if(src1.size() != 0){
      if(auto *parentOp = src1[0].getDefiningOp()){
        if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
          auto inputs =  (&castOp)->getInputs();
          src1 = ValueRange(inputs);
        }
      }
    }

    if(resultTypes.size() > 1){
      SmallVector<Value> retVec;
      Value ret;
      auto newType = resultTypes[0];
      auto size = resultTypes.size();

      for(int i = 0; i < size; i++){
        SmallVector<Value> args;
        Value operand0 = src0[i];
        Value operand1;

        if(srcNum==2){
          operand1 = src1[i];
        }

        /// process mathOp which srcType more than 1d
        // auto srcType = src0[i].getType();
        // if(isa<VectorType>(srcType)){
        //   auto shape = srcType.cast<VectorType>().getShape();
        //   auto srcElemType = srcType.cast<VectorType>().getElementType();
        //   auto retElemType = newType.cast<VectorType>().getElementType();
        //   int nElem = 1;
        //   for(auto s : shape) 
        //     nElem *= s;
        //   auto castSrcType = VectorType::get(nElem, srcElemType);
        //   operand0 = rewriter.create<vector::ShapeCastOp>(loc, castSrcType, operand0);
        //   if(srcNum==2){
        //     operand1 = rewriter.create<vector::ShapeCastOp>(loc, castSrcType, operand1);
        //   }
        //   newType = VectorType::get(nElem, retElemType);
        // }

        args.push_back(operand0);
        if(srcNum==2){
          args.push_back(operand1);
        }
        ret = rewriter.create<dstOp>(loc, newType, args);

        // if(isa<VectorType>(srcType)){
        //   newType = resultTypes[0];
        //   ret = rewriter.create<vector::ShapeCastOp>(loc, newType, ret);
        // }
        retVec.push_back(ret);
      }

      ValueRange newValueRange(retVec);
      auto resultTys = op->getResultTypes();
      auto castOp = urcast(resultTys, newValueRange)->getResults();
      newValueRange = ValueRange(castOp);
      rewriter.replaceOp(op, castOp);
    } else {
      addNamedAttrs(
          rewriter.replaceOpWithNewOp<dstOp>(op, retType, adaptor.getOperands()),
          adaptor.getAttributes());
    }
    dbgInfo("[GenericOpPattern] End");
    return success();
  }
};

class ArithConstantOpPattern : public ConvertTritonGPUToXeGPUPattern<arith::ConstantOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<arith::ConstantOp>::ConvertTritonGPUToXeGPUPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto context = op.getContext();
    Type type = op.getType();
    Attribute layout = type.cast<RankedTensorType>().getEncoding();

    llvm::SmallVector<mlir::Type> resultTypes;
    auto result = this->getTypeConverter()->convertType(op.getType(), resultTypes);
    auto retType = resultTypes[0];

    dbgInfo("[ArithConstantOpPattern]retType", retType);

    auto value = adaptor.getValue();
    auto denseValue = value.dyn_cast<DenseElementsAttr>();

    SmallVector<Value> constVals;
    if(layout.isa<GenericEncodingAttr>()){
      if(resultTypes.size() > 1){
        auto newType = resultTypes[0];
        auto newValue = denseValue.reshape(newType.cast<ShapedType>());
        auto size = resultTypes.size();
        for(int i=0;i<size;i++){
          Value constVal = rewriter.create<spirv::ConstantOp>(loc, newType, newValue);
          constVals.push_back(constVal);
        }

        ValueRange newValueRange(constVals);
        auto resultTys = op->getResultTypes();
        newValueRange = urcast(resultTys, newValueRange)->getResults();
        rewriter.replaceOp(op, newValueRange);
      } else {
        auto newValue = denseValue.reshape(retType.cast<ShapedType>());
        addNamedAttrs(
          rewriter.replaceOpWithNewOp<spirv::ConstantOp>(op, retType, newValue),
          adaptor.getAttributes());
      }
    }

    return success();
  }
};

class ArithTruncFOpPattern : public ConvertTritonGPUToXeGPUPattern<arith::TruncFOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<arith::TruncFOp>::ConvertTritonGPUToXeGPUPattern;

  LogicalResult
  matchAndRewrite(arith::TruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("[ArithTruncFOpPattern]");
    Location loc = op.getLoc();
    auto context = op.getContext();
    Type type = op.getType();
    Attribute layout = type.cast<RankedTensorType>().getEncoding();

    llvm::SmallVector<mlir::Type> resultTypes;
    auto result = this->getTypeConverter()->convertType(op.getType(), resultTypes);
    auto retType = resultTypes[0];

    Type elemType;
    Type newRetType;
    if(isa<VectorType>(retType)){
      elemType = retType.cast<VectorType>().getElementType();
      auto shape = retType.cast<VectorType>().getShape();
      if(elemType == bf16Type){
        elemType = i16Type;
      }
      newRetType = VectorType::get(shape, elemType);
    }

    dbgInfo("[ArithTruncFOpPattern]retType", retType);

    auto src = adaptor.getIn();
    ValueRange in(src);

    if(auto *parentOp = in[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        in = (&castOp)->getInputs();
      }
    }

    SmallVector<Value> truncFVals;
    if(layout.isa<GenericEncodingAttr>()){
      if(resultTypes.size() > 1){
        auto size = resultTypes.size();
        for(int i = 0; i < size;i++){
          Value truncFVal = rewriter.create<arith::TruncFOp>(loc, retType, in[i]);
          if(elemType == i16Type){
            truncFVal = urcast(newRetType, truncFVal)->getResults()[0];
          }
          truncFVals.push_back(truncFVal);
        }

        ValueRange newValueRange(truncFVals);
        auto resultTys = op->getResultTypes();
        auto cast = urcast(resultTys, newValueRange)->getResults();
        newValueRange = ValueRange(cast);
        rewriter.replaceOp(op, newValueRange);

      } else {
        Value truncFVal = rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, retType, in[0]);
        if(elemType == i16Type){
          truncFVal = urcast(newRetType, truncFVal)->getResults()[0];
        }
        rewriter.replaceOp(op, truncFVal);
        // addNamedAttrs(
        //   rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, retType, in[0]),
        //   adaptor.getAttributes());
      }
    }

    return success();
  }
};

class ReturnOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<triton::ReturnOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<triton::ReturnOp>::ConvertTritonGPUToXeGPUPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    rewriter.create<mlir::gpu::ReturnOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }
};

class LoadOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<triton::LoadOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<triton::LoadOp>::ConvertTritonGPUToXeGPUPattern;
  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("[LoadOpToXeGPUPattern]");
    auto loc = op->getLoc();
    auto context = rewriter.getContext();

    RankedTensorType tensorType = op.getResult().getType().cast<RankedTensorType>();
    auto encoding = tensorType.getEncoding();
    auto shape = tensorType.getShape();
    auto layout = encoding.cast<GenericEncodingAttr>();
    auto mmaFlag = layout.getMmaFlag();
    auto threadShape = layout.getThreadShape();
    auto elemShape = layout.getElemPerThread();

    auto xegpuPtr = adaptor.getPtr();
    ValueRange desc(xegpuPtr);
    if(auto *parentOp = desc[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        desc = (&castOp)->getInputs();
      }
    }

    int nElem = desc.size();
    dbgInfo("nElem", nElem);

    auto descType = desc[0].getType();
    Value desc0 = desc[0];

    auto value = op.getResult();
    llvm::SmallVector<mlir::Type> resultTypes;
    auto result = this->getTypeConverter()->convertType(value.getType(), resultTypes);
    Type loadType = resultTypes[0];

    auto newShape = loadType.dyn_cast<VectorType>().getShape();
    int elemNum = newShape[0];

    bool isTrans = 0;
    Value ptr = op.getPtr();
    while(auto *parentOp = ptr.getDefiningOp()){
      if(auto advanceOp = dyn_cast<AdvanceOp>(parentOp)){
        ptr = advanceOp.getPtr();
      }else{ 
        if(auto makeTensorOp = dyn_cast<MakeTensorPtrOp>(parentOp)){
          auto order = makeTensorOp.getOrder();
          isTrans = order[0] == 0;
        }
        break;
      }
    }

    ptr = op.getPtr();
    if(auto arg = dyn_cast<BlockArgument>(ptr)){
      auto ownerOp = arg.getOwner()->getParentOp();
      auto forOp = cast<scf::ForOp>(ownerOp);
      Value init = forOp.getInitArgs()[arg.getArgNumber() - 1];
      while (auto op = init.getDefiningOp()) {
        if(auto advanceOp = dyn_cast<AdvanceOp>(op)){
          init = advanceOp.getPtr();
        } else{
          if (auto makeTensorOp = dyn_cast<MakeTensorPtrOp>(op)) {
            auto order = makeTensorOp.getOrder();
            isTrans = order[0] == 0;
          }
          break;
        }
      }
    }

    dbgInfo("[LoadOpToXeGPUPattern] isTrans: ", isTrans);

    if(mmaFlag != -1){
      auto vectorType = loadType.dyn_cast<VectorType>();
      Type elemType = vectorType.getElementType();

      if(elemType == bf16Type){
        elemType = i16Type;
      }

      int dim0 = threadShape[0] * elemShape[0];
      int dim1 = threadShape[1] * elemShape[1];
      if(mmaFlag == 0){
        int loadM = shape[0] / threadShape[2];
        loadM = std::min(std::max(loadM, dim0), 32);

        loadType = VectorType::get(ArrayRef<int64_t>{loadM, dim1 / 2, 2}, elemType);
      } else if(mmaFlag == 1) {
        int loadK = shape[0];
        loadK = std::min(std::max(loadK, dim0), 32);
        //todo
        loadK = dim0;
        if(isTrans){
          loadType = VectorType::get(ArrayRef<int64_t>{dim1, dim0, 1}, elemType);
        }else{
          loadType = VectorType::get(ArrayRef<int64_t>{loadK / 2, dim1, 2}, elemType);
        }
      } else if(mmaFlag == 3){
        loadType = VectorType::get(ArrayRef<int64_t>{16, 16, 1}, elemType);
      } else{
      }
    }

    auto L1_hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto L2_hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto L3_hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);

    if(resultTypes.size() > 1){
      SmallVector<Value> newValues;
      for(int i = 0; i <nElem; i++){
        Value ret;
        if(mmaFlag == 0){ //load matrix A for gemm
          auto vnni = IntegerAttr::get(i32_ty, 1);
          ret = rewriter.create<xegpu::LoadNDOp>(loc, loadType, desc[i], vnni, DenseI64ArrayAttr{}, L1_hint, L2_hint, L3_hint);
        } else if(mmaFlag == 1){ //load matrix B for gemm
          if(isTrans){
            ::mlir::DenseI64ArrayAttr trans = ::mlir::DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{1, 0});
            ret = rewriter.create<xegpu::LoadNDOp>(loc, loadType, desc[i], IntegerAttr{}, trans, L1_hint, L2_hint, L3_hint);
          }else{
            ::mlir::IntegerAttr vnni = IntegerAttr::get(i32_ty, 0);
            ret = rewriter.create<xegpu::LoadNDOp>(loc, loadType, desc[i], vnni, DenseI64ArrayAttr{}, L1_hint, L2_hint, L3_hint);
          }
          dbgInfo("[LoadOpToXeGPUPattern] LoadNDOp: ", ret);
        } else{

        }

        newValues.push_back(ret);
      }

      ValueRange newValueRange(newValues);
      auto resultTys = op->getResultTypes();
      auto cast = urcast(resultTys, newValueRange)->getResults();
      rewriter.replaceOp(op, cast);
    } else {
      if(mmaFlag == 3){
        rewriter.create<xegpu::PrefetchNDOp>(loc, desc[0], L1_hint, L2_hint, L3_hint);
        rewriter.eraseOp(op);
      }
      else{
        Value mask = adaptor.getMask();
        if(!mask) {
          auto maskType = mlir::VectorType::get(elemNum, i32Type);
          DenseElementsAttr constData = DenseElementsAttr::get(maskType, ArrayRef<int>(std::vector<int>(elemNum, 1)));
          mask = rewriter.create<spirv::ConstantOp>(loc, maskType, constData);
        }
        dbgInfo("[LoadOpToXeGPUPattern] Load1D");
        dbgInfo("[LoadOpToXeGPUPattern] desc0", desc0);
        Value ret = rewriter.create<xegpu::LoadGatherOp>(loc, loadType, 
              desc0, mask, IntegerAttr{}, DenseI64ArrayAttr{}, L1_hint, L2_hint, L3_hint);
        rewriter.replaceOp(op, ret);
      }
    }

    return success();
  }
};

class StoreOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<triton::StoreOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<triton::StoreOp>::ConvertTritonGPUToXeGPUPattern;
  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("[StoreOpToXeGPUPattern]");
    auto loc = op->getLoc();
    auto context = rewriter.getContext();

    auto ptrRange = adaptor.getPtr();
    dbgInfo("[StoreOpToXeGPUPattern]ptrRange: ", ptrRange);
    ValueRange desc(ptrRange);
    if(auto *parentOp = desc[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        desc = (&castOp)->getInputs();
      }
    }

    auto valueRange(adaptor.getValue());
    ValueRange value(valueRange);
    if(auto *parentOp = value[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        value = (&castOp)->getInputs();
      }
    }

    auto newType = value[0].getType();
    auto shape = newType.dyn_cast<VectorType>().getShape();
    int elemNum = shape[0];

    auto L1_hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);
    auto L2_hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);
    auto L3_hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);

    if(shape.size() > 1){
      for(int i = 0; i < desc.size(); i++){
        rewriter.create<xegpu::StoreNDOp>(loc, desc[i], value[i], L1_hint, L2_hint, L3_hint);
      }
      rewriter.eraseOp(op);
    } else {
      Value mask; 
      if(Value m = adaptor.getMask()){
        mask = m;
      } else {
        auto maskType = mlir::VectorType::get(shape[0], i1Type);
        DenseElementsAttr constData = DenseElementsAttr::get(maskType, ArrayRef<int>(std::vector<int>(shape[0], 1)));
        mask = rewriter.create<spirv::ConstantOp>(loc, maskType, constData);
      }

      Value ret = rewriter.create<xegpu::StoreScatterOp>(loc, value[0], desc[0], mask, L1_hint, L2_hint, L3_hint).getODSResults(0)[0];
      rewriter.replaceOp(op, ret);
    }
    // auto module = op.getOperation()->getParentOfType<ModuleOp>();
    // dbgInfo("[Module] after StoreOp")
    // module.print(llvm::outs());
    return success();
  }
};

class MakeRangeOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<triton::MakeRangeOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<triton::MakeRangeOp>::ConvertTritonGPUToXeGPUPattern;
  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("[MakeRangeOpToXeGPUPattern]");
    auto loc = op->getLoc();
    auto context = rewriter.getContext();

    RankedTensorType tensorType = op.getResult().getType().cast<RankedTensorType>();
    auto encoding = tensorType.getEncoding();
    auto shape = tensorType.getShape();
    auto layout = encoding.cast<GenericEncodingAttr>();
    auto mmaFlag = layout.getMmaFlag();
    auto threadShape = layout.getThreadShape();

    Value subgroubId = rewriter.create<::mlir::gpu::SubgroupIdOp>(loc, rewriter.getIndexType());
    Value sgId = urcast(i64_ty, subgroubId).getResult(0);
    sgId = rewriter.create<spirv::UConvertOp>(loc, i32_ty, sgId);

    llvm::SmallVector<mlir::Type> resultTypes;
    Type retType = op.getResult().getType();
    auto result = this->getTypeConverter()->convertType(retType, resultTypes);
    auto newShape = resultTypes[0].cast<VectorType>().getShape();
    int nElem = newShape[0];

    int sgSize;
    if(threadShape.size() <= 2)
      sgSize = threadShape[1];
    else
      sgSize = threadShape[2] * threadShape[3];
    sgId = urem(sgId, i32_val(sgSize));

    auto sgAddr = rewriter.create<arith::MulIOp>(loc, i32_ty, sgId, i32_val(nElem));
    auto type = mlir::VectorType::get(nElem, i32_ty);

    //avoid spirv.CompositeConstruct
    Value sgAddrs = rewriter.create<spirv::UndefOp>(loc, type);
    auto idx0 = i32_val(0);
    sgAddrs =
        rewriter.create<spirv::VectorInsertDynamicOp>(loc, sgAddrs, sgAddr, idx0);
    SmallVector<int32_t> indices(nElem, 0);
    sgAddrs = rewriter.create<spirv::VectorShuffleOp>(
          loc, type, sgAddrs, sgAddrs, rewriter.getI32ArrayAttr(indices));

    dbgInfo("[MakeRangeOpToXeGPUPattern]sgAddrs: ", sgAddrs);
    std::vector<int> values(nElem, 0);
    for(int i = 0; i < nElem; i++){
      values[i] = i;
    }
    ArrayRef<int> arrayRef(values);
    DenseElementsAttr constantData = DenseElementsAttr::get(type, arrayRef);
    Value offsets = rewriter.create<arith::ConstantOp>(loc, type, constantData);
    Value sgOffsets = rewriter.create<arith::AddIOp>(loc, type, sgAddrs, offsets);

    rewriter.replaceOp(op, sgOffsets);
    return success();
  }
};

class AddPtrOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<triton::AddPtrOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<triton::AddPtrOp>::ConvertTritonGPUToXeGPUPattern;
  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("[AddPtrOpToXeGPUPattern]");
    auto loc = op->getLoc();
    auto context = rewriter.getContext();

    Value curDesc;
    ::llvm::ArrayRef<int64_t> shape;
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();
    dbgInfo("[AddPtrOpToXeGPUPattern]ptr: ", ptr);
    dbgInfo("[AddPtrOpToXeGPUPattern]offset: ", offset);

    Type offsetType = offset.getType();
    if(isa<IntegerType>(offsetType)){
      return success();
    }

    ptr = op.getPtr();
    auto ptrType = ptr.getType();
    bool isPtr = ptrType.isa<mlir::MemRefType>();

    llvm::SmallVector<mlir::Type> resultTypes;
    auto value = op.getResult();
    auto result = this->getTypeConverter()->convertType(offsetType, resultTypes);
    auto newType = resultTypes[0];
    //dbgInfo("[SplatOpToXeGPUPattern]resultTypes[0]: ", resultTypes[0]);
    auto size = resultTypes[0].cast<VectorType>().getShape()[0];

    while(auto curOp = ptr.getDefiningOp()){
      if(auto createDescOp = dyn_cast_or_null<CreateDescOp>(curOp)){
        curDesc = ptr;
        Value ret = rewriter.create<xegpu::UpdateOffsetOp>(loc, ptr.getType(), curDesc, offset);
        rewriter.replaceOp(op, ret);
        return success();
      } else if(auto addPtrOp = dyn_cast_or_null<triton::AddPtrOp>(curOp)){
        Value originalOffset = addPtrOp.getOffset();
        ptr = addPtrOp.getPtr();
        if(auto castOp = ptr.getDefiningOp()){
          ptr = castOp->getOperand(0);
        }
        ptrType = ptr.getType();

        Value originalOffsetVec = rewriter.create<spirv::UndefOp>(loc, offsetType);
        auto idx0 = i32_val(0);
        originalOffsetVec = rewriter.create<spirv::VectorInsertDynamicOp>(loc, 
                                              originalOffsetVec, originalOffset, idx0);
        SmallVector<int32_t> indices(size, 0);
        originalOffsetVec = rewriter.create<spirv::VectorShuffleOp>(
              loc, offsetType, originalOffsetVec, originalOffsetVec, rewriter.getI32ArrayAttr(indices));

        //Value originalOffsetVec = rewriter.create<vector::SplatOp>(loc, offsetType, originalOffset);
        offset = rewriter.create<arith::AddIOp>(loc, offsetType, offset, originalOffsetVec);

        break;
      }
      ptr = curOp->getOperand(0);
      ptrType = ptr.getType();
    }

    Type elemType = op.getPtr().getType().cast<RankedTensorType>().getElementType();
    if(isa<triton::PointerType>(elemType)){
      elemType = elemType.cast<triton::PointerType>().getPointeeType();
    }
    dbgInfo("[AddPtrOpToXeGPUPattern]elemType", elemType);

    //get shape
    if(auto type = offset.getType().cast<VectorType>()){
      shape = type.getShape();
    }

    dbgInfo("[AddPtrOpToXeGPUPattern]shape.size()", shape.size());
    dbgInfo("[AddPtrOpToXeGPUPattern]shape[0]", shape[0]);
    auto tensorDescType = ::mlir::triton::xegpu::TensorDescType::get(context, shape, elemType, ScatteredAttr::get(context));
    auto memory_scope = MemoryScopeAttr::get(context, triton::xegpu::MemoryScope::GLOBAL);
    auto vert_size = IntegerAttr::get(i32_ty, 1);

    Value ret = rewriter.create<xegpu::CreateDescOp>(loc, tensorDescType, ptr, offset, memory_scope, vert_size);
    dbgInfo("[AddPtrOpToXeGPUPattern]ret", ret);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

class CmpIOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<triton::gpu::CmpIOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<triton::gpu::CmpIOp>::ConvertTritonGPUToXeGPUPattern;
  LogicalResult
  matchAndRewrite(triton::gpu::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("[CmpIOpToXeGPUPattern]");
    auto loc = op->getLoc();
    auto context = rewriter.getContext();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    Type oprandType = this->getTypeConverter()->convertType(lhs.getType());

    Value ret =  rewriter.create<arith::CmpIOp>(loc, op.getPredicate(), lhs, rhs);
    rewriter.replaceOp(op, ret);

    return success();
  }
};

class SplatOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<triton::SplatOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<triton::SplatOp>::ConvertTritonGPUToXeGPUPattern;
  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("[SplatOpToXeGPUPattern]");
    auto loc = op->getLoc();
    auto context = rewriter.getContext();
    auto src = adaptor.getSrc();

    dbgInfo("[SplatOpToXeGPUPattern]src", src);
    dbgInfo("[SplatOpToXeGPUPattern]src.getType()", src.getType());
    if(src.getType().isa<mlir::MemRefType>()){
      auto memref = op.getSrc();
      dbgInfo("[SplatOpToXeGPUPattern]memref", memref);
      rewriter.replaceOp(op, memref);
      return success();
    }

    auto tensorTy = op.getResult().getType().cast<RankedTensorType>();
    auto tensorShape = tensorTy.getShape();
    auto elemType = tensorTy.getElementType();
    auto layout = tensorTy.getEncoding().dyn_cast<GenericEncodingAttr>();

    llvm::SmallVector<mlir::Type> resultTypes;

    auto value = op.getResult();
    auto result = this->getTypeConverter()->convertType(value.getType(), resultTypes);
    auto shape = resultTypes[0].cast<VectorType>().getShape();
    int nElem = 1;
    for(auto s : shape)
      nElem *= s;

    Type newType = resultTypes[0];
    Value ret = rewriter.create<spirv::UndefOp>(loc, newType);
    auto idx0 = i32_val(0);
    ret = rewriter.create<spirv::VectorInsertDynamicOp>(loc, ret, src, idx0);
    SmallVector<int32_t> indices(nElem, 0);

    if(tensorShape.size() == 1 && resultTypes.size() == 1){
      // dbgInfo("[SplatOpToXeGPUPattern]indices.size()", indices.size());
      ret = rewriter.create<spirv::VectorShuffleOp>(
            loc, newType, ret, ret, rewriter.getI32ArrayAttr(indices));
      // dbgInfo("[SplatOpToXeGPUPattern]1d ret", ret);
      rewriter.replaceOp(op, ret);
    } else{
      SmallVector<Value> newValues;

      for(int i = 0;i < resultTypes.size(); i++){
        Value shuffledRet = rewriter.create<spirv::VectorShuffleOp>(
            loc, newType, ret, ret, rewriter.getI32ArrayAttr(indices));
        newValues.push_back(shuffledRet);
      }
      ValueRange newValueRange(newValues);
      auto resultTys = op->getResultTypes();
      auto cast = urcast(resultTys, newValueRange)->getResults();
      newValueRange = ValueRange(cast);
      rewriter.replaceOp(op, newValueRange);
    }

    return success();
  }
};

class GetProgramIdOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<triton::GetProgramIdOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<triton::GetProgramIdOp>::ConvertTritonGPUToXeGPUPattern;
  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("GetProgramIdOpToSPIRVConversion");
    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);

    Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(
            loc, rewriter.getIndexType(), dims[op.getAxisAsInt()]);
    // Value blockId_idx = rewriter.create<::mlir::arith::TruncIOp>(
    //         loc, i32_ty, blockId);
    Value cast = urcast(i64_ty, blockId).getResult(0);

    rewriter.replaceOpWithNewOp<spirv::UConvertOp>(
            op, i32_ty, cast);
    // rewriter.replaceOp(op, cast);

    return success();
  }
private:
  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
};

class ConvertLayoutOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<ConvertLayoutOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<ConvertLayoutOp>::ConvertTritonGPUToXeGPUPattern;

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ValueRange src{adaptor.getSrc()};

    if(auto *parentOp = src[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        src = (&castOp)->getInputs();
      }
    }

    auto resultTys = op->getResultTypes();
    auto castOp = urcast(resultTys, src);
    ValueRange newValueRange{castOp->getResults()};
    rewriter.replaceOp(op, newValueRange);

    return success();
  }
};

Value getI32Value(ConversionPatternRewriter &rewriter, Value src){
  Location loc = src.getDefiningOp()->getLoc();
  if(src.getDefiningOp() && 
    isa<arith::ExtSIOp>(src.getDefiningOp())){
      return src.getDefiningOp()->getOperand(0);
  } else if(src.getType()==i64_ty){
    return rewriter.create<spirv::UConvertOp>(loc, i32_ty, src);
  } else {
    return src;
  }
}

SmallVector<mlir::OpFoldResult> getI32SmallVector(ConversionPatternRewriter &rewriter, 
                                                 ::mlir::Operation::operand_range src){
  SmallVector<mlir::OpFoldResult> I32ValueRange;
  for(auto v : src){
    I32ValueRange.push_back(getI32Value(rewriter, v));
  }
  return I32ValueRange;
}

SmallVector<Value> getI32Vevtor(ConversionPatternRewriter &rewriter, 
                                  ::mlir::Operation::operand_range &src){
  SmallVector<Value> I32ValueRange;
  for(auto v : src){
    auto I32Value = getI32Value(rewriter, v);
    I32ValueRange.push_back(I32Value);
  }
  return I32ValueRange;
}

class MakeTensorPtrOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<MakeTensorPtrOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<MakeTensorPtrOp>::ConvertTritonGPUToXeGPUPattern;

  LogicalResult
  matchAndRewrite(MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("[MakeTensorPtrOpToXeGPUPattern]");
    Location loc = op->getLoc();
    auto context = rewriter.getContext();

    auto blockPtr = adaptor.getBase();
    auto order = adaptor.getOrder();
    auto ptrType = op.getResult().getType().cast<triton::PointerType>();
    auto tensorType = ptrType.getPointeeType().cast<RankedTensorType>();
    auto genericLayout = tensorType.getEncoding().cast<GenericEncodingAttr>();
    auto mmaFlag = genericLayout.getMmaFlag();
    auto threadShape = genericLayout.getThreadShape();
    auto elemShape = genericLayout.getElemPerThread();
    auto elemStride = genericLayout.getElemStride();
    auto elemType = tensorType.getElementType();
    auto blockShape = tensorType.getShape();

    auto tensorShape = op.getShape();
    auto tensorStride = op.getStrides();
    auto tensorOffsets = op.getOffsets();
    auto NdOrder = op.getOrder();

    bool isTrans = order[0] == 0;

    bool isLoad = 1;
    Value desc = op.getODSResults(0)[0];
    for(Operation *user : desc.getUsers()){
      if(auto storeOp = llvm::dyn_cast<StoreOp>(user)){
        isLoad = 0;
      } 
    }

    if(elemType == bf16Type){
      elemType = i16Type;
    }

    Optional<spirv::StorageClass> storageClass = spirv::StorageClass::CrossWorkgroup;
    auto spirvPtrType = spirv::PointerType::get(elemType, *storageClass);
    Value addr;

    auto base = op.getBase();
    if(auto *parentOp = base.getDefiningOp()){
      if(auto addPtrOp = dyn_cast<AddPtrOp>(parentOp)){
        auto ptr = addPtrOp.getPtr();
        auto offset = addPtrOp.getOffset();
        auto offsetType = offset.getType();

        addr = urcast(spirvPtrType, ptr).getResult(0);
        addr = rewriter.create<spirv::ConvertPtrToUOp>(loc, i64_ty, addr);

        int byteWidth = elemType.getIntOrFloatBitWidth() / 8;

        if(offsetType != i64_ty)
          offset = zext(i64_ty, mul(offset, i32_val(byteWidth)));
        else
          offset = mul(offset, zext(i64_ty, i32_val(byteWidth)));

        addr = add(addr, offset);
      }
    } else{
      addr = urcast(spirvPtrType, blockPtr).getResult(0);
      addr = rewriter.create<spirv::ConvertPtrToUOp>(loc, i64_ty ,addr);
    }

    SmallVector<mlir::OpFoldResult> NdOffset = getI32SmallVector(rewriter, tensorOffsets);

    //wa: error when return the rvalue to ValueRange;
    SmallVector<Value> NdShape = getI32Vevtor(rewriter, tensorShape);
    SmallVector<Value> NdStride = getI32Vevtor(rewriter, tensorStride);

    if(isTrans){
      Value tmp = NdShape[0];
      NdShape[0] = NdShape[1];
      NdShape[1] = tmp;
    }

    //for PVC
    const int blockM = 8;
    const int blockK = 16;
    const int blockN = 16;
    Value offsetM = i32_val(blockM);
    Value offsetK = i32_val(blockK);
    Value offsetN = i32_val(blockN);

    Type resultTy;

    SmallVector<Value> newValues;
    xegpu::CreateNdDescOp descOp;
    TensorDescType tensorDescType;

    Value subgroubId = rewriter.create<::mlir::gpu::SubgroupIdOp>(loc, rewriter.getIndexType());
    Value sgId = urcast(i64_ty, subgroubId).getResult(0);
    sgId = rewriter.create<spirv::UConvertOp>(loc, i32_ty, sgId);

    int sgSize = threadShape[2] * threadShape[3];
    sgId = urem(sgId, i32_val(sgSize));

    Value gidM;
    Value gidN;
    if(threadShape[3] > 1){
      gidM = udiv(sgId, i32_val(threadShape[3]));
      gidN = urem(sgId, i32_val(threadShape[3]));
    } else{
      gidM = sgId;
      gidN = i32_val(0);
    }

    int dim0 = threadShape[0] * elemShape[0];
    int dim1 = threadShape[1] * elemShape[1];

    SmallVector<mlir::OpFoldResult> curNdOffset = NdOffset;
    if(isLoad){
      if(mmaFlag == 0){ //load matrix A
        int tensorM = elemShape[2] * elemStride[2];
        int tensorK = elemShape[3] * elemStride[3];

        int blockM = blockShape[0] / threadShape[2];
        blockM = std::min(std::max(blockM, dim0), 32);
        int blockK = dim1;
        Value offsetM = i32_val(blockM);
        Value offsetK = i32_val(blockK);

        tensorDescType = TensorDescType::get(context, ::llvm::ArrayRef<int64_t>{blockM, blockK}, elemType, 
                                                      MemoryScopeAttr::get(context, MemoryScope::GLOBAL));
        int numRepM = tensorM / blockM;
        int numRepK = tensorK / blockK;

        Value sgStartM = mul(gidM, i32_val(tensorM));
        Value baseM = NdOffset[0].dyn_cast<mlir::Value>();
        NdOffset[0] = add(sgStartM, baseM).getResult();

        for(int i = 0; i < numRepM; i++){
          for(int j = 0; j < numRepK; j++){
            Value baseM = NdOffset[0].dyn_cast<mlir::Value>();
            curNdOffset[0] = add(baseM, mul(offsetM, i32_val(i))).getResult();
            Value baseK = NdOffset[1].dyn_cast<mlir::Value>();
            curNdOffset[1] = add(baseK, mul(offsetK, i32_val(j))).getResult();

            descOp = rewriter.create<xegpu::CreateNdDescOp>(loc, tensorDescType, addr,
                    curNdOffset, NdShape, NdStride, NdOrder, 
                    triton::xegpu::MemoryScope::GLOBAL, true);

            newValues.push_back(descOp);
          }

        }
        resultTy = spirv::StructType::get(SmallVector<Type>(numRepM * numRepK, i64_ty));
      } else if(mmaFlag == 1){ //load matrix B
        int tensorK = elemShape[2] * elemStride[2];
        int tensorN = elemShape[3] * elemStride[3];

        int blockK = blockShape[0];
        int blockN = dim1;
        if(!isTrans){
          blockK = dim0;
          //blockK = std::min(std::max(blockK, dim0), 32);
        }else{
          blockK = dim0;
          blockN = dim1;
        }
        
        dbgInfo("[MakeTensorPtrOpToXeGPUPattern]blockK", blockK);
        dbgInfo("[MakeTensorPtrOpToXeGPUPattern]blockN", blockN);
        tensorDescType = TensorDescType::get(context, ::llvm::ArrayRef<int64_t>{blockK, blockN}, elemType, 
                                                      MemoryScopeAttr::get(context, MemoryScope::GLOBAL));
        Value offsetK = i32_val(blockK);
        Value offsetN = i32_val(blockN);

        int numRepK = tensorK / blockK;
        int numRepN = tensorN / blockN;
        dbgInfo("[MakeTensorPtrOpToXeGPUPattern]numRepK", numRepK);
        dbgInfo("[MakeTensorPtrOpToXeGPUPattern]numRepN", numRepN);

        if(threadShape[3] > 1){
          Value sgStartN = mul(gidN, i32_val(tensorN));
          Value baseN = NdOffset[1].dyn_cast<mlir::Value>();
          NdOffset[1] = add(sgStartN, baseN).getResult();
        }

        if(isTrans){
          for(int i = 0; i < numRepK; i++){
            for(int j = 0; j < numRepN; j++){
              Value baseK = NdOffset[0].dyn_cast<mlir::Value>();
              Value baseN = NdOffset[1].dyn_cast<mlir::Value>();

              curNdOffset[1] = add(baseK, mul(offsetK, i32_val(i))).getResult();
              curNdOffset[0] = add(baseN, mul(offsetN, i32_val(j))).getResult();

              descOp = rewriter.create<xegpu::CreateNdDescOp>(loc, tensorDescType, addr,
                      curNdOffset, NdShape, NdStride, NdOrder, 
                      triton::xegpu::MemoryScope::GLOBAL, true);
              newValues.push_back(descOp);
            }
          }

        }else{
          for(int i = 0; i < numRepK; i++){
            for(int j = 0; j < numRepN; j++){
              Value baseK = NdOffset[0].dyn_cast<mlir::Value>();
              Value baseN = NdOffset[1].dyn_cast<mlir::Value>();

              curNdOffset[0] = add(baseK, mul(offsetK, i32_val(i))).getResult();
              curNdOffset[1] = add(baseN, mul(offsetN, i32_val(j))).getResult();

              descOp = rewriter.create<xegpu::CreateNdDescOp>(loc, tensorDescType, addr,
                      curNdOffset, NdShape, NdStride, NdOrder, 
                      triton::xegpu::MemoryScope::GLOBAL, true);
              newValues.push_back(descOp);
            }
          }
        }
        resultTy = spirv::StructType::get(SmallVector<Type>(numRepK * numRepN, i64_ty));
      } else if(mmaFlag == 3){ //prefetch
        //todo
        tensorDescType = TensorDescType::get(context, ::llvm::ArrayRef<int64_t>{blockShape[0], blockShape[1]}, elemType, 
                                                      MemoryScopeAttr::get(context, MemoryScope::GLOBAL));
        //dbgInfo("[MakeTensorPtrOp] Prefetch tensorDescType: "<<tensorDescType<<"");
        descOp = rewriter.create<xegpu::CreateNdDescOp>(loc, tensorDescType, addr,
                    curNdOffset, NdShape, NdStride, NdOrder, 
                    triton::xegpu::MemoryScope::GLOBAL, true);
        newValues.push_back(descOp);
      } else{

      }
    } else {
      int tensorM = elemShape[2] * elemStride[2];
      int tensorN = elemShape[3] * elemStride[3];

      int blockM = dim0;
      int blockN = dim1;

      tensorDescType = TensorDescType::get(context, ::llvm::ArrayRef<int64_t>{blockM, blockN}, elemType, 
                                                      MemoryScopeAttr::get(context, MemoryScope::GLOBAL));
      Value sgStartM = mul(gidM, i32_val(tensorM));
      Value sgStartN = mul(gidN, i32_val(tensorN));
      Value baseM = NdOffset[0].dyn_cast<mlir::Value>();
      Value baseN = NdOffset[1].dyn_cast<mlir::Value>();
      NdOffset[0] = add(sgStartM, baseM).getResult();
      NdOffset[1] = add(sgStartN, baseN).getResult();

      int numRepM = tensorM / blockM;
      int numRepN = tensorN / blockN;

      for(int i = 0; i < numRepM; i++){
        for(int j = 0; j < numRepN; j++){
          Value baseM = NdOffset[0].dyn_cast<mlir::Value>();
          curNdOffset[0] = add(baseM, mul(offsetM, i32_val(i))).getResult();

          Value baseN = NdOffset[1].dyn_cast<mlir::Value>();
          curNdOffset[1] = add(baseN, mul(offsetN, i32_val(j))).getResult();

          descOp = rewriter.create<xegpu::CreateNdDescOp>(loc, tensorDescType, addr,
                  curNdOffset, NdShape, NdStride, NdOrder, 
                  triton::xegpu::MemoryScope::GLOBAL, true);

          newValues.push_back(descOp);
        }

      }
      resultTy = spirv::StructType::get(SmallVector<Type>(numRepM * numRepN, i64_ty));
    }

    ValueRange newValueRange{newValues};

    if(newValues.size() > 1){
      auto resultTys = op->getResultTypes();
      newValueRange = urcast(resultTys, newValueRange)->getResults();
    }

    rewriter.replaceOp(op, newValueRange);
    dbgInfo("[MakeTensorPtrOp] Finished");
    return success();
  }
};

class DotOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<DotOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<DotOp>::ConvertTritonGPUToXeGPUPattern;

  LogicalResult
  matchAndRewrite(DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("[DotOpToXeGPUPattern] start");
    Location loc = op->getLoc();
    auto context = op.getContext();

    auto instBrrier = rewriter.create<xegpu::CompilerHintOp>(loc);

    Value matAValue = adaptor.getA();
    Value matBValue = adaptor.getB();
    Value matCValue = adaptor.getC();

    ValueRange matARange;
    ValueRange matBRange;
    ValueRange matCRange;

    if(auto *parentOp = matAValue.getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        auto cast = (&castOp)->getInputs();
        matARange = ValueRange(cast);
      }
    }

    if(auto *parentOp = matBValue.getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        auto cast = (&castOp)->getInputs();
        matBRange = ValueRange(cast);
      }
    }

    if(auto *parentOp = matCValue.getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        auto cast = (&castOp)->getInputs();
        matCRange = ValueRange(cast);
      }
    }

    // dbgInfo("[DotOpToXeGPUPattern] matARange.size(): "<<matARange.size()<<"");
    // dbgInfo("[DotOpToXeGPUPattern] matARange[0]: "<<matARange[0]<<"");
    // dbgInfo("[DotOpToXeGPUPattern] matBRange.size(): "<<matBRange.size()<<"");
    // dbgInfo("[DotOpToXeGPUPattern] matBRange[0]: "<<matBRange[0]<<"");

    // Split matrix A
    auto matAType = matARange[0].getType().cast<VectorType>();
    auto aLoadShape = matAType.getShape();
    auto aLoadM = aLoadShape[0];

    auto aTensorType = op.getA().getType().cast<RankedTensorType>();
    auto elemType = aTensorType.getElementType();
    auto aEncoding = aTensorType.getEncoding().template cast<DotOperandEncodingAttr>().getParent();
    auto aLayout = llvm::dyn_cast<GenericEncodingAttr>(aEncoding);
    auto aMmaFlag = aLayout.getMmaFlag();
    auto aThreadShape = aLayout.getThreadShape();
    auto aElemShape = aLayout.getElemPerThread();
    auto aDpasM = aThreadShape[0] * aElemShape[0];
    auto aDpasK = aThreadShape[1] * aElemShape[1];
    auto aLoadSize = aDpasM * aDpasK;
    auto aCombinedLoadNum = aLoadM / aDpasM;

    // Split matrix B
    auto matBType = matBRange[0].getType().cast<VectorType>();
    auto bLoadShape = matBType.getShape();

    auto bTensorType = op.getB().getType().cast<RankedTensorType>();
    auto bEncoding = bTensorType.getEncoding().template cast<DotOperandEncodingAttr>().getParent();
    auto bLayout = llvm::dyn_cast<GenericEncodingAttr>(bEncoding);
    auto bThreadShape = bLayout.getThreadShape();
    auto bElemShape = bLayout.getElemPerThread();
    auto bDpasK = bThreadShape[0] * bElemShape[0];
    auto bDpasN = bThreadShape[1] * bElemShape[1];
    auto bLoadSize = bDpasK * bDpasN;

    auto B = matBRange[0];
    bool isTrans = 0;

    if(auto loadOp = dyn_cast<LoadNDOp>(B.getDefiningOp())){
      if(auto trans = loadOp.getTranspose()){
        llvm::ArrayRef<long int> transpose = trans.value();
        if(transpose.size() != 0)
          isTrans = transpose[0] == 1;
      }
    }

    int bCombinedLoadNum = 1;
    VectorType bDpasType;
    if(isTrans){
      bDpasType = VectorType::get({bDpasK, bLoadShape[1], 1}, elemType);
    } else{
      bDpasType = VectorType::get({bDpasK / 2, bLoadShape[1], bLoadShape[2]}, elemType);
      auto bLoadK = bLoadShape[0] * bLoadShape[2];
      bCombinedLoadNum = bLoadK / bDpasK;
    }

    int M = aElemShape[2];
    int N = bElemShape[3];
    int K = aElemShape[3];

    if(elemType == bf16Type){
      elemType = i16Type;
    }

    auto aSize = matARange.size();
    auto newMatANum = aSize * aCombinedLoadNum;
    auto aBlockK = aElemShape[3];
    auto aBlockM = aSize / aElemShape[3];

    // dbgInfo("[DotOpToXeGPUPattern] aCombinedLoadNum: ", aCombinedLoadNum);
    // dbgInfo("[DotOpToXeGPUPattern] aLoadShape.size(): ", aLoadShape.size());

    int matASize = aLoadShape.size() == 3 ? newMatANum : aSize;
    SmallVector<Value> matAVec(matASize);

    //from load todo
    if(aLoadShape.size() == 3){
      auto aDpasType = VectorType::get({aDpasM, aLoadShape[1], aLoadShape[2]}, elemType);
      SmallVector<int32_t, 2> aIndices(aLoadSize);
      for(int m = 0; m < aBlockM; m++){
        for(int k = 0; k < aBlockK; k++){
          for(int j = 0; j < aCombinedLoadNum; j++){
            auto blockIdx = m * aBlockK + k;
            uint64_t offset = j * aLoadSize;
            std::iota(aIndices.begin(), aIndices.end(), offset);
            Value slice = rewriter.create<spirv::VectorShuffleOp>(loc, aDpasType, 
                    matARange[blockIdx], matARange[blockIdx], rewriter.getI32ArrayAttr(aIndices));
            matAVec[(m * aCombinedLoadNum + j) * aBlockK + k] = slice;
          }
        }
      }
    } else if(aLoadShape.size() == 2){  //from other compute ops
      auto aDpasType = VectorType::get({aLoadShape[0], aLoadShape[1] / 2, 2}, elemType);
      for(int i = 0;i<aSize;i++){
        Value A = rewriter.create<vector::ShapeCastOp>(loc, aDpasType, matARange[i]);
        matAVec[i] = A;
      }
    }
    matARange = ValueRange(matAVec);
    dbgInfo("[DotOpToXeGPUPattern] matARange[0]", matARange[0]);

    auto bSize = matBRange.size();
    auto newMatBNum = matBRange.size() * bCombinedLoadNum;
    auto bBlockN = bElemShape[3];
    auto bBlockK = bSize / bElemShape[3];
    // dbgInfo("[DotOpToXeGPUPattern]matBRange.size()", matBRange.size());
    // dbgInfo("[DotOpToXeGPUPattern]bCombinedLoadNum", bCombinedLoadNum);
    // dbgInfo("[DotOpToXeGPUPattern]bBlockN", bBlockN);
    // dbgInfo("[DotOpToXeGPUPattern]bBlockK", bBlockK);
    SmallVector<Value> spiltB(newMatBNum);
    SmallVector<int32_t, 2> bIndices(bLoadSize);
    for(int k = 0; k < bBlockK; k++){
      for(int n = 0; n < bBlockN; n++){
        for(int j = 0; j < bCombinedLoadNum; j++){
          auto blockIdx = k * bBlockN + n;
          uint64_t offset = j * bLoadSize;
          std::iota(bIndices.begin(), bIndices.end(), offset);
          Value slice;
          if(bCombinedLoadNum!=1)
            slice = rewriter.create<spirv::VectorShuffleOp>(loc, bDpasType, 
                    matBRange[blockIdx], matBRange[blockIdx], rewriter.getI32ArrayAttr(bIndices));
          else
            slice = matBRange[blockIdx];
          spiltB[(k * bCombinedLoadNum + j) * bBlockN + n] = slice;
        }
      }
    }
    matBRange = ValueRange(spiltB);

    SmallVector<Value> results(M * N);
    Type resultTy = matCRange[0].getType();

    //todo
    for (int n = 0; n < N; n++) {
      for (int m = 0; m < M; m++) {
        auto matC = matCRange[m * N + n];
        for (int k = 0; k < K; k++) {
          auto matA = matARange[m * K + k];
          auto matB = matBRange[k * N + n];
          matC = rewriter.create<xegpu::DpasOp>(loc, resultTy, matA, matB, matC);
        }
        results[m * N + n] = matC;
      }
    }

    instBrrier = rewriter.create<xegpu::CompilerHintOp>(loc);

    ValueRange newResults(results);
    auto resultTys = op->getResultTypes();
    auto cast = urcast(resultTys, newResults)->getResults();
    newResults = ValueRange(cast);
    rewriter.replaceOp(op, newResults);
    dbgInfo("[DotOpToXeGPUPattern] end");
    return success();
  }
};

class AdvanceOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<AdvanceOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<AdvanceOp>::ConvertTritonGPUToXeGPUPattern;

  LogicalResult
  matchAndRewrite(AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("[AdvanceOpToXeGPUPattern]");
    Location loc = op.getLoc();
    auto context = op.getContext();
    Value descs = adaptor.getPtr();
    ValueRange tensorDesc = ValueRange(descs);
    //dbgInfo("[AdvanceOpToXeGPUPattern] tensorDesc[0]: ", tensorDesc[0]);
    Type tensorType = tensorDesc[0].getType();
    Value ptr = op.getPtr();
    auto layout  = ptr.getType().cast<triton::PointerType>()
                    .getPointeeType().cast<RankedTensorType>()
                    .getEncoding().cast<GenericEncodingAttr>();
    auto mmaFlag = layout.getMmaFlag();
    auto offs = adaptor.getOffsets();
    ValueRange offsets(offs);

    if(auto *parentOp = tensorDesc[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        tensorDesc = (&castOp)->getInputs();
        tensorType = tensorDesc[0].getType();
      }
    }

    bool isTrans = 0;

    if(auto *parentOp = ptr.getDefiningOp()){
      if(auto makeTensorOp = dyn_cast<MakeTensorPtrOp>(parentOp)){
        auto order = makeTensorOp.getOrder();
        if(order[0] == 0)
          isTrans = 1;
      }
    }

    if(auto arg = dyn_cast<BlockArgument>(ptr)){
      auto ownerOp = arg.getOwner()->getParentOp();
      auto forOp = cast<scf::ForOp>(ownerOp);
      Value init = forOp.getInitArgs()[arg.getArgNumber() - 1];
      while (auto op = init.getDefiningOp()) {
        if (auto makeTensorOp = dyn_cast<MakeTensorPtrOp>(op)) {
          auto order = makeTensorOp.getOrder();
          if(order[0] == 0)
            isTrans = 1;
          break;
        }else if(auto advanceOp = dyn_cast<AdvanceOp>(op)){
          init = advanceOp.getPtr();
        }
      }
    }

    dbgInfo("[AdvanceOpToXeGPUPattern]offsets[0]", offsets[0]);
    dbgInfo("[AdvanceOpToXeGPUPattern]offsets[1]", offsets[1]);

    Value offset0 = offsets[0];
    Value offset1 = offsets[1];
    SmallVector<Value> tmp;
    if(isTrans){
      tmp.push_back(offset1);
      tmp.push_back(offset0);
      offsets = ValueRange(tmp);
    }

    dbgInfo("[AdvanceOpToXeGPUPattern]isTrans", isTrans);
    dbgInfo("[AdvanceOpToXeGPUPattern]offsets[0]", offsets[0]);
    dbgInfo("[AdvanceOpToXeGPUPattern]offsets[1]", offsets[1]);

    SmallVector<Value> tensorDescVec;
    for(int i=0;i<tensorDesc.size();i++){
      tensorDescVec.push_back(tensorDesc[i]);
    }

    int size = tensorDesc.size();
    SmallVector<Value> advancedDescList;
    if(size > 1){
      for(int i = 0; i < size; i++){
        Value advancedDesc = rewriter.create<UpdateNDOffsetOp>(loc, tensorType, tensorDescVec[i], offsets);
        advancedDescList.push_back(advancedDesc);
      }
      ValueRange newValueRange(advancedDescList);
      auto resultTys = op->getResultTypes();
      auto castOp = urcast(resultTys, newValueRange)->getResults();
      newValueRange = ValueRange(castOp);
      rewriter.replaceOp(op, newValueRange);
    } else {
      Value advancedDesc = rewriter.create<UpdateNDOffsetOp>(loc, tensorType, tensorDesc[0], offsets);
      rewriter.replaceOp(op, advancedDesc);
    }
    dbgInfo("[AdvanceOpToXeGPUPattern] End");
    return success();
  }
};

class ExpandDimsOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<triton::ExpandDimsOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<triton::ExpandDimsOp>::ConvertTritonGPUToXeGPUPattern;
  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("[ExpandDimsOpToXeGPUPattern]");
    auto context = rewriter.getContext();
    Location loc = op->getLoc();
    auto src = adaptor.getSrc();
    auto type = src.getType();
    auto vectorType = type.cast<mlir::VectorType>();
    auto shape = vectorType.getShape();
    auto elemType = vectorType.getElementType();

    RankedTensorType tensorType = op.getResult().getType().cast<RankedTensorType>();
    auto encoding = tensorType.getEncoding();
    auto layout = encoding.cast<GenericEncodingAttr>();
    auto elemShape = layout.getElemPerThread();

    llvm::SmallVector<mlir::Type> resultTypes;
    auto value = op.getResult();
    auto result = this->getTypeConverter()->convertType(value.getType(), resultTypes);
    auto newType = resultTypes[0];
    dbgInfo("[ExpandDimsOpToXeGPUPattern]resultTypes.size()", resultTypes.size());
    dbgInfo("[ExpandDimsOpToXeGPUPattern]newType", newType);

    int blockM = elemShape[2];
    int blockN = elemShape[3];
    int vectorSize = shape[0] / elemShape[2];
    SmallVector<Value> newValues;
    for(int i = 0; i < blockM; i++){
      Type curVecType = mlir::VectorType::get(vectorSize, elemType);
      SmallVector<int32_t, 2> indices(vectorSize);
      uint64_t offset = vectorSize * i;
      std::iota(indices.begin(), indices.end(), offset);
      Value slice = rewriter.create<spirv::VectorShuffleOp>(loc, curVecType, src, src, rewriter.getI32ArrayAttr(indices));

      for(int j = 0; j < blockN; j++){
        newValues.push_back(slice);
      }
    }
    dbgInfo("[ExpandDimsOpToXeGPUPattern]newValues.size()", newValues.size());
    dbgInfo("[ExpandDimsOpToXeGPUPattern]newValues[0]", newValues[0]);

    ValueRange newValueRange(newValues);
    auto resultTys = op->getResultTypes();
    auto cast = urcast(resultTys, newValueRange)->getResults();
    rewriter.replaceOp(op, cast);
    return success();
  }
};

class BroadcastOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<triton::BroadcastOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<triton::BroadcastOp>::ConvertTritonGPUToXeGPUPattern;
  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("[BroadCastOpToXeGPUPattern]");
    auto context = rewriter.getContext();
    Location loc = op->getLoc();
    ValueRange src(adaptor.getSrc());

    if(auto *parentOp = src[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        src = (&castOp)->getInputs();
      }
    }

    llvm::SmallVector<mlir::Type> resultTypes;
    auto value = op.getResult();
    auto result = this->getTypeConverter()->convertType(value.getType(), resultTypes);
    auto newType = resultTypes[0];
    auto shape = newType.cast<VectorType>().getShape();

    int nElem = 1;
    for(auto s : shape){
      nElem *= s;
    }

    auto srcType = src[0].getType();
    ArrayRef<int64_t> srcShape{1};
    bool isVevtorSrc = 0;
    Type castType;
    if(isa<VectorType>(srcType)){
      isVevtorSrc = 1;
      srcShape = srcType.cast<VectorType>().getShape();
      Type elemType = srcType.cast<VectorType>().getElementType();
      castType = VectorType::get(srcShape[0], elemType);
    }

    SmallVector<Value> newValues;
    for(int i = 0; i < resultTypes.size();i++){
      Value in = src[i];
      Value ret;
      if(isVevtorSrc){
        SmallVector<int32_t> indices(nElem, 0);
        for(int d0 = 0; d0 < shape[0]; d0++){
          for(int d1 = 0; d1 < shape[1]; d1++){
            indices[d0 * shape[1] + d1] = d0;
          }
        }
        //dbgInfo("[SplatOpToXeGPUPattern]indices.size(): "<<indices.size()<<"");
        ret = rewriter.create<spirv::VectorShuffleOp>(
              loc, newType, in, in, rewriter.getI32ArrayAttr(indices));
      } else{
        ret = rewriter.create<vector::BroadcastOp>(loc, newType, in);
      }

      newValues.push_back(ret);
    }

    ValueRange newValueRange(newValues);
    auto resultTys = op->getResultTypes();
    auto cast = urcast(resultTys, newValueRange)->getResults();
    rewriter.replaceOp(op, cast);

    return success();
  }
};

class ExternElementwiseOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<triton::ExternElementwiseOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<triton::ExternElementwiseOp>::ConvertTritonGPUToXeGPUPattern;
  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("[ExternElementwiseOpToXeGPUPattern]");
    auto loc = op->getLoc();
    auto context = op.getContext();

    llvm::SmallVector<mlir::Type> resultTypes;
    auto value = op.getResult();
    auto result = this->getTypeConverter()->convertType(value.getType(), resultTypes);
    auto newType = resultTypes[0];

    auto args = adaptor.getArgs();
    int argNums = args.size();

    ValueRange argsRange(args);
    Value arg0 = argsRange[0];
    ValueRange argsRange0(arg0);
    ValueRange argsRange1;

    ValueRange arg0Cast;
    if(auto *parentOp = arg0.getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        arg0Cast = (&castOp)->getInputs();
        if(arg0Cast.size() > 1)
          argsRange0 = ValueRange(arg0Cast);
      }
    }

    ValueRange arg1Cast;
    if(argNums == 2){
      argsRange1 = ValueRange(argsRange[1]);
      if(auto *parentOp = argsRange[1].getDefiningOp()){
        if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
          arg1Cast = (&castOp)->getInputs();
          argsRange1 = ValueRange(arg1Cast);
        }
      }
    }

    ::mlir::StringAttr libname = op.getLibnameAttr();
    ::mlir::StringAttr libpath = op.getLibpathAttr();
    ::mlir::StringAttr symbol = op.getSymbolAttr();
    ::mlir::BoolAttr pure = op.getPureAttr();

    if(resultTypes.size() > 1){
      SmallVector<Value> newValues;
      for(int i = 0 ;i < resultTypes.size(); i++){
        Value newValue;
        if(argNums==1){
          newValue = rewriter.create<triton::ExternElementwiseOp>(loc, newType, argsRange0[i], libname, libpath, symbol, pure);
        }
        else{
          SmallVector<Value> argsVec{argsRange0[i], argsRange1[i]};
          ValueRange args(argsVec);
          newValue = rewriter.create<triton::ExternElementwiseOp>(loc, newType, args, libname, libpath, symbol, pure);
        }
        newValues.push_back(newValue);
      }
      
      ValueRange newValueRange(newValues);
      auto resultTys = op->getResultTypes();
      auto cast = urcast(resultTys, newValueRange)->getResults();

      rewriter.replaceOp(op, cast);
    } else{
      SmallVector<Value> argsVec;
      argsVec.push_back(argsRange0[0]);
      if(argNums == 2)
        argsVec.push_back(argsRange0[1]);

      ValueRange args(argsVec);
      Value newValue = rewriter.create<triton::ExternElementwiseOp>(loc, newType, args, libname, libpath, symbol, pure);

      rewriter.replaceOp(op, newValue);
    }
    return success();
  }
};

void populateTritonGPUToXeGPUPatterns(
    TritonGPUToXeGPUTypeConverter &typeConverter, RewritePatternSet &patterns) {
  dbgInfo("[populateXeGPUToVCIntrinsicsPatterns]");
  auto context = patterns.getContext();
  patterns.add<LoadOpToXeGPUPattern, StoreOpToXeGPUPattern,
               MakeRangeOpToXeGPUPattern, AddPtrOpToXeGPUPattern,
               CmpIOpToXeGPUPattern, SplatOpToXeGPUPattern,
               GetProgramIdOpToXeGPUPattern, ReturnOpToXeGPUPattern,
               ConvertLayoutOpToXeGPUPattern, MakeTensorPtrOpToXeGPUPattern,
               DotOpToXeGPUPattern, AdvanceOpToXeGPUPattern,
               ExpandDimsOpToXeGPUPattern, BroadcastOpToXeGPUPattern,
               ExternElementwiseOpToXeGPUPattern>(
      typeConverter, context);

  //process arith op
  patterns.add<
      ArithConstantOpPattern, ArithTruncFOpPattern,
      GenericOpPattern<arith::AddIOp>,
      GenericOpPattern<arith::SubIOp>, GenericOpPattern<arith::MulIOp>,
      GenericOpPattern<arith::DivUIOp>, GenericOpPattern<arith::DivSIOp>,
      GenericOpPattern<arith::CeilDivUIOp>,
      GenericOpPattern<arith::CeilDivSIOp>,
      GenericOpPattern<arith::FloorDivSIOp>, GenericOpPattern<arith::RemUIOp>,
      GenericOpPattern<arith::RemSIOp>, GenericOpPattern<arith::AndIOp>,
      GenericOpPattern<arith::OrIOp>, GenericOpPattern<arith::XOrIOp>,
      GenericOpPattern<arith::ShLIOp>, GenericOpPattern<arith::ShRUIOp>,
      GenericOpPattern<arith::ShRSIOp>, // NegFOp
      // Floating point
      GenericOpPattern<arith::AddFOp>, GenericOpPattern<arith::SubFOp>,
      // MaxMin
      GenericOpPattern<arith::MaxFOp, spirv::CLFMaxOp>, 
      GenericOpPattern<arith::MaxSIOp>,
      GenericOpPattern<arith::MaxUIOp>, GenericOpPattern<arith::MinFOp>,
      GenericOpPattern<arith::MinSIOp>, GenericOpPattern<arith::MinUIOp>,
      // Floating point
      GenericOpPattern<arith::MulFOp>, GenericOpPattern<arith::DivFOp>,
      GenericOpPattern<arith::RemFOp>,
      // Cast Ops
      // GenericOpPattern<arith::TruncIOp>, GenericOpPattern<arith::TruncFOp>,
      GenericOpPattern<arith::ExtUIOp>, GenericOpPattern<arith::ExtSIOp>,
      GenericOpPattern<arith::ExtFOp>, GenericOpPattern<arith::SIToFPOp>,
      GenericOpPattern<arith::FPToSIOp>, GenericOpPattern<arith::FPToUIOp>,
      GenericOpPattern<arith::UIToFPOp>>(typeConverter, context);

  //process math op
  patterns.add<
      GenericOpPattern<math::ExpOp>, GenericOpPattern<math::CosOp>,
      GenericOpPattern<math::SinOp>, GenericOpPattern<math::LogOp>,
      GenericOpPattern<math::SqrtOp>>(typeConverter, context);
}