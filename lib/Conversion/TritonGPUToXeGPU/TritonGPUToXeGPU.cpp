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
//#include "../TritonGPUToLLVM/Utility.h"
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

template <class Op> class GenericOpPattern : public OpConversionPattern<Op> {
public:
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    //llvm::outs() << "\n\n[GenericOpPattern] op: " << op << "\n";
    Location loc = op.getLoc();
    Type type = op.getType();
    auto context = op.getContext();
    TritonGPUToXeGPUTypeConverter xeGPUTypeConverter(*context);
    llvm::SmallVector<mlir::Type> resultTypes;
    auto result = xeGPUTypeConverter.convertType(op.getType(), resultTypes);
    auto retType = resultTypes[0];

    int srcNum = 1;
    auto operands = adaptor.getOperands();
    auto operand0 = operands[0];

    ValueRange src0{operand0};
    ValueRange src1;
    if(operands.size() > 1){
      auto operand1 = operands[1];
      src1 = ValueRange{operand1};
      srcNum = 2;
    }

    if(auto *parentOp = src0[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        auto inputs =  (&castOp)->getInputs();
        src0 = ValueRange{inputs};
      }
    }

    if(src1.size() != 0){
      if(auto *parentOp = src1[0].getDefiningOp()){
        if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
          auto inputs =  (&castOp)->getInputs();
          src1 = ValueRange{inputs};
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
        args.push_back(src0[i]);
        if(srcNum==2){
          args.push_back(src1[i]);
        } else {
          ret = rewriter.create<Op>(loc, newType, args);
        }
        retVec.push_back(ret);
      }

      ValueRange newValueRange(retVec);
      auto resultTys = op->getResultTypes();
      auto castOp = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newValueRange)->getResults();
      newValueRange = ValueRange{castOp};
      rewriter.replaceOp(op, castOp);
    } else {
      addNamedAttrs(
          rewriter.replaceOpWithNewOp<Op>(op, retType, adaptor.getOperands()),
          adaptor.getAttributes());
    }
    return success();
  }
};

class ArithConstantOpPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto context = op.getContext();
    Type type = op.getType();
    Attribute layout = type.cast<RankedTensorType>().getEncoding();

    TritonGPUToXeGPUTypeConverter xeGPUTypeConverter(*context);
    llvm::SmallVector<mlir::Type> resultTypes;
    auto result = xeGPUTypeConverter.convertType(op.getType(), resultTypes);
    auto retType = resultTypes[0];

    llvm::outs() << "\n\nArithConstant retType: " << retType << "\n";

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

        mlir::ValueRange newValueRange(constVals);
        auto resultTys = op->getResultTypes();
        newValueRange = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newValueRange)->getResults();
        // llvm::outs()<<"\n\nresultTys: "<<resultTys<<"\n";
        // llvm::outs()<<"\n\nnewValueRange: "<<newValueRange[0]<<"\n";
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

class ArithTruncFOpPattern : public OpConversionPattern<arith::TruncFOp> {
public:
  using OpConversionPattern<arith::TruncFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::TruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs() << "\n\nArithTruncFPattern\n";
    Location loc = op.getLoc();
    auto context = op.getContext();
    Type type = op.getType();
    Attribute layout = type.cast<RankedTensorType>().getEncoding();

    TritonGPUToXeGPUTypeConverter xeGPUTypeConverter(*context);
    llvm::SmallVector<mlir::Type> resultTypes;
    auto result = xeGPUTypeConverter.convertType(op.getType(), resultTypes);
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
    // llvm::outs() << "\n\nArithTruncFPattern retType: " << retType << "\n";
    // llvm::outs() << "\n\nArithConstant retType: " << resultTypes.size() << "\n";

    auto src = adaptor.getIn();
    ValueRange in{src};

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
            truncFVal = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, newRetType, truncFVal)->getResults()[0];
          }
          truncFVals.push_back(truncFVal);
        }

        mlir::ValueRange newValueRange(truncFVals);
        auto resultTys = op->getResultTypes();
        auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newValueRange)->getResults();
        newValueRange = ValueRange(cast);
        // llvm::outs()<<"\n\nresultTys: "<<resultTys<<"\n";
        // llvm::outs()<<"\n\nnewValueRange: "<<newValueRange[0]<<"\n";
        rewriter.replaceOp(op, newValueRange);

      } else {
        Value truncFVal = rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, retType, in[0]);
        if(elemType == i16Type){
          truncFVal = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, newRetType, truncFVal)->getResults()[0];
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

class ReturnOpToXeGPUPattern : public OpConversionPattern<triton::ReturnOp> {
public:
  using OpConversionPattern<triton::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    rewriter.create<mlir::gpu::ReturnOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }
};

class LoadOpToXeGPUPattern : public OpConversionPattern<triton::LoadOp> {
public:
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\n[LoadOpToXeGPUPattern]: \n";
    auto loc = op->getLoc();
    auto context = rewriter.getContext();

    ValueRange desc{adaptor.getPtr()};
    int nElem = desc.size();
    auto descType = desc[0].getType();
    //llvm::outs()<<"\n\n[LoadOpToXeGPUPattern] desc[0]: "<<desc[0]<<"\n";

    RankedTensorType tensorType = op.getResult().getType().cast<RankedTensorType>();
    auto encoding = tensorType.getEncoding();
    auto shape = tensorType.getShape();
    auto layout = encoding.cast<GenericEncodingAttr>();
    auto mmaFlag = layout.getMmaFlag();

    if(auto *parentOp = desc[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        desc = (&castOp)->getInputs();
        descType = desc[0].getType();
        nElem = desc.size();
      }
    }
    Value desc0 = desc[0];
    auto value = op.getResult();

    TritonGPUToXeGPUTypeConverter xeGPUTypeConverter(*context);
    llvm::SmallVector<mlir::Type> resultTypes;
    auto result = xeGPUTypeConverter.convertType(value.getType(), resultTypes);

    Type loadType = resultTypes[0];
    int elemNum = loadType.dyn_cast<VectorType>().getShape()[0];

    if(mmaFlag != -1){
      auto vectorType = loadType.dyn_cast<VectorType>();
      Type elemType = vectorType.getElementType();

      if(elemType == bf16Type){
        elemType = i16Type;
      }

      if(mmaFlag == 0){
        loadType = VectorType::get(ArrayRef<int64_t>{32, 8, 2}, elemType);
      } else if(mmaFlag == 1) {
        loadType = VectorType::get(ArrayRef<int64_t>{16, 16, 2}, elemType);
      } else if(mmaFlag == 3){
        loadType = VectorType::get(ArrayRef<int64_t>{16, 16, 1}, elemType);
      } else{
      }
    }

    // llvm::outs()<<"\n\n[LoadOpToXeGPUPattern]loadType: "<<loadType<<"\n";
    // llvm::outs()<<"\n\n[LoadOpToXeGPUPattern]nElem: "<<nElem<<"\n";
    auto L1_hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto L2_hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto L3_hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    //llvm::outs()<<"\n\n[LoadOpToXeGPUPattern] desc[0]: "<<desc[0]<<"\n";

    if(nElem > 1){
      SmallVector<Value> newValues;
      for(int i = 0; i < nElem; i++){
        Value ret;
        if(mmaFlag == 0){ //load matrix A for gemm
          auto vnni = IntegerAttr::get(i32_ty, 1);
          ret = rewriter.create<xegpu::LoadNDOp>(loc, loadType, desc[i], vnni, DenseI64ArrayAttr{}, L1_hint, L2_hint, L3_hint);
        } else if(mmaFlag == 1){ //load matrix B for gemm
          auto vnni = IntegerAttr::get(i32_ty, 0);
          ret = rewriter.create<xegpu::LoadNDOp>(loc, loadType, desc[i], vnni, DenseI64ArrayAttr{}, L1_hint, L2_hint, L3_hint);
        } else if(mmaFlag == 3){ //prefetch for gemm

        } else{

        }

        newValues.push_back(ret);
      }

      mlir::ValueRange newValueRange{newValues};
      auto resultTys = op->getResultTypes();
      auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newValueRange)->getResults();
      newValueRange = ValueRange{cast};
      rewriter.replaceOp(op, newValueRange);
    } else {
      if(shape.size() == 1){
        Value mask = adaptor.getMask();
        if(!mask) {
          auto maskType = mlir::VectorType::get(elemNum, i32Type);
          DenseElementsAttr constData = DenseElementsAttr::get(maskType, ArrayRef<int>(std::vector<int>(elemNum, 1)));
          mask = rewriter.create<spirv::ConstantOp>(loc, maskType, constData);
        }

        Value ret = rewriter.create<xegpu::LoadGatherOp>(loc, loadType, desc0, mask, IntegerAttr{}, DenseI64ArrayAttr{}, L1_hint, L2_hint, L3_hint);
        // llvm::outs()<<"\n\nxegpu::LoadGatherOp: " << ret <<"\n";
        rewriter.replaceOp(op, ret);
      } else if(mmaFlag == 3){
        rewriter.create<xegpu::PrefetchNDOp>(loc, desc0, L1_hint, L2_hint, L3_hint);
        rewriter.eraseOp(op);
      }
    }

    return success();
  }
};

class StoreOpToXeGPUPattern : public OpConversionPattern<triton::StoreOp> {
public:
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\nStoreOpToXeGPUPattern: \n";
    auto loc = op->getLoc();
    auto context = rewriter.getContext();

    ValueRange desc{adaptor.getPtr()};
    if(auto *parentOp = desc[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        desc = (&castOp)->getInputs();
      }
    }

    Value gatherStoreDesc = desc[0];

    ValueRange value{adaptor.getValue()};
    if(auto *parentOp = value[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        value = (&castOp)->getInputs();
      }
    }

    Value data = value[0];

    auto newType = data.getType();
    int elemNum = newType.dyn_cast<VectorType>().getShape()[0];

    auto L1_hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);
    auto L2_hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);
    auto L3_hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);

    if(desc.size() > 1){
      for(int i = 0; i < desc.size(); i++){
        rewriter.create<xegpu::StoreNDOp>(loc, desc[i], value[i], L1_hint, L2_hint, L3_hint);
      }
      rewriter.eraseOp(op);
    } else {
      Value mask; 
      if(Value m = adaptor.getMask()){
        mask = m;
      } else {
        auto maskType = mlir::VectorType::get(elemNum, i32Type);
        DenseElementsAttr constData = DenseElementsAttr::get(maskType, ArrayRef<int>(std::vector<int>(elemNum, 1)));
        mask = rewriter.create<spirv::ConstantOp>(loc, maskType, constData);
      }

      Value ret = rewriter.create<xegpu::StoreScatterOp>(loc, data, gatherStoreDesc, mask, L1_hint, L2_hint, L3_hint).getODSResults(0)[0];
      rewriter.replaceOp(op, ret);
    }
    auto module = op.getOperation()->getParentOfType<ModuleOp>();
    llvm::outs()<<"\n\n[Module] after StoreOp: ";
    module.print(llvm::outs());
    return success();
  }
};

class MakeRangeOpToXeGPUPattern : public OpConversionPattern<triton::MakeRangeOp> {
public:
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\nMakeRangeOpToXeGPUPattern: \n";
    auto loc = op->getLoc();
    auto context = rewriter.getContext();

    Value subgroubId = rewriter.create<::mlir::gpu::SubgroupIdOp>(loc, rewriter.getIndexType());
    Value sgId = rewriter.create<UnrealizedConversionCastOp>(loc, i64_ty, subgroubId).getResult(0);
    sgId = rewriter.create<spirv::UConvertOp>(loc, i32_ty, sgId);

    //Value sgId = zext(i32_ty, subgroubId);
    //to do replace 32 with subgroup nums
    auto subGroupNums = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(32));
    sgId = urem(sgId, subGroupNums);
    auto module = op.getOperation()->getParentOfType<ModuleOp>();
    int sgSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(module);
    auto sgAddr = rewriter.create<arith::MulIOp>(loc, i32_ty, sgId, 
      rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(sgSize)));
    auto type = mlir::VectorType::get(sgSize, i32Type);

    //avoid spirv.CompositeConstruct
    Value sgAddrs = rewriter.create<spirv::UndefOp>(loc, v32i32Type);
    auto idx0 = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(0));
    sgAddrs =
        rewriter.create<spirv::VectorInsertDynamicOp>(loc, sgAddrs, sgAddr, idx0);
    SmallVector<int32_t, 32> indices(32, 0);
    sgAddrs = rewriter.create<spirv::VectorShuffleOp>(
          loc, v32i32Type, sgAddrs, sgAddrs, rewriter.getI32ArrayAttr(indices));

    llvm::outs()<<"\n\nsgAddrs: "<<sgAddrs<<"\n";
    std::vector<int> values(sgSize, 0);
    for(int i = 0; i < sgSize;i++){
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

class AddPtrOpToXeGPUPattern : public OpConversionPattern<triton::AddPtrOp> {
public:
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\nAddPtrOpToXeGPUPattern: \n";
    auto loc = op->getLoc();
    auto context = rewriter.getContext();

    Value curDesc;
    ::llvm::ArrayRef<int64_t> shape;
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();

    Type offsetType = offset.getType();
    if(offsetType.isa<IntegerType>()){
      return success();
    }

    auto ptrType = ptr.getType();
    bool isPtr = ptrType.isa<mlir::MemRefType>();

    while(auto curOp = ptr.getDefiningOp()){
      // llvm::outs()<<"\n\ncurOp: "<<*curOp<<"\n";
      // llvm::outs()<<"\n\nptr: "<<ptr<<"\n";
      // llvm::outs()<<"\n\nptrType: "<<ptrType<<"\n";
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
        llvm::outs() << "\n\nptr: "<< ptr << "\n";

        Value originalOffsetVec = rewriter.create<spirv::UndefOp>(loc, offsetType);
        auto idx0 = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(0));
        originalOffsetVec = rewriter.create<spirv::VectorInsertDynamicOp>(loc, 
                                              originalOffsetVec, originalOffset, idx0);
        SmallVector<int32_t, 32> indices(32, 0);
        originalOffsetVec = rewriter.create<spirv::VectorShuffleOp>(
              loc, offsetType, originalOffsetVec, originalOffsetVec, rewriter.getI32ArrayAttr(indices));

        //Value originalOffsetVec = rewriter.create<vector::SplatOp>(loc, offsetType, originalOffset);
        offset = rewriter.create<arith::AddIOp>(loc, offsetType, offset, originalOffsetVec);
        llvm::outs() << "\n\noffset: "<< offset << "\n";
        break;
      }
      ptr = curOp->getOperand(0);
      ptrType = ptr.getType();
    }

    llvm::outs() << "\n\nptrType: "<< ptrType << "\n";
    auto elemType = ptrType.dyn_cast<mlir::MemRefType>().getElementType();
    llvm::outs() << "\n\nelemType: "<< elemType << "\n";
    auto newType = mlir::VectorType::get(32, elemType);
    

    //get shape
    if(auto rankType = op.getPtr().getType().dyn_cast<RankedTensorType>()){
      shape = rankType.getShape();
    }
    
    int size = shape.size()==0 ? 1 : shape.size();
    std::vector<int64_t> newShape(size, 1);
    if(shape.size()==0){
      newShape[0] = 32;
    } else {
      for(int i = 0;i < shape.size(); i++){
        newShape[0] = shape[0] / 32; 
      }
    }

    auto tensorDescType = ::mlir::triton::xegpu::TensorDescType::get(context, newShape, elemType, ScatteredAttr::get(context));
    auto memory_scope = MemoryScopeAttr::get(context, triton::xegpu::MemoryScope::GLOBAL);
    auto vert_size = IntegerAttr::get(i32_ty, 1);

    Value ret = rewriter.create<xegpu::CreateDescOp>(loc, tensorDescType, ptr, offset, memory_scope, vert_size);
    llvm::outs() << "\n\nret: "<< ret << "\n";
    rewriter.replaceOp(op, ret);
    return success();
  }
};

class CmpIOpToXeGPUPattern : public OpConversionPattern<triton::gpu::CmpIOp> {
public:
  using OpConversionPattern<triton::gpu::CmpIOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::gpu::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\nCmpIOpToXeGPUPattern\n";
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

class SplatOpToXeGPUPattern : public OpConversionPattern<triton::SplatOp> {
public:
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\n[SplatOpToXeGPUPattern]\n";
    auto loc = op->getLoc();
    auto context = rewriter.getContext();

    auto src = adaptor.getSrc();

    if(src.getType().isa<mlir::MemRefType>()){
      auto memref = op.getSrc();
      rewriter.replaceOp(op, memref);
      return success();
    }

    auto tensorTy = op.getResult().getType().cast<RankedTensorType>();
    auto tensorShape = tensorTy.getShape();
    auto elemType = tensorTy.getElementType();
    auto layout = tensorTy.getEncoding().dyn_cast<GenericEncodingAttr>();

    auto module = op.getOperation()->getParentOfType<ModuleOp>();
    int sgSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(module);

    //auto subGroupNum = tensorShape[0] / sgSize;
    auto newType = mlir::VectorType::get(sgSize, elemType);

    // Segmentation fault
    // Value ret = rewriter.create<vector::SplatOp>(loc, newType, src);

    // avoid spirv.CompositeConstruct
    Value ret = rewriter.create<spirv::UndefOp>(loc, newType);
    auto idx0 = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(0));
    ret = rewriter.create<spirv::VectorInsertDynamicOp>(loc, ret, src, idx0);
    SmallVector<int32_t, 32> indices(32, 0);
    ret = rewriter.create<spirv::VectorShuffleOp>(
          loc, newType, ret, ret, rewriter.getI32ArrayAttr(indices));

    llvm::outs()<<"\n\n[SplatOpToXeGPUPattern]ret: "<<ret<<"\n";
    rewriter.replaceOp(op, ret);
    return success();
  }
};

class GetProgramIdOpToXeGPUPattern : public OpConversionPattern<triton::GetProgramIdOp> {
public:
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\nGetProgramIdOpToSPIRVConversion\n";
    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);

    Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(
            loc, rewriter.getIndexType(), dims[op.getAxisAsInt()]);
    // Value blockId_idx = rewriter.create<::mlir::arith::TruncIOp>(
    //         loc, i32_ty, blockId);
    Value cast = rewriter.create<UnrealizedConversionCastOp>(loc, i64_ty, blockId).getResult(0);

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

class ConvertLayoutOpToXeGPUPattern : public OpConversionPattern<ConvertLayoutOp> {
public:
  using OpConversionPattern<ConvertLayoutOp>::OpConversionPattern;

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
    auto castOp = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, src);
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

class MakeTensorPtrOpToXeGPUPattern : public OpConversionPattern<MakeTensorPtrOp> {
public:
  using OpConversionPattern<MakeTensorPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\nMakeTensorPtrOpToXeGPUPattern: \n";
    Location loc = op->getLoc();
    auto context = rewriter.getContext();

    auto blockPtr = adaptor.getBase();
    auto ptrType = op.getResult().getType().cast<triton::PointerType>();
    auto tensorType = ptrType.getPointeeType().cast<RankedTensorType>();
    auto genericLayout = tensorType.getEncoding().cast<GenericEncodingAttr>();
    auto mmaFlag = genericLayout.getMmaFlag();
    auto elemType = tensorType.getElementType();

    auto blockShape = tensorType.getShape();
    auto tensorShape = op.getShape();
    auto tensorStride = op.getStrides();
    auto tensorOffsets = op.getOffsets();

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
    Value addr = rewriter.create<UnrealizedConversionCastOp>(loc, spirvPtrType, blockPtr).getResult(0);
    addr = rewriter.create<spirv::ConvertPtrToUOp>(loc, i64_ty ,addr);

    SmallVector<mlir::OpFoldResult> NdOffset = getI32SmallVector(rewriter, tensorOffsets);

    //wa: error when return the rvalue to ValueRange;
    SmallVector<Value> NdShape = getI32Vevtor(rewriter, tensorShape);
    SmallVector<Value> NdStride = getI32Vevtor(rewriter, tensorStride);

    //for PVC
    const int blockM = 8;
    const int blockK = 16;
    const int blockN = 16;

    Value offsetM = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(blockM));
    Value offsetK = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(blockK));
    Value offsetN = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(blockN));

    Type resultTy;

    SmallVector<Value> newValues;
    xegpu::CreateNdDescOp descOp;
    TensorDescType tensorDescType;

    Value subgroubId = rewriter.create<::mlir::gpu::SubgroupIdOp>(loc, rewriter.getIndexType());
    Value sgId = rewriter.create<UnrealizedConversionCastOp>(loc, i64_ty, subgroubId).getResult(0);
    sgId = rewriter.create<spirv::UConvertOp>(loc, i32_ty, sgId);

    Value gid = udiv(sgId, i32_val(32));
    sgId = urem(sgId, i32_val(32));

    Value gidM = udiv(sgId, i32_val(4));
    Value gidN = urem(sgId, i32_val(4));

    int numRepM = 32 / blockM;
    int numRepK = 32 / blockK;
    int numRepN = 64 / blockN;

    SmallVector<mlir::OpFoldResult> curNdOffset = NdOffset;
    if(isLoad){
      if(mmaFlag == 0){ //load matrix A
        int blockM = 32;
        int blockK = 16;
        Value offsetM = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(blockM));
        Value offsetK = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(blockK));

        tensorDescType = TensorDescType::get(context, ::llvm::ArrayRef<int64_t>{blockM, blockK}, elemType, 
                                                      MemoryScopeAttr::get(context, MemoryScope::GLOBAL));
        //llvm::outs()<<"\n\n[MakeTensorPtrOp] LoadA tensorDescType: "<<tensorDescType<<"\n";
        int numRepM = 32 / blockM;
        int numRepK = 32 / blockK;

        Value sgStartM = mul(gidM, i32_val(32));
        Value baseM = NdOffset[0].dyn_cast<mlir::Value>();
        NdOffset[0] = add(sgStartM, baseM).getResult();

        for(int i = 0; i < numRepM; i++){
          for(int j = 0; j < numRepK; j++){
            Value baseM = NdOffset[0].dyn_cast<mlir::Value>();
            curNdOffset[0] = add(baseM, mul(offsetM, i32_val(i))).getResult();
            Value baseK = NdOffset[1].dyn_cast<mlir::Value>();
            curNdOffset[1] = add(baseK, mul(offsetK, i32_val(j))).getResult();

            descOp = rewriter.create<xegpu::CreateNdDescOp>(loc, tensorDescType, addr,
                    curNdOffset, NdShape, NdStride, 
                    triton::xegpu::MemoryScope::GLOBAL, true);

            newValues.push_back(descOp);
          }

        }
        resultTy = spirv::StructType::get(SmallVector<Type>(numRepM * numRepK, i64_ty));
      } else if(mmaFlag == 1){ //load matrix B
        const int blockK = 32;
        const int blockN = 16;
        tensorDescType = TensorDescType::get(context, ::llvm::ArrayRef<int64_t>{blockK, blockN}, elemType, 
                                                      MemoryScopeAttr::get(context, MemoryScope::GLOBAL));
        //llvm::outs()<<"\n\n[MakeTensorPtrOp] LoadB tensorDescType: "<<tensorDescType<<"\n";
        Value offsetK = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(blockK));
        Value offsetN = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(blockN));

        int numRepK = 32 / blockK;
        int numRepN = 64 / blockN;

        Value sgStartN = mul(gidN, i32_val(64));
        Value baseN = NdOffset[1].dyn_cast<mlir::Value>();
        NdOffset[1] = add(sgStartN, baseN).getResult();

        for(int i = 0; i < numRepK; i++){
          for(int j = 0; j < numRepN; j++){

            Value baseK = NdOffset[0].dyn_cast<mlir::Value>();
            curNdOffset[0] = add(baseK, mul(offsetK, i32_val(i))).getResult();
            Value baseN = NdOffset[1].dyn_cast<mlir::Value>();
            curNdOffset[1] = add(baseN, mul(offsetN, i32_val(j))).getResult();

            descOp = rewriter.create<xegpu::CreateNdDescOp>(loc, tensorDescType, addr,
                    curNdOffset, NdShape, NdStride, 
                    triton::xegpu::MemoryScope::GLOBAL, true);

            newValues.push_back(descOp);
          }

        }
        resultTy = spirv::StructType::get(SmallVector<Type>(numRepK * numRepN, i64_ty));
      } else if(mmaFlag == 3){ //prefetch
        //todo
        tensorDescType = TensorDescType::get(context, ::llvm::ArrayRef<int64_t>{blockShape[0], blockShape[1]}, elemType, 
                                                      MemoryScopeAttr::get(context, MemoryScope::GLOBAL));
        //llvm::outs()<<"\n\n[MakeTensorPtrOp] Prefetch tensorDescType: "<<tensorDescType<<"\n";
        descOp = rewriter.create<xegpu::CreateNdDescOp>(loc, tensorDescType, addr,
                    curNdOffset, NdShape, NdStride, 
                    triton::xegpu::MemoryScope::GLOBAL, true);
        newValues.push_back(descOp);
      } else{

      }
    } else {
      tensorDescType = TensorDescType::get(context, ::llvm::ArrayRef<int64_t>{blockM, blockN}, elemType, 
                                                      MemoryScopeAttr::get(context, MemoryScope::GLOBAL));
      Value sgStartM = mul(gidM, i32_val(32));
      Value sgStartN = mul(gidN, i32_val(64));
      Value baseM = NdOffset[0].dyn_cast<mlir::Value>();
      Value baseN = NdOffset[1].dyn_cast<mlir::Value>();
      NdOffset[0] = add(sgStartM, baseM).getResult();
      NdOffset[1] = add(sgStartN, baseN).getResult();

      for(int i = 0; i < numRepM; i++){
        for(int j = 0; j < numRepN; j++){
          Value baseM = NdOffset[0].dyn_cast<mlir::Value>();
          curNdOffset[0] = add(baseM, mul(offsetM, i32_val(i))).getResult();

          Value baseN = NdOffset[1].dyn_cast<mlir::Value>();
          curNdOffset[1] = add(baseN, mul(offsetN, i32_val(j))).getResult();

          descOp = rewriter.create<xegpu::CreateNdDescOp>(loc, tensorDescType, addr,
                  curNdOffset, NdShape, NdStride, 
                  triton::xegpu::MemoryScope::GLOBAL, true);

          newValues.push_back(descOp);
        }

      }
      resultTy = spirv::StructType::get(SmallVector<Type>(numRepM * numRepN, i64_ty));
    }

    mlir::ValueRange newValueRange{newValues};

    if(newValues.size()>1){
      auto resultTys = op->getResultTypes();
      newValueRange = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newValueRange)->getResults();
    }

    rewriter.replaceOp(op, newValueRange);
    llvm::outs()<<"\n\n[MakeTensorPtrOp] Finished\n";
    return success();
  }
};

class DotOpToXeGPUPattern : public OpConversionPattern<DotOp> {
public:
  using OpConversionPattern<DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\n[DotOpToXeGPUPattern] start\n";
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
        matARange = (&castOp)->getInputs();
      }
    }

    if(auto *parentOp = matBValue.getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        matBRange = (&castOp)->getInputs();
      }
    }

    if(auto *parentOp = matCValue.getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        matCRange = (&castOp)->getInputs();
      }
    }

    // llvm::outs() << "\n\n[DotOpToXeGPUPattern] matARange[0]: "<<matARange[0]<<"\n";
    // llvm::outs() << "\n\n[DotOpToXeGPUPattern] matBRange[0]: "<<matBRange[0]<<"\n";

    // Split matrix A
    auto matAType = matARange[0].getType().cast<VectorType>();
    auto aLoadShape = matAType.getShape();
    auto aLoadM = aLoadShape[0];

    auto aTensorType = op.getA().getType().cast<RankedTensorType>();
    auto elemType = aTensorType.getElementType();
    auto aEncoding = aTensorType.getEncoding().template cast<DotOperandEncodingAttr>().getParent();
    auto aLayout = llvm::dyn_cast<GenericEncodingAttr>(aEncoding);
    auto aThreadShape = aLayout.getThreadShape();
    auto aThreadStride = aLayout.getThreadStride();
    auto aElemShape = aLayout.getElemPerThread();
    auto aDpasM = aThreadShape[0] * aElemShape[0];
    auto aDpasK = aThreadShape[1] * aElemShape[1];
    auto aLoadSize = aDpasM * aDpasK;

    if(elemType == bf16Type){
      elemType = i16Type;
    }

    auto aCombinedLoadNum = aLoadM / aDpasM;
    auto aDpasType = VectorType::get({aDpasM, aLoadShape[1], aLoadShape[2]}, elemType);

    auto newMatANum = matARange.size() * aCombinedLoadNum;
    SmallVector<Value> spiltA(newMatANum);
    SmallVector<int32_t, 2> aIndices(aLoadSize);
    for(int i = 0; i < matARange.size(); i++){
      for(int j = 0; j < aCombinedLoadNum; j++){
        uint64_t offset = j * aLoadSize;
        std::iota(aIndices.begin(), aIndices.end(), offset);
        Value slice = rewriter.create<spirv::VectorShuffleOp>(loc, aDpasType, matARange[i], matARange[i], rewriter.getI32ArrayAttr(aIndices));
        spiltA[j * 2 + i] = slice;
      }
    }
    matARange = ValueRange{spiltA};

    // Split matrix B
    auto matBType = matBRange[0].getType().cast<VectorType>();
    auto bLoadShape = matBType.getShape();
    auto bLoadK = bLoadShape[0] * bLoadShape[2];

    auto bTensorType = op.getB().getType().cast<RankedTensorType>();
    auto bEncoding = bTensorType.getEncoding().template cast<DotOperandEncodingAttr>().getParent();
    auto bLayout = llvm::dyn_cast<GenericEncodingAttr>(bEncoding);
    auto bThreadShape = bLayout.getThreadShape();
    auto bThreadStride = bLayout.getThreadStride();
    auto bElemShape = bLayout.getElemPerThread();
    auto bDpasK = bThreadShape[0] * bElemShape[0];
    auto bDpasN = bThreadShape[1] * bElemShape[1];
    auto bLoadSize = bDpasK * bDpasN;

    auto bCombinedLoadNum = bLoadK / bDpasK;
    auto bDpasType = VectorType::get({bDpasK / 2, bLoadShape[1], bLoadShape[2]}, elemType);

    auto newMatBNum = matBRange.size() * bCombinedLoadNum;
    SmallVector<Value> spiltB(newMatBNum);
    SmallVector<int32_t, 2> bIndices(bLoadSize);
    for(int i = 0; i < matBRange.size(); i++){
      for(int j = 0; j < bCombinedLoadNum; j++){
        uint64_t offset = j * bLoadSize;
        std::iota(bIndices.begin(), bIndices.end(), offset);
        Value slice = rewriter.create<spirv::VectorShuffleOp>(loc, bDpasType, matBRange[i], matBRange[i], rewriter.getI32ArrayAttr(bIndices));
        spiltB[j * 4 + i] = slice;
      }
    }
    matBRange = ValueRange{spiltB};

    auto size = matARange.size();
    SmallVector<Value> results(4*4);

    Type resultTy = matCRange[0].getType();
    Type resultStructTy = spirv::StructType::get(SmallVector<Type>(4*4, resultTy));

    //todo
    for (int n = 0; n < 4; n++) {
      for (int m = 0; m < 4; m++) {
        auto matC = matCRange[m*4+n];
        for (int k = 0; k < 2; k++) {
          auto matA = matARange[m*2+k];
          auto matB = matBRange[k*4+n];
          matC = rewriter.create<xegpu::DpasOp>(loc, resultTy, matA, matB, matC);
        }
        results[m*4+n] = matC;
      }
    }

    instBrrier = rewriter.create<xegpu::CompilerHintOp>(loc);

    mlir::ValueRange newResults{results};
    auto resultTys = op->getResultTypes();
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newResults)->getResults();
    newResults = ValueRange{cast};
    rewriter.replaceOp(op, newResults);
    llvm::outs()<<"\n\n[DotOpToXeGPUPattern] end\n";
    return success();
  }
};

class AdvanceOpToXeGPUPattern : public OpConversionPattern<AdvanceOp> {
public:
  using OpConversionPattern<AdvanceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\n[AdvanceOpToXeGPUPattern]\n";
    Location loc = op.getLoc();
    auto context = op.getContext();
    Value descs = adaptor.getPtr();
    ValueRange tensorDesc = ValueRange{descs};
    //llvm::outs()<<"\n\n[AdvanceOpToXeGPUPattern] tensorDesc[0]: " << tensorDesc[0] <<"\n";
    Type tensorType = tensorDesc[0].getType();
    Value ptr = op.getPtr();
    auto layout  = ptr.getType().cast<triton::PointerType>()
                    .getPointeeType().cast<RankedTensorType>()
                    .getEncoding().cast<GenericEncodingAttr>();
    auto mmaFlag = layout.getMmaFlag();
    ValueRange offsets{adaptor.getOffsets()};
    //llvm::outs()<<"\n\n[AdvanceOpToXeGPUPattern] tensorDesc[i]: " << tensorDesc[0] <<"\n";
    if(auto *parentOp = tensorDesc[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        tensorDesc = (&castOp)->getInputs();
        tensorType = tensorDesc[0].getType();
      }
    }

    SmallVector<Value> tensorDescVec;
    for(int i=0;i<tensorDesc.size();i++){
      tensorDescVec.push_back(tensorDesc[i]);
    }

    int size = tensorDesc.size();
    SmallVector<Value> advancedDescList;
    //if(size > 1 || mmaFlag == 3){
    if(size > 1){
      for(int i = 0; i < size; i++){
        Value advancedDesc = rewriter.create<UpdateNDOffsetOp>(loc, tensorType, tensorDescVec[i], offsets);
        advancedDescList.push_back(advancedDesc);
      }
      mlir::ValueRange newValueRange{advancedDescList};
      //llvm::outs()<<"\n\n[AdvanceOpToXeGPUPattern] newValueRange: " << newValueRange[0] <<"\n";
      auto resultTys = op->getResultTypes();
      auto castOp = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newValueRange)->getResults();
      newValueRange = ValueRange{castOp};
      rewriter.replaceOp(op, newValueRange);
    } else {
      Value advancedDesc = rewriter.create<UpdateNDOffsetOp>(loc, tensorType, tensorDesc[0], offsets);
      rewriter.replaceOp(op, advancedDesc);
    }
    llvm::outs()<<"\n\n[AdvanceOpToXeGPUPattern] End\n";
    return success();
  }
};

void populateTritonGPUToXeGPUPatterns(
    TritonGPUToXeGPUTypeConverter &typeConverter, RewritePatternSet &patterns) {
  llvm::outs()<<"\n\npopulateXeGPUToVCIntrinsicsPatterns\n";
  auto context = patterns.getContext();
  patterns.add<LoadOpToXeGPUPattern, StoreOpToXeGPUPattern,
               MakeRangeOpToXeGPUPattern, AddPtrOpToXeGPUPattern,
               CmpIOpToXeGPUPattern, SplatOpToXeGPUPattern,
               GetProgramIdOpToXeGPUPattern, ReturnOpToXeGPUPattern,
               ConvertLayoutOpToXeGPUPattern, MakeTensorPtrOpToXeGPUPattern,
               DotOpToXeGPUPattern, AdvanceOpToXeGPUPattern>(
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
      GenericOpPattern<arith::MaxFOp>, GenericOpPattern<arith::MaxSIOp>,
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