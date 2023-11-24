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
    llvm::outs() << "\n\n[GenericOpPattern] op: " << op << "\n";
    Location loc = op.getLoc();
    Type type = op.getType();
    auto context = op.getContext();
    TritonGPUToXeGPUTypeConverter xeGPUTypeConverter(*context);
    llvm::SmallVector<mlir::Type> resultTypes;
    auto result = xeGPUTypeConverter.convertType(op.getType(), resultTypes);
    auto retType = resultTypes[0];
    llvm::outs()<<"\n\n[GenericOpPattern]retType: "<<retType<<"\n";
    int srcNum = 1;
    auto operands = adaptor.getOperands();
    Value operand0 = operands[0];
    ValueRange src0(operand0);

    //llvm::outs()<<"\n\n[GenericOpPattern]operands.size(): "<<operands.size()<<"\n";
    llvm::outs()<<"\n\n[GenericOpPattern]src0[0]: "<<src0[0]<<"\n";
    if(auto *parentOp = src0[0].getDefiningOp()){
      //llvm::outs()<<"\n\n*parentOp: "<<*parentOp<<"\n";
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
      llvm::outs()<<"\n\n[GenericOpPattern]src1[0]: "<<src1[0]<<"\n";
    }

    if(src1.size() != 0){
      if(auto *parentOp = src1[0].getDefiningOp()){
        if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
          auto inputs =  (&castOp)->getInputs();
          src1 = ValueRange(inputs);
        }
      }
    }

    llvm::outs()<<"\n\n[GenericOpPattern]resultTypes.size(): "<<resultTypes.size()<<"\n";
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

        auto srcType = src0[i].getType();
        if(isa<VectorType>(srcType)){
          auto shape = srcType.cast<VectorType>().getShape();
          auto srcElemType = srcType.cast<VectorType>().getElementType();
          auto retElemType = newType.cast<VectorType>().getElementType();
          int nElem = 1;
          for(auto s : shape) 
            nElem *= s;
          auto castSrcType = VectorType::get(nElem, srcElemType);
          //operand0 = rewriter.create<vector::ShapeCastOp>(loc, castSrcType, operand0);
          operand0 = urcast(castSrcType, operand0)->getResults()[0];
          if(srcNum==2){
            //operand1 = rewriter.create<vector::ShapeCastOp>(loc, castSrcType, operand1);
            operand1 = urcast(castSrcType, operand1)->getResults()[0];
          }
          newType = VectorType::get(nElem, retElemType);
        }

        args.push_back(operand0);
        if(srcNum==2){
          args.push_back(operand1);
        }
        ret = rewriter.create<Op>(loc, newType, args);

        if(isa<VectorType>(srcType)){
          newType = resultTypes[0];
          //ret = rewriter.create<vector::ShapeCastOp>(loc, newType, ret);
          ret = urcast(newType, ret)->getResults()[0];
        }
        retVec.push_back(ret);
      }

      ValueRange newValueRange(retVec);
      auto resultTys = op->getResultTypes();
      auto castOp = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newValueRange)->getResults();
      newValueRange = ValueRange(castOp);
      rewriter.replaceOp(op, castOp);
    } else {
      addNamedAttrs(
          rewriter.replaceOpWithNewOp<Op>(op, retType, adaptor.getOperands()),
          adaptor.getAttributes());
    }
    llvm::outs()<<"\n\n[GenericOpPattern] End\n";
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
    auto threadShape = layout.getThreadShape();
    auto elemShape = layout.getElemPerThread();
    auto elemStride = layout.getElemStride();

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

      int dim0 = threadShape[0] * elemShape[0];
      int dim1 = threadShape[1] * elemShape[1];
      if(mmaFlag == 0){
        int loadM = shape[0] / threadShape[2];
        loadM = std::min(std::max(loadM, dim0), 32);

        loadType = VectorType::get(ArrayRef<int64_t>{loadM, dim1 / 2, 2}, elemType);
      } else if(mmaFlag == 1) {
        int loadK = shape[0];
        loadK = std::min(std::max(loadK, dim0), 32);

        loadType = VectorType::get(ArrayRef<int64_t>{loadK / 2, dim1, 2}, elemType);
      } else if(mmaFlag == 3){
        loadType = VectorType::get(ArrayRef<int64_t>{16, 16, 1}, elemType);
      } else{
      }
    }

    llvm::outs()<<"\n\n[LoadOpToXeGPUPattern]loadType: "<<loadType<<"\n";
    llvm::outs()<<"\n\n[LoadOpToXeGPUPattern]nElem: "<<nElem<<"\n";
    auto L1_hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto L2_hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto L3_hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);

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
    llvm::outs()<<"\n\n[StoreOpToXeGPUPattern]\n";
    auto loc = op->getLoc();
    auto context = rewriter.getContext();

    auto ptrRange = adaptor.getPtr();
    llvm::outs()<<"\n\n[StoreOpToXeGPUPattern]ptrRange: " << ptrRange <<"\n";
    ValueRange desc(ptrRange);
    if(auto *parentOp = desc[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        desc = (&castOp)->getInputs();
      }
    }

    Value gatherStoreDesc = desc[0];

    auto valueRange = adaptor.getValue();
    ValueRange value(valueRange);
    if(auto *parentOp = value[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        value = (&castOp)->getInputs();
      }
    }

    Value data = value[0];
    llvm::outs()<<"\n\n[StoreOpToXeGPUPattern]data: " << data <<"\n";
    llvm::outs()<<"\n\n[StoreOpToXeGPUPattern]value.size(): " << value.size() <<"\n";

    auto newType = data.getType();
    int elemNum = newType.dyn_cast<VectorType>().getShape()[0];

    auto L1_hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);
    auto L2_hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);
    auto L3_hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);

    llvm::outs()<<"\n\n[StoreOpToXeGPUPattern]desc[0]: " << desc[0] <<"\n";
    llvm::outs()<<"\n\n[StoreOpToXeGPUPattern]desc.size(): " << desc.size() <<"\n";
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
        auto maskType = mlir::VectorType::get(elemNum, i1Type);
        DenseElementsAttr constData = DenseElementsAttr::get(maskType, ArrayRef<int>(std::vector<int>(elemNum, 1)));
        mask = rewriter.create<spirv::ConstantOp>(loc, maskType, constData);
      }

      Value ret = rewriter.create<xegpu::StoreScatterOp>(loc, data, gatherStoreDesc, mask, L1_hint, L2_hint, L3_hint).getODSResults(0)[0];
      rewriter.replaceOp(op, ret);
    }
    // auto module = op.getOperation()->getParentOfType<ModuleOp>();
    // llvm::outs()<<"\n\n[Module] after StoreOp: ";
    // module.print(llvm::outs());
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

    TritonGPUToXeGPUTypeConverter xeGPUTypeConverter(*context);
    llvm::SmallVector<mlir::Type> resultTypes;

    Type retType = op.getResult().getType();
    auto result = xeGPUTypeConverter.convertType(retType, resultTypes);
    auto newShape = resultTypes[0].cast<VectorType>().getShape();
    int nElem = newShape[0];

    //to do replace 32 with subgroup nums
    auto subGroupNums = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(32));
    sgId = urem(sgId, subGroupNums);
    // auto module = op.getOperation()->getParentOfType<ModuleOp>();
    // int sgSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(module);

    auto sgAddr = rewriter.create<arith::MulIOp>(loc, i32_ty, sgId, 
      rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(nElem)));
    auto type = mlir::VectorType::get(nElem, i32_ty);

    //avoid spirv.CompositeConstruct
    Value sgAddrs = rewriter.create<spirv::UndefOp>(loc, type);
    auto idx0 = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(0));
    sgAddrs =
        rewriter.create<spirv::VectorInsertDynamicOp>(loc, sgAddrs, sgAddr, idx0);
    SmallVector<int32_t> indices(nElem, 0);
    sgAddrs = rewriter.create<spirv::VectorShuffleOp>(
          loc, type, sgAddrs, sgAddrs, rewriter.getI32ArrayAttr(indices));

    llvm::outs()<<"\n\nsgAddrs: "<<sgAddrs<<"\n";
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

class AddPtrOpToXeGPUPattern : public OpConversionPattern<triton::AddPtrOp> {
public:
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\n[AddPtrOpToXeGPUPattern]\n";
    auto loc = op->getLoc();
    auto context = rewriter.getContext();

    Value curDesc;
    ::llvm::ArrayRef<int64_t> shape;
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();
    llvm::outs()<<"\n\n[AddPtrOpToXeGPUPattern]ptr: " << ptr <<"\n";
    llvm::outs()<<"\n\n[AddPtrOpToXeGPUPattern]offset: " << offset <<"\n";

    Type offsetType = offset.getType();
    if(isa<IntegerType>(offsetType)){
      // Type ptrType = ptr.getType();
      // Value addr = ptr;
      // if(!isa<IntegerType>(ptrType)){
      //   Type elemType = ptrType.cast<MemRefType>().getElementType();
      //   Optional<spirv::StorageClass> storageClass = spirv::StorageClass::CrossWorkgroup;
      //   auto spirvPtrType = spirv::PointerType::get(elemType, *storageClass);
      //   addr = rewriter.create<UnrealizedConversionCastOp>(loc, spirvPtrType, ptr).getResult(0);
      //   addr = rewriter.create<spirv::ConvertPtrToUOp>(loc, i64_ty, addr);
      // }

      // if(offsetType != i64_ty)
      //   offset = zext(i64_ty, offset);
      // addr = add(addr, offset);
      // rewriter.replaceOp(op, addr);
      return success();
    }

    ptr = op.getPtr();
    auto ptrType = ptr.getType();
    bool isPtr = ptrType.isa<mlir::MemRefType>();

    llvm::SmallVector<mlir::Type> resultTypes;
    TritonGPUToXeGPUTypeConverter xeGPUTypeConverter(*context);
    auto value = op.getResult();
    auto result = xeGPUTypeConverter.convertType(offsetType, resultTypes);
    auto newType = resultTypes[0];
    llvm::outs()<<"\n\n[SplatOpToXeGPUPattern]resultTypes[0]: "<<resultTypes[0]<<"\n";
    auto size = resultTypes[0].cast<VectorType>().getShape()[0];

    while(auto curOp = ptr.getDefiningOp()){
      // llvm::outs()<<"\n\n[SplatOpToXeGPUPattern]curOp: "<<*curOp<<"\n";
      // llvm::outs()<<"\n\n[SplatOpToXeGPUPattern]ptr: "<<ptr<<"\n";
      // llvm::outs()<<"\n\n[SplatOpToXeGPUPattern]ptrType: "<<ptrType<<"\n";
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
        llvm::outs() << "\n\n[AddPtrOpToXeGPUPattern]ptr: "<< ptr << "\n";

        Value originalOffsetVec = rewriter.create<spirv::UndefOp>(loc, offsetType);
        auto idx0 = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(0));
        originalOffsetVec = rewriter.create<spirv::VectorInsertDynamicOp>(loc, 
                                              originalOffsetVec, originalOffset, idx0);
        SmallVector<int32_t> indices(size, 0);
        originalOffsetVec = rewriter.create<spirv::VectorShuffleOp>(
              loc, offsetType, originalOffsetVec, originalOffsetVec, rewriter.getI32ArrayAttr(indices));

        //Value originalOffsetVec = rewriter.create<vector::SplatOp>(loc, offsetType, originalOffset);
        offset = rewriter.create<arith::AddIOp>(loc, offsetType, offset, originalOffsetVec);
        llvm::outs() << "\n\n[AddPtrOpToXeGPUPattern]offset: "<< offset << "\n";
        break;
      }
      ptr = curOp->getOperand(0);
      ptrType = ptr.getType();
    }

    Type elemType = op.getPtr().getType().cast<RankedTensorType>().getElementType();
    if(isa<triton::PointerType>(elemType)){
      elemType = elemType.cast<triton::PointerType>().getPointeeType();
    }
    llvm::outs() << "\n\n[AddPtrOpToXeGPUPattern]elemType: "<< elemType << "\n";

    //get shape
    if(auto type = offset.getType().cast<VectorType>()){
      shape = type.getShape();
    }

    llvm::outs() << "\n\n[AddPtrOpToXeGPUPattern]shape.size(): "<< shape.size() << "\n";
    llvm::outs() << "\n\n[AddPtrOpToXeGPUPattern]shape[0]: "<< shape[0] << "\n";
    auto tensorDescType = ::mlir::triton::xegpu::TensorDescType::get(context, shape, elemType, ScatteredAttr::get(context));
    auto memory_scope = MemoryScopeAttr::get(context, triton::xegpu::MemoryScope::GLOBAL);
    auto vert_size = IntegerAttr::get(i32_ty, 1);

    Value ret = rewriter.create<xegpu::CreateDescOp>(loc, tensorDescType, ptr, offset, memory_scope, vert_size);
    llvm::outs() << "\n\n[AddPtrOpToXeGPUPattern]ret: "<< ret << "\n";
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

    llvm::outs()<<"\n\n[SplatOpToXeGPUPattern]src: " << src <<"\n";
    llvm::outs()<<"\n\n[SplatOpToXeGPUPattern]src.getType(): " << src.getType() <<"\n";
    if(src.getType().isa<mlir::MemRefType>()){
      auto memref = op.getSrc();
      llvm::outs()<<"\n\n[SplatOpToXeGPUPattern]memref: " << memref <<"\n";
      rewriter.replaceOp(op, memref);
      return success();
    }

    auto tensorTy = op.getResult().getType().cast<RankedTensorType>();
    auto tensorShape = tensorTy.getShape();
    auto elemType = tensorTy.getElementType();
    auto layout = tensorTy.getEncoding().dyn_cast<GenericEncodingAttr>();

    TritonGPUToXeGPUTypeConverter xeGPUTypeConverter(*context);
    llvm::SmallVector<mlir::Type> resultTypes;

    auto value = op.getResult();
    auto result = xeGPUTypeConverter.convertType(value.getType(), resultTypes);
    auto shape = resultTypes[0].cast<VectorType>().getShape();
    int nElem = shape[0];

    Type newType;
    Value ret;
    if(tensorShape.size() == 1 && resultTypes.size() == 1){
      newType = resultTypes[0];

      ret = rewriter.create<spirv::UndefOp>(loc, newType);
      auto idx0 = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(0));
      ret = rewriter.create<spirv::VectorInsertDynamicOp>(loc, ret, src, idx0);
      SmallVector<int32_t> indices(nElem, 0);
      llvm::outs()<<"\n\n[SplatOpToXeGPUPattern]indices.size(): "<<indices.size()<<"\n";
      ret = rewriter.create<spirv::VectorShuffleOp>(
            loc, newType, ret, ret, rewriter.getI32ArrayAttr(indices));
      llvm::outs()<<"\n\n[SplatOpToXeGPUPattern]1d ret: "<<ret<<"\n";
      rewriter.replaceOp(op, ret);
    } else{
      SmallVector<Value> newValues;
      newType = resultTypes[0];

      // auto srcType = src.getType();
      // ArrayRef<int64_t> srcShape{1};
      // bool isVevtorSrc = 0;
      // Type castType;
      // if(isa<VectorType>(srcType)){
      //   isVevtorSrc = 1;
      //   srcShape = srcType.cast<VectorType>().getShape();
      //   castType = VectorType::get(srcShape[0], elemType);
      // }

      for(int i = 0;i < resultTypes.size(); i++){
        // if(isVevtorSrc && (srcShape.size() > 1))
        //   src = rewriter.create<vector::ShapeCastOp>(loc, castType, src);
        ret = rewriter.create<vector::BroadcastOp>(loc, newType, src);
        newValues.push_back(ret);
      }
      mlir::ValueRange newValueRange(newValues);
      auto resultTys = op->getResultTypes();
      auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newValueRange)->getResults();
      newValueRange = ValueRange(cast);
      rewriter.replaceOp(op, newValueRange);
    }

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
    auto threadShape = genericLayout.getThreadShape();
    auto elemShape = genericLayout.getElemPerThread();
    auto elemStride = genericLayout.getElemStride();
    auto threadStride = genericLayout.getThreadStride();
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
    Value addr;

    // UnrealizedConversionCastOp castOp;

    // if(auto *parentOp = addr.getDefiningOp()){
    //   castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp);
    //   addr = (&castOp)->getInputs()[0];
    // }

    //castOp.erase();
    auto base = op.getBase();
    if(auto *parentOp = base.getDefiningOp()){
      if(auto addPtrOp = dyn_cast<AddPtrOp>(parentOp)){
        auto ptr = addPtrOp.getPtr();
        auto offset = addPtrOp.getOffset();
        auto offsetType = offset.getType();

        addr = rewriter.create<UnrealizedConversionCastOp>(loc, spirvPtrType, ptr).getResult(0);
        addr = rewriter.create<spirv::ConvertPtrToUOp>(loc, i64_ty, addr);

        if(offsetType != i64_ty)
          offset = zext(i64_ty, offset);
        addr = add(addr, offset);
      }
    } else{
      addr = rewriter.create<UnrealizedConversionCastOp>(loc, spirvPtrType, blockPtr).getResult(0);
      addr = rewriter.create<spirv::ConvertPtrToUOp>(loc, i64_ty ,addr);
    }


    //addr = rewriter.create<UnrealizedConversionCastOp>(loc, i64_ty, blockPtr).getResult(0);
    // Type blockPtrType = blockPtr.getType();
    // if(!isa<IntegerType>(blockPtrType)){
    //   addr = rewriter.create<UnrealizedConversionCastOp>(loc, spirvPtrType, blockPtr).getResult(0);
    //   addr = rewriter.create<spirv::ConvertPtrToUOp>(loc, i64_ty ,addr);
    // }

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
        Value offsetM = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(blockM));
        Value offsetK = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(blockK));

        tensorDescType = TensorDescType::get(context, ::llvm::ArrayRef<int64_t>{blockM, blockK}, elemType, 
                                                      MemoryScopeAttr::get(context, MemoryScope::GLOBAL));
        //llvm::outs()<<"\n\n[MakeTensorPtrOp] LoadA tensorDescType: "<<tensorDescType<<"\n";
        int numRepM = tensorM / blockM;
        int numRepK = tensorK / blockK;

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
        int tensorK = elemShape[2] * elemStride[2];
        int tensorN = elemShape[3] * elemStride[3];

        int blockK = blockShape[0];
        blockK = std::min(std::max(blockK, dim0), 32);
        int blockN = dim1;
        tensorDescType = TensorDescType::get(context, ::llvm::ArrayRef<int64_t>{blockK, blockN}, elemType, 
                                                      MemoryScopeAttr::get(context, MemoryScope::GLOBAL));
        //llvm::outs()<<"\n\n[MakeTensorPtrOp] LoadB tensorDescType: "<<tensorDescType<<"\n";
        Value offsetK = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(blockK));
        Value offsetN = rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(blockN));

        int numRepK = tensorK / blockK;
        int numRepN = tensorN / blockN;

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
                  curNdOffset, NdShape, NdStride, 
                  triton::xegpu::MemoryScope::GLOBAL, true);

          newValues.push_back(descOp);
        }

      }
      resultTy = spirv::StructType::get(SmallVector<Type>(numRepM * numRepN, i64_ty));
    }

    mlir::ValueRange newValueRange{newValues};

    if(newValues.size() > 1){
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

    llvm::outs() << "\n\n[DotOpToXeGPUPattern] matARange.size(): "<<matARange.size()<<"\n";
    llvm::outs() << "\n\n[DotOpToXeGPUPattern] matARange[0]: "<<matARange[0]<<"\n";
    llvm::outs() << "\n\n[DotOpToXeGPUPattern] matBRange.size(): "<<matBRange.size()<<"\n";
    llvm::outs() << "\n\n[DotOpToXeGPUPattern] matBRange[0]: "<<matBRange[0]<<"\n";

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
    auto aThreadStride = aLayout.getThreadStride();
    auto aElemShape = aLayout.getElemPerThread();
    auto aDpasM = aThreadShape[0] * aElemShape[0];
    auto aDpasK = aThreadShape[1] * aElemShape[1];
    auto aLoadSize = aDpasM * aDpasK;
    auto aCombinedLoadNum = aLoadM / aDpasM;

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

    llvm::outs() <<"\n\n[DotOpToXeGPUPattern] aCombinedLoadNum: " << aCombinedLoadNum <<"\n";
    llvm::outs() <<"\n\n[DotOpToXeGPUPattern] aLoadShape.size(): " << aLoadShape.size() <<"\n";

    int matASize = aLoadShape.size() == 3 ? newMatANum : aSize;
    SmallVector<Value> matAVec(matASize);

    if(aLoadShape.size() == 3){
      auto aDpasType = VectorType::get({aDpasM, aLoadShape[1], aLoadShape[2]}, elemType);
      SmallVector<int32_t, 2> aIndices(aLoadSize);
      for(int m = 0; m < aBlockM; m++){
        for(int k = 0; k < aBlockK; k++){
          for(int j = 0; j < aCombinedLoadNum; j++){
            auto blockIdx = m * aBlockK + k;
            uint64_t offset = j * aLoadSize;
            std::iota(aIndices.begin(), aIndices.end(), offset);
            Value slice = rewriter.create<spirv::VectorShuffleOp>(loc, aDpasType, matARange[blockIdx], matARange[blockIdx], rewriter.getI32ArrayAttr(aIndices));
            matAVec[(m * aCombinedLoadNum + j) * aBlockK + k] = slice;
          }
        }
      }
    } else if(aLoadShape.size() == 2){
      auto aDpasType = VectorType::get({aLoadShape[0], aLoadShape[1] / 2, 2}, elemType);
      for(int i = 0;i<aSize;i++){
        //Value A = rewriter.create<vector::ShapeCastOp>(loc, aDpasType, matARange[i]);
        Value A = urcast(aDpasType, matARange[i])->getResults()[0];
        matAVec[i] = A;
      }
    }
    matARange = ValueRange(matAVec);
    llvm::outs() <<"\n\n[DotOpToXeGPUPattern] matARange[0]: " << matARange[0] <<"\n";

    auto bSize = matBRange.size();
    auto newMatBNum = matBRange.size() * bCombinedLoadNum;
    auto bBlockN = bElemShape[3];
    auto bBlockK = bSize / bElemShape[3];
    llvm::outs() <<"\n\n[DotOpToXeGPUPattern]matBRange.size(): " << matBRange.size() <<"\n";
    llvm::outs() <<"\n\n[DotOpToXeGPUPattern]bCombinedLoadNum: " << bCombinedLoadNum <<"\n";
    llvm::outs() <<"\n\n[DotOpToXeGPUPattern]bBlockN: " << bBlockN <<"\n";
    llvm::outs() <<"\n\n[DotOpToXeGPUPattern]bBlockK: " << bBlockK <<"\n";
    SmallVector<Value> spiltB(newMatBNum);
    SmallVector<int32_t, 2> bIndices(bLoadSize);
    for(int k = 0; k < bBlockK; k++){
      for(int n = 0; n < bBlockN; n++){
        for(int j = 0; j < bCombinedLoadNum; j++){
          auto blockIdx = k * bBlockN + n;
          uint64_t offset = j * bLoadSize;
          std::iota(bIndices.begin(), bIndices.end(), offset);
          Value slice = rewriter.create<spirv::VectorShuffleOp>(loc, bDpasType, matBRange[blockIdx], matBRange[blockIdx], rewriter.getI32ArrayAttr(bIndices));
          spiltB[(k * bCombinedLoadNum + j) * bBlockN + n] = slice;
          // spiltB[j * matBRange.size() + i] = slice;
        }
      }
    }
    matBRange = ValueRange(spiltB);

    auto size = matARange.size();

    llvm::outs() <<"\n\n[DotOpToXeGPUPattern]M: " << M 
                                         <<" N: " << N
                                         <<" K: " << K <<"\n";
    SmallVector<Value> results(M * N);

    Type resultTy = matCRange[0].getType();
    Type resultStructTy = spirv::StructType::get(SmallVector<Type>(4*4, resultTy));

    //todo
    for (int n = 0; n < N; n++) {
      for (int m = 0; m < M; m++) {
        auto matC = matCRange[m * N + n];
        for (int k = 0; k < K; k++) {
          // llvm::outs() <<"\n\n[DotOpToXeGPUPattern] k: " << k <<"\n";
          auto matA = matARange[m * K + k];
          // llvm::outs() <<"\n\n[DotOpToXeGPUPattern] matA: " << matA <<"\n";
          auto matB = matBRange[k * N + n];
          // llvm::outs() <<"\n\n[DotOpToXeGPUPattern] matB: " << matB <<"\n";
          matC = rewriter.create<xegpu::DpasOp>(loc, resultTy, matA, matB, matC);
        }
        results[m * N + n] = matC;
      }
    }

    instBrrier = rewriter.create<xegpu::CompilerHintOp>(loc);

    mlir::ValueRange newResults(results);
    auto resultTys = op->getResultTypes();
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newResults)->getResults();
    newResults = ValueRange(cast);
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

class ExpandDimsOpToXeGPUPattern : public OpConversionPattern<triton::ExpandDimsOp> {
public:
  using OpConversionPattern<triton::ExpandDimsOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\n[ExpandDimsOpToXeGPUPattern]\n";
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
    auto mmaFlag = layout.getMmaFlag();
    auto threadShape = layout.getThreadShape();
    auto elemShape = layout.getElemPerThread();
    auto elemStride = layout.getElemStride();

    TritonGPUToXeGPUTypeConverter xeGPUTypeConverter(*context);
    llvm::SmallVector<mlir::Type> resultTypes;

    auto value = op.getResult();
    auto result = xeGPUTypeConverter.convertType(value.getType(), resultTypes);
    auto newType = resultTypes[0];
    llvm::outs()<<"\n\n[ExpandDimsOpToXeGPUPattern]resultTypes.size(): " << resultTypes.size() << "\n";
    llvm::outs()<<"\n\n[ExpandDimsOpToXeGPUPattern]newType: " << newType << "\n";

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
        //Value ret = rewriter.create<vector::ShapeCastOp>(loc, newType, slice);
        // Value ret = urcast(newType, slice)->getResults()[0];
        //Value ret = rewriter.create<vector::BroadcastOp>(loc, newType, slice);
        newValues.push_back(slice);
      }
    }
    llvm::outs()<<"\n\n[ExpandDimsOpToXeGPUPattern]newValues.size(): " << newValues.size() << "\n";
    llvm::outs()<<"\n\n[ExpandDimsOpToXeGPUPattern]newValues[0]: " << newValues[0] << "\n";

    mlir::ValueRange newValueRange(newValues);
    auto resultTys = op->getResultTypes();
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newValueRange)->getResults();
   
    newValueRange = ValueRange(cast);
    rewriter.replaceOp(op, newValueRange);
    return success();
  }
};

class BroadcastOpToXeGPUPattern : public OpConversionPattern<triton::BroadcastOp> {
public:
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\n[BroadCastOpToXeGPUPattern]\n";
    auto context = rewriter.getContext();
    Location loc = op->getLoc();
    ValueRange src(adaptor.getSrc());
    llvm::outs()<<"\n\n[BroadCastOpToXeGPUPattern]src: "<<src[0]<<"\n";

    if(auto *parentOp = src[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        src = (&castOp)->getInputs();
      }
    }

    TritonGPUToXeGPUTypeConverter xeGPUTypeConverter(*context);
    llvm::SmallVector<mlir::Type> resultTypes;

    auto value = op.getResult();
    auto result = xeGPUTypeConverter.convertType(value.getType(), resultTypes);
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
      if(isVevtorSrc && (srcShape.size() > 1)){
        // in = rewriter.create<vector::ShapeCastOp>(loc, castType, in);
        ret = rewriter.create<spirv::UndefOp>(loc, newType);

        SmallVector<int32_t> indices(nElem, 0);
        for(int d0=0;d0<shape[0];d0++){
          for(int d1=0;d1<shape[1];d1++){
            indices[d0*shape[1]+d1] = d0;
          }
        }
        llvm::outs()<<"\n\n[SplatOpToXeGPUPattern]indices.size(): "<<indices.size()<<"\n";
        ret = rewriter.create<spirv::VectorShuffleOp>(
              loc, newType, in, in, rewriter.getI32ArrayAttr(indices));
      } else{
        ret = rewriter.create<vector::BroadcastOp>(loc, newType, in);
      }

      newValues.push_back(ret);
    }

    llvm::outs()<<"\n\n[BroadCastOpToXeGPUPattern]newValues.size(): " << newValues.size() << "\n";
    llvm::outs()<<"\n\n[BroadCastOpToXeGPUPattern]newValues[0]: " << newValues[0] << "\n";

    mlir::ValueRange newValueRange(newValues);
    auto resultTys = op->getResultTypes();
    auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newValueRange)->getResults();
    newValueRange = ValueRange(cast);
    rewriter.replaceOp(op, newValueRange);

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
               DotOpToXeGPUPattern, AdvanceOpToXeGPUPattern,
               ExpandDimsOpToXeGPUPattern, BroadcastOpToXeGPUPattern>(
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