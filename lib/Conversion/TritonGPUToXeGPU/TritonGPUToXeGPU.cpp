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
    Location loc = op.getLoc();
    Type type = op.getType();
    Type retType = this->getTypeConverter()->convertType(op.getType());

    if(isa<spirv::StructType>(retType)){
      SmallVector<Value> retVec;
      auto newType = retType.cast<spirv::StructType>().getElementType(0);
      auto size = retType.cast<spirv::StructType>().getNumElements();
      auto newValues = unpackLLElements(loc, adaptor.getOperands()[0], rewriter);
      for(int i=0;i<size;i++){
          Value ret = rewriter.create<Op>(loc, newType, newValues[i]);
          //llvm::outs() << "\n\nret: " << ret << "\n";
          retVec.push_back(ret);
      }
      auto retStruct = packLLElements(loc, retVec, rewriter, retType);
      //llvm::outs() << "\n\nretStruct: " << retStruct << "\n";
      rewriter.replaceOp(op, retStruct);
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
        llvm::outs()<<"\n\nresultTys: "<<resultTys<<"\n";
        llvm::outs()<<"\n\nnewValueRange: "<<newValueRange[0]<<"\n";
        rewriter.replaceOp(op, newValueRange);

        // auto constValStruct = packLLElements(loc, constVals, rewriter, retType);
        // llvm::outs() << "\n\nconstValStruct: " << constValStruct << "\n";
        // rewriter.replaceOp(op, constValStruct);
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

    llvm::outs() << "\n\nArithTruncFPattern retType: " << retType << "\n";
    llvm::outs() << "\n\nArithConstant retType: " << resultTypes.size() << "\n";

    ValueRange in = adaptor.getIn();


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
          truncFVals.push_back(truncFVal);
        }

        mlir::ValueRange newValueRange(truncFVals);
        auto resultTys = op->getResultTypes();
        newValueRange = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newValueRange)->getResults();
        llvm::outs()<<"\n\nresultTys: "<<resultTys<<"\n";
        llvm::outs()<<"\n\nnewValueRange: "<<newValueRange[0]<<"\n";
        rewriter.replaceOp(op, newValueRange);

      } else {
        addNamedAttrs(
          rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, retType, in[0]),
          adaptor.getAttributes());
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
    llvm::outs()<<"\n\nLoadOpToXeGPUPattern: \n";
    auto loc = op->getLoc();
    auto context = rewriter.getContext();

    ValueRange desc{adaptor.getPtr()};
    int nElem = desc.size();
    auto descType = desc[0].getType();
    llvm::outs()<<"\n\ndesc[0]: "<<desc[0]<<"\n";

    if(auto *parentOp = desc[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        desc = (&castOp)->getInputs();
        descType = desc[0].getType();
        nElem = desc.size();
      }
    }
    auto gatherLoadDesc = desc[0];
    auto value = op.getResult();
    bool isMMA = 0;
    bool isLoadA = 0;
    // llvm::outs()<<"\n\nvalue: "<<value<<"\n";
    for(Operation *user : value.getUsers()){
      if(auto cvtOp = dyn_cast<ConvertLayoutOp>(user)){
        auto ret = cvtOp.getResult();
        if(auto tensorType = ret.getType().cast<RankedTensorType>()){
          if(auto encoding = tensorType.getEncoding().cast<DotOperandEncodingAttr>()){
            isMMA = 1;
            auto idx = encoding.getOpIdx();
            isLoadA = idx == 0;
          }
        }
      }
    }

    TritonGPUToXeGPUTypeConverter xeGPUTypeConverter(*context);
    llvm::SmallVector<mlir::Type> resultTypes;
    auto result = xeGPUTypeConverter.convertType(value.getType(), resultTypes);
    
    // auto newType = xeGPUTypeConverter.convertType(value.getType());
    // llvm::outs()<<"\n\nvalue.getType(): "<<value.getType()<<"\n";
    // llvm::outs()<<"\n\nresultTypes[0]: "<<resultTypes[0]<<"\n";
    // llvm::outs()<<"\n\nresultTypes.size(): "<<resultTypes.size()<<"\n";

    int elemNum;
    Type elemType;
    if(resultTypes.size() > 1){
      // Type newType = resultTypes[0];
      elemType = resultTypes[0];
      elemNum = elemType.dyn_cast<VectorType>().getShape()[0];
      auto vectorType = elemType.dyn_cast<VectorType>();

      if(isLoadA){
        elemType = VectorType::get(ArrayRef<int64_t>{8, 8, 2}, vectorType.getElementType());
      } else {
        elemType = VectorType::get(ArrayRef<int64_t>{8, 16, 2}, vectorType.getElementType());
      }

    } else {
      elemNum = resultTypes[0].dyn_cast<VectorType>().getShape()[0];
      elemType = resultTypes[0];
    }
    llvm::outs()<<"\n\nelemType: "<<elemType<<"\n";
    auto L1_hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto L2_hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto L3_hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    llvm::outs()<<"\n\nnElem: "<<nElem<<"\n";
    if(nElem > 1){
      SmallVector<Value> newValues;
      for(int i = 0; i < nElem; i++){
        Value ret;
        if(isLoadA){
          auto vnni = IntegerAttr::get(i32_ty, 1);
          //llvm::outs()<<"\n\ni: "<<i<<" desc[i]: "<<desc[i]<<"\n";
          ret = rewriter.create<xegpu::LoadNDOp>(loc, elemType, desc[i], vnni, DenseI64ArrayAttr{}, L1_hint, L2_hint, L3_hint);
        } else {
          auto vnni = IntegerAttr::get(i32_ty, 0);
          //llvm::outs()<<"\n\ni: "<<i<<" desc[i]: "<<desc[i]<<"\n";
          ret = rewriter.create<xegpu::LoadNDOp>(loc, elemType, desc[i], vnni, DenseI64ArrayAttr{}, L1_hint, L2_hint, L3_hint);
        }
        
        newValues.push_back(ret);
      }

      mlir::ValueRange newValueRange(newValues);
      auto resultTys = op->getResultTypes();
      newValueRange = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newValueRange)->getResults();
      rewriter.replaceOp(op, newValueRange);
    } else {
      Value mask = adaptor.getMask();
      if(!mask) {
        auto maskType = mlir::VectorType::get(elemNum, i32Type);
        DenseElementsAttr constData = DenseElementsAttr::get(maskType, ArrayRef<int>(std::vector<int>(elemNum, 1)));
        mask = rewriter.create<spirv::ConstantOp>(loc, maskType, constData);
      }

      Value ret = rewriter.create<xegpu::LoadGatherOp>(loc, elemType, gatherLoadDesc, mask, IntegerAttr{}, DenseI64ArrayAttr{}, L1_hint, L2_hint, L3_hint);
      llvm::outs()<<"\n\nxegpu::LoadGatherOp: " << ret <<"\n";
      rewriter.replaceOp(op, ret);
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
    // llvm::outs()<<"\n\ndesc: " << desc[0] <<"\n";
    if(auto *parentOp = desc[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        desc = (&castOp)->getInputs();
      }
    }

    Value gatherStoreDesc = desc[0];

    ValueRange value{adaptor.getValue()};
    // llvm::outs()<<"\n\nvalue: " << value[0] <<"\n";
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
    llvm::outs()<<"\n\nSplatOpToXeGPUPattern: \n";
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

    // DenseElementsAttr constData;
    // if(isa<FloatType>(elemType)){
    //   constData = DenseElementsAttr::get(newType, ArrayRef<float>(std::vector<float>(sgSize, 0.0f)));
    // } else {
    //   constData = DenseElementsAttr::get(newType, ArrayRef<int>(std::vector<int>(sgSize, 0)));
    // }
    // Value ret = rewriter.create<spirv::ConstantOp>(loc, newType, constData);
    // for(int i = 0; i < 32; i++){
    //   Value idx = rewriter.create<spirv::ConstantOp>(loc, i32Type, IntegerAttr::get(i32Type, i));
    //   ret = rewriter.create<spirv::VectorInsertDynamicOp>(loc, newType, ret, src, idx);
    // }

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
    ValueRange src = adaptor.getSrc();
    //llvm::outs()<<"\n\nconvertLayout Op src[0]: "<<src[0]<<"\n";

    if(auto *parentOp = src[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        src = (&castOp)->getInputs();
      }
    }

    ValueRange newValueRange;
    auto resultTys = op->getResultTypes();
    newValueRange = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, src)->getResults();
    rewriter.replaceOp(op, newValueRange);

    //llvm::outs()<<"\n\nconvertLayout newValueRange: "<<newValueRange[0]<<"\n";
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
    auto elemType = tensorType.getElementType();
    auto blockShape = tensorType.getShape();
    auto tensorShape = op.getShape();
    auto tensorStride = op.getStrides();
    auto tensorOffsets = op.getOffsets();

    bool isLoad = 1;
    bool isLoadA = 1; //-1 : not mma ; 0 : load B ; 1:load A

    Value desc = op.getODSResults(0)[0];
    for(Operation *user : desc.getUsers()){
      if(auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(user)){
        for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
          if (desc == forOp.getOperands()[i + 3]) {
            auto descInForOp = forOp.getRegionIterArgs()[i];
            for(Operation *userInLoop : descInForOp.getUsers()){
              if(auto loadOp = llvm::dyn_cast<triton::LoadOp>(userInLoop)){
                auto convertOp = llvm::dyn_cast<ConvertLayoutOp>(*loadOp.getODSResults(0)[0].getUsers().begin());
                auto dotLayout = llvm::dyn_cast<DotOperandEncodingAttr>(convertOp.getResult()
                                                .getType().cast<RankedTensorType>().getEncoding());
                isLoadA = dotLayout.getOpIdx() == 0;
              }
            }
          }
        }
      } else if(auto storeOp = llvm::dyn_cast<StoreOp>(user)){
        isLoad = 0;
      } else if(auto loadOp = llvm::dyn_cast<LoadOp>(user)){
        //todo
      }
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
      if(isLoadA){
        tensorDescType = TensorDescType::get(context, ::llvm::ArrayRef<int64_t>{blockM, blockK}, elemType, 
                                                      MemoryScopeAttr::get(context, MemoryScope::GLOBAL));

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
      } else {
        tensorDescType = TensorDescType::get(context, ::llvm::ArrayRef<int64_t>{blockK, blockN}, elemType, 
                                                      MemoryScopeAttr::get(context, MemoryScope::GLOBAL));
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

    mlir::ValueRange newValueRange(newValues);

    auto resultTys = op->getResultTypes();
    newValueRange = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newValueRange)->getResults();

    rewriter.replaceOp(op, newValueRange);
    return success();
  }
};

class DotOpToXeGPUPattern : public OpConversionPattern<DotOp> {
public:
  using OpConversionPattern<DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\nDotOpToXeGPUPattern: \n";
    Location loc = op->getLoc();

    Value matAValue = adaptor.getA();
    Value matBValue = adaptor.getB();
    Value matCValue = adaptor.getC();

    ValueRange matARange;
    ValueRange matBRange;
    ValueRange matCRange;

    // llvm::outs()<<"\n\nmatARange:"<<matAValue<<" \n";
    // llvm::outs()<<"\n\nmatBRange:"<<matBValue<<" \n";
    // llvm::outs()<<"\n\nmatCRange:"<<matCValue<<" \n";

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

    // llvm::outs()<<"\n\nmatARange:"<<matARange.size()<<" \n";
    // llvm::outs()<<"\n\nmatBRange:"<<matBRange.size()<<" \n";
    // llvm::outs()<<"\n\nmatCRange:"<<matCRange.size()<<" \n";

    auto size = matARange.size();
    SmallVector<Value> results(4*4);

    Type resultTy = matCRange[0].getType();
    Type resultStructTy = spirv::StructType::get(SmallVector<Type>(4*4, resultTy));

    //todo
    for (int m = 0; m < 4; m++) {
      for (int n = 0; n < 4; n++) {
        auto matC = matCRange[m*4+n];
        for (int k = 0; k < 2; k++) {
          auto matA = matARange[m*2+k];
          auto matB = matBRange[k*4+n];
          matC = rewriter.create<xegpu::DpasOp>(loc, resultTy, matA, matB, matC);
        }
        results[m*4+n] = matC;
      }
    }

    mlir::ValueRange newResults(results);
    auto resultTys = op->getResultTypes();
    newResults = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newResults)->getResults();
    rewriter.replaceOp(op, newResults);

    return success();
  }
};

class AdvanceOpToXeGPUPattern : public OpConversionPattern<AdvanceOp> {
public:
  using OpConversionPattern<AdvanceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"\n\nAdvanceOpToXeGPUPattern: \n";
    Location loc = op.getLoc();
    auto context = op.getContext();
    ValueRange tensorDesc = adaptor.getPtr();
    // llvm::outs()<<"\n\ntensorDesc[0]: \n" << tensorDesc[0] <<"\n";
    auto tensorType = tensorDesc[0].getType();
    auto offsets = adaptor.getOffsets();

    if(auto *parentOp = tensorDesc[0].getDefiningOp()){
      if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
        tensorDesc = (&castOp)->getInputs();
        tensorType = tensorDesc[0].getType();
      }
    }

    Type structType = tensorType;

    SmallVector<Value> advancedDescList;
    if(tensorDesc.size() > 1){
      for(int i = 0; i < tensorDesc.size(); i++){
        Value advancedDesc = rewriter.create<UpdateNDOffsetOp>(loc, tensorType, tensorDesc[i], offsets);
        advancedDescList.push_back(advancedDesc);
      }
      mlir::ValueRange newValueRange(advancedDescList);

      auto resultTys = op->getResultTypes();
      newValueRange = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newValueRange)->getResults();
      rewriter.replaceOp(op, newValueRange);
    } else if(isa<spirv::StructType>(tensorType)){
      auto tensorDescStruct = tensorDesc[0];
      auto descList = unpackLLElements(loc, tensorDescStruct, rewriter);
      for(auto desc : descList){
        auto cast = rewriter.create<UnrealizedConversionCastOp>(loc, tensorType, desc).getResult(0);
        Value advancedDesc = rewriter.create<UpdateNDOffsetOp>(loc, tensorType, desc, offsets);
        cast = rewriter.create<UnrealizedConversionCastOp>(loc, i64_ty, advancedDesc).getResult(0);
        advancedDescList.push_back(cast);
        Value advancedDescStruct = packLLElements(loc, advancedDescList, rewriter, structType);
        rewriter.replaceOp(op, advancedDescStruct);
      }
    } else {
    }

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