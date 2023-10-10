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

#include "TritonGPUToXeGPU.h"
#include "triton/Dialect/XeGPU/IR/XeGPUOps.h"
//#include "../TritonGPUToLLVM/Utility.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;
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
    llvm::outs()<<"\n\nGenericOpPattern op: "<<op<<"\n";
    Type retType = this->getTypeConverter()->convertType(op.getType());
    addNamedAttrs(
        rewriter.replaceOpWithNewOp<Op>(op, retType, adaptor.getOperands()),
        adaptor.getAttributes());
    return success();
  }
};

class ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    llvm::outs() << "\n\nretType: " << retType << "\n";
    auto value = adaptor.getValue();

    if (auto denseValue = value.dyn_cast<DenseElementsAttr>()){
      denseValue = denseValue.reshape(retType.cast<ShapedType>());
      llvm::outs() << "\n\ndenseValue: " << denseValue << "\n";
      op.setValueAttr(denseValue);
    }

    auto denseValue = value.dyn_cast<DenseElementsAttr>();
    auto newValue = denseValue.reshape(retType.cast<ShapedType>());
    llvm::outs() << "\n\nnewValue: " << newValue << "\n";

    addNamedAttrs(
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, retType, newValue),
        adaptor.getAttributes());
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
    auto loc = op->getLoc();
    auto context = rewriter.getContext();

    Value desc;
    if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(adaptor.getPtr().getDefiningOp())){
      desc = (&castOp)->getInputs()[0];
    }

    llvm::outs()<<"\n\ntdesc: "<<desc<<"\n";
    auto mask = adaptor.getMask();
    auto value = op.getResult();
    auto newType = this->getTypeConverter()->convertType(value.getType());

    auto L1_hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto L2_hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);
    auto L3_hint = CacheReadHintAttr::get(context, CacheReadHint::CACHED);

    auto vnni = IntegerAttr::get(i32_ty, 0);
    auto transpose = rewriter.getDenseI64ArrayAttr({1, 0});

    Value ret = rewriter.create<xegpu::LoadGatherOp>(loc, newType, desc, mask, IntegerAttr{}, DenseI64ArrayAttr{}, L1_hint, L2_hint, L3_hint);
    llvm::outs()<<"\n\nxegpu::LoadGatherOp: " << ret <<"\n";
    rewriter.replaceOp(op, ret);
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

    Value desc;
    if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(adaptor.getPtr().getDefiningOp())){
      desc = (&castOp)->getInputs()[0];
    }

    auto value = adaptor.getValue();
    auto mask = adaptor.getMask();
    auto newType = value.getType();

    auto L1_hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);
    auto L2_hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);
    auto L3_hint = CacheWriteHintAttr::get(context, CacheWriteHint::WRITE_BACK);

    Value ret = rewriter.create<xegpu::StoreScatterOp>(loc, value, desc, mask, L1_hint, L2_hint, L3_hint).getODSResults(0)[0];
    llvm::outs()<<"\n\nxegpu::StoreGatherOp: " << ret <<"\n";
    rewriter.replaceOp(op, ret);
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

    auto subgroubId = rewriter.create<::mlir::gpu::SubgroupIdOp>(loc, rewriter.getIndexType());
    auto cast = rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{i64_ty}, ValueRange{subgroubId});
    auto sgId = zext(i32_ty, cast.getResult(0));
    auto module = op.getOperation()->getParentOfType<ModuleOp>();
    int sgSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(module);
    auto sgAddr = rewriter.create<arith::MulIOp>(loc, i32_ty, sgId, 
      rewriter.create<arith::ConstantOp>(loc, i32_ty, rewriter.getI32IntegerAttr(sgSize)));
    auto type = mlir::VectorType::get(sgSize, i32Type);

    //segmentation fault
    //Value sgAddrs = rewriter.create<vector::SplatOp>(loc, type, sgAddr);

    //avoid spirv.CompositeConstruct
    DenseElementsAttr constData = DenseElementsAttr::get(v32i32Type, ArrayRef<int>(std::vector<int>(32,0)));
    Value sgAddrs = rewriter.create<spirv::ConstantOp>(loc, v32i32Type, constData);
    for(int i = 0; i < 32; i++){
      Value idx = rewriter.create<spirv::ConstantOp>(loc, i32Type, IntegerAttr::get(i32Type, i));
      sgAddrs = rewriter.create<spirv::VectorInsertDynamicOp>(loc, v32i32Type, sgAddrs, sgAddr, idx);
    }

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
    llvm::outs()<<"\n\nafter MakeRangeOpToXeGPUPattern: \n";
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
        Type originalOffsetType = originalOffset.getType();
        Value originalOffsetVec = rewriter.create<vector::SplatOp>(loc, offsetType, originalOffset);
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
    }
    else{
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
    DenseElementsAttr constData = DenseElementsAttr::get(v32i32Type, ArrayRef<int>(std::vector<int>(32,0)));
    Value ret = rewriter.create<spirv::ConstantOp>(loc, v32i32Type, constData);
    for(int i = 0; i < 32; i++){
      Value idx = rewriter.create<spirv::ConstantOp>(loc, i32Type, IntegerAttr::get(i32Type, i));
      ret = rewriter.create<spirv::VectorInsertDynamicOp>(loc, v32i32Type, ret, src, idx);
    }

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

    return success();
  }
private:
  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
};

void populateTritonGPUToXeGPUPatterns(
    TritonGPUToXeGPUTypeConverter &typeConverter, RewritePatternSet &patterns) {
  llvm::outs()<<"\n\npopulateXeGPUToVCIntrinsicsPatterns\n";
  auto context = patterns.getContext();
  patterns.add<LoadOpToXeGPUPattern, StoreOpToXeGPUPattern,
               MakeRangeOpToXeGPUPattern, AddPtrOpToXeGPUPattern,
               CmpIOpToXeGPUPattern, SplatOpToXeGPUPattern,
               GetProgramIdOpToXeGPUPattern, ReturnOpToXeGPUPattern>(
      typeConverter, context);

  //process arith op
  patterns.add<
      ArithConstantPattern, GenericOpPattern<arith::AddIOp>,
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
      GenericOpPattern<arith::TruncIOp>, GenericOpPattern<arith::TruncFOp>,
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