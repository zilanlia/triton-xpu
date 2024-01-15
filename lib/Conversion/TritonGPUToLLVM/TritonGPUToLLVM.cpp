#include "TritonGPUToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getSRegValue;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

struct ReturnOpConversion : public ConvertOpToLLVMPattern<triton::ReturnOp> {
  using ConvertOpToLLVMPattern<triton::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numArguments = op.getNumOperands();

    // Currently, Triton kernel function always return nothing.
    // TODO(Superjomn) add support for non-inline device function
    if (numArguments > 0) {
      return rewriter.notifyMatchFailure(
          op, "Only kernel function with nothing returned is supported.");
    }

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), ValueRange(),
                                                op->getAttrs());
    return success();
  }
};

struct BroadcastOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::BroadcastOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::BroadcastOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Following the order of indices in the legacy code, a broadcast of:
    //   [s(0), s(1) ... s(k-1),    1, s(k+1), s(k+2) ... s(n-1)]
    // =>
    //   [s(0), s(1) ... s(k-1), s(k), s(k+1), s(k+2) ... s(n-1)]
    //
    // logically maps to a broadcast within a thread's scope:
    //   [cta(0)..cta(k-1),     1,cta(k+1)..cta(n-1),spt(0)..spt(k-1),
    //   1,spt(k+1)..spt(n-1)]
    // =>
    //   [cta(0)..cta(k-1),cta(k),cta(k+1)..cta(n-1),spt(0)..spt(k-1),spt(k),spt(k+1)..spt(n-1)]
    //
    // regardless of the order of the layout
    //
    Location loc = op->getLoc();
    Value src = adaptor.getSrc();
    Value result = op.getResult();
    auto srcTy = op.getSrc().getType().cast<RankedTensorType>();
    auto resultTy = result.getType().cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding();
    auto resultLayout = resultTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto resultShape = resultTy.getShape();
    unsigned rank = srcTy.getRank();

    assert(rank == resultTy.getRank());
    auto order = triton::gpu::getOrder(srcLayout);
    auto srcOffsets = emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultTy);
    SmallVector<Value> srcVals =
        getTypeConverter()->unpackLLElements(loc, src, rewriter, srcTy);

    DenseMap<SmallVector<unsigned>, Value, SmallVectorKeyInfo> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); i++) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }

    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      for (size_t j = 0; j < srcShape.size(); j++)
        if (srcShape[j] == 1)
          offset[j] = 0;
      resultVals.push_back(srcValues.lookup(offset));
    }

    Value resultStruct =
        getTypeConverter()->packLLElements(loc, resultVals, rewriter, resultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct PrintOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::PrintOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::PrintOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    SmallVector<Value, 16> operands;
    for (size_t i = 0; i < op.getNumOperands(); i++) {
      auto sub_operands = getTypeConverter()->unpackLLElements(
          loc, adaptor.getOperands()[i], rewriter, op.getOperand(i).getType());
      for (auto elem : sub_operands) {
        operands.push_back(elem);
      }
    }
    std::string formatStr;
    llvm::raw_string_ostream os(formatStr);
    os << op.getPrefix();
    if (!operands.empty()) {
      os << getFormatSubstr(operands[0]);
    }

    for (size_t i = 1; i < operands.size(); ++i) {
      os << ", " << getFormatSubstr(operands[i]);
    }
    llPrintf(formatStr, operands, rewriter);
    rewriter.eraseOp(op);
    return success();
  }

  std::string getFormatSubstr(Value value) const {
    Type type = value.getType();
    if (type.isa<LLVM::LLVMPointerType>()) {
      return "%p";
    } else if (type.isBF16() || type.isF16() || type.isF32() || type.isF64()) {
      return "%f";
    } else if (type.isSignedInteger()) {
      if (type.getIntOrFloatBitWidth() == 64)
        return "%lli";
      else
        return "%i";
    } else if (type.isUnsignedInteger() || type.isSignlessInteger()) {
      if (type.getIntOrFloatBitWidth() == 64)
        return "%llu";
      else
        return "%u";
    }
    assert(false && "not supported type");
    return "";
  }

  // declare vprintf(i8*, i8*) as external function
  static LLVM::LLVMFuncOp
  getVprintfDeclaration(ConversionPatternRewriter &rewriter) {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    StringRef funcName("vprintf");
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);

    auto *context = rewriter.getContext();

    SmallVector<Type> argsType{ptr_ty(IntegerType::get(context, 8)),
                               ptr_ty(IntegerType::get(context, 8))};
    auto funcType = LLVM::LLVMFunctionType::get(i32_ty, argsType);

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context), funcName,
                                             funcType);
  }

  // extend integer to int32, extend float to float64
  // this comes from vprintf alignment requirements.
  static std::pair<Type, Value>
  promoteValue(ConversionPatternRewriter &rewriter, Value value) {
    auto *context = rewriter.getContext();
    auto type = value.getType();
    Value newOp = value;
    Type newType = type;
    auto loc = UnknownLoc::get(context);

    bool bUnsigned = type.isUnsignedInteger();
    if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
      if (bUnsigned) {
        newType = ui32_ty;
        newOp = zext(newType, value);
      } else {
        newType = i32_ty;
        newOp = sext(newType, value);
      }
    } else if (type.isBF16() || type.isF16() || type.isF32()) {
      newType = f64_ty;
      newOp = fpext(newType, value);
    }

    return {newType, newOp};
  }

  static void llPrintf(StringRef msg, ValueRange args,
                       ConversionPatternRewriter &rewriter) {
    assert(!msg.empty() && "printf with empty string not support");
    Type int8Ptr = ptr_ty(i8_ty);

    auto *ctx = rewriter.getContext();
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    auto funcOp = getVprintfDeclaration(rewriter);
    auto loc = UnknownLoc::get(ctx);

    Value one = i32_val(1);
    Value zero = i32_val(0);

    llvm::SmallString<64> msgNewline(msg);
    msgNewline.push_back('\n');
    msgNewline.push_back('\0');
    Value prefixString =
        LLVM::addStringToModule(loc, rewriter, "printfFormat_", msgNewline);
    Value bufferPtr = null(int8Ptr);

    SmallVector<Value, 16> newArgs;
    if (args.size() >= 1) {
      SmallVector<Type> argTypes;
      for (auto arg : args) {
        Type newType;
        Value newArg;
        std::tie(newType, newArg) = promoteValue(rewriter, arg);
        argTypes.push_back(newType);
        newArgs.push_back(newArg);
      }

      Type structTy = LLVM::LLVMStructType::getLiteral(ctx, argTypes);
      auto allocated =
          rewriter.create<LLVM::AllocaOp>(loc, ptr_ty(structTy), one,
                                          /*alignment=*/0);

      for (const auto &entry : llvm::enumerate(newArgs)) {
        auto index = i32_val(entry.index());
        auto fieldPtr = gep(ptr_ty(argTypes[entry.index()]), allocated,
                            ArrayRef<Value>{zero, index});
        store(entry.value(), fieldPtr);
      }
      bufferPtr = bitcast(allocated, int8Ptr);
    }

    SmallVector<Value> operands{prefixString, bufferPtr};
    call(funcOp, operands);
  }
};

struct AssertOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AssertOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AssertOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto elems = getTypeConverter()->unpackLLElements(
        loc, adaptor.getCondition(), rewriter, op.getCondition().getType());
    auto elemTy = elems[0].getType();
    Value condition = int_val(elemTy.getIntOrFloatBitWidth(), 0);
    for (auto elem : elems) {
      if (elemTy.isSignedInteger() || elemTy.isSignlessInteger()) {
        condition =
            or_(condition,
                icmp_eq(elem, rewriter.create<LLVM::ConstantOp>(
                                  loc, elemTy, rewriter.getZeroAttr(elemTy))));
      } else {
        assert(false && "Unsupported type for assert");
        return failure();
      }
    }
    llAssert(op, condition, adaptor.getMessage(), adaptor.getFile(),
             adaptor.getFunc(), adaptor.getLine(), rewriter);
    rewriter.eraseOp(op);
    return success();
  }

  // op: the op at which the assert is inserted. Unlike printf, we need to
  // know about the op to split the block.
  static void llAssert(Operation *op, Value condition, StringRef message,
                       StringRef file, StringRef func, int line,
                       ConversionPatternRewriter &rewriter) {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    auto ctx = rewriter.getContext();
    auto loc = op->getLoc();

    // #block1
    // if (condition) {
    //   #block2
    //   __assertfail(message);
    // }
    // #block3
    Block *prevBlock = op->getBlock();
    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);

    auto funcOp = getAssertfailDeclaration(rewriter);
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    Value messageString =
        LLVM::addStringToModule(loc, rewriter, "assertMessage_", message);
    Value fileString =
        LLVM::addStringToModule(loc, rewriter, "assertFile_", file);
    Value funcString =
        LLVM::addStringToModule(loc, rewriter, "assertFunc_", func);
    Value lineNumber = i32_val(line);
    Value charSize = int_val(sizeof(size_t) * 8, sizeof(char));

    SmallVector<Value> operands = {messageString, fileString, lineNumber,
                                   funcString, charSize};
    auto ret = call(funcOp, operands);

    // Split a block after the call.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<cf::BranchOp>(loc, thenBlock);
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<cf::CondBranchOp>(loc, condition, ifBlock, thenBlock);
  }

  static LLVM::LLVMFuncOp
  getAssertfailDeclaration(ConversionPatternRewriter &rewriter) {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    StringRef funcName("__assertfail");
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);

    // void __assert_fail(const char * assertion, const char * file, unsigned
    // int line, const char * function);
    auto *ctx = rewriter.getContext();
    SmallVector<Type> argsType{ptr_ty(i8_ty), ptr_ty(i8_ty), i32_ty,
                               ptr_ty(i8_ty),
                               rewriter.getIntegerType(sizeof(size_t) * 8)};
    auto funcType = LLVM::LLVMFunctionType::get(void_ty(ctx), argsType);

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                             funcType);
  }
};

struct MakeRangeOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::MakeRangeOp> {

  MakeRangeOpConversion(
      TritonGPUToLLVMTypeConverter &converter,
      ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::MakeRangeOp>(
            converter, indexCacheInfo, benefit) {}

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto rankedTy = op.getResult().getType().cast<RankedTensorType>();
    auto shape = rankedTy.getShape();
    auto layout = rankedTy.getEncoding();

    auto elemTy = rankedTy.getElementType();
    assert(elemTy.isInteger(32));
    Value start = createIndexAttrConstant(rewriter, loc, elemTy, op.getStart());
    auto idxs = emitIndices(loc, rewriter, layout, rankedTy);
    unsigned elems = idxs.size();
    SmallVector<Value> retVals(elems);
    // TODO: slice layout has more elements than expected.
    // Unexpected behavior for make range, but generally OK when followed by
    // expand dims + broadcast. very weird behavior otherwise potentially.
    for (const auto &multiDim : llvm::enumerate(idxs)) {
      assert(multiDim.value().size() == 1);
      retVals[multiDim.index()] = add(multiDim.value()[0], start);
    }
    Value result =
        getTypeConverter()->packLLElements(loc, retVals, rewriter, rankedTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct GetProgramIdOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::GetProgramIdOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::GetProgramIdOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // It is not easy to get the compute capability here, so we use numCTAs to
    // decide the semantic of GetProgramIdOp. If numCTAs = 1, then
    // GetProgramIdOp is converted to "%ctaid", otherwise it is converted to
    // "%clusterid".
    // auto moduleOp = op->getParentOfType<ModuleOp>();
    // assert(moduleOp && "Parent ModuleOp not found for GetProgramIdOp");
    // int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Location loc = op->getLoc();
    // assert(op.getAxisAsInt() < 3);
    // std::string sreg = numCTAs == 1 ? "%ctaid." : "%clusterid.";
    // sreg.append(1, 'x' + op.getAxisAsInt()); // 0 -> 'x', 1 -> 'y', 2 -> 'z'

    // Value programId = getSRegValue(rewriter, loc, sreg);
    // rewriter.replaceOp(op, programId);

    ::llvm::StringRef funcName = "llvm.pisa.groupid.x";
    Value gid = rewriter.create<LLVM::CallIntrinsicOp>(loc, i32_ty, funcName, ::mlir::ValueRange{}).getResults()[0];
    rewriter.replaceOp(op, gid);

    return success();
  }
};

struct GetNumProgramsOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // It is not easy to get the compute capability here, so we use numCTAs to
    // decide the semantic of GetNumProgramsOp. If numCTAs = 1, then
    // GetNumProgramsOp is converted to "%nctaid", otherwise it is converted to
    // "%nclusterid".
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for GetProgramIdOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Location loc = op->getLoc();
    assert(op.getAxis() < 3);
    std::string sreg = numCTAs == 1 ? "%nctaid." : "%nclusterid.";
    sreg.append(1, 'x' + op.getAxis()); // 0 -> 'x', 1 -> 'y', 2 -> 'z'

    Value numPrograms = getSRegValue(rewriter, loc, sreg);
    rewriter.replaceOp(op, numPrograms);
    return success();
  }
};

// TODO[goostavz]: GetThreadIdOp/GetClusterCTAIdOp is a temporary solution
// before async dialect is done. These concepts should appear in ttgpu
// level, and they are planned to be deprecated along with ttgpu.mbarrier_xxx
// ops.
struct GetThreadIdOpConversion : public ConvertTritonGPUOpToLLVMPattern<
                                     triton::nvidia_gpu::GetThreadIdOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::GetThreadIdOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::GetThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, getThreadId(rewriter, op->getLoc()));
    return success();
  }
};

struct GetClusterCTAIdOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::nvidia_gpu::GetClusterCTAIdOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::GetClusterCTAIdOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::GetClusterCTAIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, getClusterCTAId(rewriter, op->getLoc()));
    return success();
  }
};

struct AddPtrOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AddPtrOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AddPtrOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType();
    auto offsetTy = op.getOffset().getType();
    auto ptrTy = op.getPtr().getType();
    auto resultTensorTy = resultTy.dyn_cast<RankedTensorType>();
    if (resultTensorTy) {
      unsigned elems = getTotalElemsPerThread(resultTy);
      Type elemTy =
          getTypeConverter()->convertType(resultTensorTy.getElementType());
      auto ptrs = getTypeConverter()->unpackLLElements(loc, adaptor.getPtr(),
                                                       rewriter, ptrTy);
      auto offsets = getTypeConverter()->unpackLLElements(
          loc, adaptor.getOffset(), rewriter, offsetTy);
      SmallVector<Value> resultVals(elems);
      for (unsigned i = 0; i < elems; ++i) {
        resultVals[i] = gep(elemTy, ptrs[i], offsets[i]);
      }
      Value view = getTypeConverter()->packLLElements(loc, resultVals, rewriter,
                                                      resultTy);
      rewriter.replaceOp(op, view);
    } else {
      assert(resultTy.isa<triton::PointerType>());
      Type llResultTy = getTypeConverter()->convertType(resultTy);
      Value result = gep(llResultTy, adaptor.getPtr(), adaptor.getOffset());
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

struct AllocTensorOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::AllocTensorOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AllocTensorOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AllocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getResult());
    auto resultTy = op.getType().dyn_cast<RankedTensorType>();
    auto llvmElemTy =
        getTypeConverter()->convertType(resultTy.getElementType());
    auto elemPtrTy = ptr_ty(llvmElemTy, 3);
    smemBase = bitcast(smemBase, elemPtrTy);
    auto sharedLayout = resultTy.getEncoding().cast<SharedEncodingAttr>();
    auto order = sharedLayout.getOrder();
    // Workaround for 3D tensors
    // TODO: we need to modify the pipeline pass to give a proper shared
    // encoding to 3D tensors
    SmallVector<unsigned> newOrder;
    if (resultTy.getShape().size() == 3)
      newOrder = {1 + order[0], 1 + order[1], 0};
    else
      newOrder = SmallVector<unsigned>(order.begin(), order.end());

    auto shapePerCTA = getShapePerCTA(sharedLayout, resultTy.getShape());
    auto smemObj =
        SharedMemoryObject(smemBase, shapePerCTA, newOrder, loc, rewriter);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct ExtractSliceOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::ExtractSliceOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::ExtractSliceOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = extract_slice %src[%offsets]
    Location loc = op->getLoc();
    auto srcTy = op.getSource().getType().dyn_cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    assert(srcLayout && "Unexpected resultLayout in ExtractSliceOpConversion");
    assert(op.hasUnitStride() &&
           "Only unit stride supported by ExtractSliceOpConversion");

    // newBase = base + offset
    // Triton supports either static and dynamic offsets
    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, adaptor.getSource(), rewriter);
    SmallVector<Value, 4> opOffsetVals;
    SmallVector<Value, 4> offsetVals;
    auto mixedOffsets = op.getMixedOffsets();
    for (auto i = 0; i < mixedOffsets.size(); ++i) {
      if (op.isDynamicOffset(i))
        opOffsetVals.emplace_back(adaptor.getOffsets()[i]);
      else
        opOffsetVals.emplace_back(i32_val(op.getStaticOffset(i)));
      offsetVals.emplace_back(add(smemObj.offsets[i], opOffsetVals[i]));
    }
    // Compute the offset based on the original strides of the shared memory
    // object
    auto offset = dot(rewriter, loc, opOffsetVals, smemObj.strides);
    // newShape = rank_reduce(shape)
    // Triton only supports static tensor sizes
    SmallVector<Value, 4> strideVals;
    for (auto i = 0; i < op.getStaticSizes().size(); ++i) {
      if (op.getStaticSize(i) == 1) {
        offsetVals.erase(offsetVals.begin() + i);
      } else {
        strideVals.emplace_back(smemObj.strides[i]);
      }
    }

    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto elemPtrTy = ptr_ty(llvmElemTy, 3);
    smemObj = SharedMemoryObject(gep(elemPtrTy, smemObj.base, offset),
                                 strideVals, offsetVals);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct AsyncWaitOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AsyncWaitOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PTXBuilder ptxBuilder;
    auto &asyncWaitOp = *ptxBuilder.create<>("cp.async.wait_group");
    auto num = op->getAttrOfType<IntegerAttr>("num").getInt();
    asyncWaitOp(ptxBuilder.newConstantOperand(num));

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto voidTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, voidTy);

    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncCommitGroupOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::AsyncCommitGroupOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AsyncCommitGroupOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    PTXBuilder ptxBuilder;
    ptxBuilder.create<>("cp.async.commit_group")->operator()();
    ptxBuilder.launch(rewriter, op.getLoc(), void_ty(op.getContext()));
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncBulkWaitOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::AsyncBulkWaitOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AsyncBulkWaitOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncBulkWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PTXBuilder ptxBuilder;
    auto &asyncBulkWaitOp = *ptxBuilder.create<>("cp.async.bulk.wait_group");
    auto num = op->getAttrOfType<IntegerAttr>("num").getInt();
    asyncBulkWaitOp(ptxBuilder.newConstantOperand(num));

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto voidTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, voidTy);

    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncBulkCommitGroupOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::gpu::AsyncBulkCommitGroupOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::AsyncBulkCommitGroupOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncBulkCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    PTXBuilder ptxBuilder;
    ptxBuilder.create<>("cp.async.bulk.commit_group")->operator()();
    ptxBuilder.launch(rewriter, op.getLoc(), void_ty(op.getContext()));
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

namespace mlir {
namespace LLVM {

void vprintf(StringRef msg, ValueRange args,
             ConversionPatternRewriter &rewriter) {
  PrintOpConversion::llPrintf(msg, args, rewriter);
}

void vprintf_array(Value thread, ArrayRef<Value> arr, std::string info,
                   std::string elem_repr, ConversionPatternRewriter &builder) {
  std::string fmt = info + " t-%d ";
  std::vector<Value> new_arr({thread});
  for (int i = 0; i < arr.size(); ++i) {
    fmt += elem_repr + ((i == arr.size() - 1) ? "" : ", ");
    new_arr.push_back(arr[i]);
  }

  vprintf(fmt, new_arr, builder);
}

} // namespace LLVM
} // namespace mlir

void populateTritonGPUToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &moduleAllocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<AddPtrOpConversion>(typeConverter, benefit);
  patterns.add<AllocTensorOpConversion>(typeConverter, moduleAllocation,
                                        benefit);
  patterns.add<AsyncCommitGroupOpConversion>(typeConverter, benefit);
  patterns.add<AsyncWaitOpConversion>(typeConverter, benefit);
  patterns.add<AsyncBulkCommitGroupOpConversion>(typeConverter, benefit);
  patterns.add<AsyncBulkWaitOpConversion>(typeConverter, benefit);
  patterns.add<BroadcastOpConversion>(typeConverter, benefit);
  patterns.add<ExtractSliceOpConversion>(typeConverter, moduleAllocation,
                                         benefit);
  patterns.add<GetProgramIdOpConversion>(typeConverter, benefit);
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  patterns.add<GetThreadIdOpConversion>(typeConverter, benefit);
  patterns.add<GetClusterCTAIdOpConversion>(typeConverter, benefit);
  patterns.add<MakeRangeOpConversion>(typeConverter, indexCacheInfo, benefit);
  patterns.add<ReturnOpConversion>(typeConverter, benefit);
  patterns.add<PrintOpConversion>(typeConverter, benefit);
  patterns.add<AssertOpConversion>(typeConverter, benefit);
}
