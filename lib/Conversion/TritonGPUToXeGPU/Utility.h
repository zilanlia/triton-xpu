#ifndef TRITON_CONVERSION_TRITONGPU_TO_SPIRV_UTILITY_H
#define TRITON_CONVERSION_TRITONGPU_TO_SPIRV_UTILITY_H

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"

// Shortcuts for some commonly used LLVM ops to keep code simple and intuitive
// Operators
#define urcast(...) rewriter.create<mlir::UnrealizedConversionCastOp>(loc, __VA_ARGS__)
#define inttoptr(...) rewriter.create<spirv::ConvertUToPtrOp>(loc, __VA_ARGS__)
#define ptrtoint(...) rewriter.create<spirv::ConvertPtrToUOp>(loc, __VA_ARGS__)
#define zext(...) rewriter.create<spirv::UConvertOp>(loc, __VA_ARGS__)
#define sext(...) rewriter.create<spirv::SConvertOp>(loc, __VA_ARGS__)
#define fpext(...) rewriter.create<spirv::FConvertOp>(loc, __VA_ARGS__)
#define udiv(...) rewriter.create<spirv::UDivOp>(loc, __VA_ARGS__)
#define urem(...) rewriter.create<spirv::UModOp>(loc, __VA_ARGS__)
#define add(...) rewriter.create<spirv::IAddOp>(loc, __VA_ARGS__)
#define sub(...) rewriter.create<spirv::ISubOp>(loc, __VA_ARGS__)
#define fadd(...) rewriter.create<spirv::FAddOp>(loc, __VA_ARGS__)
#define mul(...) rewriter.create<spirv::IMulOp>(loc, __VA_ARGS__)
#define fmul(...) rewriter.create<spirv::FMulOp>(loc, __VA_ARGS__)
#define smax(...) rewriter.create<spirv::CLSMaxOp>(loc, __VA_ARGS__)
#define umax(...) rewriter.create<spirv::CLUMaxOp>(loc, __VA_ARGS__)
#define fmax(...) rewriter.create<spirv::CLFMaxOp>(loc, __VA_ARGS__)
#define smin(...) rewriter.create<spirv::CLSMinOp>(loc, __VA_ARGS__)
#define umin(...) rewriter.create<spirv::CLUMinOp>(loc, __VA_ARGS__)
#define fmin(...) rewriter.create<spirv::CLFMinOp>(loc, __VA_ARGS__)
#define logic_and(...) rewriter.create<spirv::LogicalAndOp>(loc, __VA_ARGS__)
#define and_(...) rewriter.create<spirv::BitwiseAndOp>(loc, __VA_ARGS__)
#define xor_(...) rewriter.create<spirv::BitwiseXorOp>(loc, __VA_ARGS__)
#define logic_or(...) rewriter.create<spirv::LogicalOrOp>(loc, __VA_ARGS__)
#define or_(...) rewriter.create<spirv::BitwiseOrOp>(loc, __VA_ARGS__)
#define bitcast(val__, type__)   ({                                       \
    Value srcVal__ = (val__);                                                  \
    Type dstType__ = (type__);                                                 \
    Type srcType__ = srcVal__.getType();                                       \
    Value toVal__ = (srcType__ != dstType__) ?                                 \
       rewriter.create<spirv::BitcastOp>(loc, dstType__, srcVal__) : srcVal__; \
    toVal__;                                                                   \
  })
#define gep(...) rewriter.create<spirv::PtrAccessChainOp>(loc, __VA_ARGS__, ValueRange{})
#define ptr_ty(...) spirv::PointerType::get(__VA_ARGS__)
#define insert_val(...) rewriter.create<spirv::CompositeInsertOp>(loc, __VA_ARGS__)
#define extract_val(...) rewriter.create<spirv::CompositeExtractOp>(loc, __VA_ARGS__)
#define insert_element(...)                                                    \
  rewriter.create<spirv::VectorInsertDynamicOp>(loc, __VA_ARGS__)
#define extract_element(...)                                                   \
  rewriter.create<spirv::VectorExtractDynamicOp>(loc, __VA_ARGS__)
#define load(...) rewriter.create<spirv::LoadOp>(loc, __VA_ARGS__)
#define store(val, ptr) rewriter.create<spirv::StoreOp>(loc, ptr, val)
#define fcmp_ogt(lhs, rhs)                                                     \
  rewriter.create<spirv::FOrdGreaterThanOp>(loc, lhs, rhs)
#define fcmp_olt(lhs, rhs)                                                     \
  rewriter.create<spirv::FOrdLessThanOp>(loc, lhs, rhs)
#define icmp_eq(...)                                                           \
  rewriter.create<spirv::IEqualOp>(loc, __VA_ARGS__)
#define logic_cmp_eq(...)                                                       \
  rewriter.create<spirv::LogicalEqualOp>(loc, __VA_ARGS__)
#define icmp_ne(...)                                                           \
  rewriter.create<spirv::INotEqualOp>(loc, __VA_ARGS__)
#define icmp_slt(...)                                                          \
  rewriter.create<spirv::SLessThanOp>(loc, __VA_ARGS__)
#define icmp_sle(...)                                                          \
  rewriter.create<spirv::SLessThanEqualOp>(loc, __VA_ARGS__)
#define icmp_sgt(...)                                                          \
  rewriter.create<spirv::SGreaterThanOp>(loc, __VA_ARGS__)
#define icmp_sge(...)                                                          \
  rewriter.create<spirv::SGreaterThanEqualOp>(loc, __VA_ARGS__)
#define icmp_ult(...)                                                          \
  rewriter.create<spirv::ULessThanOp>(loc, __VA_ARGS__)
#define icmp_ule(...)                                                          \
  rewriter.create<spirv::ULessThanEqualOp>(loc, __VA_ARGS__)
#define icmp_ugt(...)                                                          \
  rewriter.create<spirv::UGreaterThanOp>(loc, __VA_ARGS__)
#define icmp_uge(...)                                                          \
  rewriter.create<spirv::UGreaterThanEqualOp>(loc, __VA_ARGS__)
#define select(...) rewriter.create<spirv::SelectOp>(loc, __VA_ARGS__)
#define address_of(...) rewriter.create<spirv::AddressOfOp>(loc, __VA_ARGS__)
#define barrier() rewriter.create<mlir::gpu::BarrierOp>(loc)
#define undef(...) rewriter.create<spirv::UndefOp>(loc, __VA_ARGS__)
#define call(...) rewriter.create<spirv::FunctionCallOp>(loc, __VA_ARGS__)
#define joint_matrix_load(...)                                                 \
  rewriter.create<spirv::INTELJointMatrixLoadOp>(loc, __VA_ARGS__)
#define joint_matrix_store(...)                                                \
  rewriter.create<spirv::INTELJointMatrixStoreOp>(loc, __VA_ARGS__)
#define joint_matrix_mad(...)                                                  \
  rewriter.create<spirv::INTELJointMatrixMadOp>(loc, __VA_ARGS__)

#define stage 3

// Types
#define i64_ty rewriter.getIntegerType(64)
#define ui64_ty rewriter.getIntegerType(64, false)
#define i32_ty rewriter.getIntegerType(32)
#define i16_ty rewriter.getIntegerType(16)
#define ui32_ty rewriter.getIntegerType(32, false)
#define f16_ty rewriter.getF16Type()
#define bf16_ty rewriter.getBF16Type()
#define i8_ty rewriter.getIntegerType(8)
#define i1_ty rewriter.getI1Type()
#define f32_ty rewriter.getF32Type()
#define f64_ty rewriter.getF64Type()
#define vec_ty(type, num) VectorType::get(num, type)
#define f32_val(...) spirv::createConstantF32(loc, rewriter, __VA_ARGS__)
#define f64_val(...) spirv::createConstantF64(loc, rewriter, __VA_ARGS__)
#define void_ty(ctx) spirv::VoidType::get(ctx)
#define struct_ty(...) spirv::StructType::get(__VA_ARGS__)
#define array_ty(elemTy, count) spirv::LLVMArrayType::get(elemTy, count)
#define joint_matrix_ty(elemTy, scope, rows, cols, layout)                     \
  spirv::JointMatrixINTELType::get(elemTy, scope, rows, cols, layout);

//Types
#define f32Type mlir::FloatType::getF32(context)
#define f16Type mlir::FloatType::getF16(context)
#define bf16Type mlir::FloatType::getBF16(context)
#define i1Type mlir::IntegerType::get(context, 1)
#define i8Type mlir::IntegerType::get(context, 8)
#define i16Type mlir::IntegerType::get(context, 16)
#define i32Type mlir::IntegerType::get(context, 32)
#define i64Type mlir::IntegerType::get(context, 64)
#define v1i64Type mlir::VectorType::get(1, i64Type)
#define v4i64Type mlir::VectorType::get(4, i64Type)
#define v8i32Type mlir::VectorType::get(8, i32Type)
#define v16i32Type mlir::VectorType::get(16, i32Type)
#define v32i32Type mlir::VectorType::get(32, i32Type)
#define v32f32Type mlir::VectorType::get(32, f32Type)
#define v64f16Type mlir::VectorType::get(64, f16Type)
#define v64i32Type mlir::VectorType::get(64, i32Type)
#define v64f32Type mlir::VectorType::get(64, f32Type)
#define fp32PtrType spirv::PointerType::get(f32Type, spirv::StorageClass::CrossWorkgroup)
#define fp16PtrType spirv::PointerType::get(f16Type, spirv::StorageClass::CrossWorkgroup)
#define i16PtrType spirv::PointerType::get(i16Type, spirv::StorageClass::CrossWorkgroup)

// Constants
#define f32_val(...) spirv::createConstantF32(loc, rewriter, __VA_ARGS__)
#define f64_val(...) spirv::createConstantF64(loc, rewriter, __VA_ARGS__)
#define i32_val(value) rewriter.create<spirv::ConstantOp>(loc, i32_ty, IntegerAttr::get(i32_ty, value))
#define int_val(width, val)                                                    \
  spirv::createSPIRVIntegerConstant(rewriter, loc, width, val)
#define idx_val(...)                                                           \
  spirv::createIndexConstant(rewriter, loc, this->getTypeConverter(),           \
                            __VA_ARGS__)
#define tid_val() getThreadId(rewriter, loc)
#define i8_val(value) rewriter.create<spirv::ConstantOp>(loc, i8_ty, rewriter.getI8IntegerAttr(value))
#define i1_val(value) rewriter.create<spirv::ConstantOp>(loc, i1_ty, rewriter.getBoolAttr(value))

// Attributes
#define i32_arr_attr(...) rewriter.getI32ArrayAttr({__VA_ARGS__})
#define i64_arr_attr(...) rewriter.getI64ArrayAttr({__VA_ARGS__})

namespace mlir {
namespace triton {

// Delinearize supposing order is [0, 1, .. , n]
template <typename T>
llvm::SmallVector<T> getMultiDimIndexImpl(T linearIndex,
                                          llvm::ArrayRef<T> shape) {
  // shape: {a, b, c, d}  ->  accMul: {1, a, a*b, a*b*c}
  size_t rank = shape.size();
  T accMul = product(shape.drop_back());
  T linearRemain = linearIndex;
  llvm::SmallVector<T> multiDimIndex(rank);
  for (int i = rank - 1; i >= 0; --i) {
    multiDimIndex[i] = linearRemain / accMul;
    linearRemain = linearRemain % accMul;
    if (i != 0) {
      accMul = accMul / shape[i - 1];
    }
  }
  return multiDimIndex;
}

template <typename T>
llvm::SmallVector<T> getMultiDimIndex(T linearIndex, llvm::ArrayRef<T> shape,
                                      llvm::ArrayRef<unsigned> order) {
  size_t rank = shape.size();
  assert(rank == order.size());
  auto reordered = reorder(shape, order);
  auto reorderedMultiDim = getMultiDimIndexImpl<T>(linearIndex, reordered);
  llvm::SmallVector<T> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

// Linearize supposing order is [0, 1, .. , n]
template <typename T>
T getLinearIndexImpl(llvm::ArrayRef<T> multiDimIndex, llvm::ArrayRef<T> shape) {
  assert(multiDimIndex.size() == shape.size());
  // shape: {a, b, c, d}  ->  accMul: {1, a, a*b, a*b*c}
  size_t rank = shape.size();
  T accMul = product(shape.drop_back());
  T linearIndex = 0;
  for (int i = rank - 1; i >= 0; --i) {
    linearIndex += multiDimIndex[i] * accMul;
    if (i != 0) {
      accMul = accMul / shape[i - 1];
    }
  }
  return linearIndex;
}

template <typename T>
T getLinearIndex(llvm::ArrayRef<T> multiDimIndex, llvm::ArrayRef<T> shape,
                 llvm::ArrayRef<unsigned> order) {
  assert(shape.size() == order.size());
  return getLinearIndexImpl<T>(reorder(multiDimIndex, order),
                               reorder(shape, order));
}

} // namespace triton

namespace spirv {

Value createConstantI32(Location loc, PatternRewriter &rewriter, int32_t v);

/// Create a 32-bit float constant.
Value createConstantF32(Location loc, PatternRewriter &rewriter, float v);

/// Create a 64-bit float constant.
Value createConstantF64(Location loc, PatternRewriter &rewriter, float v);

/// Create an index type constant.
Value createIndexConstant(OpBuilder &builder, Location loc,
                          TypeConverter *converter, int64_t value);

/// Create an integer constant of \param width bits.
Value createSPIRVIntegerConstant(OpBuilder &builder, Location loc, short width,
                                int64_t value);

/// Helper function to get strides from a given shape and its order
SmallVector<Value>
getStridesFromShapeAndOrder(ArrayRef<int64_t> shape, ArrayRef<unsigned> order,
                            Location loc, ConversionPatternRewriter &rewriter);
struct SharedMemoryObject {
  Value base; // i32 ptr. The start address of the shared memory object.
  // We need to store strides as Values but not integers because the
  // extract_slice instruction can take a slice at arbitrary offsets.
  // Take $a[16:32, 16:32] as an example, though we know the stride of $a[0] is
  // 32, we need to let the instruction that uses $a to be aware of that.
  // Otherwise, when we use $a, we only know that the shape of $a is 16x16. If
  // we store strides into an attribute array of integers, the information
  // cannot pass through block argument assignment because attributes are
  // associated with operations but not Values.
  // TODO(Keren): We may need to figure out a way to store strides as integers
  // if we want to support more optimizations.
  SmallVector<Value>
      strides; // i32 int. The strides of the shared memory object.
  SmallVector<Value> offsets; // i32 int. The offsets of the shared memory
  // objects from the originally allocated object.

  SharedMemoryObject(Value base, ArrayRef<Value> strides,
                     ArrayRef<Value> offsets)
      : base(base), strides(strides.begin(), strides.end()),
        offsets(offsets.begin(), offsets.end()) {}

  SharedMemoryObject(Value base, ArrayRef<int64_t> shape,
                     ArrayRef<unsigned> order, Location loc,
                     ConversionPatternRewriter &rewriter)
      : base(base) {
    strides = getStridesFromShapeAndOrder(shape, order, loc, rewriter);
    offsets.append(order.size(), i32_val(0));
  }

  SmallVector<Value> getElems() const {
    SmallVector<Value> elems;
    elems.push_back(base);
    elems.append(strides.begin(), strides.end());
    elems.append(offsets.begin(), offsets.end());
    return elems;
  }

  SmallVector<Type> getTypes() const {
    SmallVector<Type> types;
    types.push_back(base.getType());
    types.append(strides.size(), IntegerType::get(base.getContext(), 32));
    types.append(offsets.size(), IntegerType::get(base.getContext(), 32));
    return types;
  }

  Value getCSwizzleOffset(int order) const {
    assert(order >= 0 && order < strides.size());
    return offsets[order];
  }

  Value getBaseBeforeSwizzle(int order, Location loc,
                             ConversionPatternRewriter &rewriter) const {
    Value cSwizzleOffset = getCSwizzleOffset(order);
    Value offset = sub(i32_val(0), cSwizzleOffset);
    Type type = base.getType();
    return gep(type, base, offset);
  }
};

SharedMemoryObject
getSharedMemoryObjectFromStruct(Location loc, Value llvmStruct,
                                ConversionPatternRewriter &rewriter);

void storeShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                  Value val, Value pred);

Value shflSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
               int i);

Value addStringToModule(Location loc, ConversionPatternRewriter &rewriter,
                        StringRef key, StringRef content);

} // namespace spirv
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

/// Mapping between SPIR-V storage classes to Triton memory spaces.
///
#define STORAGE_SPACE_MAP_LIST(MAP_FN)                                         \
  MAP_FN(spirv::StorageClass::CrossWorkgroup, 1)                               \
  MAP_FN(spirv::StorageClass::Workgroup, 3)                                    \

#if 0
MAP_FN(spirv::StorageClass::StorageBuffer, 0)                                \
  MAP_FN(spirv::StorageClass::Uniform, 4)                                      \
  MAP_FN(spirv::StorageClass::Private, 5)                                      \
  MAP_FN(spirv::StorageClass::Function, 6)                                     \
  MAP_FN(spirv::StorageClass::PushConstant, 7)                                 \
  MAP_FN(spirv::StorageClass::UniformConstant, 8)                              \
  MAP_FN(spirv::StorageClass::Input, 9)                                        \
  MAP_FN(spirv::StorageClass::Output, 10)                                      \
  MAP_FN(spirv::StorageClass::CrossWorkgroup, 11)                              \
  MAP_FN(spirv::StorageClass::AtomicCounter, 12)                               \
  MAP_FN(spirv::StorageClass::Image, 13)                                       \
  MAP_FN(spirv::StorageClass::CallableDataKHR, 14)                             \
  MAP_FN(spirv::StorageClass::IncomingCallableDataKHR, 15)                     \
  MAP_FN(spirv::StorageClass::RayPayloadKHR, 16)                               \
  MAP_FN(spirv::StorageClass::HitAttributeKHR, 17)                             \
  MAP_FN(spirv::StorageClass::IncomingRayPayloadKHR, 18)                       \
  MAP_FN(spirv::StorageClass::ShaderRecordBufferKHR, 19)                       \
  MAP_FN(spirv::StorageClass::PhysicalStorageBuffer, 20)                       \
  MAP_FN(spirv::StorageClass::CodeSectionINTEL, 21)                            \
  MAP_FN(spirv::StorageClass::DeviceOnlyINTEL, 22)                             \
  MAP_FN(spirv::StorageClass::HostOnlyINTEL, 23)
#endif

enum class LSCDataOrder : int8_t {
  Invalid,
  NonTranspose,
  Transpose
};

enum class LSCCacheHint : int8_t {
  Default,
  Uncached,
  Cached,
  Writeback,
  Writethrough,
  Streaming,
  Readinvalidate
};

#endif
