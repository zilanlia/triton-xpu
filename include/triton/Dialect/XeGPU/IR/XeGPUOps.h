//===- XeGPUOps.h - XeGPU dialect  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the imex Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _XEGPU_OPS_H_INCLUDED_
#define _XEGPU_OPS_H_INCLUDED_

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/CopyOpInterface.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/ShapedOpInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {

/// Return the list of Range (i.e. offset, size, stride). Each Range
/// entry contains either the dynamic value or a ConstantIndexOp constructed
/// with `b` at location `loc`.
SmallVector<Range, 8> getOrCreateRanges(OffsetSizeAndStrideOpInterface op,
                                        OpBuilder &b, Location loc);

} // namespace mlir

namespace mlir {
namespace triton {
namespace xegpu {

class TensorDescType;

} // namespace xegpu
} // namespace triton
} // namespace mlir 

namespace mlir {
namespace triton {
namespace xegpu {

class BaseTensorDescType: public mlir::Type, public mlir::ShapedType::Trait<BaseTensorDescType> {
public:
  using Type::Type;

  /// Returns the element type of this tensor type.
  mlir::Type getElementType() const;

  /// Returns if this type is ranked, i.e. it has a known number of dimensions.
  bool hasRank() const;

  /// Returns the shape of this tensor type.
  llvm::ArrayRef<int64_t> getShape() const;

  /// Clone this type with the given shape and element type. If the
  /// provided shape is `None`, the current shape of the type is used.
  BaseTensorDescType cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                     mlir::Type elementType) const;

  /// Return true if the specified element type is ok in a tensor.
  static bool isValidElementType(Type type);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Allow implicit conversion to ShapedType.
  operator mlir::ShapedType() const { return cast<mlir::ShapedType>(); }
};

} // namespace xegpu
} // namespace triton
} // namespace mlir

#include <triton/Dialect/XeGPU/IR/XeGPUOpsDialect.h.inc>
#include <triton/Dialect/XeGPU/IR/XeGPUOpsEnums.h.inc>
#define GET_ATTRDEF_CLASSES
#include <triton/Dialect/XeGPU/IR/XeGPUOpsAttrs.h.inc>
#define GET_TYPEDEF_CLASSES
#include <triton/Dialect/XeGPU/IR/XeGPUOpsTypes.h.inc>
#define GET_OP_CLASSES
#include <triton/Dialect/XeGPU/IR/XeGPUOps.h.inc>

#endif // _XeGPU_OPS_H_INCLUDED_
