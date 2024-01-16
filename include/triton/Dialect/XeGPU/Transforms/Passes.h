//===-- Passes.h - XeGPU pass declaration file --------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header file defines prototypes that expose pass constructors for the
/// XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _XEGPU_PASSES_H_INCLUDED_
#define _XEGPU_PASSES_H_INCLUDED_

#include <mlir/Pass/Pass.h>

namespace mlir {
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T> class OperationPass;
class RewritePatternSet;
} // namespace mlir

namespace mlir {

//===----------------------------------------------------------------------===//
/// XeGPU passes.
//===----------------------------------------------------------------------===//

std::unique_ptr<::mlir::Pass> createXeGPUToSPIRVWithVCIntrinsicsPass();
std::unique_ptr<::mlir::Pass> createXeGPUToSPIRVWithJointMatrixPass();

/// Populate the given list with patterns that eliminate XeGPU ops
void populateXeGPUToSPIRVWithVCIntrinsicsPatterns(
    ::mlir::LLVMTypeConverter &converter, ::mlir::RewritePatternSet &patterns);
/// Populate the given list with patterns that eliminate XeGPU ops
void populateXeGPUToSPIRVWithJointMatrixPatterns(
    ::mlir::LLVMTypeConverter &converter, ::mlir::RewritePatternSet &patterns);

std::unique_ptr<Pass> createXeGPUOptimizeDotOpPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include <triton/Dialect/XeGPU/Transforms/Passes.h.inc>

} // namespace mlir

#endif // _XEGPU_PASSES_H_INCLUDED_
