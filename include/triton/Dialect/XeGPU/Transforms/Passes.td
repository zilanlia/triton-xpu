//===-- Passes.td - XeGPU pass definition file --------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines passes/transformations of the XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#ifndef XEGPU_PASSES
#define XEGPU_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// XeGPUToSPIRVWithVCIntrinsics pass
//===----------------------------------------------------------------------===//

// def XeGPUToSPIRVWithVCIntrinsics: Pass<"xegpu-spirvWithVCIntrinsics", "::mlir::func::FuncOp"> {
//   let summary = "Add a pass for lowering XeGPU dialect ops to spirv with VC intrinsics";
//   let description = [{
//
//     #### Input invariant
//
//
//     #### Output IR
//
//   }];
//   let constructor = "::imex::createXeGPUToSPIRVWithVCIntrinsicsPass()";
//   let dependentDialects = ["::imex::xegpu::XeGPUDialect"];
//   let options = [];
// }

//===----------------------------------------------------------------------===//
// XeGPUToSPIRVWithJointMatrix pass
//===----------------------------------------------------------------------===//

// def XeGPUToSPIRVWithJointMatrix: Pass<"xegpu-spirvWithJointMatrix", "::mlir::func::FuncOp"> {
//   let summary = "Add a pass for lowering XeGPU dialect ops to spirv with Joint matrix extension";
//   let description = [{
//
//     #### Input invariant
//
//
//     #### Output IR
//
//   }];
//   let constructor = "::imex::createXeGPUToSPIRVWithJointMatrixPass()";
//   let dependentDialects = ["::imex::xegpu::XeGPUDialect"];
//   let options = [];
// }

def XeGPUOptimizeDotOp : Pass<"xegpu-optimize-dotOp", "mlir::ModuleOp"> {
  let summary = "optimize dotOp";

  let description = [{
    optimize dotOp
  }];

  let constructor = "mlir::createXeGPUOptimizeDotOpPass()";

  let dependentDialects = ["mlir::triton::xegpu::XeGPUDialect"];
}

#endif // _XEGPU_PASSES_TD_INCLUDED_
