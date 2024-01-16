//===- GPUToSPIRVPass.cpp -  --------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file extends upstream GPUToSPIRV Pass that converts GPU ops to SPIR-V
/// by adding more conversion patterns like SCF, math and control flow. This
/// pass only converts gpu.func ops inside gpu.module op.
///
//===----------------------------------------------------------------------===//
#include "triton/Conversion/TritonGPUToXeGPU/TritonGPUToXeGPUPass.h"
#include "TypeConverter.h"
#include "triton/Dialect/XeGPU/IR/XeGPUOps.h"

// #include "../PassDetail.h"

#include <mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "TritonGPUToXeGPU.h"
#include "ReduceOpToXeGPU.h"
#include "SCFOpToXeGPU.h"

using namespace mlir;
using namespace mlir::triton;
// using namespace mlir::triton::xegpu;

#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonGPUToXeGPU/Passes.h.inc"

class TritonXeGPUFuncConversionTarget : public ConversionTarget {
public:
  explicit TritonXeGPUFuncConversionTarget(MLIRContext &ctx, TritonGPUToXeGPUTypeConverter& typeConverter)
          : ConversionTarget(ctx) {
    addIllegalOp<mlir::func::FuncOp>();
    addIllegalOp<mlir::triton::FuncOp>();
    addLegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::gpu::GPUFuncOp>();
    addLegalOp<mlir::gpu::GPUModuleOp>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct XeGPUFuncOpConversion : public OpConversionPattern<triton::FuncOp> {
  XeGPUFuncOpConversion(TritonGPUToXeGPUTypeConverter &converter, MLIRContext *context, int numWarps,
                   PatternBenefit benefit)
      : OpConversionPattern<triton::FuncOp>(converter, context, benefit), numWarps(numWarps) {}

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp mod = dyn_cast<ModuleOp>(funcOp->getParentOp());

    if (!mod)
      return failure();

    auto fnType = funcOp.getFunctionType();
    if (fnType.getNumResults() > 1)
      return failure();

    int num_inputs = fnType.getNumInputs();
    TypeConverter::SignatureConversion signatureConverter(num_inputs);
    for (const auto &argType : enumerate(fnType.getInputs())) {
      auto convertedType = getTypeConverter()->convertType(argType.value());
      if (!convertedType)
        return failure();
      signatureConverter.addInputs(argType.index(), convertedType);
    }

    Type resultType;
    if (fnType.getNumResults() == 1) {
      resultType = getTypeConverter()->convertType(fnType.getResult(0));
      if (!resultType)
        return failure();
    }

    // Create gpu.module
    auto gpuModule = rewriter.create<mlir::gpu::GPUModuleOp>(
      funcOp.getLoc(), funcOp.getName()
    );

    // Create gpu.func.
    rewriter.setInsertionPointToStart(gpuModule.getBody());
    auto newFuncOp = rewriter.create<mlir::gpu::GPUFuncOp>(
            rewriter.getUnknownLoc(), funcOp.getName(),
            rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                                     resultType ? TypeRange(resultType)
                                                : TypeRange()));

    auto& blocks = newFuncOp.getBody().getBlocks();
    auto& block = *(blocks.begin());

    rewriter.eraseBlock(&block);

    // Copy over all attributes other than the function name and type.
    for (const auto &namedAttr : funcOp->getAttrs()) {
        if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
            namedAttr.getName() != SymbolTable::getSymbolAttrName() &&
            namedAttr.getName() != funcOp.getArgAttrsAttrName())
        newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }
    // mlir::BoolAttr kernelFlagAttr = mlir::BoolAttr::get(rewriter.getContext(), true);
    // newFuncOp->setAttr("kernel", kernelFlagAttr);
    newFuncOp->setAttr("gpu.kernel", rewriter.getUnitAttr());

    mlir::spirv::EntryPointABIAttr entryPointABI = 
          mlir::spirv::EntryPointABIAttr::get(rewriter.getContext(), nullptr, std::nullopt);
    newFuncOp->setAttr("spirv.entry_point_abi", entryPointABI);

    // ArrayAttr attrs = funcOp.getAllArgAttrs();
    // for(int i = 0; i < attrs.size(); i++) {
    //   if (attrs[i].isa<mlir::DictionaryAttr>()) {
    //     newFuncOp.setArgAttrs(i, attrs[i].dyn_cast<mlir::DictionaryAttr>());
    //   }
    // }

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
    // rewriter.cloneRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
    //                         newFuncOp.end());

    if (failed(rewriter.convertRegionTypes(
            &newFuncOp.getBody(), *getTypeConverter(), &signatureConverter)))
      return failure();

    rewriter.eraseOp(funcOp);

    return success();
  }
private:
  int numWarps{0};
};

class TritonGPUToXeGPUConversionTarget : public ConversionTarget {
public:
  explicit TritonGPUToXeGPUConversionTarget(MLIRContext &ctx, TritonGPUToXeGPUTypeConverter& typeConverter)
          : ConversionTarget(ctx) {
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalOp<triton::GetProgramIdOp>();
    addIllegalOp<triton::SplatOp>();
    addIllegalOp<triton::DotOp>();
    addIllegalOp<triton::gpu::ConvertLayoutOp>();
    addLegalOp<vector::SplatOp>();

    addLegalDialect<mlir::spirv::SPIRVDialect>();
    addLegalDialect<triton::xegpu::XeGPUDialect>();
    addLegalDialect<mlir::vector::VectorDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addLegalDialect<mlir::gpu::GPUDialect>();

    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addLegalOp<arith::CmpIOp>();

    addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect,
                              mlir::memref::MemRefDialect>(
    [&](Operation *op) {
      if (typeConverter.isLegal(op))
        return true;
      return false;
    });

    addDynamicallyLegalOp<scf::ForOp>([](scf::ForOp forOp) -> bool {
      for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
        Value arg = forOp.getRegionIterArgs()[i];
        auto type = arg.getType();
        if(isa<triton::PointerType>(type)){
          return false;
        }
      }
      return true;
    });

    addDynamicallyLegalOp<scf::YieldOp>([](scf::YieldOp yieldOp) -> bool {
      for (auto arg : yieldOp.getResults()) {
        auto type = arg.getType();
        if(isa<triton::PointerType>(type)){
          return false;
        }
      }
      return true;
    });

    addDynamicallyLegalOp<triton::ExternElementwiseOp>([](triton::ExternElementwiseOp op) -> bool {
      auto result = op.getResult();
      Type retType = result.getType();
      if(isa<VectorType>(retType))
        return true;
      return false;
    });
  }
};

class TritonGPUToXeGPUPass : public ConvertTritonGPUToXeGPUBase<TritonGPUToXeGPUPass> {
public:
  explicit TritonGPUToXeGPUPass() {}
  void runOnOperation() override;
};

void TritonGPUToXeGPUPass::runOnOperation() {
  mlir::MLIRContext *context = &getContext();
  auto tritonGPUModule = getOperation();

  mlir::RewritePatternSet patterns(context);

  TritonGPUToXeGPUTypeConverter xeGPUTypeConverter(*context);
  TritonGPUToXeGPUConversionTarget xeGPUTarget(*context, xeGPUTypeConverter);

  TritonXeGPUFuncConversionTarget xeGPUFuncTarget(*context, xeGPUTypeConverter);
  RewritePatternSet func_patterns(context);
  func_patterns.add<XeGPUFuncOpConversion>(xeGPUTypeConverter, context, 0, 1);

  if (failed(
          applyPartialConversion(tritonGPUModule, xeGPUFuncTarget, std::move(func_patterns))))
    return signalPassFailure();

  populateTritonGPUToXeGPUPatterns(xeGPUTypeConverter, patterns);
  populateReduceOpToXeGPUPatterns(xeGPUTypeConverter, patterns);
  populateScfOpToXeGPUPatterns(xeGPUTypeConverter, patterns);

  if (failed(applyPartialConversion(tritonGPUModule, xeGPUTarget, std::move(patterns)))){
      return signalPassFailure();
  }
};

namespace mlir {
namespace triton {

std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertTritonGPUToXeGPUPass() {
  return std::make_unique<TritonGPUToXeGPUPass>();
}

} // namespace triton
} // namespace mlir

