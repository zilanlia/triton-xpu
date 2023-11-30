#include "mlir/Dialect/Arith/IR/Arith.h"
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

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Transforms/OneToNTypeConversion.h>

#include "SCFOpToXeGPU.h"
#include "triton/Dialect/XeGPU/IR/XeGPUOps.h"
#include "TritonGPUToXeGPUBase.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::spirv;
using namespace mlir::triton::xegpu;

class ForOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<scf::ForOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<scf::ForOp>::ConvertTritonGPUToXeGPUPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto context = op.getContext();

    llvm::SmallVector<mlir::Value> convertedArgs;
    for (Value values: adaptor.getInitArgs()){
      if(auto *parentOp = values.getDefiningOp()){
        if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
          ValueRange args = (&castOp)->getInputs();
          for(auto arg : args){
            auto argOp = arg.getDefiningOp();
            if(auto argCastOp = dyn_cast<UnrealizedConversionCastOp>(argOp)){
              Value originaArg = (&argCastOp)->getInputs()[0];
              convertedArgs.push_back(originaArg);
            }else{
              convertedArgs.push_back(arg);
            }
          }
        }else{
          convertedArgs.push_back(values);
        }
      }else{
        convertedArgs.push_back(values);
      }
    }

    TritonGPUToXeGPUTypeConverter xeGPUTypeConverter(*context);
    auto argumentTys = op.getRegion().getArgumentTypes();
    mlir::OneToNTypeMapping argumentMapping(argumentTys);
    if (mlir::failed(xeGPUTypeConverter.computeTypeMapping(argumentTys, argumentMapping))) {
      op.emitOpError("Failed to compute the type mapping for arguments.\n");
      return mlir::failure();
    }

    if(mlir::failed(rewriter.convertRegionTypes(&op.getRegion(), xeGPUTypeConverter, &argumentMapping))) {
      op.emitOpError("Failed to convert region types.\n");
      return mlir::failure();
    }

    auto newOp = rewriter.create<mlir::scf::ForOp>(loc, op.getLowerBound(), 
                                      op.getUpperBound(), op.getStep(), convertedArgs);

    newOp.getBody()->erase();
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(), newOp.getRegion().end());

    SmallVector<Value> newResult;
    auto results = newOp.getResults();
    ValueRange newValues{results};
    auto resultTys = op->getResultTypes();
    mlir::OneToNTypeMapping resultMapping(resultTys);
    llvm::SmallVector<mlir::Value> recastValues;
    if (mlir::failed(xeGPUTypeConverter.computeTypeMapping(resultTys, resultMapping))) {
      llvm_unreachable("It is an unexpected failure of failing to convert the result types.");
    } else {
      mlir::TypeRange originalTypes = resultMapping.getOriginalTypes();
      recastValues.reserve(originalTypes.size());
      auto convertedValueIt = newValues.begin();
      for (auto [idx, originalType] : llvm::enumerate(originalTypes)) {
        mlir::TypeRange convertedTypes = resultMapping.getConvertedTypes(idx);
        size_t numConvertedValues = convertedTypes.size();

        // Non-identity conversion: cast back to source type.
        mlir::ValueRange tmp{convertedValueIt, convertedValueIt + numConvertedValues};
        auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(loc,
            originalType, tmp).getResults();
        mlir::ValueRange recastValue{cast};
        assert(recastValue.size() == 1);
        recastValues.push_back(recastValue.front());

        convertedValueIt += numConvertedValues;
      }
    }
    rewriter.replaceOp(op, recastValues);

    return success();
  }
};

class YieldOpToXeGPUPattern : public ConvertTritonGPUToXeGPUPattern<scf::YieldOp> {
public:
  using ConvertTritonGPUToXeGPUPattern<scf::YieldOp>::ConvertTritonGPUToXeGPUPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    dbgInfo("[YieldOpToXeGPUPattern]");
    llvm::SmallVector<mlir::Value> convertedResults;
    for (Value values: adaptor.getResults()){
      if(auto *parentOp = values.getDefiningOp()){
        if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
          ValueRange args = (&castOp)->getInputs();
          for(auto arg : args){
            auto argOp = arg.getDefiningOp();
            if(auto argCastOp = dyn_cast<UnrealizedConversionCastOp>(argOp)){
              Value originaArg = (&argCastOp)->getInputs()[0];
              convertedResults.push_back(originaArg);
            }else{
              convertedResults.push_back(arg);
            }
          }
        }else{
          convertedResults.push_back(values);
        }
      }else{
        convertedResults.push_back(values);
      }
    }

    auto newOp = rewriter.create<mlir::scf::YieldOp>(op.getLoc(), convertedResults).getResults();
    dbgInfo("[YieldOpToXeGPUPattern]newYieldOP", newOp[0]);
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

void populateScfOpToXeGPUPatterns(
    TritonGPUToXeGPUTypeConverter &typeConverter, RewritePatternSet &patterns) {
  dbgInfo("[populateScfOpToXeGPUPatterns]");
  auto context = patterns.getContext();
  patterns.add<ForOpToXeGPUPattern, YieldOpToXeGPUPattern>(typeConverter, context);
}