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

#include "ScfOpToXeGPU.h"
#include "triton/Dialect/XeGPU/IR/XeGPUOps.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::spirv;
using namespace mlir::triton::xegpu;

class ForOpToXeGPUPattern : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto context = op.getContext();

    llvm::SmallVector<mlir::Value> convertedArgs;
    for (Value values: adaptor.getInitArgs()){
      if(auto *parentOp = values.getDefiningOp()){
        if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
          ValueRange tmp = (&castOp)->getInputs();
          convertedArgs.append(tmp.begin(), tmp.end());
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
    ValueRange newValues = newOp.getResults();
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
        mlir::ValueRange recastValue = rewriter.create<mlir::UnrealizedConversionCastOp>(loc,
            originalType, tmp).getResults();
        assert(recastValue.size() == 1);
        recastValues.push_back(recastValue.front());

        convertedValueIt += numConvertedValues;
      }
    }
    rewriter.replaceOp(op, recastValues);

    return success();
  }
};

class YieldOpToXeGPUPattern : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Value> convertedResults;
    for (Value values: adaptor.getResults()){
      if(auto *parentOp = values.getDefiningOp()){
        if(auto castOp = dyn_cast<UnrealizedConversionCastOp>(parentOp)){
          ValueRange tmp = (&castOp)->getInputs();
          convertedResults.append(tmp.begin(), tmp.end());
        }else{
          convertedResults.push_back(values);
        }
      }else{
        convertedResults.push_back(values);
      }
    }

    auto newOp = rewriter.create<mlir::scf::YieldOp>(op.getLoc(), convertedResults).getResults();
    rewriter.replaceOp(op, newOp);
    return mlir::success();
  }
};

void populateScfOpToXeGPUPatterns(
    TritonGPUToXeGPUTypeConverter &typeConverter, RewritePatternSet &patterns) {
  llvm::outs()<<"\n\npopulateScfOpToXeGPUPatterns\n";
  auto context = patterns.getContext();
  patterns.add<ForOpToXeGPUPattern, YieldOpToXeGPUPattern>(typeConverter, context);
}