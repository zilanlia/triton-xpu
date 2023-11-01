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
      llvm::outs() <<"\n\nvalues in forloop: "<< values<<"\n";
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
      // llvm::outs() << "\n\nvalues: " << values << "\n";
    }

    TritonGPUToXeGPUTypeConverter xeGPUTypeConverter(*context);
    auto argumentTys = op.getRegion().getArgumentTypes();
    for(auto arg : argumentTys){
      llvm::outs()<<"\n\nargumentTys arg: "<<arg<<"\n";
    }
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
      // newValues = buildUnrealizedCast(rewriter, op->getResultTypes(), newValues);
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
        llvm::outs()<<"\n\nrecastValue.front(): "<<recastValue.front()<<"\n";
        llvm::outs()<<"\n\nnumConvertedValue: "<<numConvertedValues<<"\n";
      }
    }
    rewriter.replaceOp(op, recastValues);

    //.getResults();
    
    // mlir::ValueRange newValueRange0(newValues.begin(), newValues.begin() + 16);
    // Value arg0 = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys[0], newValueRange0).getODSOperands(0)[0];
    // newResult.push_back(arg0);

    // mlir::ValueRange newValueRange1(newValues.begin() + 16, newValues.begin() + 24);
    // Value arg1 = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys[1], newValueRange1).getODSOperands(0)[0];
    // newResult.push_back(arg1);

    // mlir::ValueRange newValueRange2(newValues.begin() + 24, newValues.begin() + 32);
    // Value arg2 = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys[2], newValueRange2).getODSOperands(0)[0];
    // newResult.push_back(arg2);

    // mlir::ValueRange newResultRange(newResult);
    // Value args = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resultTys, newResultRange).getODSOperands(0)[0];

    // llvm::outs()<<"\n\narg0 : "<<arg0<<"\n";
    // llvm::outs()<<"\n\narg1 : "<<arg1<<"\n";
    // llvm::outs()<<"\n\narg2 : "<<arg2<<"\n";
    // llvm::outs()<<"\n\nargs : "<<args<<"\n";

    // llvm::outs()<<"\n\nresultTys[0] : "<<resultTys[0]<<"\n";
    // llvm::outs()<<"\n\nresultTys[1] : "<<resultTys[1]<<"\n";
    // llvm::outs()<<"\n\nresultTys[2] : "<<resultTys[2]<<"\n";

    // rewriter.replaceOp(op, newResultRange);
    // rewriter.replaceOp(op, args);

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
      llvm::outs()<<"\n\nyield values: "<<values<<"\n";
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

    llvm::outs()<<"\n\nyield newOp.size(): "<<newOp.size()<<"\n";

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