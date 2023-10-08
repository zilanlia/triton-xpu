#ifndef XEGPU_CONVERSION_PASSES
#define XEGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonGPUToXeGPU : Pass<"convert-tritongpu-to-xegpu", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to XeGPU";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonGPUToXeGPUPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::xegpu::XeGPUDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::spirv::SPIRVDialect"];

    let options = [
    Option<"computeCapability", "compute-capability",
            "int32_t", /*default*/"80",
            "device compute capability">
    ];
}


#endif
