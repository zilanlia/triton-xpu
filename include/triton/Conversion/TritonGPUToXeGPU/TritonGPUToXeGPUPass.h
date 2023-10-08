#ifndef TRITON_TRITONGPUTOXEGPU_PASS_H_
#define TRITON_TRITONGPUTOXEGPU_PASS_H_

#include <memory>

namespace mlir {
class Pass;
struct ScfToSPIRVContextImpl;
class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
/// Create a pass
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertTritonGPUToXeGPUPass();

std::string
translateTritonGPUToXeGPUVIR(mlir::ModuleOp module,
                        int computeCapability);
} // namespace triton
} // namespace mlir

#endif
