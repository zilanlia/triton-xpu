add_mlir_dialect_library(XeGPUTransforms
  OptimizeDotOp.cpp

  DEPENDS
  XeGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  XeGPUIR
  MLIRTransformUtils
)
