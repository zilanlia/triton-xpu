add_mlir_conversion_library(TritonGPUToXeGPU
  TritonGPUToXeGPU.cpp
  TritonGPUToXeGPUPass.cpp
  TypeConverter.cpp
  ReduceOpToXeGPU.cpp
  SCFOpToXeGPU.cpp
  Utility.cpp

  ADDITIONAL_HEADER_DIRS
   ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonGPUToXeGPU

  DEPENDS
  XeGPUConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSPIRVDialect
  MLIRSPIRVConversion
  MLIRSupport
  MLIRTransforms
  MLIRGPUOps
  MLIRGPUToNVVMTransforms
  MLIRGPUTransforms
  XeGPUIR
  TritonGPUIR
  TritonAnalysis
  TritonGPUTransforms
  )
