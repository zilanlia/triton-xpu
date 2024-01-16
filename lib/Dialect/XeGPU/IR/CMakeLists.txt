add_mlir_dialect_library(XeGPUIR
  XeGPUDialect.cpp
  XeGPUOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/triton/Dialect/XeGPU

  DEPENDS
  XeGPUTableGen
  XeGPUOpsAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
