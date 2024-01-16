#ifndef TRITON_CONVERSION_TRITONGPU_TO_XEGPU_H
#define TRITON_CONVERSION_TRITONGPU_TO_XEGPU_H

#include "TypeConverter.h"

class ConvertTritonGPUToXeGPUPatternBase {
public:
  explicit ConvertTritonGPUToXeGPUPatternBase(
      TritonGPUToXeGPUTypeConverter &typeConverter)
      : converter(&typeConverter) {}

  TritonGPUToXeGPUTypeConverter *getTypeConverter() const { return converter; }

protected:
  TritonGPUToXeGPUTypeConverter *converter;
};

template <typename SourceOp>
class ConvertTritonGPUToXeGPUPattern
    : public OpConversionPattern<SourceOp>,
      public ConvertTritonGPUToXeGPUPatternBase {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ConvertTritonGPUToXeGPUPattern(
      TritonGPUToXeGPUTypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<SourceOp>(typeConverter, context),
        ConvertTritonGPUToXeGPUPatternBase(typeConverter) {}

protected:
  TritonGPUToXeGPUTypeConverter *getTypeConverter() const {
    TritonGPUToXeGPUTypeConverter *ret =
        ((ConvertTritonGPUToXeGPUPatternBase *)this)->getTypeConverter();
    return (TritonGPUToXeGPUTypeConverter *)ret;
  }
};

#endif