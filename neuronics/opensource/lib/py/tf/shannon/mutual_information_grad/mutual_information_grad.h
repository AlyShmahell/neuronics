#ifndef MutualInformationGrad_OP_H_
#define MutualInformationGrad_OP_H_

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include "base.h"
using namespace tensorflow;

template <typename Device, typename dtype>
struct MutualInformationGradFunctor {
  void operator()(
    const Device& d,
    const float grad,
    const dtype x,
    const dtype y
  );
};

#if GOOGLE_CUDA
template <typename dtype>
struct MutualInformationGradFunctor<Eigen::GpuDevice, dtype> {
  void operator()(
    const Eigen::GpuDevice& d,
    const float grad,
    const dtype x,
    const dtype y
  );
};
#endif

#endif // MutualInformationGrad_OP_H_
