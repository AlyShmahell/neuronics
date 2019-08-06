#ifndef Entropy_OP_H_
#define Entropy_OP_H_

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include "base.h"
using namespace tensorflow;

template <typename Device, typename dtype>
struct EntropyFunctor {
  void operator()(
    const Device& d,
    const dtype x
  );
};

#if GOOGLE_CUDA
template <typename dtype>
struct EntropyFunctor<Eigen::GpuDevice, dtype> {
  void operator()(
    const Eigen::GpuDevice& d,
    const dtype x
  );
};
#endif

#endif // Entropy_OP_H_