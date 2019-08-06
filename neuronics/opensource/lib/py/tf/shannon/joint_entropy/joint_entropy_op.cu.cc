#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "joint_entropy_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

template <typename dtype>
void JointEntropyFunctor<GPUDevice, dtype>::operator()(
    const GPUDevice& d,
    const dtype x,
    const dtype y
  ) {
  int block_count = 16;
  int thread_per_block = 1024;
  JointEntropyCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(
        x, y
      );
    cudaDeviceSynchronize();
}

template struct JointEntropyFunctor<GPUDevice, float>;
template struct JointEntropyFunctor<GPUDevice, double>;
template struct JointEntropyFunctor<GPUDevice, int32>;
template struct JointEntropyFunctor<GPUDevice, int32>;
template struct JointEntropyFunctor<GPUDevice, string>;

#endif // GOOGLE_CUDA
