#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "conditional_entropy_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

template <typename dtype>
void MutualInformationFunctor<GPUDevice, dtype>::operator()(
    const GPUDevice& d,
    const dtype x,
    const dtype y
  ) {
  int block_count = 16;
  int thread_per_block = 1024;
  MutualInformationCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(
        x, y
      );
    cudaDeviceSynchronize();
}

template struct MutualInformationFunctor<GPUDevice, float>;
template struct MutualInformationFunctor<GPUDevice, double>;
template struct MutualInformationFunctor<GPUDevice, int32>;
template struct MutualInformationFunctor<GPUDevice, int32>;
template struct MutualInformationFunctor<GPUDevice, string>;

#endif // GOOGLE_CUDA
