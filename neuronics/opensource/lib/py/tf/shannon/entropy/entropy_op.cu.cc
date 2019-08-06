#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "entropy_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

template <typename dtype>
void EntropyFunctor<GPUDevice, dtype>::operator()(
    const GPUDevice& d,
    const dtype x
  ) {
  int block_count = 16;
  int thread_per_block = 1024;
  EntropyCudaKernel<GPUDevice>
      <<<block_count, thread_per_block, 0, d.stream()>>>(
        x
      );
    cudaDeviceSynchronize();
}

template struct EntropyFunctor<GPUDevice, float>;
template struct EntropyFunctor<GPUDevice, double>;
template struct EntropyFunctor<GPUDevice, int32>;
template struct EntropyFunctor<GPUDevice, int32>;
template struct EntropyFunctor<GPUDevice, string>;

#endif // GOOGLE_CUDA
