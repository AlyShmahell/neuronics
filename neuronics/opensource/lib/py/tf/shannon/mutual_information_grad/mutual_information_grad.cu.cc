#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "mutual_information_grad.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

template <typename dtype>
void MutualInformationGradFunctor<GPUDevice, dtype>::operator()(
    const GPUDevice& d,
    const float grad,
    const dtype x,
    const dtype y
  ) {
  int block_count = 16;
  int thread_per_block = 1024;
  MutualInformationGradCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(
        grad, x, y
      );
    cudaDeviceSynchronize();
}

template struct MutualInformationGradFunctor<GPUDevice, float>;
template struct MutualInformationGradFunctor<GPUDevice, double>;
template struct MutualInformationGradFunctor<GPUDevice, int32>;
template struct MutualInformationGradFunctor<GPUDevice, int64>;
template struct MutualInformationGradFunctor<GPUDevice, string>;

#endif // GOOGLE_CUDA
