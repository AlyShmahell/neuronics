#include "entropy_op.h"

using namespace tensorflow;

REGISTER_OP("Entropy")
    .Attr("dtype: {float, double, int32, int64, string} = DT_INT64")
    .Input("x: dtype")
    .Output("output: double")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    })
    .Doc(R"doc(
Calculates the conditional entropy of a Tensor.
output: A Tensor with one element representing the conditional entropy of the input elements.
)doc");

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename dtype>
struct EntropyFunctor<CPUDevice, dtype> {
  void operator()(
    const CPUDevice& d,
    const dtype x
  ) {}
};

// OpKernel definition.
// template parameter  is the datatype of the tensors.
template <typename Device, typename dtype>
class EntropyOp : public OpKernel {
 public:
  explicit EntropyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    DCHECK_EQ(1, context->num_inputs());
        const Tensor &input = context->input(0);
        const TensorShape &input_shape = input.shape();
        DCHECK_EQ(input_shape.dims(), 2);
        const int input_dim0size = input_shape.dim_size(0);
        const int input_dim1size = input_shape.dim_size(1);
        auto input_tensor = input.flat<dtype>();
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, {input_dim0size}, &output));
        auto output_tensor = output->flat<double>();
        for (int sample_index = 0; sample_index < input_dim0size; sample_index++)
        {
            std::map<dtype, long double> sample_frequencies;
            typename std::map<dtype, long double>::iterator sample_frequency_iterator;
            for (int i = input_dim1size * sample_index; i < input_dim1size * (sample_index + 1); i++)
            {
                long double sample_frequency = (long double)(1) / (long double)(input_dim1size);
                sample_frequency_iterator = sample_frequencies.find(input_tensor(i));
                if (sample_frequency_iterator == sample_frequencies.end())
                    sample_frequencies.insert(std::pair<dtype, long double>(input_tensor(i), sample_frequency));
                else
                    sample_frequencies[input_tensor(i)] += sample_frequency;
            }
            output_tensor(sample_index) = entropy(sample_frequencies);
        }
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(dtype)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Entropy").Device(DEVICE_CPU).TypeConstraint<dtype>("dtype"), \
      EntropyOp<CPUDevice, dtype>);
REGISTER_CPU(float);
REGISTER_CPU(double);
REGISTER_CPU(int32);
REGISTER_CPU(int64);
REGISTER_CPU(string);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(dtype)                                          \
  extern template EntropyFunctor<GPUDevice, dtype>;                  \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Entropy").Device(DEVICE_GPU).TypeConstraint<dtype>("dtype"), \
      EntropyOp<GPUDevice, dtype>);
REGISTER_GPU(float);
REGISTER_GPU(double);
REGISTER_GPU(int32);
REGISTER_GPU(int64);
REGISTER_GPU(string);
#endif // GOOGLE_CUDA