#include "conditional_entropy_op.h"

using namespace tensorflow;

REGISTER_OP("ConditionalEntropy")
    .Attr("dtype: {float, double, int32, int64, string} = DT_INT64")
    .Input("x: dtype")
    .Input("y: dtype")
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
struct ConditionalEntropyFunctor<CPUDevice, dtype> {
  void operator()(
    const CPUDevice& d,
    const dtype x,
    const dtype y
  ) {}
};

// OpKernel definition.
// template parameter  is the datatype of the tensors.
template <typename Device, typename dtype>
class ConditionalEntropyOp : public OpKernel {
 public:
  explicit ConditionalEntropyOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
        DCHECK_EQ(2, context->num_inputs());
        const Tensor &input0 = context->input(0);
        const TensorShape &input0_shape = input0.shape();
        DCHECK_EQ(input0_shape.dims(), 2);
        const int input0_dim0size = input0_shape.dim_size(0);
        const int input0_dim1size = input0_shape.dim_size(1);
        auto input0_tensor = input0.flat<dtype>();
        const Tensor &input1 = context->input(1);
        const TensorShape &input1_shape = input1.shape();
        DCHECK_EQ(input1_shape.dims(), 2);
        const int input1_dim0size = input1_shape.dim_size(0);
        const int input1_dim1size = input1_shape.dim_size(1);
        auto input1_tensor = input1.flat<dtype>();
        DCHECK_EQ(input0_dim0size, input1_dim0size);
        DCHECK_EQ(input0_dim1size, input1_dim1size);
        Tensor *output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, {input0_dim0size}, &output));
        auto output_tensor = output->flat<double>();
        for (int sample_index = 0; sample_index < input0_dim0size; sample_index++)
        {
            std::map<string, long double> joint_frequencies;
            typename std::map<string, long double>::iterator joint_frequency_iterator;
            std::map<dtype, long double> condition_frequencies;
            typename std::map<dtype, long double>::iterator condition_frequency_iterator;
            for (int i = input0_dim1size * sample_index; i < input0_dim1size * (sample_index + 1); i++)
            {
                long double joint_frequency = (long double)(1) / (long double)(input0_dim1size);
                joint_frequency_iterator = joint_frequencies.find(join(input0_tensor(i), input1_tensor(i)));
                if (joint_frequency_iterator == joint_frequencies.end())
                    joint_frequencies.insert(std::pair<string, long double>(join(input0_tensor(i), input1_tensor(i)), joint_frequency));
                else
                    joint_frequencies[join(input0_tensor(i), input1_tensor(i))] += joint_frequency;
                long double condition_frequency = (long double)(1) / (long double)(input1_dim1size);
                condition_frequency_iterator = condition_frequencies.find(input1_tensor(i));
                if (condition_frequency_iterator == condition_frequencies.end())
                    condition_frequencies.insert(std::pair<dtype, long double>(input1_tensor(i), condition_frequency));
                else
                    condition_frequencies[input1_tensor(i)] += condition_frequency;
            }
            output_tensor(sample_index) = entropy(joint_frequencies) - entropy(condition_frequencies);
        }
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(dtype)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ConditionalEntropy").Device(DEVICE_CPU).TypeConstraint<dtype>("dtype"), \
      ConditionalEntropyOp<CPUDevice, dtype>);
REGISTER_CPU(float);
REGISTER_CPU(double);
REGISTER_CPU(int32);
REGISTER_CPU(int64);
REGISTER_CPU(string);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(dtype)                                          \
  extern template ConditionalEntropyFunctor<GPUDevice, dtype>;                  \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ConditionalEntropy").Device(DEVICE_GPU).TypeConstraint<dtype>("dtype"), \
      ConditionalEntropyOp<GPUDevice, dtype>);
REGISTER_GPU(float);
REGISTER_GPU(double);
REGISTER_GPU(int32);
REGISTER_GPU(int64);
REGISTER_GPU(string);
#endif // GOOGLE_CUDA