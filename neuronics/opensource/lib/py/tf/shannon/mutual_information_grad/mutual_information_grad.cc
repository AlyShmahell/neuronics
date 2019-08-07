#include "mutual_information_grad.h"

using namespace tensorflow;

REGISTER_OP("MutualInformationGrad")
    .Attr("dtype: {float, double, int32, int64, string} = DT_INT64")
    .Input("grad: float32")
    .Input("x: dtype")
    .Input("y: dtype")
    .Output("grad_x: float32")
    .Output("grad_y: float32")
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
struct MutualInformationGradFunctor<CPUDevice, dtype> {
  void operator()(
    const CPUDevice& d,
    const float grad,
    const dtype x,
    const dtype y
  ) {}
};

// OpKernel definition.
// template parameter  is the datatype of the tensors.
template <typename Device, typename dtype>
class MutualInformationGradOp : public OpKernel {
 public:
  explicit MutualInformationGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
        DCHECK_EQ(3, context->num_inputs());
        const Tensor& grad = context->input(0);
        const Tensor &input0 = context->input(1);
        const TensorShape &input0_shape = input0.shape();
        DCHECK_EQ(input0_shape.dims(), 2);
        const int input0_dim0size = input0_shape.dim_size(0);
        const int input0_dim1size = input0_shape.dim_size(1);
        auto input0_tensor = input0.flat<dtype>();
        const Tensor &input1 = context->input(2);
        const TensorShape &input1_shape = input1.shape();
        DCHECK_EQ(input1_shape.dims(), 2);
        const int input1_dim0size = input1_shape.dim_size(0);
        const int input1_dim1size = input1_shape.dim_size(1);
        auto input1_tensor = input1.flat<dtype>();
        DCHECK_EQ(input0_dim0size, input1_dim0size);
        DCHECK_EQ(input0_dim1size, input1_dim1size);
        Tensor *grad_x = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, {input0_dim0size}, &grad_x));
        auto grad_x_tensor = grad_x->flat<double>();
        Tensor *grad_y = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, {input0_dim0size}, &grad_y));
        auto grad_y_tensor = grad_y->flat<double>();
        for (int sample_index = 0; sample_index < input0_dim0size; sample_index++)
        {
            std::map<string, long double> x_y_frequencies;
            typename std::map<string, long double>::iterator x_y_frequency_iterator;
            std::map<dtype, long double> y_frequencies;
            typename std::map<dtype, long double>::iterator y_frequency_iterator;
            std::map<dtype, long double> x_frequencies;
            typename std::map<dtype, long double>::iterator x_frequency_iterator;
            for (int i = input0_dim1size * sample_index; i < input0_dim1size * (sample_index + 1); i++)
            {
                long double x_y_frequency = (long double)(1) / (long double)(input1_dim1size);
                x_y_frequency_iterator = x_y_frequencies.find(join(input0_tensor(i), input1_tensor(i)));
                if (x_y_frequency_iterator == x_y_frequencies.end())
                    x_y_frequencies.insert(std::pair<string, long double>(join(input0_tensor(i), input1_tensor(i)), x_y_frequency));
                else
                    x_y_frequencies[join(input0_tensor(i), input1_tensor(i))] += x_y_frequency;
                long double y_frequency = (long double)(1) / (long double)(input1_dim1size);
                y_frequency_iterator = y_frequencies.find(input1_tensor(i));
                if (y_frequency_iterator == y_frequencies.end())
                    y_frequencies.insert(std::pair<dtype, long double>(input1_tensor(i), y_frequency));
                else
                    y_frequencies[input1_tensor(i)] += y_frequency;
                long double x_frequency = (long double)(1) / (long double)(input1_dim1size);
                x_frequency_iterator = x_frequencies.find(input0_tensor(i));
                if (x_frequency_iterator == x_frequencies.end())
                    x_frequencies.insert(std::pair<dtype, long double>(input0_tensor(i), x_frequency));
                else
                    x_frequencies[input0_tensor(i)] += x_frequency;
            }
            grad_x_tensor(sample_index) = dxentropy(x_frequencies);
            grad_y_tensor(sample_index) = dxentropy(y_frequencies);
        }
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(dtype)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("MutualInformationGrad").Device(DEVICE_CPU).TypeConstraint<dtype>("dtype"), \
      MutualInformationGradOp<CPUDevice, dtype>);
REGISTER_CPU(float);
REGISTER_CPU(double);
REGISTER_CPU(int32);
REGISTER_CPU(int64);
REGISTER_CPU(string);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(dtype)                                          \
  extern template MutualInformationGradFunctor<GPUDevice, dtype>;                  \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("MutualInformationGrad").Device(DEVICE_GPU).TypeConstraint<dtype>("dtype"), \
      MutualInformationGradOp<GPUDevice, dtype>);
REGISTER_GPU(float);
REGISTER_GPU(double);
REGISTER_GPU(int32);
REGISTER_GPU(int64);
REGISTER_GPU(string);
#endif // GOOGLE_CUDA