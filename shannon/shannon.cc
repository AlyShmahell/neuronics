#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

#include <iostream>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdint.h>
#include <string>

#include "format_id.h"

using namespace tensorflow;

template <typename T>
std::string join(T a, T b)
{
    char buffer1[32];
    char buffer2[32];
    snprintf(buffer1, sizeof(buffer1), format_id<T>::Get(), a);
    snprintf(buffer2, sizeof(buffer2), format_id<T>::Get(), b);
    return std::string(buffer1) + "|" + std::string(buffer2);
}

template <typename T>
long double entropy(std::map<T, long double> frequencies)
{
    long double result = 0.0;
    for (auto frequency : frequencies)
    {
        result += -frequency.second * log2(frequency.second);
    }
    return result;
}

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

REGISTER_OP("JointEntropy")
    .Attr("dtype: {float, double, int32, int64, string} = DT_INT64")
    .Input("x: dtype")
    .Input("y: dtype")
    .Output("output: double")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    })
    .Doc(R"doc(
Calculates the joint entropy of a Tensor.
output: A Tensor with one element representing the joint entropy of the input elements.
)doc");

REGISTER_OP("MutualInformation")
    .Attr("dtype: {float, double, int32, int64, string} = DT_INT64")
    .Input("x: dtype")
    .Input("y: dtype")
    .Output("output: double")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    })
    .Doc(R"doc(
                Calculates the mutual information of a Tensor.
                output: A Tensor with one element representing the mutual information of the input elements.
              )doc");

template <typename dtype>
class EntropyOp : public OpKernel
{

public:
    explicit EntropyOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
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

template <typename dtype>
class ConditionalEntropyOp : public OpKernel
{

public:
    explicit ConditionalEntropyOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
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

template <typename dtype>
class JointEntropyOp : public OpKernel
{

public:
    explicit JointEntropyOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
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
            std::map<string, long double> x_y_frequencies;
            typename std::map<string, long double>::iterator x_y_frequency_iterator;
            for (int i = input0_dim1size * sample_index; i < input0_dim1size * (sample_index + 1); i++)
            {
                long double x_y_frequency = (long double)(1) / (long double)(input0_dim1size);
                x_y_frequency_iterator = x_y_frequencies.find(join(input0_tensor(i), input1_tensor(i)));
                if (x_y_frequency_iterator == x_y_frequencies.end())
                    x_y_frequencies.insert(std::pair<string, long double>(join(input0_tensor(i), input1_tensor(i)), x_y_frequency));
                else
                    x_y_frequencies[join(input0_tensor(i), input1_tensor(i))] += x_y_frequency;
            }
            output_tensor(sample_index) = entropy(x_y_frequencies);
        }
    }
};

template <typename dtype>
class MutualInformationOp : public OpKernel
{

public:
    explicit MutualInformationOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
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
            output_tensor(sample_index) = entropy(y_frequencies) + entropy(x_frequencies) - entropy(x_y_frequencies);
        }
    }
};


REGISTER_KERNEL_BUILDER(Name("Entropy").Device(DEVICE_CPU).TypeConstraint<int32>("dtype"), EntropyOp<int32>);
REGISTER_KERNEL_BUILDER(Name("Entropy").Device(DEVICE_CPU).TypeConstraint<int64>("dtype"), EntropyOp<int64>);
REGISTER_KERNEL_BUILDER(Name("Entropy").Device(DEVICE_CPU).TypeConstraint<float>("dtype"), EntropyOp<float>);
REGISTER_KERNEL_BUILDER(Name("Entropy").Device(DEVICE_CPU).TypeConstraint<double>("dtype"), EntropyOp<double>);
REGISTER_KERNEL_BUILDER(Name("Entropy").Device(DEVICE_CPU).TypeConstraint<string>("dtype"), EntropyOp<string>);

REGISTER_KERNEL_BUILDER(Name("ConditionalEntropy").Device(DEVICE_CPU).TypeConstraint<int32>("dtype"), ConditionalEntropyOp<int32>);
REGISTER_KERNEL_BUILDER(Name("ConditionalEntropy").Device(DEVICE_CPU).TypeConstraint<int64>("dtype"), ConditionalEntropyOp<int64>);
REGISTER_KERNEL_BUILDER(Name("ConditionalEntropy").Device(DEVICE_CPU).TypeConstraint<float>("dtype"), ConditionalEntropyOp<float>);
REGISTER_KERNEL_BUILDER(Name("ConditionalEntropy").Device(DEVICE_CPU).TypeConstraint<double>("dtype"), ConditionalEntropyOp<double>);
REGISTER_KERNEL_BUILDER(Name("ConditionalEntropy").Device(DEVICE_CPU).TypeConstraint<string>("dtype"), ConditionalEntropyOp<string>);

REGISTER_KERNEL_BUILDER(Name("JointEntropy").Device(DEVICE_CPU).TypeConstraint<int32>("dtype"), JointEntropyOp<int32>);
REGISTER_KERNEL_BUILDER(Name("JointEntropy").Device(DEVICE_CPU).TypeConstraint<int64>("dtype"), JointEntropyOp<int64>);
REGISTER_KERNEL_BUILDER(Name("JointEntropy").Device(DEVICE_CPU).TypeConstraint<float>("dtype"), JointEntropyOp<float>);
REGISTER_KERNEL_BUILDER(Name("JointEntropy").Device(DEVICE_CPU).TypeConstraint<double>("dtype"), JointEntropyOp<double>);
REGISTER_KERNEL_BUILDER(Name("JointEntropy").Device(DEVICE_CPU).TypeConstraint<string>("dtype"), JointEntropyOp<string>);

REGISTER_KERNEL_BUILDER(Name("MutualInformation").Device(DEVICE_CPU).TypeConstraint<int32>("dtype"), MutualInformationOp<int32>);
REGISTER_KERNEL_BUILDER(Name("MutualInformation").Device(DEVICE_CPU).TypeConstraint<int64>("dtype"), MutualInformationOp<int64>);
REGISTER_KERNEL_BUILDER(Name("MutualInformation").Device(DEVICE_CPU).TypeConstraint<float>("dtype"), MutualInformationOp<float>);
REGISTER_KERNEL_BUILDER(Name("MutualInformation").Device(DEVICE_CPU).TypeConstraint<double>("dtype"), MutualInformationOp<double>);
REGISTER_KERNEL_BUILDER(Name("MutualInformation").Device(DEVICE_CPU).TypeConstraint<string>("dtype"), MutualInformationOp<string>);
