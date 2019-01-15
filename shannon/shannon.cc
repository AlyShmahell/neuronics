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


template<typename T>
std::string join(T a, T b)
{
    char buffer1[32];
    char buffer2[32];
    snprintf(buffer1, sizeof(buffer1), format_id<T>::Get(), a);
    snprintf(buffer2, sizeof(buffer2), format_id<T>::Get(), b);
    return std::string(buffer1) + "|" + std::string(buffer2);
}


template<typename T>
long double entropy(std::map<T, long double> frequencies)
{
    long double result = 0.0;
    for(auto frequency: frequencies)
    {
        result += -frequency.second*log2(frequency.second);
    }
    return result;
}


REGISTER_OP("ConditionalEntropy")
.Attr("dtype: {float, double, int32, int64, string} = DT_INT64")
.Input("x: dtype")
.Input("y: dtype")
.Output("output: double")
.SetShapeFn([](shape_inference::InferenceContext* c)
{
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
.SetShapeFn([](shape_inference::InferenceContext* c)
{
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
.SetShapeFn([](shape_inference::InferenceContext* c)
{
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
.SetShapeFn([](shape_inference::InferenceContext* c)
{
    c->set_output(0, c->input(0));
    return Status::OK();
})
.Doc(R"doc(
Calculates the mutual information of a Tensor.
output: A Tensor with one element representing the mutual information of the input elements.
)doc");


template <typename dtype>
class EntropyOp : public OpKernel {

 public:

    explicit EntropyOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // check if the number of inputs is 1
        DCHECK_EQ(1, context->num_inputs());
        // get the x tensor
        const Tensor& x = context->input(0);
        // create output tensor
        Tensor* output = NULL;
        // allocate output tensor size
        OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &output));
        // get the shape of x
        const TensorShape& x_shape = x.shape();
        // check if x has the correct number of dimentions
        DCHECK_EQ(x_shape.dims(), 1);
        // get the x Eigen tensors
        auto x_tensor = x.flat<dtype>();
        // get the output Eigen tensors
        auto output_tensor = output->flat<double>();
        // X Frequency Definition
        std::map<dtype, long double> x_frequencies;
        typename std::map<dtype, long double>::iterator x_frequency_iterator;
        // Frequency Calculation
        for (int i = 0; i < x_tensor.size(); i++)
        {
            /**
             * X Frequency Calculation
             */
            // Single Frequency
            long double x_frequency = (long double)(1)/(long double)(x_tensor.size());
            x_frequency_iterator = x_frequencies.find(x_tensor(i));
            if(x_frequency_iterator == x_frequencies.end())
                // Frequency Injection
                x_frequencies.insert(std::pair<dtype, long double>(x_tensor(i), x_frequency)) ;
            else 
                // Frequency Accumilation
                x_frequencies[x_tensor(i)] += x_frequency;
        }
        // get result and allocate it to output
        output_tensor(0) = entropy(x_frequencies);
    }
};


template <typename dtype>
class ConditionalEntropyOp : public OpKernel {

 public:

    explicit ConditionalEntropyOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // check if the number of inputs is 2
        DCHECK_EQ(2, context->num_inputs());
        // get the x tensor
        const Tensor& x = context->input(0);
        // get the y tensor
        const Tensor& y = context->input(1);
        // create output tensor
        Tensor* output = NULL;
        // allocate output tensor size
        OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &output));
        // get the shape of x
        const TensorShape& x_shape = x.shape();
        // get the shape of y
        const TensorShape& y_shape = y.shape();
        // check if x has the correct number of dimentions
        DCHECK_EQ(x_shape.dims(), 1);
        // check if y has the correct number of dimentions
        DCHECK_EQ(y_shape.dims(), 1);
        // check if x and y have the same sizes
        DCHECK_EQ(x_shape.dim_size(0), y_shape.dim_size(0));
        // get the x Eigen tensors
        auto x_tensor = x.flat<dtype>();
        // get the y Eigen tensors
        auto y_tensor = y.flat<dtype>();
        // get the output Eigen tensors
        auto output_tensor = output->flat<double>();
        // X|Y Frequency Definition
        std::map<string, long double> x_y_frequencies;
        typename std::map<string, long double>::iterator x_y_frequency_iterator;
        // Y Frequency Definition
        std::map<dtype, long double> y_frequencies;
        typename std::map<dtype, long double>::iterator y_frequency_iterator;
        // Frequency Calculation
        for (int i = 0; i < x_tensor.size(); i++)
        {
            /**
             * X|Y Frequency Calculation
             */
            // Single Frequency
            long double x_y_frequency = (long double)(1)/(long double)(x_tensor.size());
            // Accumilation Check
            x_y_frequency_iterator = x_y_frequencies.find(join(x_tensor(i), y_tensor(i)));
            if(x_y_frequency_iterator == x_y_frequencies.end())
                // Frequency Injection
                x_y_frequencies.insert(std::pair<string, long double>(join(x_tensor(i), y_tensor(i)), x_y_frequency)) ;
            else 
                // Frequency Accumilation
                x_y_frequencies[join(x_tensor(i), y_tensor(i))] += x_y_frequency;
            /**
             * Y Frequency Calculation
             */
            // Single Frequency
            long double y_frequency = (long double)(1)/(long double)(y_tensor.size());
            y_frequency_iterator = y_frequencies.find(y_tensor(i));
            if(y_frequency_iterator == y_frequencies.end())
                // Frequency Injection
                y_frequencies.insert(std::pair<dtype, long double>(y_tensor(i), y_frequency)) ;
            else 
                // Frequency Accumilation
                y_frequencies[y_tensor(i)] += y_frequency;
        }
        // get result and allocate it to output
        output_tensor(0) = entropy(x_y_frequencies) - entropy(y_frequencies);
    }
};


template <typename dtype>
class JointEntropyOp : public OpKernel {

 public:

    explicit JointEntropyOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // check if the number of inputs is 2
        DCHECK_EQ(2, context->num_inputs());
        // get the x tensor
        const Tensor& x = context->input(0);
        // get the y tensor
        const Tensor& y = context->input(1);
        // create output tensor
        Tensor* output = NULL;
        // allocate output tensor size
        OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &output));
        // get the shape of x
        const TensorShape& x_shape = x.shape();
        // get the shape of y
        const TensorShape& y_shape = y.shape();
        // check if x has the correct number of dimentions
        DCHECK_EQ(x_shape.dims(), 1);
        // check if y has the correct number of dimentions
        DCHECK_EQ(y_shape.dims(), 1);
        // check if x and y have the same sizes
        DCHECK_EQ(x_shape.dim_size(0), y_shape.dim_size(0));
        // get the x Eigen tensors
        auto x_tensor = x.flat<dtype>();
        // get the y Eigen tensors
        auto y_tensor = y.flat<dtype>();
        // get the output Eigen tensors
        auto output_tensor = output->flat<double>();
        // X|Y Frequency Definition
        std::map<string, long double> x_y_frequencies;
        typename std::map<string, long double>::iterator x_y_frequency_iterator;
        // Frequency Calculation
        for (int i = 0; i < x_tensor.size(); i++)
        {
            /**
             * X|Y Frequency Calculation
             */
            // Single Frequency
            long double x_y_frequency = (long double)(1)/(long double)(x_tensor.size());
            // Accumilation Check
            x_y_frequency_iterator = x_y_frequencies.find(join(x_tensor(i), y_tensor(i)));
            if(x_y_frequency_iterator == x_y_frequencies.end())
                // Frequency Injection
                x_y_frequencies.insert(std::pair<string, long double>(join(x_tensor(i), y_tensor(i)), x_y_frequency)) ;
            else 
                // Frequency Accumilation
                x_y_frequencies[join(x_tensor(i), y_tensor(i))] += x_y_frequency;
        }
        // get result and allocate it to output
        output_tensor(0) = entropy(x_y_frequencies);
    }
};


template <typename dtype>
class MutualInformationOp : public OpKernel {

 public:

    explicit MutualInformationOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // check if the number of inputs is 2
        DCHECK_EQ(2, context->num_inputs());
        // get the x tensor
        const Tensor& x = context->input(0);
        // get the y tensor
        const Tensor& y = context->input(1);
        // create output tensor
        Tensor* output = NULL;
        // allocate output tensor size
        OP_REQUIRES_OK(context, context->allocate_output(0, {1}, &output));
        // get the shape of x
        const TensorShape& x_shape = x.shape();
        // get the shape of y
        const TensorShape& y_shape = y.shape();
        // check if x has the correct number of dimentions
        DCHECK_EQ(x_shape.dims(), 1);
        // check if y has the correct number of dimentions
        DCHECK_EQ(y_shape.dims(), 1);
        // check if x and y have the same sizes
        DCHECK_EQ(x_shape.dim_size(0), y_shape.dim_size(0));
        // get the x Eigen tensors
        auto x_tensor = x.flat<dtype>();
        // get the y Eigen tensors
        auto y_tensor = y.flat<dtype>();
        // get the output Eigen tensors
        auto output_tensor = output->flat<double>();
        // X|Y Frequency Definition
        std::map<string, long double> x_y_frequencies;
        typename std::map<string, long double>::iterator x_y_frequency_iterator;
        // Y Frequency Definition
        std::map<dtype, long double> y_frequencies;
        typename std::map<dtype, long double>::iterator y_frequency_iterator;
        // X Frequency Definition
        std::map<dtype, long double> x_frequencies;
        typename std::map<dtype, long double>::iterator x_frequency_iterator;
        // Frequency Calculation
        for (int i = 0; i < x_tensor.size(); i++)
        {
            /**
             * X|Y Frequency Calculation
             */
            // Single Frequency
            long double x_y_frequency = (long double)(1)/(long double)(x_tensor.size());
            // Accumilation Check
            x_y_frequency_iterator = x_y_frequencies.find(join(x_tensor(i), y_tensor(i)));
            if(x_y_frequency_iterator == x_y_frequencies.end())
                // Frequency Injection
                x_y_frequencies.insert(std::pair<string, long double>(join(x_tensor(i), y_tensor(i)), x_y_frequency)) ;
            else 
                // Frequency Accumilation
                x_y_frequencies[join(x_tensor(i), y_tensor(i))] += x_y_frequency;
            /**
             * Y Frequency Calculation
             */
            // Single Frequency
            long double y_frequency = (long double)(1)/(long double)(x_tensor.size());
            y_frequency_iterator = y_frequencies.find(y_tensor(i));
            if(y_frequency_iterator == y_frequencies.end())
                // Frequency Injection
                y_frequencies.insert(std::pair<dtype, long double>(y_tensor(i), y_frequency)) ;
            else 
                // Frequency Accumilation
                y_frequencies[y_tensor(i)] += y_frequency;
            /**
             * X Frequency Calculation
             */
            // Single Frequency
            long double x_frequency = (long double)(1)/(long double)(x_tensor.size());
            x_frequency_iterator = x_frequencies.find(x_tensor(i));
            if(x_frequency_iterator == x_frequencies.end())
                // Frequency Injection
                x_frequencies.insert(std::pair<dtype, long double>(x_tensor(i), x_frequency)) ;
            else 
                // Frequency Accumilation
                x_frequencies[x_tensor(i)] += x_frequency;
        }
        // get result and allocate it to output
        output_tensor(0) = entropy(y_frequencies) + entropy(x_frequencies) - entropy(x_y_frequencies);
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
