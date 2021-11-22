#include <torch/extension.h>

/* Actual Tensor Core Function */
//torch::Tensor tensor_core_dgrad(
        //torch::Tensor& err_in, 
        //torch::Tensor& weight);

torch::Tensor tensor_core_wgrad(
        torch::Tensor& err_in, 
        torch::Tensor& act_in,
        torch::IntArrayRef weight_shape);

torch::Tensor tensor_core_int8_conv(
        torch::Tensor& input, 
        torch::Tensor& weight);

torch::Tensor tensor_core_int8_conv_optimized(
        torch::Tensor& input, 
        torch::Tensor& weight);

torch::Tensor tensor_core_int4_conv(
        torch::Tensor& input, 
        torch::Tensor& weight);

torch::Tensor tensor_core_int4_conv_optimized(
        torch::Tensor& input, 
        torch::Tensor& weight);

// accept stride and padding
torch::Tensor tensor_core_sp_conv(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h,
        int32_t padding_h);

// accept stride and padding
torch::Tensor tensor_core_sp_conv_optimized(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride_h,
        int32_t padding_h);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

/* Extension Interface */
// stride 1, padding 1, dilation 1, kernel 3x3
torch::Tensor int8_conv(torch::Tensor input, 
        torch::Tensor weight){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_int8_conv(input, weight);
}

// stride, padding, group
torch::Tensor sp_conv(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h,
        int32_t padding_h){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_sp_conv(input, weight, stride_h, padding_h);
}

torch::Tensor sp_conv_optimized(torch::Tensor input, 
        torch::Tensor weight,
        int32_t stride_h,
        int32_t padding_h){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_sp_conv_optimized(input, weight, stride_h, padding_h);
}


torch::Tensor int8_conv_optimized(torch::Tensor input, 
        torch::Tensor weight){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_int8_conv_optimized(input, weight);
}

torch::Tensor int4_conv(torch::Tensor input, 
        torch::Tensor weight){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_int4_conv(input, weight);
}

torch::Tensor int4_conv_optimized(torch::Tensor input, 
        torch::Tensor weight){
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    return tensor_core_int4_conv_optimized(input, weight);
}

torch::Tensor weight_grad(torch::Tensor err_in,
        torch::Tensor act_in,
        torch::IntArrayRef weight_shape){
    CHECK_INPUT(err_in);
    CHECK_INPUT(act_in);
    return tensor_core_wgrad(err_in, act_in, weight_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_conv", &int8_conv, "int8 convolution forward Nvidia GPU tensor core");
  m.def("sp_conv", &sp_conv, "int8 convolution forward Nvidia GPU tensor core");
  m.def("sp_conv_optimized", &sp_conv_optimized, "int8 convolution forward Nvidia GPU tensor core");
  m.def("int8_conv_optimized", &int8_conv_optimized, "int8 convolution forward Nvidia GPU tensor core");
  m.def("int4_conv", &int4_conv, "int4 convolution forward Nvidia GPU tensor core");
  m.def("int4_conv_optimized", &int4_conv_optimized, "int4 convolution forward Nvidia GPU tensor core");
  m.def("weight_grad", &weight_grad, "int4 convolution forward Nvidia GPU tensor core");
}
