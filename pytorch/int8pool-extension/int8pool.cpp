#include <torch/extension.h>

/* Actual Tensor Core Function */
torch::Tensor tensor_core_int8_pool(
        torch::Tensor& input, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding);

torch::Tensor tensor_core_int8_pool_backward(
        torch::Tensor& y, 
        torch::Tensor& dy, 
        torch::Tensor& x, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding);

torch::Tensor tensor_core_average_pool(
        torch::Tensor& input, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding);

torch::Tensor tensor_core_average_pool_backward(
        torch::Tensor& y, 
        torch::Tensor& dy, 
        torch::Tensor& x, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/* Extension Interface */
torch::Tensor int8_pool(
        torch::Tensor input, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding){
    CHECK_INPUT(input);
    return tensor_core_int8_pool(input, kernel_size, stride, padding);
}

torch::Tensor int8_pool_backward(
        torch::Tensor y, 
        torch::Tensor dy, 
        torch::Tensor x, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding){
    CHECK_INPUT(y);
    CHECK_INPUT(dy);
    CHECK_INPUT(x);
    return tensor_core_int8_pool_backward(y, dy, x, kernel_size, stride, padding);
}

torch::Tensor average_pool(
        torch::Tensor input, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding){
    CHECK_INPUT(input);
    return tensor_core_average_pool(input, kernel_size, stride, padding);
}

torch::Tensor average_pool_backward(
        torch::Tensor y, 
        torch::Tensor dy, 
        torch::Tensor x, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding){
    CHECK_INPUT(y);
    CHECK_INPUT(dy);
    CHECK_INPUT(x);
    return tensor_core_average_pool_backward(y, dy, x, kernel_size, stride, padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_pool", &int8_pool, "int8 max pooling forward Nvidia GPU tensor core");
  m.def("int8_pool_backward", &int8_pool_backward, "int8 max pooling forward Nvidia GPU tensor core");
  m.def("average_pool", &average_pool, "average pooling forward Nvidia GPU tensor core");
  m.def("average_pool_backward", &average_pool_backward, "average pooling backward Nvidia GPU tensor core");
}
