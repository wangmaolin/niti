#include <torch/extension.h>

torch::Tensor im2col_cuda_int(
        const torch::Tensor& input,
        torch::IntArrayRef kernel_size,
        torch::IntArrayRef dilation,
        torch::IntArrayRef padding,
        torch::IntArrayRef stride);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor int_im2col(
        torch::Tensor input,
        torch::IntArrayRef kernel_size,
        torch::IntArrayRef dilation,
        torch::IntArrayRef padding,
        torch::IntArrayRef stride){
    CHECK_INPUT(input);
    return im2col_cuda_int(input, kernel_size, dilation, padding, stride);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("im2col", &int_im2col, "image to column cuda function for integers");
}

