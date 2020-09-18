/*
 * This example demonstrates how to use the cuBLASGEmmEx 
 * to perform int8 matrix multiply using tensor core
 * m,n,k must be multiply of 4
 * This example requires compute capability 7.2 or greater.
 */
#include <torch/extension.h>

torch::Tensor tensor_core_int8_mm(torch::Tensor lhs,torch::Tensor rhs);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor int8_mm(torch::Tensor lhs,torch::Tensor rhs) 
{
    CHECK_INPUT(lhs);
    CHECK_INPUT(rhs);
    return tensor_core_int8_mm(lhs, rhs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_mm", &int8_mm, "int8 matrix multiply using Nvidia GPU tensor core");
}

