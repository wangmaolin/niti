#include <torch/extension.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <ATen/cudnn/Handle.h>

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

torch::Tensor tensor_core_int8_pool(
        torch::Tensor& input, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding){

    cudnnHandle_t cudnnHandle = at::native::getCudnnHandle();

    cudnnTensorDescriptor_t xDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));

    int32_t n_in = input.size(0);
    int32_t h_in = input.size(1);
    int32_t w_in = input.size(2);
    int32_t c_in = input.size(3);
    checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_INT8, 
                n_in, c_in, h_in, w_in));

    cudnnPoolingDescriptor_t poolDesc;
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
    checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 
                kernel_size, kernel_size, padding, padding, stride, stride));
    
    int32_t n_out;
    int32_t h_out;
    int32_t w_out;
    int32_t c_out;
    checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc,xDesc,&n_out,&c_out,&h_out,&w_out));

    cudnnTensorDescriptor_t yDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_INT8, 
                n_out, c_out, h_out, w_out));

    auto y = torch::empty({n_out, h_out, w_out, c_out}, torch::dtype(torch::kInt8).device(torch::kCUDA, 0));
   
    float alpha = 1.0;
    float beta = 0.0;

    checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc,
                &alpha, xDesc, input.data<int8_t>(),
                &beta, yDesc, y.data<int8_t>()));

     checkCUDNN(cudnnDestroyTensorDescriptor(yDesc));
     checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
     checkCUDNN(cudnnDestroyTensorDescriptor(xDesc));

     return y;
}

torch::Tensor tensor_core_int8_pool_backward(
        torch::Tensor& y, 
        torch::Tensor& dy, 
        torch::Tensor& x, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding){
    cudnnHandle_t cudnnHandle = at::native::getCudnnHandle();

    cudnnTensorDescriptor_t yDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
    int32_t n_y = y.size(0);
    int32_t h_y = y.size(1);
    int32_t w_y = y.size(2);
    int32_t c_y = y.size(3);
    checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_HALF, 
                n_y, c_y, h_y, w_y));

    cudnnTensorDescriptor_t xDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));
    int32_t n_x= x.size(0);
    int32_t h_x= x.size(1);
    int32_t w_x= x.size(2);
    int32_t c_x= x.size(3);
    checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_HALF, 
                n_x, c_x, h_x, w_x));

    cudnnPoolingDescriptor_t poolDesc;
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
    checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 
                kernel_size, kernel_size, padding, padding, stride, stride));

    auto dx = torch::empty({n_x, h_x, w_x, c_x}, torch::dtype(torch::kHalf).device(torch::kCUDA, 0));

    float alpha = 1.0;
    float beta = 0.0;
    checkCUDNN(cudnnPoolingBackward(cudnnHandle,poolDesc,
                &alpha, yDesc, 
                y.data<at::Half>(),
                yDesc,
                dy.data<at::Half>(),
                xDesc,
                x.data<at::Half>(),
                &beta,
                xDesc,
                dx.data<at::Half>()));

    checkCUDNN(cudnnDestroyTensorDescriptor(yDesc));
    checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(xDesc));

    return dx;
}

torch::Tensor tensor_core_average_pool(
        torch::Tensor& input, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding){

    cudnnHandle_t cudnnHandle = at::native::getCudnnHandle();

    cudnnTensorDescriptor_t xDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));

    int32_t n_in = input.size(0);
    int32_t h_in = input.size(1);
    int32_t w_in = input.size(2);
    int32_t c_in = input.size(3);
    checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_HALF, 
                n_in, c_in, h_in, w_in));

    cudnnPoolingDescriptor_t poolDesc;
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
    checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_PROPAGATE_NAN, 
                kernel_size, kernel_size, padding, padding, stride, stride));
    
    int32_t n_out;
    int32_t h_out;
    int32_t w_out;
    int32_t c_out;
    checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc,xDesc,&n_out,&c_out,&h_out,&w_out));

    cudnnTensorDescriptor_t yDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_HALF, 
                n_out, c_out, h_out, w_out));

    auto y = torch::empty({n_out, h_out, w_out, c_out}, torch::dtype(torch::kHalf).device(torch::kCUDA, 0));
   
    float alpha = 1.0;
    float beta = 0.0;

    checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc,
                &alpha, xDesc, input.data<at::Half>(),
                &beta, yDesc, y.data<at::Half>()));

     checkCUDNN(cudnnDestroyTensorDescriptor(yDesc));
     checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
     checkCUDNN(cudnnDestroyTensorDescriptor(xDesc));

     return y;
}

torch::Tensor tensor_core_average_pool_backward(
        torch::Tensor& y, 
        torch::Tensor& dy, 
        torch::Tensor& x, 
        int32_t kernel_size,
        int32_t stride,
        int32_t padding){
    cudnnHandle_t cudnnHandle = at::native::getCudnnHandle();

    cudnnTensorDescriptor_t yDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
    int32_t n_y = y.size(0);
    int32_t h_y = y.size(1);
    int32_t w_y = y.size(2);
    int32_t c_y = y.size(3);
    checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_HALF, 
                n_y, c_y, h_y, w_y));

    cudnnTensorDescriptor_t xDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));
    int32_t n_x= x.size(0);
    int32_t h_x= x.size(1);
    int32_t w_x= x.size(2);
    int32_t c_x= x.size(3);
    checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_HALF, 
                n_x, c_x, h_x, w_x));

    cudnnPoolingDescriptor_t poolDesc;
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
    checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_PROPAGATE_NAN, 
                kernel_size, kernel_size, padding, padding, stride, stride));

    auto dx = torch::empty({n_x, h_x, w_x, c_x}, torch::dtype(torch::kHalf).device(torch::kCUDA, 0));

    float alpha = 1.0;
    float beta = 0.0;
    checkCUDNN(cudnnPoolingBackward(cudnnHandle,poolDesc,
                &alpha, yDesc, 
                y.data<at::Half>(),
                yDesc,
                dy.data<at::Half>(),
                xDesc,
                x.data<at::Half>(),
                &beta,
                xDesc,
                dx.data<at::Half>()));

    checkCUDNN(cudnnDestroyTensorDescriptor(yDesc));
    checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(xDesc));

    return dx;
}
