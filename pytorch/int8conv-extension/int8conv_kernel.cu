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

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

void tensor_core_find_algo(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride,
        int32_t padding,
        int32_t dilation,
        int32_t float_flag){

    /* only support n_in and c_in multiply of 4 */

    cudnnHandle_t cudnnHandle = at::native::getCudnnHandle();

    cudnnDataType_t input_type;
    cudnnDataType_t output_type;
    cudnnDataType_t conv_type;
    cudnnTensorFormat_t input_format;
    cudnnTensorFormat_t output_format;

    if (float_flag == 0){
        conv_type = CUDNN_DATA_INT32;

        //input_type = CUDNN_DATA_INT8;
        //input_format = CUDNN_TENSOR_NHWC;
        input_type = CUDNN_DATA_INT8x4;
        input_format = CUDNN_TENSOR_NCHW_VECT_C;

        //output_type = CUDNN_DATA_INT8x4;
        //output_format = CUDNN_TENSOR_NCHW_VECT_C;

        output_type = CUDNN_DATA_FLOAT;
        //output_type = CUDNN_DATA_INT8;
        output_format = CUDNN_TENSOR_NHWC;
    }
    else{
        conv_type = CUDNN_DATA_FLOAT;

        input_type = CUDNN_DATA_FLOAT;
        input_format = CUDNN_TENSOR_NHWC;

        output_type = CUDNN_DATA_FLOAT;
        output_format = CUDNN_TENSOR_NHWC;
    }

    int32_t n_in = input.size(0);
    int32_t h_in = input.size(1);
    int32_t w_in = input.size(2);
    int32_t c_in = input.size(3);
    cudnnTensorDescriptor_t xDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, 
                input_format, 
                input_type, 
                n_in, c_in, h_in, w_in));
    //std::cout<<n_in<<' '<<h_in<<' '<<w_in<<' '<<c_in<<' '<<std::endl;

    int32_t n_weight= weight.size(0);
    int32_t h_weight = weight.size(1);
    int32_t w_weight = weight.size(2);
    int32_t c_weight = weight.size(3);
    cudnnFilterDescriptor_t wDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(wDesc, 
                input_type, 
                input_format, 
                n_weight, c_weight, h_weight, w_weight));

    //std::cout<<n_weight<<' '<<h_weight<<' '<<w_weight<<' '<<c_weight<<' '<<std::endl;

    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding, padding, stride, stride, dilation, dilation, 
                CUDNN_CROSS_CORRELATION,
                conv_type));

    //std::cout<<"create conv descriptor"<<std::endl;

    int32_t n_out;
    int32_t h_out;
    int32_t w_out;
    int32_t c_out;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n_out, &c_out, &h_out, &w_out));
    //std::cout<<n_out<<' '<<h_out<<' '<<w_out<<' '<<c_out<<' '<<std::endl;

    cudnnTensorDescriptor_t yDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, 
                output_format, 
                output_type, 
                n_out, c_out, h_out, w_out));

    //std::cout<<"create y tensor"<<std::endl;
    //auto y = torch::empty({n_out, h_out, w_out, c_out}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    cudnnConvolutionFwdAlgoPerf_t perfResults[3];
    int32_t algo_cnt;
    checkCUDNN(cudnnFindConvolutionForwardAlgorithm(
                cudnnHandle, 
                xDesc, 
                wDesc, 
                convDesc, 
                yDesc, 
                3, 
                &algo_cnt, 
                //&perfResults));
                perfResults));

    std::cout<<"float flag: "<<float_flag<<std::endl;
    std::cout<<"conv algorithm count: "<<algo_cnt<<std::endl;
    for (int i=0; i<algo_cnt;i++){
        std::cout<<"algo: "<<perfResults[i].algo<<std::endl;
        std::cout<<"time: "<<perfResults[i].time<<std::endl;
        std::cout<<"memory: "<<perfResults[i].memory<<std::endl;
        std::cout<<"mathType: "<<perfResults[i].mathType<<std::endl;
        std::cout<<"==============="<<std::endl;
    }

    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    // this is algo 0
    //std::cout<<"algo: "<<CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM<<std::endl;
    // this is algo 1
    //std::cout<<"algo: "<<CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM<<std::endl;
    // according to find algo function, should use algo 1
}

torch::Tensor tensor_core_int8_conv(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride,
        int32_t padding,
        int32_t dilation){

    /* only support n_in and c_in multiply of 4 */

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

    int32_t n_weight= weight.size(0);
    int32_t h_weight = weight.size(1);
    int32_t w_weight = weight.size(2);
    int32_t c_weight = weight.size(3);
    cudnnFilterDescriptor_t wDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(wDesc, 
                CUDNN_DATA_INT8, 
                CUDNN_TENSOR_NHWC, 
                n_weight, c_weight, h_weight, w_weight));


    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding, padding, stride, stride, dilation, dilation, 
                CUDNN_CROSS_CORRELATION,
                CUDNN_DATA_INT32));

    int32_t n_out;
    int32_t h_out;
    int32_t w_out;
    int32_t c_out;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n_out, &c_out, &h_out, &w_out));

    cudnnTensorDescriptor_t yDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_FLOAT, 
                n_out, c_out, h_out, w_out));

    //std::cout<<"create y tensor"<<std::endl;
    auto y = torch::empty({n_out, h_out, w_out, c_out}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
   
    float alpha = 1.0;
    //float alpha = 1;
    float beta = 0.0;

    //size_t ws_size = 355968;
    size_t ws_size;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,xDesc,wDesc,convDesc,yDesc,algo,&ws_size));
    auto workspace = torch::empty({static_cast<int64_t>(ws_size)}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                &alpha,xDesc,input.data<int8_t>(),
                wDesc,weight.data<int8_t>(),
                convDesc,
                algo,
                workspace.data<int32_t>(),
                ws_size,
                &beta,yDesc,
                y.data<float>()));

     checkCUDNN(cudnnDestroyTensorDescriptor(yDesc));
     checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
     checkCUDNN(cudnnDestroyFilterDescriptor(wDesc));
     checkCUDNN(cudnnDestroyTensorDescriptor(xDesc));

     return y;
}

torch::Tensor tensor_core_group_conv(
        torch::Tensor& input, 
        torch::Tensor& weight,
        int32_t stride,
        int32_t padding,
        int32_t dilation,
        int32_t groups){

    /* only support n_in and c_in multiply of 4 */

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

    int32_t n_weight= weight.size(0);
    int32_t h_weight = weight.size(1);
    int32_t w_weight = weight.size(2);
    int32_t c_weight = weight.size(3);
    cudnnFilterDescriptor_t wDesc;
    checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
    checkCUDNN(cudnnSetFilter4dDescriptor(wDesc, 
                CUDNN_DATA_INT8, 
                CUDNN_TENSOR_NHWC, 
                n_weight, c_weight, h_weight, w_weight));

    cudnnConvolutionDescriptor_t convDesc;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding, padding, stride, stride, dilation, dilation, 
                CUDNN_CROSS_CORRELATION,
                CUDNN_DATA_INT32));

    checkCUDNN(cudnnSetConvolutionGroupCount(convDesc,groups));

    int32_t n_out;
    int32_t h_out;
    int32_t w_out;
    int32_t c_out;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n_out, &c_out, &h_out, &w_out));

    cudnnTensorDescriptor_t yDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(yDesc, 
                CUDNN_TENSOR_NHWC, 
                CUDNN_DATA_FLOAT, 
                n_out, c_out, h_out, w_out));

    //std::cout<<"create y tensor"<<std::endl;
    auto y = torch::empty({n_out, h_out, w_out, c_out}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    //cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
   
    float alpha = 1.0;
    //float alpha = 1;
    float beta = 0.0;

    //size_t ws_size = 355968;
    size_t ws_size;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,xDesc,wDesc,convDesc,yDesc,algo,&ws_size));
    auto workspace = torch::empty({static_cast<int64_t>(ws_size)}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                &alpha,xDesc,input.data<int8_t>(),
                wDesc,weight.data<int8_t>(),
                convDesc,
                algo,
                workspace.data<int32_t>(),
                ws_size,
                &beta,yDesc,
                y.data<float>()));

     checkCUDNN(cudnnDestroyTensorDescriptor(yDesc));
     checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
     checkCUDNN(cudnnDestroyFilterDescriptor(wDesc));
     checkCUDNN(cudnnDestroyTensorDescriptor(xDesc));

     return y;
}
