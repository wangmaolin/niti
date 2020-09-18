#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>
#include <helper_cuda.h>

int roundoff( int v, int d ) {
	return ( v + d - 1 ) / d * d;
}

torch::Tensor tensor_core_int8_mm(torch::Tensor lhs,torch::Tensor rhs)
{
    int32_t alpha = 1;
    int32_t beta = 0;
    /* only support m,n,k multiply of 4 */
    int m = lhs.size(0);
    int k = lhs.size(1);
    int n = rhs.size(1);

    int lda = k;
    int ldb = n;
    int ldc = m;
    // create the result tensor in a transposed way
    
    //auto results=torch::zeros({n,m},torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    auto results=torch::empty({n,m},torch::dtype(torch::kInt32).device(torch::kCUDA, 0));
    
    cublasHandle_t handle;
	checkCudaErrors(cublasCreate( &handle ));
    // Pytorch is row major, cublas is column major
    // need to use TT version gemm 
    cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_T,
            m,n,k,&alpha,
            lhs.data<int8_t>(),CUDA_R_8I,lda,
            rhs.data<int8_t>(),CUDA_R_8I,ldb,
            &beta,results.data<int32_t>(),
            CUDA_R_32I,ldc,CUDA_R_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    checkCudaErrors(cublasDestroy( handle )); 
    // the result here is column major
    // need to tranpose it for pytorch usage
    return results.transpose(0,1);
}

