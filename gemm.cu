#include <cstdio>
#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>
#include <stdio.h>
using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}
__global__ void wmma_example(half *a, half *b, float* c, int M, int N, int K, float alpha, float beta) {
    int lda = M;
    int ldb = K;
    int ldc = M;

    int warpM = (blockIdx.y * blockDim.x + threadIdx.x)/warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bcol = warpN * WMMA_N;
        if (aRow < M && aCol < K && bRow < K && bcol < N) {
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bcol * ldb, ldb);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;
   if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, c+cRow+cCol*ldc, ldc, wmma::mem_col_major);
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   }
}
__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}
#define MATRIX_M 16384
#define MATRIX_N 16384
#define MATRIX_K 16384
int main() {
   float *a_fp32;
   float *b_fp32;
   half *a_fp16;
   half *b_fp16;

   float *c;
   float *c_cublas;
   float *c_wmma;

   float *c_host_cublas;
   float *c_host_wmma;

   curandGenerator_t gen;
   cublasHandle_t cublasHandle;
   
   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;
   
   cudaEvent_t startcublas;
   cudaEvent_t stopcublas;

   cudaEventCreate(&startcublas);
   cudaEventCreate(&stopcublas);

   cudaEventCreate(&startWMMA);
   cudaEventCreate(&stopWMMA);
   cublasCreate_v2(&cublasHandle);
   cublasSetMathMode(cublasHandle, CUBLAS_PEDANTIC_MATH);
   cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

   cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));

   // 分配结果空间
   c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));


   curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

   curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
   curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));

   convertFp32ToFp16<<<(MATRIX_M * MATRIX_K + 255)/256, 256>>>(a_fp16, a_fp32, MATRIX_M*MATRIX_K);
   convertFp32ToFp16<<<(MATRIX_K * MATRIX_N + 255)/256, 256>>>(b_fp16, b_fp32, MATRIX_K*MATRIX_N);

   curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));
   cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
   cudaErrCheck(cudaMemcpy(c_wmma, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));

    float alpha = 2.0f;
   float beta = 2.0f;


    printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

    dim3 gridDim;
    dim3 blockDim;
    
    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
    cudaErrCheck(cudaEventRecord(startWMMA));
    wmma_example<<<gridDim, blockDim>>>(a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K ,alpha,  beta);
    cudaEventRecord(stopWMMA);
    cudaEventSynchronize(stopWMMA);
    float time;
    cudaEventElapsedTime(&time, startWMMA, stopWMMA);
    printf("tesnor core time cost:%f:ms\n", time);
    cudaErrCheck(cudaEventRecord(startcublas));
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                    MATRIX_M, MATRIX_N, MATRIX_K,
                    &alpha,
                    a_fp16, CUDA_R_16F, MATRIX_M,
                    b_fp16, CUDA_R_16F, MATRIX_K,
                    &beta,
                    c_cublas, CUDA_R_32F, MATRIX_M,
                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    cudaErrCheck(cudaEventRecord(stopcublas));
    cudaErrCheck(cudaEventSynchronize(stopcublas));
    cudaEventElapsedTime(&time, startcublas, stopcublas);

    printf("blas time cost:%f:ms\n", time);

}