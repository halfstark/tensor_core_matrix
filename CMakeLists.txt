cmake_minimum_required(VERSION 3.26)

project(tensor CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(cuBLAS REQUIRED)
include_directories(/home/star/workspace/cuda/cuda_blas/include)
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86 -allow-unsupported-compiler")

add_executable(tensor gemm.cu)
target_link_libraries(tensor ${CUBLAS_LIBRARIES} CUDA::curand)