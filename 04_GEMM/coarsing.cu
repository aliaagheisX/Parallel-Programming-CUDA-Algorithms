// #include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>

#define I_TILE_DIM 8
#define CEIL_DIV(X, Y) (X+Y-1)/Y
#define INDEX(Y, X, W) (X + Y * W)

__global__ void matrix_mult_coarsing_kernel(const float *A, const float *B, float *C, const int N, const int K, const int M) {

    const uint row = threadIdx.y + blockIdx.y * blockDim.y;
    const uint col = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float Ash[I_TILE_DIM][I_TILE_DIM];
    __shared__ float Bsh[I_TILE_DIM][I_TILE_DIM];
    
    float tempVal = 0.0;
    
    for(int tile = 0; tile < K; tile += I_TILE_DIM) {
        
        int tiledRow = tile + threadIdx.y;
        int tiledCol = tile + threadIdx.x;
        // Load A tile
        if (row < N && tiledCol < K)
            Ash[threadIdx.y][threadIdx.x] = A[INDEX(row, tiledCol, K)];
        else
            Ash[threadIdx.y][threadIdx.x] = 0.0f;
        // Load B tile
        if (tiledRow < K && col < M)
            Bsh[threadIdx.y][threadIdx.x] = B[INDEX(tiledRow, col, M)];
        else
            Bsh[threadIdx.y][threadIdx.x] = 0.0f;

            
        __syncthreads();
        
        if(row < N && col < M) {
            for(int i = 0;i < I_TILE_DIM;i++) {
                tempVal += Ash[threadIdx.y][i] * Bsh[i][threadIdx.x];
            }
        }
        __syncthreads();
    }
    
    if(row < N && col < M) {
        C[INDEX(row, col, M)] = tempVal;
    }
}


at::Tensor matrix_mult_coarsing(at::Tensor A, at::Tensor B) {
    const auto N = A.size(0);
    const auto K = A.size(1);
    const auto M = B.size(1);

    // auto result_dim = at::
    auto C = at::empty_like(A);

    // int shared_mem_size = I_TILE_DIM * I_TILE_DIM * sizeof(float);
    dim3 threads_per_block(I_TILE_DIM, I_TILE_DIM);
    dim3 number_of_blocks(CEIL_DIV(M, I_TILE_DIM),
                          CEIL_DIV(N, I_TILE_DIM));

    

    matrix_mult_coarsing_kernel<<<number_of_blocks, threads_per_block>>> (
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N, 
        K, 
        M
    );
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel failed to launch: ", cudaGetErrorString(err));
    }


    return C;
}


// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("matrix_mult_simple", &matrix_mult_simple, "matrix_mult_simple");
// }