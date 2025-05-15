#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

// Kernel definitions
#define TILE_DIM 32
#define NCOLS 200
#define NROWS 300
#define GET_IDX(y, x, ncols) (x + y * ncols)
#define CEIL_DIV(X, Y) ((X+Y-1)/Y)

// CUDA kernel (as provided)
__global__ void matrix_transpose_kernel(int *matrix, int *matrix_out) {
    __shared__ int sh_mem[TILE_DIM][TILE_DIM];
    int global_x = threadIdx.x + blockIdx.x * TILE_DIM;
    int global_y = threadIdx.y + blockIdx.y * TILE_DIM;
    
    if(global_x < NCOLS && global_y < NROWS) {
        sh_mem[threadIdx.y][threadIdx.x] = matrix[GET_IDX(global_y, global_x, NCOLS)];
    }
    
    __syncthreads();

    int out_global_x = threadIdx.x + blockIdx.y * TILE_DIM;
    int out_global_y = threadIdx.y + blockIdx.x * TILE_DIM;

    if (out_global_x < NROWS && out_global_y < NCOLS) {
        matrix_out[GET_IDX(out_global_y, out_global_x, NROWS)] = sh_mem[threadIdx.x][threadIdx.y];
    }
}

// CPU function to compute matrix transpose for validation
void cpu_matrix_transpose(int *matrix, int *matrix_out, int nrows, int ncols) {
    for (int y = 0; y < nrows; ++y) {
        for (int x = 0; x < ncols; ++x) {
            matrix_out[GET_IDX(x, y, nrows)] = matrix[GET_IDX(y, x, ncols)];
        }
    }
}

int main() {
    // Host memory
    int *h_matrix = new int[NROWS * NCOLS];
    int *h_matrix_out = new int[NROWS * NCOLS];
    int *h_expected_out = new int[NROWS * NCOLS];

    // Device memory
    int *d_matrix, *d_matrix_out;
    cudaMalloc(&d_matrix, NROWS * NCOLS * sizeof(int));
    cudaMalloc(&d_matrix_out, NROWS * NCOLS * sizeof(int));

    // Initialize input matrix with unique values
    for (int y = 0; y < NROWS; ++y) {
        for (int x = 0; x < NCOLS; ++x) {
            h_matrix[GET_IDX(y, x, NCOLS)] = y * NCOLS + x;
        }
    }

    // Compute expected output on CPU
    cpu_matrix_transpose(h_matrix, h_expected_out, NROWS, NCOLS);

    // Copy input matrix to device
    cudaMemcpy(d_matrix, h_matrix, NROWS * NCOLS * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid(CEIL_DIV(NCOLS, TILE_DIM), CEIL_DIV(NROWS, TILE_DIM));
    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_matrix_out);
    cudaDeviceSynchronize();

    // Copy output back to host
    cudaMemcpy(h_matrix_out, d_matrix_out, NROWS * NCOLS * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify results
    bool passed = true;
    for (int i = 0; i < NROWS * NCOLS; ++i) {
        if (h_matrix_out[i] != h_expected_out[i]) {
            std::cerr << "Mismatch at index " << i << ": expected " << h_expected_out[i]
                      << ", got " << h_matrix_out[i] << std::endl;
            passed = false;
            break;
        }
    }

    // Print result
    if (passed) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }

    // Cleanup
    delete[] h_matrix;
    delete[] h_matrix_out;
    delete[] h_expected_out;
    cudaFree(d_matrix);
    cudaFree(d_matrix_out);

    return passed ? 0 : 1;
}