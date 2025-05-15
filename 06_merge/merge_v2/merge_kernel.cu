#pragma once
#include <stdint.h>

#define ELE_PER_TH 6
#define BLOCK_DIM 512
#define ELE_PER_BLOCK (ELE_PER_TH*BLOCK_DIM)

__host__ __device__ int find_corank(int* A, int* B, uint32_t m, uint32_t n, uint32_t k) {
    uint32_t l = k > n ? k - n : 0; 
    uint32_t r = k < m ? k : m; 
    uint32_t i, j;
    while(l <= r) {
        i = (l + r) / 2;
        j = k - i;
        if(j > 0 && i < m  && B[j - 1] > A[i])
            l = i + 1;
        else if(i > 0 && j < n &&  A[i - 1] > B[j])
            r = i - 1;
        else
            return i;
    }
    return l;
}

__global__ void merge_sorted_array_kernel_v2(int* A, int* B, int* C, int *lasti, const uint32_t k_mx, uint32_t m, uint32_t n) {
    __shared__ int A_sh[ELE_PER_BLOCK];
    __shared__ int B_sh[ELE_PER_BLOCK];
    __shared__ int C_sh[ELE_PER_BLOCK];
    __shared__ uint32_t block_start_k, block_next_k;
    __shared__ uint32_t block_start_i, block_next_i;
    __shared__ uint32_t block_start_j, block_next_j;
    if(threadIdx.x == 0) {
        //TODO: haha
        if(blockIdx.x == 0){
            *lasti = find_corank(A, B, m, n, k_mx);
        }
        block_start_k = ELE_PER_TH * (blockIdx.x * blockDim.x);
        block_next_k  = block_start_k + ELE_PER_BLOCK > k_mx ? k_mx : block_start_k + ELE_PER_BLOCK ;

        block_start_i = find_corank(A, B, m, n, block_start_k);
        block_next_i = find_corank(A, B, m, n, block_next_k);
        
        block_start_j = block_start_k - block_start_i;
        block_next_j = block_next_k - block_next_i;
    }
    __syncthreads();
    uint32_t m_sh = block_next_i - block_start_i;
    uint32_t n_sh = block_next_j - block_start_j;
    for(int i = threadIdx.x; i < m_sh; i += BLOCK_DIM) {
        A_sh[i] = A[block_start_i + i];
    }
    for(int j = threadIdx.x;  j < n_sh; j += BLOCK_DIM) {
        B_sh[j] = B[block_start_j + j];
    }
    __syncthreads();
    
    
    uint32_t k = threadIdx.x * ELE_PER_TH;
    if(k < n_sh + m_sh) {

        uint32_t i = find_corank(A_sh, B_sh, m_sh, n_sh, k);
        uint32_t j = k - i;

        for(int d = 0; d < ELE_PER_TH && k + d < n_sh + m_sh; d++) {
            if(j >= (block_next_j - block_start_j)) C_sh[k + d] = A_sh[i++];
            else if(i >= (block_next_i - block_start_i)) C_sh[k + d] = B_sh[j++];
            else if(A_sh[i] <= B_sh[j]) C_sh[k + d] = A_sh[i++];
            else C_sh[k + d] = B_sh[j++];
        }

    }
    __syncthreads();
    for(int i = threadIdx.x; i < n_sh + m_sh; i+=BLOCK_DIM)
        C[block_start_k + i] = C_sh[i];
}