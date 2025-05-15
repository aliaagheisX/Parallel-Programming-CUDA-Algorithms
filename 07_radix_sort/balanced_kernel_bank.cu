#include<stdint-gcc.h>
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define BLOCK_DIM 256
#define CONFLICT_FREE_OFFSET(n)((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

__device__ void add_prev_block_bank(int* res, int* sh_mem, int actualBlockIdx, const uint32_t sz, const int arr_idx, int *blocks_finished) {
    // handle block 0
    __syncthreads();
    // wait for previous
    __shared__ int prev_sum;
    while(atomicAdd(blocks_finished, 0) != actualBlockIdx) {__nanosleep(100);}
    if(threadIdx.x == 0) {
        prev_sum = actualBlockIdx == 0 ? 0 : res[2*blockDim.x - 1 + 2 * blockDim.x * (actualBlockIdx - 1)];
    }
    __syncthreads();
    // let's go & sum previous
    // 3. write output
    int ai = threadIdx.x; ai += CONFLICT_FREE_OFFSET(ai);
    int bi = threadIdx.x + blockDim.x; bi += CONFLICT_FREE_OFFSET(bi);
    // write once in result
    if(arr_idx < sz)
        res[arr_idx] = sh_mem[ai] + prev_sum;
    if(arr_idx + blockDim.x < sz)
        res[arr_idx + blockDim.x] = sh_mem[bi] + prev_sum;

    __syncthreads();
    if(threadIdx.x == blockDim.x - 1) {
        __threadfence();
        atomicAdd(blocks_finished, 1);
    }    
}


__global__ void balanced_bank_kernel(int* arr, int* res, const uint32_t sz, int* block_counter, int *blocks_finished) {
    // 1. get actual blockIdx
    __shared__ int actualBlockIdx;
    if(threadIdx.x == 0) {
        actualBlockIdx = atomicAdd(block_counter, 1);
    }
    __syncthreads(); // ensure all threads see updated



    const int arr_idx = threadIdx.x + 2 * blockDim.x * actualBlockIdx; // blockDim.x - 1 + 2 * blockDim.x * actualBlockIdx
    const int th = threadIdx.x;
    __shared__ int sh_mem[2 * BLOCK_DIM + 2 * BLOCK_DIM / NUM_BANKS]; // add padding to sh_mem
    // 1. get data into sh_mem
    int ai = th; ai += CONFLICT_FREE_OFFSET(ai);
    int bi = th + blockDim.x; bi += CONFLICT_FREE_OFFSET(bi);
    sh_mem[ai] = arr_idx >= sz ? 0 : arr[arr_idx]; // [arr[0], arr[1], arr[2], arr[3]]
    sh_mem[bi] = arr_idx  + blockDim.x >= sz ? 0 : arr[arr_idx  + blockDim.x];
    __syncthreads();
    // 2. upsweep
    uint64_t offset = 1;
    for(uint32_t d = blockDim.x; d > 0; d>>= 1) {
        ai = 2*offset*(th + 1) - 1; ai += CONFLICT_FREE_OFFSET(ai);
        bi = 2*offset*(th + 1) - offset - 1; bi += CONFLICT_FREE_OFFSET(bi);
        if(th < d) {
            sh_mem[ai] += sh_mem[bi];
        }
        offset <<= 1;
        __syncthreads();
    }

    // 3. downsweep
    for(uint32_t d = 1; d <= blockDim.x; d<<= 1) {
        offset >>= 1;
        ai = 2*offset*(th + 1) + offset - 1;  ai += CONFLICT_FREE_OFFSET(ai);
        bi = 2*offset*(th + 1) - 1;  bi += CONFLICT_FREE_OFFSET(bi);
        if(ai < 2 * BLOCK_DIM + 2 * BLOCK_DIM / NUM_BANKS)  {
            sh_mem[ai] += sh_mem[bi];
        }
        __syncthreads();
    }
    
    add_prev_block_bank(res, sh_mem, actualBlockIdx, sz, arr_idx, blocks_finished);
}