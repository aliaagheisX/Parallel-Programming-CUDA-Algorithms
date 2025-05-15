#include <unistd.h>
#include<iostream>
#include<stdint-gcc.h>
#include<error.h>
#include<algorithm>

// 1. radix sort [x]
// 2. arg radix sort [x]
// 3. ASC, DESC [x]
// 4. float, signed numbers [x]
// 5. transpose SUM to allocate for every batch -> compine
// 6. streams ---> kill me please

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CUDA_CHECK_LAST() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}


#define BLOCK_DIM 4
// #define MAX_BATCH_SZ 32764
#define COARSING_FACTOR 2
#define RADIX 2
#define BUCKET_SZ 4 // 1<<RADIX
// get ith bit in x
#define GET_BIT(x,i) ((x >> i) & 1LL)
// get mask 000111 if radix = 3
#define MASK_ZERO ((1 << (RADIX)) - 1)
// get mask of iter 000111000 if iter = 1
#define MASK_ITER(iter) (MASK_ZERO << (iter*RADIX))
// get radix for certain iter
#define GET_RADIX_KEY(x,iter) ((x>>(iter*RADIX)) & MASK_ZERO)

#define CEIL_DIV(X, Y) ((X + Y - 1)/(Y))
#define CEIL_DIVI(X, Y) int((X + Y - 1)/(Y))

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n)((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

#define GET_IDX(y, x, cols) (x + y*cols)


__global__  void int_to_uint32(int* arr, uint32_t* res, int N) {
    // Flip the sign bit to make negative numbers come before positive ones in unsigned space
    int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx < N) {
        res[global_idx] = static_cast<uint32_t>(arr[global_idx]) ^ 0x80000000;
    }
}

__global__  void uint32_to_int(uint32_t* arr, int* res, int N) {
    // reverse process
    int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx < N) {
        res[global_idx] = static_cast<int32_t>(arr[global_idx] ^ 0x80000000);
    }
}

__global__  void float_to_uint32(float* arr, uint32_t* res, int N) {
    // reverse process
    int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx < N) {
        uint32_t bits = *reinterpret_cast<uint32_t*>(&arr[global_idx]);
        res[global_idx] = (bits & 0x80000000) ? ~bits : (bits ^ 0x80000000);
    }
}

__global__  void uint32_to_float(uint32_t* arr, float* res, int N) {
    // reverse process
    int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(global_idx < N) {
        uint32_t x = arr[global_idx];
        uint32_t bits = (x & 0x80000000) ? (x ^ 0x80000000) : ~x;
        res[global_idx] = *reinterpret_cast<float*>(&bits);
    }
}

struct SortOp
{
    __device__ virtual inline int get_radix(int x, int iter) const = 0;
    __device__ virtual inline int get_bit(int x, int bit) const = 0;
};
struct AscOp : SortOp
{
    __device__ inline int get_radix(int x, int iter) const override { return GET_RADIX_KEY(x, iter);  }
    __device__ inline int get_bit(int x, int bit) const override { return GET_BIT(x, bit); };
};
struct DescOp : SortOp
{
    __device__ inline int get_radix(int x, int iter) const override { return MASK_ZERO - GET_RADIX_KEY(x, iter);  }
    __device__ inline int get_bit(int x, int bit) const override { return 1-GET_BIT(x, bit); };
};

__device__ void add_prev_block_bank(uint32_t* res, uint32_t* sh_mem, uint32_t actualBlockIdx, const uint32_t sz, const uint32_t arr_idx, uint32_t *blocks_finished) {
    // handle block 0
    __syncthreads();
    // wait for previous
    __shared__ int prev_sum;
    while(atomicAdd(blocks_finished, 0) != actualBlockIdx) {}
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


__global__ void balanced_bank_kernel(uint32_t* arr, uint32_t* res, const uint32_t sz, uint32_t* block_counter, uint32_t *blocks_finished) {
    // 1. get actual blockIdx
    __shared__ int actualBlockIdx;
    if(threadIdx.x == 0) {
        actualBlockIdx = atomicAdd(block_counter, 1);
    }
    __syncthreads(); // ensure all threads see updated



    const uint32_t arr_idx = threadIdx.x + 2 * blockDim.x * actualBlockIdx; // blockDim.x - 1 + 2 * blockDim.x * actualBlockIdx
    const uint32_t th = threadIdx.x;
    __shared__ uint32_t sh_mem[2 * BLOCK_DIM + 2 * BLOCK_DIM / NUM_BANKS]; // add padding to sh_mem
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


// __device__ int COUNTER[CEIL_DIV(MAX_BATCH_SZ, BLOCK_DIM) * BUCKET_SZ];//should be accessed [BLOCKIdx * BACKET_SZ + radix]

__device__ bool scan_inefficent(uint32_t sh_mem[][BLOCK_DIM+1]) {
    bool buff_idx = 0; // buffer
    const uint32_t thidx = threadIdx.x + 1;
    for(uint32_t offset = 1; offset <= BLOCK_DIM; offset <<= 1) {
        if(thidx >= offset) {
            sh_mem[!buff_idx][thidx] = sh_mem[buff_idx][thidx] + sh_mem[buff_idx][thidx - offset];
        } else {
            sh_mem[!buff_idx][thidx] = sh_mem[buff_idx][thidx];
        }
        buff_idx = !buff_idx;
        __syncthreads();
    }
    return buff_idx;
}

template <typename OP>
__device__ void one_bit_sort(uint32_t* sh_A, uint32_t* res, uint32_t* old_idxs, uint32_t *new_idxs, uint32_t sh_mem[][BLOCK_DIM+1], const uint32_t bit, OP& op) {
    // 1. count
    sh_mem[0][0] = 0;
    sh_mem[0][threadIdx.x+1] = op.get_bit(sh_A[threadIdx.x], bit);
    __syncthreads();
    // 2. scan
    bool buff_idx = scan_inefficent(sh_mem);
    // 3. gather
    // for zeros it's my index - num of ones left me
    // for one it's (total_size - one in total + ones on left) 
    int num_ones = sh_mem[buff_idx][BLOCK_DIM];
    int ones_left = sh_mem[buff_idx][threadIdx.x];

    int new_idx = op.get_bit(sh_A[threadIdx.x], bit) ? 
                  (BLOCK_DIM - num_ones + ones_left) : // For 1s: place after all 0s
                  (threadIdx.x - ones_left);
    res[ new_idx ] = sh_A[threadIdx.x];
    new_idxs[ new_idx ] = old_idxs[threadIdx.x];
}
// global_counter[radix][blockIdx.x]
__device__ void update_glob_buckets(uint32_t* A, uint32_t* local_counter, const uint32_t iter, const uint32_t N, const uint32_t global_idx, uint32_t* global_counter, SortOp& op) {
    if(global_idx < N) {
        int radix = op.get_radix(A[threadIdx.x], iter);
        atomicAdd(&local_counter[radix], 1);
    }
    __syncthreads();
    if(threadIdx.x < BUCKET_SZ) {
        global_counter[GET_IDX(threadIdx.x, blockIdx.x, gridDim.x)] = local_counter[threadIdx.x];
    }
}

template <typename OP>
__global__ void radix_sort_local_kerenl(uint32_t *A, uint32_t *res, uint32_t* old_idxs, uint32_t *new_idxs, uint32_t* global_counter, const uint32_t N, const uint32_t iter) {
    OP op;
    __shared__ uint32_t sh_mem[2][BLOCK_DIM+1]; // 2 blocks swaped every time (+1 since it's exclusive sum)
    __shared__ uint32_t sh_A[2][BLOCK_DIM];
    __shared__ uint32_t sh_idxs[2][BLOCK_DIM];
    __shared__ uint32_t local_counter[BUCKET_SZ];
    // 1. get data into local memory
    if(threadIdx.x == 0) {
        sh_mem[0][0] = 0;
        sh_mem[1][0] = 0;
    }
    if(threadIdx.x < BUCKET_SZ) local_counter[threadIdx.x] = 0;
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    sh_A[0][threadIdx.x] = global_idx < N ? A[global_idx] : 0;
    sh_idxs[0][threadIdx.x] = (global_idx < N) ? old_idxs[global_idx] : 0;
    // 2. perform one bit radix
    bool input_idx = 0;
    #pragma unroll
    for(int r = 0;r < RADIX;r++) {
        one_bit_sort(sh_A[input_idx], sh_A[!input_idx], sh_idxs[input_idx], sh_idxs[!input_idx], sh_mem, iter*RADIX + r, op);
        input_idx = !input_idx;
    } 
    // 3. write it ^ write on global counter
    update_glob_buckets(sh_A[input_idx], local_counter, iter, N, global_idx, global_counter, op);
    if(global_idx < N) {
        res[global_idx] = sh_A[input_idx][threadIdx.x];
        new_idxs[global_idx] = sh_idxs[input_idx][threadIdx.x];
    }
}

// global_counter[radix][blockIdx.x]
template <typename OP>
__global__ void radix_sort_shuffle(uint32_t* A, uint32_t* res, uint32_t* old_idxs, uint32_t* new_idxs, uint32_t* global_counter, uint32_t* global_counter_sum, const uint32_t N, const uint32_t iter) {
    OP op;
    __shared__ int local_counter[BUCKET_SZ];
    if(threadIdx.x < BUCKET_SZ) {
        local_counter[threadIdx.x] = global_counter[GET_IDX(threadIdx.x, blockIdx.x, gridDim.x)];
    }
    __syncthreads();
    // sort locally
    if(threadIdx.x == 0) {
        for(int i = 1; i < BUCKET_SZ;i++) {
            local_counter[i] += local_counter[i - 1];
        }
    }
    __syncthreads();
    //
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(global_idx < N) {
        int radix = op.get_radix(A[global_idx], iter);
        int bucket_idx = radix == 0 ? threadIdx.x : threadIdx.x - local_counter[radix - 1];
        
        int counter_idx = GET_IDX(radix, blockIdx.x, gridDim.x) - 1;
        int sum = counter_idx >= 0 ? global_counter_sum[counter_idx] : 0;
        int new_idx = bucket_idx + sum;
        // printf("blcokIdx %i radix %i bucket_idx %i sum %i newidx %i\n",blockIdx.x, radix, bucket_idx, sum, new_idx);
        // printf("old: %i new: %i\n", global_idx, new_idx);
        res[new_idx] = A[global_idx];
        new_idxs[new_idx] = old_idxs[global_idx];
    }
}

int main() {
    // const int N = BLOCK_DIM;
    // int h_input[N] = {15, 3, 7, 9, 2, 1, 5, 6, 4, 8, 10, 0, 11, 13, 12, 14, 19, 17, 16, 18, 22, 20, 21, 23, 25, 24, 27, 26, 29, 28, 31, 30};
    // int h_output[N];
    const uint32_t N = 17;
    float h_input[N] = {12.5, 3.67, 3.4, 3.45,  -15, 8, 5, 10,  9, 6, 11, 13,  4,10,7,0, -5 };
    float h_output[N];
    
    uint32_t indexs[N], indexs_out[N];
    for(int i = 0; i < N;i++) indexs[i] = i;
    
    float *d_input_buff;
    uint32_t *d_input, *d_output, *d_idxs_in, *d_idxs_out;
    CUDA_CHECK(cudaMalloc((void**)&d_input_buff, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_idxs_in, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_idxs_out, N * sizeof(int)));


    CUDA_CHECK(cudaMemcpy(d_input_buff, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    float_to_uint32<<<CEIL_DIVI(N, BLOCK_DIM), BLOCK_DIM>>>(d_input_buff, d_input, N);
    // CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idxs_in, indexs, N * sizeof(int), cudaMemcpyHostToDevice));



    const int num_iters = 32/RADIX; // Enough for sorting up to 32-bit ints (since RADIX = 2, 16 * 2 = 32 bits)

    // 1. init global counter
    uint32_t* global_counter, *global_counter_sum;
    dim3 blocks( CEIL_DIV(N, BLOCK_DIM) );
    uint32_t counter_sz = blocks.x * BUCKET_SZ;
    dim3 blocks_counter( CEIL_DIV(counter_sz, 2*BLOCK_DIM) );
    // for global counter
    CUDA_CHECK( cudaMalloc((void**)&global_counter, counter_sz * sizeof(int)) );
    CUDA_CHECK( cudaMalloc((void**)&global_counter_sum, counter_sz * sizeof(int)) );//+1 to handle exclusive
    CUDA_CHECK( cudaMemset(global_counter_sum, 0, sizeof(int)) );//+1 to handle exclusive
    // 2. init prefix sum on global counter
    uint32_t* block_counter, *blocks_finished;
    CUDA_CHECK( cudaMalloc((void**)&block_counter, sizeof(int)) );
    CUDA_CHECK( cudaMalloc((void**)&blocks_finished, sizeof(int)) );

    for (int iter = 0; iter < num_iters; ++iter) {
        CUDA_CHECK( cudaMemset(block_counter, 0, sizeof(int)) );
        CUDA_CHECK( cudaMemset(blocks_finished, 0, sizeof(int)) );
        
        radix_sort_local_kerenl<DescOp><<<blocks, BLOCK_DIM>>>(d_input, d_output, d_idxs_in, d_idxs_out, global_counter, N, iter);
        // prefix sum on global counter
        balanced_bank_kernel<<<blocks_counter, BLOCK_DIM>>>(global_counter, global_counter_sum, counter_sz, block_counter, blocks_finished);
        // update/global shuffling
        radix_sort_shuffle<DescOp><<<blocks, BLOCK_DIM>>>(d_output, d_input, d_idxs_out, d_idxs_in, global_counter, global_counter_sum, N, iter);
    }
    
    

    // CUDA_CHECK(cudaMemcpy(h_output, d_input, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(indexs_out, d_idxs_in, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    for(int i = 0; i < N;i++) h_output[i] = h_input[ indexs_out[i] ];

    std::cout << "Sorted output:\n";
    for (int i = 0; i < N; ++i) std::cout << h_output[i] << " ";
    std::cout << std::endl;

    // Validate
    std::sort(h_input, h_input + N);
    std::reverse(h_input, h_input + N);
    for (int i = 0; i < N; ++i) {
        if (h_input[i] != h_output[i]) {
            std::cerr << "Test failed at index " << i << ": expected " << h_input[i] << ", got " << h_output[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Test passed.\n";


    cudaFree(d_input);
    cudaFree(d_output);
    CUDA_CHECK(cudaFree(d_idxs_in));
    CUDA_CHECK(cudaFree(d_idxs_out));
    CUDA_CHECK(cudaFree(block_counter));
    CUDA_CHECK(cudaFree(blocks_finished));
    CUDA_CHECK(cudaFree(global_counter));
    CUDA_CHECK(cudaFree(global_counter_sum));
    
    return 0;
}