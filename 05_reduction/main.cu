#include<iostream>
#include<stdint.h>
#define N 1000000
#define BLOCK_DIM 1024
#define COARSENING_FACTOR 5
#define CEIL_DIV(X, Y) (X + Y - 1)/(Y)

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


__global__ void reduction_sum_v1(int* arr, int *res) {
    __shared__ int sh_mem[2*BLOCK_DIM];
    for(int stride = threadIdx.x; stride < 2*BLOCK_DIM; stride+=BLOCK_DIM) {
        if(stride + blockIdx.x * 2*BLOCK_DIM < N)
            sh_mem[stride] = arr[stride + blockIdx.x * 2*BLOCK_DIM];
        else
            sh_mem[stride] = 0;
    }
    __syncthreads();

    for(int d = BLOCK_DIM; d > 0; d>>=1) {//4,2,1
        if(threadIdx.x < d) {
            sh_mem[threadIdx.x] += sh_mem[threadIdx.x + d];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        atomicAdd(res, sh_mem[0]);
    }
}

// __device__ void reduce_wrap() {}
__global__ void reduction_sum_v2(int* arr, int* res) {
    __shared__ int wrap_reductions[BLOCK_DIM/32];
    __shared__ int sh_res;

    if(threadIdx.x == 0)  sh_res=0;
    // get info
    int tid = threadIdx.x;
    int wrapIdx = tid/32;
    int laneIdx = tid%32;

    int localVal = 0;
    int globalIdx = threadIdx.x + COARSENING_FACTOR * blockIdx.x * blockDim.x;
    #pragma unroll
    for(uint32_t k = 0;k < COARSENING_FACTOR;k++) {
        // get local value for each ele in wrap
        localVal = globalIdx < N ? arr[globalIdx] : 0;
        // first sum
        #pragma unroll
        for(uint32_t d = 16; d > 0; d>>=1) //4,2,1
            localVal += __shfl_down_sync(0xFFFFFFFF, localVal, d);
        
        // each wrap put it's value
        if(laneIdx == 0)  wrap_reductions[wrapIdx] = localVal;
        __syncthreads();
        // first wrap sum values
        if(wrapIdx == 0) {
            localVal = tid < BLOCK_DIM/32 ? wrap_reductions[tid] : 0;
            #pragma unroll
            for(uint32_t d = 16; d > 0; d>>=1) //4,2,1
                localVal += __shfl_down_sync(0xFFFFFFFF, localVal, d);
            
            if(laneIdx == 0) 
                sh_res += localVal;
        }
        globalIdx += blockDim.x;
    }
    if(threadIdx.x == 0) 
        atomicAdd(res, sh_res);
}

int main() {
    int arr[N]; int sum = 0;
    for(int i = 0;i < N;i++) arr[i] = i + 1, sum+= i + 1;


    int* d_arr; 
    CUDA_CHECK( cudaMalloc((void**)&d_arr, N * sizeof(int)) );
    CUDA_CHECK( cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice) );

    int *d_res, *d_res_v2;
    CUDA_CHECK( cudaMalloc((void**)&d_res, sizeof(int)) );
    CUDA_CHECK( cudaMemset(d_res, 0, sizeof(int)) );

    CUDA_CHECK( cudaMalloc((void**)&d_res_v2, sizeof(int)) );
    CUDA_CHECK( cudaMemset(d_res_v2, 0, sizeof(int)) );

    dim3 threads(BLOCK_DIM);
    dim3 blocks(CEIL_DIV(N, 2*BLOCK_DIM));
    dim3 blocks_v2(CEIL_DIV(N, COARSENING_FACTOR*BLOCK_DIM));

    printf("threads %i blocks %i\n", threads.x, blocks.x);
    reduction_sum_v1<<<blocks, threads>>>(d_arr, d_res);
    printf("threads %i blocks %i\n", threads.x, blocks_v2.x);
    reduction_sum_v2<<<blocks_v2, threads>>>(d_arr, d_res_v2);
    
    int res = 0, res_v2 = 0; 
    CUDA_CHECK( cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(&res_v2, d_res_v2, sizeof(int), cudaMemcpyDeviceToHost) );


    std::cout << "gt: " << sum << "\n";
    std::cout << "out: " << res << "\n";
    std::cout << "v2: " << res_v2 << "\n";
    return 0;
}
