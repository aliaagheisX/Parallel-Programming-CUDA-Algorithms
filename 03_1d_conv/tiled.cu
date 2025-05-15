#include<iostream>
using namespace std;

#define MAX_MASK_SZ 1000

#define O_TILE_W 256
#define I_TILE_W(mask_sz) O_TILE_W + mask_sz - 1
/*
- each block  => output tile
- num threads => output tile width
- input tile  => output tile + MASK_W - 1
*/

__constant__ int mask_d[MAX_MASK_SZ];

__global__ void conv1d_kernal_simple(int *res, int *arr, int mask_sz, int data_sz) {
    int n = (mask_sz >> 1);
    const int input_sz = O_TILE_W + mask_sz - 1;
    // initalize shared
    extern __shared__ int arr_sh[];
    
    // move to shared memory
    for(int i = threadIdx.x; i < input_sz; i+=blockDim.x){
        int k = blockDim.x * blockIdx.x - n + i;
        if(k >= 0 && k < data_sz)
            arr_sh[i] = arr[k];
        else
            arr_sh[i] = 0;
    }
    __syncthreads();
    
    // compute
    int temp = 0;
    for(int i = 0;i < mask_sz; i++) {
        if (threadIdx.x + i < input_sz) {  // Ensure we stay within bounds
            temp += mask_d[i] * arr_sh[threadIdx.x + i];
        }
    }
    __syncthreads();
    
    // write result
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < data_sz)
        res[tid] = temp;
}

int main() {
    // init setup
    // =============
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);

    int mask_size, data_size;
    cin >> mask_size >> data_size;

    int mask[mask_size], arr[data_size];
    for(int i = 0;i < mask_size;i++) cin >> mask[i];
    for(int i = 0;i < data_size;i++) cin >> arr[i];

    // to device mem
    // ===============
    cudaMemcpyToSymbolAsync(mask_d, mask, sizeof(int) * mask_size, 0, cudaMemcpyHostToDevice);

    int *arr_d;
    cudaMalloc((void**)& arr_d, sizeof(int) * data_size);
    cudaMemcpyAsync(arr_d, arr, sizeof(int) * data_size, cudaMemcpyHostToDevice);

    int *res_d;
    cudaMallocAsync(&res_d, sizeof(int)*data_size, 0);

    // kernal run
    // ===============
    dim3 threads(O_TILE_W);
    dim3 blocks((data_size + threads.x - 1) / threads.x); 
    cudaDeviceSynchronize();
    
    conv1d_kernal_simple<<<blocks, threads, (O_TILE_W + mask_size - 1)* sizeof(int)>>>(res_d, arr_d, mask_size, data_size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    // printf("GFG\n");
    // copy output
    // ===============
    int res[data_size];
    cudaMemcpy(res, res_d, sizeof(int)*data_size, cudaMemcpyDeviceToHost);

    for(int i = 0;i < data_size;i++)
        cout << res[i] << " ";
    cout << "\n";

    cudaFree(res_d);
    return 0;
}