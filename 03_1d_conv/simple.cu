#include<iostream>
using namespace std;

#define MAX_MASK_SZ 1000

__constant__ int mask_d[MAX_MASK_SZ];

__global__ void conv1d_kernal_simple(int *res, int *arr, int mask_sz, int data_sz) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = tid - (mask_sz >> 1);

    
    int temp = 0;
    for(int i = 0;i < mask_sz; i++) {
        if(start+i >= 0 && start+i < data_sz)
            temp +=  (mask_d[i] * arr[start + i]);
    }
    __syncthreads();

    if(tid < data_sz)
        res[tid] = temp;
}

int main() {
    // init setup
    // =============
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "a", stdout);

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
    dim3 threads(256);
    dim3 blocks((data_size + threads.x - 1) / threads.x); 

    cudaDeviceSynchronize();
    
    conv1d_kernal_simple<<<blocks, threads>>> (res_d, arr_d, mask_size, data_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
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