extern "C" {
__global__ void histogram_kernal(char* buffer, int* histogram, int lenBuffer, int numBins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // ==== privatization shared memory ==== 
    extern __shared__ int histogram_sh[];
    for(int i = tid;i < numBins; i += blockDim.x) histogram_sh[i] = 0;
    __syncthreads();
    
    // ==== calcute histogram ==== 
    for(int i = tid; i < lenBuffer; i += blockDim.x * gridDim.x) {
        int idx = buffer[i] - 'a';
        
        if(idx >= 0 && idx < 26)  {
            atomicAdd(&histogram[idx/4], 1);
        }
    }
    __syncthreads();
    
    // ==== accumalate ====
    for(int i = tid;i < numBins; i += blockDim.x)
        atomicAdd(&histogram[i], histogram_sh[i]); 
}
}
extern "C" {
void histogram_wrapper(char* buffer, int* histogram, int lenBuffer, int numBins) {
    dim3 threads(256);
    dim3 blocks((lenBuffer + threads.x - 1) / threads.x);

    histogram_kernal<<<blocks, threads>>>(buffer, histogram, lenBuffer, numBins);

}

}