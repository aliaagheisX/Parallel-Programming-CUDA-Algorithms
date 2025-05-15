#include <iostream>
#include <cstring>
using namespace std;

#define N 20
#define M 30


/*
    vec [in]: 1xN
    Mat [in]: NxM
    res [out]: 1xM
*/
__global__ void multVecMatrix(float * vec, float *Mat, float *res) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    res[j] = 0;
    for(int i = 0; i < N; i++) {
        res[j] += Mat[j + i * M] * vec[i];
    }
}


int main() {
    // HOST Initializaiton
    float *vec, *Mat, *res;
    float *vec_d, *Mat_d, *res_d;

    vec = (float*)malloc(N * sizeof(float));        // 1xN
    Mat = (float*)malloc(N * M * sizeof(float));    // NxM
    res = (float*)malloc(M * sizeof(float));        // 1xM

    for(int i = 0; i < N; i++) vec[i] = i + 1;

    for(int i = 0; i < N; i++) {
        for(int j = 0;j < M;j++) {
            Mat[j + i * M] = j + i * M + 1;
        }
    }

    // DEVICE Initializaiton
    cudaMalloc((void**)&vec_d, N * sizeof(float));
    cudaMalloc((void**)&Mat_d, N * M * sizeof(float));
    cudaMalloc((void**)&res_d, M * sizeof(float));
    
    cudaMemcpy(vec_d, vec, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Mat_d, Mat, N * M * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threads(16);
    dim3 blocks((M + threads.x - 1) / threads.x);
    
    multVecMatrix<<<blocks, threads>>>(vec_d, Mat_d, res_d);
    
    cudaMemcpy(res, res_d, M * sizeof(float), cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < M; i++) cout << res[i] << " ";
    
    free(vec);
    free(Mat);
    free(res);

    return 0;
}
