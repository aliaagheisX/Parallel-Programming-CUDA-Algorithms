#pragma once
#include<vector>
#include<memory>
#include<error.h>
#include<iostream>
#define MAX_BATCH_SZ 32
#define BUFFER_SZ 32

#define CEIL_DIV(X, Y) ((X + Y - 1)/(Y))
#define CEIL_DIVI(X, Y) int((X + Y - 1)/(Y))
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


struct Batch {
    int bs;
    int *data;
    // bool on_gpu = false;

    Batch() {}
    Batch(int bs): bs(bs) {
        CUDA_CHECK(cudaMalloc((void**)&data, bs*sizeof(int)));
    }
    
    int* moveToGPU() {
        return data;
    }
    int* moveToHost() {
        return data;
    }
    ~Batch() {
        CUDA_CHECK(cudaFree(data));
    }
};

struct RUN {
    int total_rows = 0;
    std::vector<std::unique_ptr<Batch>> batchs;
};
