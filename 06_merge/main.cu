#include<iostream>
#include <stdint.h>
#include <cassert>
#include <random>
#include<vector>
#include<unordered_map>
#include <stdint.h>
#include <algorithm>
#define BATCH_SIZE 32
#define ELE_PER_TH 6
#define BLOCK_DIM 512
#define ELE_PER_BLOCK (ELE_PER_TH*BLOCK_DIM)

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
__host__ __device__ int find_corank(int* A, int* B, uint32_t m, uint32_t n, uint32_t k) {
    uint32_t l = k > n ? k - n : 0; 
    uint32_t r = k < m ? k : m; 
    uint32_t i, j;
    while(l <= r) {
        i = (l + r) / 2;
        j = k - i;
        // printf("bs debug: %d %d %d %d\n", l, r, i, j);
        // good means A[i - 1] <= B[j] & B[j - 1] <= A[i] 
        // i too low -> A[i] means B[j - 1] > A[i]
        // i too high -> means A[i - 1] > B[j]
        if(j > 0 && i < m  && B[j - 1] > A[i])
            l = i + 1;
        else if(i > 0 && j < n &&  A[i - 1] > B[j])
            r = i - 1;
        else
            return i;
    }
    return l;
}

__global__ void merge_sorted_array_kernel_v1(int* A, int* B, int* C, uint32_t m, uint32_t n) {
    uint32_t k = ELE_PER_TH * (threadIdx.x + blockIdx.x * blockDim.x);
    if(k < n + m) {
        uint32_t i = find_corank(A, B, m, n, k);
        uint32_t j = k - i;

        for(int d = 0; d < ELE_PER_TH && k + d < n + m; d++) {
            if(j >= n) C[k + d] = A[i++];
            else if(i >= m) C[k + d] = B[j++];
            else if(A[i] <= B[j]) C[k + d] = A[i++];
            else C[k + d] = B[j++];
        }
    }
}

__global__ void merge_sorted_array_kernel_v2(int* A, int* B, int* C, int *lasti, const uint32_t k_mx, uint32_t m, uint32_t n) {
    __shared__ int A_sh[ELE_PER_BLOCK];
    __shared__ int B_sh[ELE_PER_BLOCK];
    __shared__ int C_sh[ELE_PER_BLOCK];
    __shared__ uint32_t block_start_k, block_next_k;
    __shared__ uint32_t block_start_i, block_next_i;
    __shared__ uint32_t block_start_j, block_next_j;
    if(threadIdx.x == 0) {
        //TODO:
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

struct Batch {
    int bs;
    int *data;
    bool on_gpu = false;

    Batch() {}
    Batch(int bs): bs(bs) {
        CUDA_CHECK(cudaMalloc((void**)&data, bs*sizeof(int)));
        on_gpu = true;
    }
    Batch(Batch* copy_data): bs(copy_data->bs), on_gpu(true) {
        CUDA_CHECK(cudaMalloc((void**)&data, bs*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(data, copy_data->data, copy_data->bs * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    int* moveToGPU() {
        return data;
    }
    int* moveToHost() {
        return data;
    }
    ~Batch() {
        if (on_gpu) {
            CUDA_CHECK(cudaFree(data));
        }
    }
};

void merge_two_runs(std::vector<Batch*>& batchs, std::vector<Batch*>& o_batchs, int run_idx, int run_sz) {
    int b1_idx = run_idx * run_sz; 
    int b1_end = std::min((run_idx + 1) * run_sz, (int)batchs.size());
    int b2_idx = (run_idx + 1) * run_sz; 
    int b2_end = std::min((run_idx + 2) * run_sz, (int)batchs.size());

    if (b2_idx >= batchs.size()) {
        for(int i = b1_idx; i < b1_end; i++) {
            o_batchs.emplace_back(batchs[i]);
        }
        return;
    }

    int b1_offset = 0, b2_offset = 0;
    while(b1_idx < b1_end || b2_idx < b2_end) {
        if(b1_idx >= b1_end) {
            while(b2_idx < b2_end) 
                o_batchs.emplace_back(batchs[b2_idx++]);
            break;
        }
        if(b2_idx >= b2_end) {
            while(b1_idx < b1_end) 
                o_batchs.emplace_back(batchs[b1_idx++]);
            break;
        }

        int m = batchs[b1_idx]->bs - b1_offset;
        int n = batchs[b2_idx]->bs - b2_offset;
        
        if (m <= 0) {
            b1_idx++;
            b1_offset = 0;
            continue;
        }
        if (n <= 0) {
            b2_idx++;
            b2_offset = 0;
            continue;
        }

        int new_bs = std::min(BATCH_SIZE, m + n);
        int* b1 = batchs[b1_idx]->data + b1_offset;
        int* b2 = batchs[b2_idx]->data + b2_offset;
        
        Batch* bout = new Batch(new_bs);
        o_batchs.push_back(bout);

        std::cout << "st:  " << b1_idx << " " << b2_idx << " " << b1_offset << " " << b2_offset << " " << m << " " << n << " " << new_bs << std::endl;
        
        int *i_end_batch_d;
        CUDA_CHECK(cudaMalloc((void**)&i_end_batch_d, sizeof(int)));
        
        dim3 blocks(CEIL_DIV(new_bs, ELE_PER_BLOCK));
        merge_sorted_array_kernel_v2<<<blocks, BLOCK_DIM>>>(b1, b2, bout->data, i_end_batch_d, new_bs, m, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK_LAST();
        
        int i_end_batch;
        CUDA_CHECK(cudaMemcpy(&i_end_batch, i_end_batch_d, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(i_end_batch_d));
        
        // Update offsets
        b1_offset += i_end_batch;
        b2_offset += (new_bs - i_end_batch);
        // std::cout << "OHOH: " << i_end_batch << " " << (new_bs - i_end_batch) << std::endl;
        // std::cout << "en1: " << b1_idx << " " << b2_idx << " " << b1_offset << " " << b2_offset << " " << m << " " << n << " " << new_bs << std::endl;

        
        // Move to next batch if current one is exhausted
        if (b1_offset >= batchs[b1_idx]->bs) {
            b1_idx++;
            b1_offset = 0;
        }
        if (b2_offset >= batchs[b2_idx]->bs) {
            b2_idx++;
            b2_offset = 0;
        }
        
    }
}


void merge_sorted_batches(std::vector<Batch*>& batchs, int total_rows) { // given batch idxs & pointer to their values
    int num_batchs = batchs.size();
    
    int num_runs = batchs.size();
    int run_sz = 1; // curr number of batchs in run
    while(num_runs > 1) {
        std::vector<Batch*> obatchs;
        for(int run = 0; run < num_runs; run+=2) {
            merge_two_runs(batchs, obatchs, run, run_sz);
        }
        
        batchs = obatchs;
        num_runs = (num_runs + 1)/2; // ceil div
        run_sz <<= 1;
    }

}


void test_merge_sorted_batches() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_dist(10, BATCH_SIZE);
    std::uniform_int_distribution<> val_dist(0, 1000);

    // Generate random sorted batches
    const int num_batches = 5;
    std::vector<Batch*> batches;
    std::vector<std::vector<int>> host_data(num_batches);
    int total_rows = 0;
    for (int i = 0; i < num_batches; ++i) {
        // int size = size_dist(gen);
        int size = BATCH_SIZE;
        total_rows += size;

        std::vector<int> data(size);
        for (int j = 0; j < size; ++j) {
            data[j] = val_dist(gen);
        }
        std::sort(data.begin(), data.end());
        
        Batch* batch = new Batch(size);
        CUDA_CHECK(cudaMemcpy(batch->data, data.data(), size * sizeof(int), cudaMemcpyHostToDevice));
        printf("batch reserved ptr %p\n", batch->data);
        batches.push_back(batch);
        host_data[i] = data;
    }

    // Print input batches
    std::cout << "Input batches:\n";
    for (int i = 0; i < num_batches; ++i) {
        std::cout << "Batch " << i << ": ";
        for (int val : host_data[i]) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }

    // Merge batches
    merge_sorted_batches(batches, total_rows);

    // Verify result
    std::vector<int> result;
    for(auto batch: batches) {
        int arr[batch->bs];
        CUDA_CHECK(cudaMemcpy(arr, batch->data, batch->bs * sizeof(int), cudaMemcpyDeviceToHost));
        for(int i = 0;i < batch->bs;i++) result.push_back(arr[i]);
    }

    // Create expected result by merging all input data on host
    std::vector<int> expected;
    for (const auto& batch : host_data) {
        expected.insert(expected.end(), batch.begin(), batch.end());
    }
    std::sort(expected.begin(), expected.end());
    freopen("out", "w", stdout);
    // Print result
    std::cout << "Merged result: " << " res size = " << result.size() << " actual = " << total_rows << std::endl;
    int i = 0;
    for (int val : result) {
        std::cout << val << " " << expected[i++] << "\n";
    }
    std::cout << "\n";

    // Verify correctness
    bool correct = (result.size() == expected.size()) && 
                   std::equal(result.begin(), result.end(), expected.begin());
    std::cout << "Merge is " << (correct ? "correct" : "incorrect") << "\n";

    // Cleanup
    for (Batch* batch : batches) {
        delete batch;
    }
}

int main() {
    test_merge_sorted_batches();
    return 0;
}