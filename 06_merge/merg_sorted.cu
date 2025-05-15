// #include<iostream>
// #include <stdint.h>
// #include <cassert>
// #include <random>
// #include<string>
// #include<unordered_map>
// #define N_BATCHS 1000
// #define BATCH_SZ 32

// #define MAX_RAND 100

// #define ELE_PER_TH 6
// #define BLOCK_DIM 512
// #define ELE_PER_BLOCK (ELE_PER_TH*BLOCK_DIM)

// #define COARSENING_FACTOR 5
// #define CEIL_DIV(X, Y) (X + Y - 1)/(Y)


// #define CUDA_CHECK(call) { \
//     cudaError_t err = call; \
//     if (err != cudaSuccess) { \
//         fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
//         exit(EXIT_FAILURE); \
//     } \
// }

// #define CUDA_CHECK_LAST() { \
//     cudaError_t err = cudaGetLastError(); \
//     if (err != cudaSuccess) { \
//         fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
//         exit(EXIT_FAILURE); \
//     } \
// }

// __host__ __device__ int find_corank(int* A, int* B, uint32_t m, uint32_t n, uint32_t k) {
//     uint32_t l = k > n ? k - n : 0; 
//     uint32_t r = k < m ? k : m; 
//     uint32_t i, j;
//     while(l <= r) {
//         i = (l + r) / 2;
//         j = k - i;
//         // good means A[i - 1] <= B[j] & B[j - 1] <= A[i] 
//         // i too low -> A[i] means B[j - 1] > A[i]
//         // i too high -> means A[i - 1] > B[j]
//         if(j > 0 && i < m  && B[j - 1] > A[i])
//             l = i + 1;
//         else if(i > 0 && j < n &&  A[i - 1] > B[j])
//             r = i - 1;
//         else
//             return i;
//     }
//     return i;
// }

// __global__ void merge_sorted_array_kernel_v1(int* A, int* B, int* C, uint32_t m, uint32_t n) {
//     uint32_t k = ELE_PER_TH * (threadIdx.x + blockIdx.x * blockDim.x);
//     if(k < n + m) {
//         uint32_t i = find_corank(A, B, m, n, k);
//         uint32_t j = k - i;

//         for(int d = 0; d < ELE_PER_TH && k + d < n + m; d++) {
//             if(j >= n) C[k + d] = A[i++];
//             else if(i >= m) C[k + d] = B[j++];
//             else if(A[i] <= B[j]) C[k + d] = A[i++];
//             else C[k + d] = B[j++];
//         }
//     }
// }

// __global__ void merge_sorted_array_kernel_v2(int* A, int* B, int* C, uint32_t m, uint32_t n) {
//     __shared__ int A_sh[ELE_PER_BLOCK];
//     __shared__ int B_sh[ELE_PER_BLOCK];
//     __shared__ int C_sh[ELE_PER_BLOCK];
//     __shared__ uint32_t block_start_k, block_next_k;
//     __shared__ uint32_t block_start_i, block_next_i;
//     __shared__ uint32_t block_start_j, block_next_j;
//     if(threadIdx.x == 0) {
//         block_start_k = ELE_PER_TH * (blockIdx.x * blockDim.x);
//         block_next_k  = block_start_k + ELE_PER_BLOCK > n + m ? n + m : block_start_k + ELE_PER_BLOCK ;

//         block_start_i = find_corank(A, B, m, n, block_start_k);
//         block_next_i = find_corank(A, B, m, n, block_next_k);
        
//         block_start_j = block_start_k - block_start_i;
//         block_next_j = block_next_k - block_next_i;
//     }
//     __syncthreads();
//     uint32_t m_sh = block_next_i - block_start_i;
//     uint32_t n_sh = block_next_j - block_start_j;
//     for(int i = threadIdx.x; i < m_sh; i += BLOCK_DIM) {
//         A_sh[i] = A[block_start_i + i];
//     }
//     for(int j = threadIdx.x;  j < n_sh; j += BLOCK_DIM) {
//         B_sh[j] = B[block_start_j + j];
//     }
//     __syncthreads();
    
    
//     uint32_t k = threadIdx.x * ELE_PER_TH;
//     if(k < n_sh + m_sh) {

//         uint32_t i = find_corank(A_sh, B_sh, m_sh, n_sh, k);
//         uint32_t j = k - i;

//         for(int d = 0; d < ELE_PER_TH && k + d < n_sh + m_sh; d++) {
//             if(j >= (block_next_j - block_start_j)) C_sh[k + d] = A_sh[i++];
//             else if(i >= (block_next_i - block_start_i)) C_sh[k + d] = B_sh[j++];
//             else if(A_sh[i] <= B_sh[j]) C_sh[k + d] = A_sh[i++];
//             else C_sh[k + d] = B_sh[j++];
//         }
//     }
//     __syncthreads();
//     for(int i = threadIdx.x; i < n_sh + m_sh; i+=BLOCK_DIM)
//         C[block_start_k + i] = C_sh[i];
// }


// void get_rand_batch(int*arr, int batch_sz) {// get sorted array of batch_sz
//     arr[0] = rand()%MAX_RAND;
//     for(int i = 1; i < batch_sz;i++) {
//         arr[i] = arr[i-1] + (rand())%MAX_RAND;
//     }   
// }

// void get_batch_out(int *arr, FILE* f, int sz) {
//     for(int i = 0;i < sz;i++) fprintf(f, "%d\n", arr[i]);
// }
// //TODO: return batch size
// int read_batch_in(int*arr, FILE* f, int sz) {
//     for(int i = 0;i < sz;i++) {
//         if(fscanf(f, "%d", &arr[i]) != 1) {
//             return i;
//         }
//     }
// }

// int main() {
//     std::unordered_map<int, const char*> idx_file;
//     int incremental_counter = 0;
//     // 1. simulate sorting batches
//     int A[BATCH_SZ];
//     for(int i = 0; i < N_BATCHS;i++) {
//         const char* fname = std::to_string(incremental_counter++).c_str();
//         idx_file[i] = fname;
//         FILE* f = fopen(fname, "w");
//         get_rand_batch(A, BATCH_SZ);
//         get_batch_out(A, f, BATCH_SZ);
//         fclose(f);
//     }
//     // run_sz  max number of batchs in run
//     int A[BATCH_SZ], B[BATCH_SZ], C[BATCH_SZ], *A_d, *B_d, *C_d;
//     CUDA_CHECK( cudaMalloc((void**)&A_d, BATCH_SZ*sizeof(int)) );
//     CUDA_CHECK( cudaMalloc((void**)&B_d, BATCH_SZ*sizeof(int)) );
//     CUDA_CHECK( cudaMalloc((void**)&C_d, BATCH_SZ*sizeof(int)) );

//     size_t run_sz = 1;
//     size_t num_batchs = N_BATCHS;
//     for(; run_sz < N_BATCHS; run_sz<<=1) {
//         int run_idx =  0;
//         for(; run_idx < num_batchs; run_idx+=2) {
//             auto f1 = fopen(idx_file[run_idx], "r"), f2 = fopen(idx_file[run_idx+1], "r");
//             // store new index
//             const char* fname = std::to_string(incremental_counter++).c_str();
//             idx_file[run_idx/2] = fname;
//             FILE* fout = fopen(std::to_string(incremental_counter++).c_str(), "w");
//             // ======= start mergine ==================
//             int i_st_batch = 0, j_st_batch = 0;
            
//             int bs1 = read_batch_in(A, f1, BATCH_SZ);
//             int bs2 = read_batch_in(B, f2, BATCH_SZ);
            
//             CUDA_CHECK( cudaMemcpy(A_d, A, bs1, cudaMemcpyHostToDevice)  );
//             CUDA_CHECK( cudaMemcpy(B_d, B, bs2, cudaMemcpyHostToDevice)  );

//             while(bs1 && bs2) {
//                 int m = bs1-i_st_batch, n = bs2-j_st_batch;
//                 int i_end_batch = find_corank(A + i_st_batch, B + j_st_batch, m, n, BATCH_SZ);
//                 int j_end_batch = BATCH_SZ - i_end_batch;

//                 dim3 blocks(CEIL_DIV(m+n, ELE_PER_BLOCK));
//                 merge_sorted_array_kernel_v2<<<blocks, BLOCK_DIM>>>(A_d, B_d, C_d, m, n);
//                 CUDA_CHECK( cudaMemcpy(C, C_d, BATCH_SZ, cudaMemcpyDeviceToHost) );
//                 get_batch_out(C, fout, BATCH_SZ);
//                 // update to get next batch out
//                 if(i_end_batch == m) {
//                     bs1 = read_batch_in(A, f1, BATCH_SZ);
//                     i_st_batch = 0;
//                 } else {
//                     i_st_batch += i_end_batch;
//                 }
//                 if(j_end_batch == n) {
//                     bs2 = read_batch_in(B, f2, BATCH_SZ);
//                     j_st_batch = 0;
//                 } else {
//                     j_st_batch += j_end_batch;
//                 }
//             }
            

//         }
//         // odd write in same
//         // if(num_batchs & 1) idx_file[num_batchs/2 + 1] = idx_file[k]; 
//         // num_batchs = CEIL_DIV(num_batchs, 2);
//     }    
//     // uint32_t i_glob_begin = 0, j_glob_begin = 0;
    
//     // int A[BATCH_SZ], B[BATCH_SZ], C[BATCH_SZ];
//     // int batch_sz1 = get_batch_in(A, fin1, BATCH_SZ);
//     // int batch_sz2 = get_batch_in(B, fin2, BATCH_SZ);

//     // int *A_d, *B_d, *C_d;
//     // uint32_t i_prev = 0, j_prev = 0;
//     // while(i_glob_begin < M && j_glob_begin < N) {
//     //     uint32_t m_max = i_glob_begin + BATCH_SZ <= M ? BATCH_SZ : M - i_glob_begin;
//     //     uint32_t n_max = j_glob_begin + BATCH_SZ <= N ? BATCH_SZ : N - j_glob_begin;
//     //     uint32_t k = BATCH_SZ > m_max + n_max ? m_max + n_max : BATCH_SZ;
//     //     // 1. read
//     //     freopen("in1.txt", "r", stdin);
//     //     for(int ii = 0;ii < m_max;++ii) std::cin>>A[ii]; 
//     //     freopen("in2.txt", "r", stdin);
//     //     for(int ii = 0;ii < n_max;++ii) std::cin>>B[ii]; 

//     //     // 2. determinne i, j, n, m
//     //     uint32_t i = find_corank(A, B, m_max, n_max, k);
//     //     uint32_t j = k - i;
//     //     // uint32_t n = 
//     //     CUDA_CHECK( cudaMalloc((void**)&A_d, sizeof(int)*i) );
//     //     CUDA_CHECK( cudaMalloc((void**)&B_d, sizeof(int)*j) );
//     //     CUDA_CHECK( cudaMalloc((void**)&C_d, sizeof(int)*k) );

//     //     // CUDA_CHECK( cudaMemcpy(A_d, A, M * sizeof(int), cudaMemcpyHostToDevice) );

//     //     // 3. update state
//     //     i_glob_begin += i;
//     //     j_glob_begin += j;
//     // }

//     // CUDA_CHECK( cudaMemcpy(B_d, B, N * sizeof(int), cudaMemcpyHostToDevice) );

    
//     // int C[M + N];
//     // CUDA_CHECK( cudaMemcpy(C, C_d, (M + N)*sizeof(int), cudaMemcpyDeviceToHost) );
    
//     // CUDA_CHECK_LAST();
//     // for(int i = 0;i < M + N;++i)
//     //     std::cout << C[i] << "\n"; 
//     // return 0;
// }