#pragma once
#include <iostream>

#include "merge_kernel.cu"
#include "include.cu"

void load_in_buffer(int* buffer, int &buffer_offset, int &run_offset, RUN* run) {
    while(buffer_offset < BUFFER_SZ && run_offset < run->total_rows) {
        int batch_idx = run_offset / MAX_BATCH_SZ;
        int in_batch_idx = run_offset % MAX_BATCH_SZ;
        int m = min(run->batchs[batch_idx]->bs - in_batch_idx,   BUFFER_SZ - buffer_offset);
        cudaMemcpy(buffer + buffer_offset, run->batchs[batch_idx]->data + in_batch_idx, m * sizeof(int), cudaMemcpyDeviceToDevice);

        buffer_offset += m; run_offset += m;
    }
}

void shift_buffer(int *buffer, int i_last, int m, int &buffer_offset) {
    int shift_m = m - i_last;
    if(shift_m > 0) {
        int* buffer_temp;
        CUDA_CHECK(cudaMalloc((void**)&buffer_temp, shift_m * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(buffer_temp, buffer + i_last, shift_m * sizeof(int), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(buffer, buffer_temp, shift_m * sizeof(int), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaFree(buffer_temp));
    }    
    buffer_offset = shift_m;
}

RUN* merge_two_runs(RUN* left, RUN* right) {
    RUN* out_run = new RUN();
    int* l_buffer, *r_buffer, *o_buffer;
    int *i_last_d;
    CUDA_CHECK( cudaMalloc((void**)&l_buffer, sizeof(int) * BUFFER_SZ) );
    CUDA_CHECK( cudaMalloc((void**)&r_buffer, sizeof(int) * BUFFER_SZ) );
    CUDA_CHECK( cudaMalloc((void**)&o_buffer, sizeof(int) * BUFFER_SZ) );
    CUDA_CHECK( cudaMalloc((void**)&i_last_d, sizeof(int)) );

    int i_last, j_last;
    int l_buffer_off = 0, r_buffer_off = 0;
    int l_run_off = 0, r_run_off = 0;
    while((l_run_off < left->total_rows && r_run_off < right->total_rows) || l_buffer_off > 0 || r_buffer_off > 0 ) {
        load_in_buffer(l_buffer, l_buffer_off, l_run_off, left);
        load_in_buffer(r_buffer, r_buffer_off, r_run_off, right);

        

        int m = l_buffer_off, n = r_buffer_off;
        int k = min(n + m, BUFFER_SZ);

        dim3 blocks(CEIL_DIV(k, ELE_PER_BLOCK));
        merge_sorted_array_kernel_v2<<<blocks, BLOCK_DIM>>>(l_buffer, r_buffer, o_buffer, i_last_d, k, m, n);
        
        auto outbatch = std::unique_ptr<Batch>(new Batch(k));
        CUDA_CHECK(cudaMemcpy(outbatch->data, o_buffer, k*sizeof(int), cudaMemcpyDeviceToDevice));
        out_run->batchs.push_back(std::move(outbatch));
        out_run->total_rows += k;

        CUDA_CHECK_LAST();

        CUDA_CHECK( cudaMemcpy(&i_last, i_last_d, sizeof(int), cudaMemcpyDeviceToHost) );
        j_last = k - i_last;
        
        shift_buffer(l_buffer, i_last, m, l_buffer_off);
        shift_buffer(r_buffer, j_last, n, r_buffer_off);
    }

    delete left, right;
    return out_run;
}

void merge_all_sorted(std::vector<RUN*>& in_runs) { 
    std::vector<RUN*> out_runs;
    while(in_runs.size() > 1) {
        for(int i = 0; i < in_runs.size(); i+=2) {
            // add last run as no one merged with
            if(i == in_runs.size() - 1) {
                out_runs.push_back(in_runs[in_runs.size() - 1]);
            } else {
                RUN* new_run = merge_two_runs(in_runs[i], in_runs[i+1]);
                out_runs.push_back(new_run);
            }
        }

        in_runs = out_runs;
        out_runs = std::vector<RUN*>();
    }
}
