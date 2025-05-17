#include <cooperative_groups.h>

#define DP_BLOCK_DIM 1024
#define DRIVER_BLOCK_DIM 128

__global__ void bfs_vertix_dp_child(int* neighbors, const int startIdx, const int endIdx, 
                                    int* levels, int curr_level,
                                    int* curr_frontier, int* sz_curr_frontier) {
    int nodeIdxfromStart = threadIdx.x + blockIdx.x * blockDim.x;
    if (startIdx + nodeIdxfromStart < endIdx) {
        int neighbor = neighbors[startIdx + nodeIdxfromStart];
        if (atomicCAS(&levels[neighbor], -1, curr_level) == -1) {
            int newIdxinCurrFrontier = atomicAdd(sz_curr_frontier, 1);
            curr_frontier[newIdxinCurrFrontier] = neighbor;
        }
    }
}

__global__ void bfs_vertix_dp(int* nodesPtr, int* neighbors, int* levels, int curr_level,
                              int* prev_frontier, int* curr_frontier,
                              const int sz_prev_frontier, int* sz_curr_frontier) {
    int nodeIdxinFrontier = threadIdx.x + blockIdx.x * blockDim.x;
    if (nodeIdxinFrontier < sz_prev_frontier) {
        // printf("AGGGG\n");
        int node = prev_frontier[nodeIdxinFrontier];
        int start = nodesPtr[node], end = nodesPtr[node + 1];

        int num_neighbors = end - start;
        if (num_neighbors < DP_BLOCK_DIM) {
            for (int i = start; i < end; i++) {
                int neighbor = neighbors[i];
                if (atomicCAS(&levels[neighbor], -1, curr_level) == -1) {
                    int newIdxinCurrFrontier = atomicAdd(sz_curr_frontier, 1);
                    curr_frontier[newIdxinCurrFrontier] = neighbor;
                }
            }
        } else {
            dim3 threads(DP_BLOCK_DIM);
            dim3 blocks((num_neighbors + DP_BLOCK_DIM - 1) / DP_BLOCK_DIM);
            bfs_vertix_dp_child<<<blocks, threads>>>(neighbors, start, end, levels, curr_level, curr_frontier, sz_curr_frontier);
        }
    }
}


__device__ int flag_sync = 0;
__global__ void bfs_swap() {
    atomicOr(&flag_sync, 1);
}

__global__ void bfs_dp_driver_kernel(int* nodesPtr, int* neighbors, int* levels, int* prev_frontier, int* curr_frontier, int* sz_prev_frontier, int* sz_curr_frontier) {
    int curr_level = 1;
    while (*sz_prev_frontier > 0) {
    //     printf("MAHAHAHAA\n");
        dim3 threads(DRIVER_BLOCK_DIM);
        dim3 blocks((*sz_prev_frontier + DRIVER_BLOCK_DIM - 1) / DRIVER_BLOCK_DIM);
        bfs_vertix_dp<<<blocks, threads>>>(nodesPtr, neighbors, levels, curr_level,
                                           prev_frontier, curr_frontier,
                                           *sz_prev_frontier, sz_curr_frontier);
        bfs_swap<<<1, 1>>>();

        while(!atomicAnd(&flag_sync, 0)) {};

        int* temp = prev_frontier;
        prev_frontier = curr_frontier;
        curr_frontier = temp;
        
        *sz_prev_frontier = *sz_curr_frontier;
        *sz_curr_frontier = 0;

        curr_level = (curr_level)+1;
    }
}