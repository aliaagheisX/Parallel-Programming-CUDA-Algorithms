#define MAX_OUT_DEGREE 50
#define BLOCK_DIM_OPT 128

__global__ void bfs_vertix_opt(int* nodesPtr, int* neighbors, int* levels, int curr_level,
																int* prev_frontier, int* curr_frontier,
																const int sz_prev_frontier, int* sz_curr_frontier) {
									
    int nodeIdxinFrontier = threadIdx.x + blockIdx.x * blockDim.x;
    if(nodeIdxinFrontier < sz_prev_frontier) {
        int node = prev_frontier[nodeIdxinFrontier];
        for(int i = nodesPtr[node]; i < nodesPtr[node+1];i++) {
            int neighbor = neighbors[i];
            // if neighbor not visited update levels & add to curr_frontier
            if(atomicCAS(&levels[neighbor], -1, curr_level) == -1) {
                int newIdxinCurrFrontier = atomicAdd(sz_curr_frontier, 1);
                curr_frontier[newIdxinCurrFrontier] = neighbor;
            }
        }
        
    }

}

	
// local frontier optimization
__global__ void bfs_vertix_reg_privatization(int* nodesPtr, int* neighbors, int* levels, int curr_level,
                                                            int* prev_frontier, int* curr_frontier,
                                                            const int sz_prev_frontier, int* sz_curr_frontier) {
                                
    int nodeIdxinFrontier = threadIdx.x + blockIdx.x * blockDim.x;
    if(nodeIdxinFrontier < sz_prev_frontier) {
        int localFrontier[MAX_OUT_DEGREE];
        int sz_local_frontier = 0;
        int node = prev_frontier[nodeIdxinFrontier];
        for(int i = nodesPtr[node]; i < nodesPtr[node+1];i++) {
            int neighbor = neighbors[i];
            // if neighbor not visited update levels & add to local_frontier
            if(atomicCAS(&levels[neighbor], -1, curr_level) == -1) {
                localFrontier[sz_local_frontier++] = neighbor;
            }
        }
        // add local to global
        int startIdx = atomicAdd(sz_curr_frontier, sz_local_frontier);
        for(int i = 0;i < sz_local_frontier;i++) {
            curr_frontier[startIdx + i] = localFrontier[i];
        }
        
    }

}
	
	
	
	
// local + shared frontier optimizations
__global__ void bfs_vertix_block_privatization(int* nodesPtr, int* neighbors, int* levels, int curr_level,
                                                            int* prev_frontier, int* curr_frontier,
                                                            const int sz_prev_frontier, int* sz_curr_frontier) {
    __shared__ int sharedFrontier[MAX_OUT_DEGREE * BLOCK_DIM_OPT];
    __shared__ int sz_shared_frontier;
    if(threadIdx.x == 0) sz_shared_frontier = 0;
    __syncthreads();
    
    int nodeIdxinFrontier = threadIdx.x + blockIdx.x * blockDim.x;
    if(nodeIdxinFrontier < sz_prev_frontier) {
        int localFrontier[MAX_OUT_DEGREE];
        int sz_local_frontier = 0;
        int node = prev_frontier[nodeIdxinFrontier];
        for(int i = nodesPtr[node]; i < nodesPtr[node+1];i++) {
            int neighbor = neighbors[i];
            // if neighbor not visited update levels & add to local_frontier
            if(atomicCAS(&levels[neighbor], -1, curr_level) == -1) {
                localFrontier[sz_local_frontier++] = neighbor;
            }
        }
        // add local to shared
        int startIdx = atomicAdd(&sz_shared_frontier, sz_local_frontier);
        for(int i = 0;i < sz_local_frontier;i++) {
            sharedFrontier[startIdx + i] = localFrontier[i];
        }
        
    }
    __syncthreads();
    // add shared to global
    int startIdx = atomicAdd(sz_curr_frontier, sz_shared_frontier);
    for(int i = threadIdx.x; i < sz_shared_frontier;i+= blockDim.x){
        curr_frontier[startIdx + i] = sharedFrontier[i];
    }

}


// local + shared frontier optimizations to call from device
__device__ void bfs_vertix_block_privatization_device(int* nodesPtr, int* neighbors, int* levels, int curr_level,
                                                            int* prev_frontier, int* curr_frontier,
                                                            const int sz_prev_frontier, int* sz_curr_frontier) {
    __shared__ int sharedFrontier[MAX_OUT_DEGREE * BLOCK_DIM_OPT];
    __shared__ int sz_shared_frontier;
    if(threadIdx.x == 0) sz_shared_frontier = 0;
    __syncthreads();
    
    int nodeIdxinFrontier = threadIdx.x + blockIdx.x * blockDim.x;
    if(nodeIdxinFrontier < sz_prev_frontier) {
        int localFrontier[MAX_OUT_DEGREE];
        int sz_local_frontier = 0;
        int node = prev_frontier[nodeIdxinFrontier];
        for(int i = nodesPtr[node]; i < nodesPtr[node+1];i++) {
            int neighbor = neighbors[i];
            // if neighbor not visited update levels & add to local_frontier
            if(atomicCAS(&levels[neighbor], -1, curr_level) == -1) {
                localFrontier[sz_local_frontier++] = neighbor;
            }
        }
        // add local to shared
        int startIdx = atomicAdd(&sz_shared_frontier, sz_local_frontier);
        for(int i = 0;i < sz_local_frontier;i++) {
            sharedFrontier[startIdx + i] = localFrontier[i];
        }
        
    }
    __syncthreads();
    // add shared to global
    int startIdx = atomicAdd(sz_curr_frontier, sz_shared_frontier);
    for(int i = threadIdx.x; i < sz_shared_frontier;i+= blockDim.x){
        curr_frontier[startIdx + i] = sharedFrontier[i];
    }

}


// wrapper to launch multiple levels
__global__ void bfs_vertix_opt3(int* nodesPtr, int* neigbors, int* levels, int *curr_level,
																int** prev_frontier, int** curr_frontier,
																int* sz_prev_frontier, int* sz_curr_frontier) {
    bfs_vertix_block_privatization_device(nodesPtr, neigbors, levels, *curr_level, *prev_frontier, *curr_frontier,*sz_prev_frontier,sz_curr_frontier);
    __syncthreads();
    // if 1 block & next level will also have one block
    while(gridDim.x == 1 && *sz_curr_frontier > 0 && *sz_curr_frontier <= blockDim.x) {
        // only 1 thread will do necessary updating
        if(threadIdx.x == 0) {
            // swap(*prev_frontier, *curr_frontier) for next
            int *temp = *prev_frontier;
            *prev_frontier = *curr_frontier; *curr_frontier = temp;
            
            // swap(sz_prev_frontier, sz_curr_frontier)
            int temp1 = *sz_prev_frontier;
            *sz_prev_frontier = *sz_curr_frontier; *sz_curr_frontier = temp1;

            *curr_level = (*curr_level) + 1;
        }
        __syncthreads();
        bfs_vertix_block_privatization_device(nodesPtr, neigbors, levels, *curr_level, *prev_frontier, *curr_frontier,*sz_prev_frontier,sz_curr_frontier);
        __syncthreads();
    } 

}