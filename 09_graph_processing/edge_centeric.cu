#include <iostream>
#include <vector>

#define BLOCK_DIM 128
#define BATCH_SIZE 4096
#define CEIL_DIV(X, Y) ((X + Y - 1) / Y)
#define CEIL_DIV_I(X, Y) int((X + Y - 1) / Y)

__global__ void bfs_edge(int* rows, int* cols, int *levels, bool* flag, int num_edges, int curr_level){
	int edge = threadIdx.x + blockIdx.x * blockDim.x;
	if(edge < num_edges) {
		int u = rows[edge], v = cols[edge];
        //* since I only keep 1 edge in mem
		if(levels[u] == curr_level - 1 && levels[v] == -1) {
			levels[v] = curr_level;
			*flag = true;
		}
        if(levels[v] == curr_level - 1&& levels[u] == -1) {
			levels[u] = curr_level;
            *flag = true;
        }
	} 
}
// stream optimization :sad
void launch_bfs_streams(int *h_nodes, int *h_neighbors, int* h_levels, int* d_levels, int num_edges, int num_vertexs, int src_vertex) {
    // prepare flag
    bool *d_flag; 
    cudaMalloc((void**)&d_flag, sizeof(bool));
    cudaMemset(d_flag, false, sizeof(bool));
    

    // divide edges on streams & allocate & copy
    std::vector<cudaStream_t> streams;
    std::vector<int*> streams_d_nodes, streams_d_neighbors;
    std::vector<int> bsz;
	for(int i = 0;i < num_edges;i+=BATCH_SIZE) {
	    cudaStream_t stream; 
        cudaStreamCreate(&stream);
		streams.push_back(stream);

        int curr_num_edges = std::min(BATCH_SIZE, num_edges - i);
		int *d_nodes, *d_neighbors;

        cudaMallocAsync((void**)&d_nodes, curr_num_edges * sizeof(int), stream);
        cudaMallocAsync((void**)&d_neighbors, curr_num_edges * sizeof(int), stream);
        cudaMemcpyAsync(d_nodes, h_nodes + i, curr_num_edges * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_neighbors, h_neighbors + i, curr_num_edges * sizeof(int), cudaMemcpyHostToDevice, stream);

        streams_d_nodes.push_back(d_nodes);
        streams_d_neighbors.push_back(d_neighbors);
        bsz.push_back(curr_num_edges);
    }

    // iterate till flag = false
    int curr_level = 1;
    bool h_flag = true;
    while(h_flag) {
        for(int i = 0; i < streams.size();i++) {
            bfs_edge<<<CEIL_DIV_I(bsz[i], BLOCK_DIM), BLOCK_DIM, 0, streams[i]>>>(
                streams_d_nodes[i], streams_d_neighbors[i], d_levels, d_flag, bsz[i], curr_level
            );
        }
        cudaDeviceSynchronize();
        cudaMemcpy(&h_flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemset(d_flag, false, sizeof(bool));
        curr_level++;
    }
    // clean up
    for(int i = 0; i < streams.size();i++) {
        cudaFreeAsync(streams_d_nodes[i], streams[i]);
        cudaFreeAsync(streams_d_neighbors[i], streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    // copy d_levels
    cudaMemcpy(h_levels, d_levels, num_vertexs * sizeof(int), cudaMemcpyDeviceToHost);
}
