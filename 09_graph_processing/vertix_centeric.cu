__global__ void bfs_vertix_top_down(int* nodePtrs, int* neighbors, int *levels, bool* flag, int num_nodes, int curr_level){
	int node_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(node_idx < num_nodes && levels[node_idx] == curr_level - 1) {
	    bool tempFlag = false;
			// loop on neighbors
			for(int i = nodePtrs[node_idx]; i < nodePtrs[node_idx + 1]; i++) {
				int neighbor = neighbors[i];
				if(levels[neighbor] == -1) {
					levels[neighbor] = curr_level;
					tempFlag = true;
				}
			}
			// set flag and don't stop iterate unless flag == false
			if(tempFlag) *flag = true;
	} 
}


__global__ void bfs_vertix_bottom_up(int* nodePtrs, int* neighbors, int *levels, bool* flag, int num_nodes, int curr_level){
	int node_idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(node_idx < num_nodes && levels[node_idx] == - 1) {
			// loop on neighbors
			for(int i = nodePtrs[node_idx]; i < nodePtrs[node_idx + 1]; i++) {
				int neighbor = neighbors[i];
				if(levels[neighbor] == curr_level - 1) {
					levels[node_idx] = curr_level;
					*flag = true;
					break;
				}
			}
	} 
}