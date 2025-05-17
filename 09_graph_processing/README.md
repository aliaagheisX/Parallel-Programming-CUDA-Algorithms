# BFS Processing
## Dataset
you can download from https://snap.stanford.edu/data/ any dataset 
```sh
gunzip roadNet-CA.txt.gz  #unzip
```
in makefile set GRAPH_FILE
```bash
GRAPH_FILE = com-youtube.ungraph.txt
```
make sure you tune inside `vertix_centeric_optimizations.cu`
```cpp
#define MAX_OUT_DEGREE 100
#define BLOCK_DIM_OPT 32
```
then run
```sh
 make run_all_no_priv # if max_out_degree very huge can't test optimizations
 make # to run all 
```


## edge_centeric
found `edge_centeric.cu` with two functions 
```cpp
__global__ void bfs_edge(int* rows, int* cols, int *levels, bool* flag, int num_edges, int curr_level);

//streaming
void launch_bfs_streams(int *h_nodes, int *h_neighbors, int* h_levels, int* d_levels, int num_edges, int num_vertexs, int src_vertex)
```

## vetex_centeric
found in `vetex_centeric.cu`
```cpp
__global__ void bfs_vertix_top_down(int* nodePtrs, int* neighbors, int *levels, bool* flag, int num_nodes, int curr_level);
__global__ void bfs_vertix_bottom_up(int* nodePtrs, int* neighbors, int *levels, bool* flag, int num_nodes, int curr_level);
```

found in `vertix_centeric_optimizations.cu`

```cpp
__global__ void bfs_vertix_opt(int* nodesPtr, int* neighbors, int* levels, int curr_level,
																int* prev_frontier, int* curr_frontier,
																const int sz_prev_frontier, int* sz_curr_frontier) ;
// local frontier optimization
__global__ void bfs_vertix_reg_privatization(int* nodesPtr, int* neighbors, int* levels, int curr_level,
                                                            int* prev_frontier, int* curr_frontier,
                                                            const int sz_prev_frontier, int* sz_curr_frontier) ;
// local + shared frontier optimizations
__global__ void bfs_vertix_block_privatization(int* nodesPtr, int* neighbors, int* levels, int curr_level,
                                                            int* prev_frontier, int* curr_frontier,
                                                            const int sz_prev_frontier, int* sz_curr_frontier);

// device function to call from bfs_vertix_opt3
__device__ void bfs_vertix_block_privatization_device(int* nodesPtr, int* neighbors, int* levels, int curr_level,
                                                            int* prev_frontier, int* curr_frontier,
                                                            const int sz_prev_frontier, int* sz_curr_frontier);

// wrapper to reduce kernel launchs
__global__ void bfs_vertix_opt3(int* nodesPtr, int* neigbors, int* levels, int *curr_level,
																int** prev_frontier, int** curr_frontier,
																int* sz_prev_frontier, int* sz_curr_frontier);
```