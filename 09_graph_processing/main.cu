#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <queue>
#include <cstring>
#include <cooperative_groups.h>

#include "edge_centeric.cu"
#include "vertix_centeric.cu"
#include "dynamic_programming.cu"
#include "vertix_centeric_optimizations.cu"
// Constants from provided files
#define CEIL_DIV(X, Y) ((X + Y - 1) / Y)
#define CEIL_DIV_I(X, Y) int((X + Y - 1) / Y)


// Utility function to check CUDA errors
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// Read graph from file (edge list format) and convert to CSR and edge list
void read_graph(const std::string& filename, std::vector<int>& h_nodePtrs, std::vector<int>& h_neighbors,
                std::vector<int>& h_rows, std::vector<int>& h_cols, int& num_vertices, int& num_edges) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    std::string line;
    // Skip header lines
    while (std::getline(file, line) && line[0] == '#');
    
    // First pass: count edges and find max vertex ID
    std::vector<std::pair<int, int>> edges;
    num_vertices = 0;
    num_edges = 0;
    do {
        int u, v;
        if (sscanf(line.c_str(), "%d %d", &u, &v) == 2) {
            edges.emplace_back(u, v);
            num_vertices = std::max(num_vertices, std::max(u, v) + 1);
            num_edges++;
        }
    } while (std::getline(file, line));

    file.close();

    // Create edge list
    h_rows.resize(num_edges);
    h_cols.resize(num_edges);
    for (int i = 0; i < num_edges; i++) {
        h_rows[i] = edges[i].first;
        h_cols[i] = edges[i].second;
    }

    // Create CSR: count degrees
    int max_degree = 0;
    std::vector<int> degrees(num_vertices, 0);
    for (const auto& edge : edges) {
        degrees[edge.first]++;
        degrees[edge.second]++; // Undirected graph
        max_degree = std::max(max_degree, degrees[edge.first]);
        max_degree = std::max(max_degree, degrees[edge.second]);
    }
    std::cout << "MAX DEGREE: " << max_degree << "\n";
    // Compute node pointers
    h_nodePtrs.resize(num_vertices + 1);
    h_nodePtrs[0] = 0;
    for (int i = 0; i < num_vertices; i++) {
        h_nodePtrs[i + 1] = h_nodePtrs[i] + degrees[i];
    }

    // Create neighbors list
    h_neighbors.resize(h_nodePtrs[num_vertices]);
    std::vector<int> offsets = h_nodePtrs;
    for (const auto& edge : edges) {
        h_neighbors[offsets[edge.first]++] = edge.second;
        h_neighbors[offsets[edge.second]++] = edge.first; // Undirected graph
    }
}

// CPU BFS for reference
void bfs_cpu(const std::vector<int>& nodePtrs, const std::vector<int>& neighbors, int num_vertices,
             int src_vertex, std::vector<int>& levels) {
    levels.assign(num_vertices, -1);
    std::queue<int> q;
    levels[src_vertex] = 0;
    q.push(src_vertex);

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int i = nodePtrs[u]; i < nodePtrs[u + 1]; i++) {
            int v = neighbors[i];
            if (levels[v] == -1) {
                levels[v] = levels[u] + 1;
                q.push(v);
            }
        }
    }
}

// Verify GPU levels against CPU levels
bool verify_levels(const std::vector<int>& cpu_levels, const int* gpu_levels, int num_vertices,
                  const std::string& kernel_name) {
    bool correct = true;
    for (int i = 0; i < num_vertices; i++) {
        // std::cout << "node i " << i << " cpu " << cpu_levels[i] << "\n";
        if (cpu_levels[i] != gpu_levels[i]) {
            std::cerr << "Mismatch in " << kernel_name << " at vertex " << i
                      << ": expected " << cpu_levels[i] << ", got " << gpu_levels[i] << std::endl;
            correct = false;
            break;
        }
    }
    if (correct) {
        std::cout << kernel_name << " passed verification." << std::endl;
    }
    return correct;
}


int main(int argc, char* argv[]) {
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // printf("Device: %s\n", prop.name);
    // printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    // printf("Cooperative launch: %s\n", 
    //     prop.cooperativeLaunch ? "Yes" : "No");

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <kernel_name> <graph_file>" << std::endl;
        std::cerr << "Kernel names: top_down, bottom_up, edge, opt, reg_priv, block_priv, opt3" << std::endl;
        return 1;
    }

    std::string kernel_name = argv[1];
    std::string graph_file = argv[2];
    std::vector<std::string> valid_kernels = {"top_down", "bottom_up", "edge", "edge_stream", "opt", "opt_dp", "opt_dp_driver", "reg_priv", "block_priv", "opt3"};
    if (std::find(valid_kernels.begin(), valid_kernels.end(), kernel_name) == valid_kernels.end()) {
        std::cerr << "Invalid kernel name: '" << kernel_name << "'" << std::endl;
        return 1;
    }

    // Read graph
    std::vector<int> h_nodePtrs, h_neighbors, h_rows, h_cols;
    int num_vertices, num_edges;
    read_graph(graph_file, h_nodePtrs, h_neighbors, h_rows, h_cols, num_vertices, num_edges);
    std::cout << "Graph loaded: " << num_vertices << " vertices, " << num_edges << " edges" << std::endl;

    const int src_vertex = 1; // Start BFS from vertex 0

    // Run CPU BFS
    std::vector<int> cpu_levels;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    bfs_cpu(h_nodePtrs, h_neighbors, num_vertices, src_vertex, cpu_levels);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU BFS time: " << cpu_time << " ms" << std::endl;
    // ============= start GPU TIME ==============
    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, num_vertices);
    float gpu_time = 0;
    std::vector<int> gpu_levels(num_vertices);

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    // Device arrays
    int *d_nodePtrs, *d_neighbors, *d_rows, *d_cols, *d_levels;
    bool *d_flag;
    CHECK_CUDA_ERROR(cudaMalloc(&d_levels, num_vertices * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_flag, sizeof(bool)));

    CHECK_CUDA_ERROR(cudaMemset(d_levels, -1, num_vertices * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_levels + src_vertex, 0, sizeof(int)));

    if(kernel_name != "edge") {
        CHECK_CUDA_ERROR(cudaMalloc(&d_nodePtrs, (num_vertices + 1) * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_neighbors, num_edges * sizeof(int) * 2)); // Undirected
        // Copy graph data
        CHECK_CUDA_ERROR(cudaMemcpy(d_nodePtrs, h_nodePtrs.data(), (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_neighbors, h_neighbors.data(), h_nodePtrs[num_vertices] * sizeof(int), cudaMemcpyHostToDevice));
    }
    

    if (kernel_name == "top_down") {
        bool h_flag = true;
        int curr_level = 1;
        while (h_flag) {
            CHECK_CUDA_ERROR(cudaMemset(d_flag, 0, sizeof(bool)));
            bfs_vertix_top_down<<<CEIL_DIV_I(num_vertices, BLOCK_DIM), BLOCK_DIM>>>(
                d_nodePtrs, d_neighbors, d_levels, d_flag, num_vertices, curr_level);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(cudaMemcpy(&h_flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost));
            curr_level++;
        }
    }
    else if (kernel_name == "bottom_up") {
        bool h_flag = true;
        int curr_level = 1;
        while (h_flag) {
            CHECK_CUDA_ERROR(cudaMemset(d_flag, 0, sizeof(bool)));
            bfs_vertix_bottom_up<<<CEIL_DIV_I(num_vertices, BLOCK_DIM), BLOCK_DIM>>>(
                d_nodePtrs, d_neighbors, d_levels, d_flag, num_vertices, curr_level);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(cudaMemcpy(&h_flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost));
            curr_level++;
        }
    }
    else if (kernel_name == "edge") {
        
        CHECK_CUDA_ERROR(cudaMalloc(&d_rows, num_edges * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_cols, num_edges * sizeof(int)));
        
        CHECK_CUDA_ERROR(cudaMemcpy(d_rows, h_rows.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_cols, h_cols.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
        
        bool h_flag = true;
        int curr_level = 1;
        while (h_flag) {
            CHECK_CUDA_ERROR(cudaMemset(d_flag, 0, sizeof(bool)));
            bfs_edge<<<CEIL_DIV_I(num_edges, BLOCK_DIM), BLOCK_DIM>>>(
                d_rows, d_cols, d_levels, d_flag, num_edges, curr_level);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(cudaMemcpy(&h_flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost));
            curr_level++;
        }
        CHECK_CUDA_ERROR(cudaFree(d_rows));
        CHECK_CUDA_ERROR(cudaFree(d_cols));
    }
    else if (kernel_name == "edge_stream") {
        launch_bfs_streams(h_rows.data(), h_cols.data(), gpu_levels.data(), d_levels, num_edges, num_vertices, src_vertex);
    }
    else {
        int *d_prev_frontier, *d_curr_frontier, *d_sz_prev_frontier, *d_sz_curr_frontier;
        CHECK_CUDA_ERROR(cudaMalloc(&d_prev_frontier, num_vertices * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_curr_frontier, num_vertices * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_sz_prev_frontier, sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_sz_curr_frontier, sizeof(int)));

        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        CHECK_CUDA_ERROR(cudaEventRecord(start));

        CHECK_CUDA_ERROR(cudaMemset(d_levels, -1, num_vertices * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMemset(d_levels + src_vertex, 0, sizeof(int)));
        int h_prev_frontier[] = {src_vertex};
        int h_sz_prev_frontier = 1;
        int h_sz_curr_frontier = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(d_prev_frontier, h_prev_frontier, sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_sz_prev_frontier, &h_sz_prev_frontier, sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_sz_curr_frontier, &h_sz_curr_frontier, sizeof(int), cudaMemcpyHostToDevice));

        if (kernel_name == "opt3") {
            int *d_curr_level;
            int **d_ptr_prev_frontier, **d_ptr_curr_frontier;

            CHECK_CUDA_ERROR(cudaMalloc(&d_curr_level, sizeof(int)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_ptr_prev_frontier, sizeof(int*)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_ptr_curr_frontier, sizeof(int*)));

            int h_curr_level = 1;
            while (h_sz_prev_frontier > 0) {
                CHECK_CUDA_ERROR(cudaMemcpy(d_curr_level, &h_curr_level, sizeof(int), cudaMemcpyHostToDevice));
                CHECK_CUDA_ERROR(cudaMemcpy(d_ptr_prev_frontier, &d_prev_frontier, sizeof(int*), cudaMemcpyHostToDevice));
                CHECK_CUDA_ERROR(cudaMemcpy(d_ptr_curr_frontier, &d_curr_frontier, sizeof(int*), cudaMemcpyHostToDevice));

                bfs_vertix_opt3<<<CEIL_DIV_I(h_sz_prev_frontier, BLOCK_DIM_OPT), BLOCK_DIM_OPT>>>(
                    d_nodePtrs, d_neighbors, d_levels, d_curr_level,
                    d_ptr_prev_frontier, d_ptr_curr_frontier,
                    d_sz_prev_frontier, d_sz_curr_frontier);

                CHECK_CUDA_ERROR(cudaDeviceSynchronize());

                CHECK_CUDA_ERROR(cudaMemcpy(&h_sz_curr_frontier, d_sz_curr_frontier, sizeof(int), cudaMemcpyDeviceToHost));
                h_sz_prev_frontier = h_sz_curr_frontier;
                h_sz_curr_frontier = 0;
                CHECK_CUDA_ERROR(cudaMemcpy(d_sz_prev_frontier, &h_sz_prev_frontier, sizeof(int), cudaMemcpyHostToDevice));
                CHECK_CUDA_ERROR(cudaMemcpy(d_sz_curr_frontier, &h_sz_curr_frontier, sizeof(int), cudaMemcpyHostToDevice));

                CHECK_CUDA_ERROR(cudaMemcpy(&h_curr_level, d_curr_level, sizeof(int), cudaMemcpyDeviceToHost));
                std::swap(d_prev_frontier, d_curr_frontier);
                h_curr_level++;
            }
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(cudaMemcpy(gpu_levels.data(), d_levels, num_vertices * sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaFree(d_curr_level));
        } 
        else if (kernel_name == "opt_dp_driver") {
            void* kernelArgs[] = { &d_nodePtrs, &d_neighbors, &d_levels, &d_prev_frontier, &d_curr_frontier, &d_sz_prev_frontier, &d_sz_curr_frontier };
            CHECK_CUDA_ERROR(cudaLaunchCooperativeKernel((void*)bfs_dp_driver_kernel, 1, 1, kernelArgs));
            // bfs_dp_driver_kernel<<<1,1>>>(d_nodePtrs, d_neighbors, d_levels, d_prev_frontier, d_curr_frontier, d_sz_prev_frontier, d_sz_curr_frontier);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(cudaMemcpy(gpu_levels.data(), d_levels, num_vertices * sizeof(int), cudaMemcpyDeviceToHost));
        }
        else {
            int curr_level = 1;
            while (h_sz_prev_frontier > 0) {
                if (kernel_name == "opt") {
                    bfs_vertix_opt<<<CEIL_DIV_I(h_sz_prev_frontier, BLOCK_DIM_OPT), BLOCK_DIM_OPT>>>(
                        d_nodePtrs, d_neighbors, d_levels, curr_level,
                        d_prev_frontier, d_curr_frontier, h_sz_prev_frontier, d_sz_curr_frontier);
                }
                else if (kernel_name == "opt_dp") {
                    bfs_vertix_dp<<<CEIL_DIV_I(h_sz_prev_frontier, BLOCK_DIM_OPT), BLOCK_DIM_OPT>>>(
                        d_nodePtrs, d_neighbors, d_levels, curr_level,
                        d_prev_frontier, d_curr_frontier, h_sz_prev_frontier, d_sz_curr_frontier);
                }
                else if (kernel_name == "reg_priv") {
                    bfs_vertix_reg_privatization<<<CEIL_DIV_I(h_sz_prev_frontier, BLOCK_DIM_OPT), BLOCK_DIM_OPT>>>(
                        d_nodePtrs, d_neighbors, d_levels, curr_level,
                        d_prev_frontier, d_curr_frontier, h_sz_prev_frontier, d_sz_curr_frontier);
                }
                else if (kernel_name == "block_priv") {
                    bfs_vertix_block_privatization<<<CEIL_DIV_I(h_sz_prev_frontier, BLOCK_DIM_OPT), BLOCK_DIM_OPT>>>(
                        d_nodePtrs, d_neighbors, d_levels, curr_level,
                        d_prev_frontier, d_curr_frontier, h_sz_prev_frontier, d_sz_curr_frontier);
                }
                CHECK_CUDA_ERROR(cudaDeviceSynchronize());
                CHECK_CUDA_ERROR(cudaMemcpy(&h_sz_curr_frontier, d_sz_curr_frontier, sizeof(int), cudaMemcpyDeviceToHost));
                h_sz_prev_frontier = h_sz_curr_frontier;
                h_sz_curr_frontier = 0;
                CHECK_CUDA_ERROR(cudaMemcpy(d_sz_prev_frontier, &h_sz_prev_frontier, sizeof(int), cudaMemcpyHostToDevice));
                CHECK_CUDA_ERROR(cudaMemcpy(d_sz_curr_frontier, &h_sz_curr_frontier, sizeof(int), cudaMemcpyHostToDevice));
                std::swap(d_prev_frontier, d_curr_frontier);
                curr_level++;
            }
        }
        CHECK_CUDA_ERROR(cudaFree(d_prev_frontier));
        CHECK_CUDA_ERROR(cudaFree(d_curr_frontier));
        CHECK_CUDA_ERROR(cudaFree(d_sz_prev_frontier));
        CHECK_CUDA_ERROR(cudaFree(d_sz_curr_frontier));
    }

    CHECK_CUDA_ERROR(cudaMemcpy(gpu_levels.data(), d_levels, num_vertices * sizeof(int), cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_levels));
    CHECK_CUDA_ERROR(cudaFree(d_flag));
    if(kernel_name != "edge") {
        CHECK_CUDA_ERROR(cudaFree(d_nodePtrs));
        CHECK_CUDA_ERROR(cudaFree(d_neighbors));
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    std::cout << "GPU " << kernel_name << " time: " << gpu_time << " ms" << std::endl;
    verify_levels(cpu_levels, gpu_levels.data(), num_vertices, kernel_name);


    return 0;
}