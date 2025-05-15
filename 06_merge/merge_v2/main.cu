#include <iostream>
#include <algorithm>
#include <random>
#include <memory>

#include "include.cu"
#include "merge_kernel.cu"
#include "externel_merge.cu"

void test_merge_sorted_batches() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_dist(10, MAX_BATCH_SZ);
    std::uniform_int_distribution<> val_dist(0, 1000);

    // Generate random sorted batches
    const int num_batches = 5;
    // std::vector<Batch*> batches;
    std::vector<RUN*> runs;
    std::vector<std::vector<int>> host_data(num_batches);
    int total_rows = 0;
    for (int i = 0; i < num_batches; ++i) {
        // int size = size_dist(gen);
        int size = i == num_batches - 1 ?  size_dist(gen) : MAX_BATCH_SZ;
        total_rows += size;

        std::vector<int> data(size);
        for (int j = 0; j < size; ++j) {
            data[j] = val_dist(gen);
        }
        std::sort(data.begin(), data.end());
        
        auto batch = std::unique_ptr<Batch>(new Batch(size));
        CUDA_CHECK(cudaMemcpy(batch->data, data.data(), size * sizeof(int), cudaMemcpyHostToDevice));

        RUN* run = new RUN();
        run->batchs.push_back(std::move(batch));
        run->total_rows = size;
        runs.push_back(run);

        // copy to host data to sort later
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
    //TODO: add
    merge_all_sorted(runs);

    // Verify result
    std::vector<int> result;
    for(const auto& batch: runs[0]->batchs) {
        int arr[batch->bs];
        CUDA_CHECK(cudaMemcpy(arr, batch->data, batch->bs * sizeof(int), cudaMemcpyDeviceToHost));
        for(int i = 0;i < batch->bs;i++) result.push_back(arr[i]);
    }

    // Create expected result by merging all input data on host
    std::vector<int> expected;
    for (const auto& batch : host_data) {
        expected.insert(expected.end(), batch.begin(), batch.end());
    }
    printf("okay\n");
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

    // Cleanup //NONEED MAKE UNIQUE
    // for (auto& batch : runs[0]->batchs) {
    //     delete batch;
    // }
}


int main() {
    test_merge_sorted_batches();
}