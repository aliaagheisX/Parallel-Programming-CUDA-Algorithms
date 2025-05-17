NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_86
# KERNEL_FILES = vertix_centeric.cu edge_centeric.cu vertix_centeric_optimizations.cu
TARGET = bfs_test
# GRAPH_FILE = roadNet-CA.txt
GRAPH_FILE = com-youtube.ungraph.txt

all: build run_all

build:
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) main.cu

run_all: run_top_down run_bottom_up run_edge run_edge_stream run_opt run_reg_priv run_block_priv run_opt3
run_all_no_priv: run_top_down run_bottom_up run_edge run_edge_stream

run_top_down: build
	@echo "Running top_down kernel..."
	./$(TARGET) top_down $(GRAPH_FILE)

run_bottom_up: build
	@echo "Running bottom_up kernel..."
	./$(TARGET) bottom_up $(GRAPH_FILE)

run_edge: build
	@echo "Running edge kernel..."
	./$(TARGET) edge $(GRAPH_FILE)

run_edge_stream: build
	@echo "Running edge kernel..."
	./$(TARGET) edge_stream $(GRAPH_FILE)

run_opt: build
	@echo "Running opt kernel..."
	./$(TARGET) opt $(GRAPH_FILE)

run_reg_priv: build
	@echo "Running reg_priv kernel..."
	./$(TARGET) reg_priv $(GRAPH_FILE)

run_block_priv: build
	@echo "Running block_priv kernel..."
	./$(TARGET) block_priv $(GRAPH_FILE)

run_opt3: build
	@echo "Running opt3 kernel..."
	./$(TARGET) opt3 $(GRAPH_FILE)

clean:
	rm -f $(TARGET)

.PHONY: all build run_all run_top_down run_bottom_up run_edge run_opt run_reg_priv run_block_priv run_opt3 clean