#!/bin/bash

# Create reports directory if it doesn't exist
mkdir -p reports

# Compile and profile CPU version
nvcc vector_matrix_mult.cpp -o vector_matrix_mult
nsys profile -o reports/cpu_report vector_matrix_mult
nsys stats --format table reports/cpu_report.nsys-rep --report cuda_api_sum -o reports/cpu_report

# Compile and profile CUDA basic version
nvcc vector_matrix_mult.cu -o vector_matrix_mult
nsys profile -o reports/cuda_basic_report vector_matrix_mult
nsys stats --format table reports/cuda_basic_report.nsys-rep --report cuda_api_sum -o reports/cuda_basic_report

# Compile and profile CUDA optimized version
nvcc vector_matrix_mult_mem_opt.cu -o vector_matrix_mult
nsys profile -o reports/cuda_opt_report vector_matrix_mult
nsys stats --format table reports/cuda_opt_report.nsys-rep --report cuda_api_sum -o reports/cuda_opt_report

# Remove .nsys-rep files to save space
rm reports/*.sqlite
