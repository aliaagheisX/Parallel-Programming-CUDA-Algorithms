#!/bin/bash

# Create reports directory if it doesn't exist
rm -rf reports
mkdir -p reports

# Compile and profile Basic version
nvcc histogram.cu -o histogram
nsys profile -o reports/cuda_basic_report histogram
nsys stats --format table reports/cuda_basic_report.nsys-rep --report cuda_api_sum -o reports/cuda_basic_report

# Compile and profile CUDA optimized version
nvcc histogram_privatization.cu -o histogram
nsys profile -o reports/cuda_opt_report histogram
nsys stats --format table reports/cuda_opt_report.nsys-rep --report cuda_api_sum -o reports/cuda_opt_report

# Remove .nsys-rep files to save space
rm reports/*.sqlite
