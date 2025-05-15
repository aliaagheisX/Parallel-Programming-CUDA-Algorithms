#!/bin/bash

# Create reports directory if it doesn't exist
rm -rf profiling
mkdir -p profiling

# Compile and profile Basic version
nvcc histogram.cu -o histogram
ncu -o ./profiling/histo histogram

# Compile and profile CUDA optimized version
nvcc histogram_privatization.cu -o histogram
ncu -o ./profiling/histogram_privatization histogram

