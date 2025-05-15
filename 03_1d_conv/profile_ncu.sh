#!/bin/bash

# Create reports directory if it doesn't exist
rm -rf profiling
mkdir -p profiling

# Compile and profile Basic version
nvcc simple.cu -o simple
ncu -o ./profiling/simple simple

# Compile and profile CUDA optimized version
nvcc tiled.cu -o tiled 
ncu -o ./profiling/tiled  tiled

