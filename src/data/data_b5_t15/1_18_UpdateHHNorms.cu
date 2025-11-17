#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define NTHREADS 512

__global__ void UpdateHHNorms(int cols, float *dV, float *dNorms) {
// Copyright 2009, Mark Seligman at Rapid Biologics, LLC.  All rights
// reserved.

int colIndex = threadIdx.x + blockIdx.x * blockDim.x;
if (colIndex < cols) {
float val = dV[colIndex];
dNorms[colIndex] -= val * val;
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int cols = 5 * 32 * 5 * 32;
    float *d_dV = NULL;
    float *d_dNorms = NULL;
    cudaMalloc(&d_dV, cols * sizeof(float));
    cudaMalloc(&d_dNorms, cols * sizeof(float));
    
    // Warmup
    cudaFree(0);
    UpdateHHNorms<<<gridBlock, threadBlock>>>(cols, d_dV, d_dNorms);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        UpdateHHNorms<<<gridBlock, threadBlock>>>(cols, d_dV, d_dNorms);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        UpdateHHNorms<<<gridBlock, threadBlock>>>(cols, d_dV, d_dNorms);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_dV);
    cudaFree(d_dNorms);
    
    return 0;
}
