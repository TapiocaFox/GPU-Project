#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 1024

__global__ void post_scan(float* in, float* add, int len) {
unsigned int t = threadIdx.x;
unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

if (blockIdx.x) {
if (start + t < len) in[start + t] += add[blockIdx.x - 1];
if (start + BLOCK_SIZE + t < len) in[start + BLOCK_SIZE + t] += add[blockIdx.x - 1];
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int len = 5 * 32 * 5 * 32;
    float *d_in = NULL;
    float *d_add = NULL;
    cudaMalloc(&d_in, len * sizeof(float));
    cudaMalloc(&d_add, 5 * 5 * sizeof(float));
    
    // Warmup
    cudaFree(0);
    post_scan<<<gridBlock, threadBlock>>>(d_in, d_add, len);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        post_scan<<<gridBlock, threadBlock>>>(d_in, d_add, len);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        post_scan<<<gridBlock, threadBlock>>>(d_in, d_add, len);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_in);
    cudaFree(d_add);
    
    return 0;
}
