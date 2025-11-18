#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 1024

#ifndef THREADS
# define THREADS 1024
#endif

__global__ void total(float * input, float * output, unsigned int len) {
__shared__ float sum[2*BLOCK_SIZE];
unsigned int i = threadIdx.x;
unsigned int j = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

float localSum = (i < len) ? input[j] : 0;
if (j + blockDim.x < len) localSum += input[j + blockDim.x];

sum[i] = localSum;
__syncthreads();

for (unsigned int step = blockDim.x / 2; step >= 1; step >>= 1) {
if (i < step) sum[i] = localSum = localSum + sum[i + step];
__syncthreads();
}

if(i == 0) output[blockIdx.x] = sum[0];
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    unsigned int len = 5 * 32 * 5 * 32;
    float *d_input = NULL;
    float *d_output = NULL;
    cudaMalloc(&d_input, len * sizeof(float));
    cudaMalloc(&d_output, 5 * 5 * sizeof(float));
    
    // Warmup
    cudaFree(0);
    total<<<gridBlock, threadBlock>>>(d_input, d_output, len);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        total<<<gridBlock, threadBlock>>>(d_input, d_output, len);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        total<<<gridBlock, threadBlock>>>(d_input, d_output, len);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
