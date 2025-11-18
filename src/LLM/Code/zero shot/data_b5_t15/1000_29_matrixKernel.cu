#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 32
#define STRIDE 160

__global__ void matrixKernel(float* d_in, float* d_out) {
// Block index
int bx = blockIdx.x;
int by = blockIdx.y;

// Thread index (current coefficient)
int tx = threadIdx.x;
int ty = threadIdx.y;

float dividend =
d_in[(by * BLOCK_SIZE + 0) * STRIDE + (bx * BLOCK_SIZE + 0)];
float divisor =
d_in[(by * BLOCK_SIZE + ty) * STRIDE + (bx * BLOCK_SIZE + tx)];

d_out[(by * BLOCK_SIZE + ty) * STRIDE + (bx * BLOCK_SIZE + tx)] =
dividend / divisor;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    float *d_in = NULL;
    float *d_out = NULL;
    cudaMalloc(&d_in, STRIDE * STRIDE * sizeof(float));
    cudaMalloc(&d_out, STRIDE * STRIDE * sizeof(float));
    
    // Warmup
    cudaFree(0);
    matrixKernel<<<gridBlock, threadBlock>>>(d_in, d_out);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        matrixKernel<<<gridBlock, threadBlock>>>(d_in, d_out);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        matrixKernel<<<gridBlock, threadBlock>>>(d_in, d_out);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}
