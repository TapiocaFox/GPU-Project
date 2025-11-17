#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define N 160

__global__ void additionMatricesKernel(int* d_a, int* d_b, int* d_c) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
int j = threadIdx.y + blockIdx.y * blockDim.y;

d_c[i * N + j] = d_a[i * N + j] + d_b[i * N + j];
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int *d_a = NULL;
    int *d_b = NULL;
    int *d_c = NULL;
    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));
    
    // Warmup
    cudaFree(0);
    additionMatricesKernel<<<gridBlock, threadBlock>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        additionMatricesKernel<<<gridBlock, threadBlock>>>(d_a, d_b, d_c);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        additionMatricesKernel<<<gridBlock, threadBlock>>>(d_a, d_b, d_c);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
