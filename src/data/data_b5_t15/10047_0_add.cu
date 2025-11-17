#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

extern "C"
__global__ void add(int n, float *a, float *b, float *sum)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i<n)
{
sum[i] = a[i] + b[i];
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int n = 5 * 32 * 5 * 32;
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_sum = NULL;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_sum, n * sizeof(float));
    
    // Warmup
    cudaFree(0);
    add<<<gridBlock, threadBlock>>>(n, d_a, d_b, d_sum);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        add<<<gridBlock, threadBlock>>>(n, d_a, d_b, d_sum);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        add<<<gridBlock, threadBlock>>>(n, d_a, d_b, d_sum);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_sum);
    
    return 0;
}
