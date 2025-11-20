#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void reduce(int *g_idata, int searchedNumber, int *ok) {

int i = blockIdx.x * blockDim.x + threadIdx.x;

__syncthreads();
if (g_idata[i] == searchedNumber) {
*ok = i;
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int size = 5 * 32 * 5 * 32;
    int searchedNumber = 42;
    int *d_g_idata = NULL;
    int *d_ok = NULL;
    cudaMalloc(&d_g_idata, size * sizeof(int));
    cudaMalloc(&d_ok, sizeof(int));
    
    // Warmup
    cudaFree(0);
    reduce<<<gridBlock, threadBlock>>>(d_g_idata, searchedNumber, d_ok);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        reduce<<<gridBlock, threadBlock>>>(d_g_idata, searchedNumber, d_ok);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        reduce<<<gridBlock, threadBlock>>>(d_g_idata, searchedNumber, d_ok);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_g_idata);
    cudaFree(d_ok);
    
    return 0;
}
