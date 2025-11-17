#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

// a simple kernel that simply increments each array element by b

// a predicate that checks whether each array elemen is set to its index plus b
__global__ void kernelAddConstant(int *g_a, const int b)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
g_a[idx] += b;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int size = 5 * 32 * 5 * 32;
    int b = 10;
    int *d_g_a = NULL;
    cudaMalloc(&d_g_a, size * sizeof(int));
    
    // Warmup
    cudaFree(0);
    kernelAddConstant<<<gridBlock, threadBlock>>>(d_g_a, b);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        kernelAddConstant<<<gridBlock, threadBlock>>>(d_g_a, b);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        kernelAddConstant<<<gridBlock, threadBlock>>>(d_g_a, b);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_g_a);
    
    return 0;
}
