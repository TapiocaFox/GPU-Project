#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

extern "C" {



}

__global__ void vsub(const float *a, const float *b, float *c)
{
int i = blockIdx.x *blockDim.x + threadIdx.x;
c[i] = a[i] - b[i];
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int size = 5 * 32 * 5 * 32;
    const float *d_a = NULL;
    const float *d_b = NULL;
    float *d_c = NULL;
    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_c, size * sizeof(float));
    
    // Warmup
    cudaFree(0);
    vsub<<<gridBlock, threadBlock>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        vsub<<<gridBlock, threadBlock>>>(d_a, d_b, d_c);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        vsub<<<gridBlock, threadBlock>>>(d_a, d_b, d_c);
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
