#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void cuda_cmp_kernel(size_t n, int* aptr, int* bptr, int* rptr) {
int i = threadIdx.x+blockIdx.x*blockDim.x;
int cmp = i<n? aptr[i]<bptr[i]: 0;
if (__syncthreads_or(cmp)) *rptr=1;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    size_t n = 5 * 32 * 5 * 32;
    int *d_aptr = NULL;
    int *d_bptr = NULL;
    int *d_rptr = NULL;
    cudaMalloc(&d_aptr, n * sizeof(int));
    cudaMalloc(&d_bptr, n * sizeof(int));
    cudaMalloc(&d_rptr, sizeof(int));
    
    // Warmup
    cudaFree(0);
    cuda_cmp_kernel<<<gridBlock, threadBlock>>>(n, d_aptr, d_bptr, d_rptr);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        cuda_cmp_kernel<<<gridBlock, threadBlock>>>(n, d_aptr, d_bptr, d_rptr);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        cuda_cmp_kernel<<<gridBlock, threadBlock>>>(n, d_aptr, d_bptr, d_rptr);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_aptr);
    cudaFree(d_bptr);
    cudaFree(d_rptr);
    
    return 0;
}
