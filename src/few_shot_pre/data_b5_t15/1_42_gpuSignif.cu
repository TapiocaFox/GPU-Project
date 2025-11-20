#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define FALSE 0
#define TRUE !FALSE

#define NUMTHREADS 16
#define THREADWORK 32

__global__ void gpuSignif(const float * gpuNumPairs, const float * gpuCorrelations, size_t n, float * gpuTScores)
{
size_t
i, start,
bx = blockIdx.x, tx = threadIdx.x;
float
radicand, cor, npairs;

start = bx * NUMTHREADS * THREADWORK + tx * THREADWORK;
for(i = 0; i < THREADWORK; i++) {
if(start+i >= n)
break;

npairs = gpuNumPairs[start+i];
cor = gpuCorrelations[start+i];
radicand = (npairs - 2.f) / (1.f - cor * cor);
gpuTScores[start+i] = cor * sqrtf(radicand);
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    size_t n = 5 * 32 * 5 * 32;
    float *d_gpuNumPairs = NULL;
    float *d_gpuCorrelations = NULL;
    float *d_gpuTScores = NULL;
    cudaMalloc(&d_gpuNumPairs, n * sizeof(float));
    cudaMalloc(&d_gpuCorrelations, n * sizeof(float));
    cudaMalloc(&d_gpuTScores, n * sizeof(float));
    
    // Warmup
    cudaFree(0);
    gpuSignif<<<gridBlock, threadBlock>>>(d_gpuNumPairs, d_gpuCorrelations, n, d_gpuTScores);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        gpuSignif<<<gridBlock, threadBlock>>>(d_gpuNumPairs, d_gpuCorrelations, n, d_gpuTScores);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        gpuSignif<<<gridBlock, threadBlock>>>(d_gpuNumPairs, d_gpuCorrelations, n, d_gpuTScores);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_gpuNumPairs);
    cudaFree(d_gpuCorrelations);
    cudaFree(d_gpuTScores);
    
    return 0;
}
