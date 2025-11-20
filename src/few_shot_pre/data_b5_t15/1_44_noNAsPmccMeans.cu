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

__global__ void noNAsPmccMeans(int nRows, int nCols, float * a, float * means)
{
int
col = blockDim.x * blockIdx.x + threadIdx.x,
inOffset = col * nRows,
outOffset = threadIdx.x * blockDim.y,
j = outOffset + threadIdx.y;
float sum = 0.f;

if(col >= nCols) return;

__shared__ float threadSums[NUMTHREADS*NUMTHREADS];

for(int i = threadIdx.y; i < nRows; i += blockDim.y)
sum += a[inOffset + i];

threadSums[j] = sum;
__syncthreads();

for(int i = blockDim.y >> 1; i > 0; i >>= 1) {
if(threadIdx.y < i) {
threadSums[outOffset+threadIdx.y]
+= threadSums[outOffset+threadIdx.y + i];
}
__syncthreads();
}
if(threadIdx.y == 0)
means[col] = threadSums[outOffset] / (float)nRows;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int nRows = 100;
    int nCols = 5 * 32 * 5 * 32;
    float *d_a = NULL;
    float *d_means = NULL;
    cudaMalloc(&d_a, nRows * nCols * sizeof(float));
    cudaMalloc(&d_means, nCols * sizeof(float));
    
    // Warmup
    cudaFree(0);
    noNAsPmccMeans<<<gridBlock, threadBlock>>>(nRows, nCols, d_a, d_means);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        noNAsPmccMeans<<<gridBlock, threadBlock>>>(nRows, nCols, d_a, d_means);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        noNAsPmccMeans<<<gridBlock, threadBlock>>>(nRows, nCols, d_a, d_means);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_a);
    cudaFree(d_means);
    
    return 0;
}
