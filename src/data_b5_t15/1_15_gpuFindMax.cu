#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define NTHREADS 512

__global__ void gpuFindMax(int n, float * data, int threadWorkLoad, int * maxIndex)
{
int
j, k,
start = threadWorkLoad * threadIdx.x,
end = start + threadWorkLoad;
__shared__ int maxIndicies[NTHREADS];

maxIndicies[threadIdx.x] = -1;

if(start >= n)
return;

int localMaxIndex = start;
for(int i = start+1; i < end; i++) {
if(i >= n)
break;
if(data[i] > data[localMaxIndex])
localMaxIndex = i;
}
maxIndicies[threadIdx.x] = localMaxIndex;
__syncthreads();

for(int i = blockDim.x >> 1; i > 0; i >>= 1) {
if(threadIdx.x < i) {
j = maxIndicies[threadIdx.x];
k = maxIndicies[i + threadIdx.x];
if((j != -1) && (k != -1) && (data[j] < data[k]))
maxIndicies[threadIdx.x] = k;
}
__syncthreads();
}
if(threadIdx.x == 0) {
*maxIndex = maxIndicies[0];
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int n = 5 * 32 * 5 * 32;
    int threadWorkLoad = 10;
    float *d_data = NULL;
    int *d_maxIndex = NULL;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_maxIndex, sizeof(int));
    
    // Warmup
    cudaFree(0);
    gpuFindMax<<<gridBlock, threadBlock>>>(n, d_data, threadWorkLoad, d_maxIndex);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        gpuFindMax<<<gridBlock, threadBlock>>>(n, d_data, threadWorkLoad, d_maxIndex);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        gpuFindMax<<<gridBlock, threadBlock>>>(n, d_data, threadWorkLoad, d_maxIndex);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_data);
    cudaFree(d_maxIndex);
    
    return 0;
}
