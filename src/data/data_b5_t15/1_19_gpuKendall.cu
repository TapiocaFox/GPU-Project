#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define NUMTHREADS 16
#define THREADWORK 32

__global__ void gpuKendall(const float * a, size_t na, const float * b, size_t nb, size_t sampleSize, double * results)
{
size_t
i, j, tests,
tx = threadIdx.x, ty = threadIdx.y,
bx = blockIdx.x, by = blockIdx.y,
rowa = bx * sampleSize, rowb = by * sampleSize;
float
discordant, concordant = 0.f,
numer, denom;

__shared__ float threadSums[NUMTHREADS*NUMTHREADS];

for(i = tx; i < sampleSize; i += NUMTHREADS) {
for(j = i+1+ty; j < sampleSize; j += NUMTHREADS) {
tests = ((a[rowa+j] >  a[rowa+i]) && (b[rowb+j] >  b[rowb+i]))
+ ((a[rowa+j] <  a[rowa+i]) && (b[rowb+j] <  b[rowb+i]))
+ ((a[rowa+j] == a[rowa+i]) && (b[rowb+j] == b[rowb+i]));
concordant = concordant + (float)tests;
}
}
threadSums[tx*NUMTHREADS+ty] = concordant;

__syncthreads();
for(i = NUMTHREADS >> 1; i > 0; i >>= 1) {
if(ty < i)
threadSums[tx*NUMTHREADS+ty] += threadSums[tx*NUMTHREADS+ty+i];
__syncthreads();
}
for(i = NUMTHREADS >> 1; i > 0; i >>= 1) {
if((tx < i) && (ty == 0))
threadSums[tx*NUMTHREADS] += threadSums[(tx+i)*NUMTHREADS];
__syncthreads();
}

if((tx == 0) && (ty == 0)) {
concordant = threadSums[0];
denom = (float)sampleSize;
denom = (denom * (denom - 1.f)) / 2.f; discordant = denom - concordant;
numer = concordant - discordant;
results[by*na+bx] = ((double)numer)/((double)denom);
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    size_t na = 5;
    size_t nb = 5;
    size_t sampleSize = 100;
    const float *d_a = NULL;
    const float *d_b = NULL;
    double *d_results = NULL;
    cudaMalloc(&d_a, na * sampleSize * sizeof(float));
    cudaMalloc(&d_b, nb * sampleSize * sizeof(float));
    cudaMalloc(&d_results, na * nb * sizeof(double));
    
    // Warmup
    cudaFree(0);
    gpuKendall<<<gridBlock, threadBlock>>>(d_a, na, d_b, nb, sampleSize, d_results);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        gpuKendall<<<gridBlock, threadBlock>>>(d_a, na, d_b, nb, sampleSize, d_results);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        gpuKendall<<<gridBlock, threadBlock>>>(d_a, na, d_b, nb, sampleSize, d_results);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_results);
    
    return 0;
}