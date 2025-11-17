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

__global__ void gpuSD(const float * vectsA, size_t na, const float * vectsB, size_t nb, size_t dim, const float * means, const float * numPairs, float * sds)
{
size_t
offset, stride,
tx = threadIdx.x,
bx = blockIdx.x, by = blockIdx.y;
float
a, b,
termA, termB;
__shared__ float
meanA, meanB, n,
threadSumsA[NUMTHREADS], threadSumsB[NUMTHREADS];

if((bx >= na) || (by >= nb))
return;

if(tx == 0) {
meanA = means[bx*nb*2+by*2];
meanB = means[bx*nb*2+by*2+1];
n = numPairs[bx*nb+by];
}
__syncthreads();

threadSumsA[tx] = 0.f;
threadSumsB[tx] = 0.f;
for(offset = tx; offset < dim; offset += NUMTHREADS) {
a = vectsA[bx * dim + offset];
b = vectsB[by * dim + offset];
if(!(isnan(a) || isnan(b))) {
termA = a - meanA;
termB = b - meanB;
threadSumsA[tx] += termA * termA;
threadSumsB[tx] += termB * termB;
}
}
__syncthreads();

for(stride = NUMTHREADS >> 1; stride > 0; stride >>= 1) {
if(tx < stride) {
threadSumsA[tx] += threadSumsA[tx + stride];
threadSumsB[tx] += threadSumsB[tx + stride];
}
__syncthreads();
}
if(tx == 0) {
sds[bx*nb*2+by*2]   = sqrtf(threadSumsA[0] / (n - 1.f));
sds[bx*nb*2+by*2+1] = sqrtf(threadSumsB[0] / (n - 1.f));
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
    size_t dim = 100;
    const float *d_vectsA = NULL;
    const float *d_vectsB = NULL;
    const float *d_means = NULL;
    const float *d_numPairs = NULL;
    float *d_sds = NULL;
    cudaMalloc(&d_vectsA, na * dim * sizeof(float));
    cudaMalloc(&d_vectsB, nb * dim * sizeof(float));
    cudaMalloc(&d_means, na * nb * 2 * sizeof(float));
    cudaMalloc(&d_numPairs, na * nb * sizeof(float));
    cudaMalloc(&d_sds, na * nb * 2 * sizeof(float));
    
    // Warmup
    cudaFree(0);
    gpuSD<<<gridBlock, threadBlock>>>(d_vectsA, na, d_vectsB, nb, dim, d_means, d_numPairs, d_sds);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        gpuSD<<<gridBlock, threadBlock>>>(d_vectsA, na, d_vectsB, nb, dim, d_means, d_numPairs, d_sds);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        gpuSD<<<gridBlock, threadBlock>>>(d_vectsA, na, d_vectsB, nb, dim, d_means, d_numPairs, d_sds);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_vectsA);
    cudaFree(d_vectsB);
    cudaFree(d_means);
    cudaFree(d_numPairs);
    cudaFree(d_sds);
    
    return 0;
}
