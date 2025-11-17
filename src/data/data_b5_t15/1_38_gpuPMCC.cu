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

__global__ void gpuPMCC(const float * vectsa, size_t na, const float * vectsb, size_t nb, size_t dim, const float * numPairs, const float * means, const float * sds, float * correlations)
{
size_t
offset, stride,
x = blockIdx.x, y = blockIdx.y,
tx = threadIdx.x;
float
a, b, n, scoreA, scoreB;
__shared__ float
meanA, meanB,
sdA, sdB,
threadSums[NUMTHREADS];

if((x >= na) || (y >= nb))
return;

if(tx == 0) {
meanA = means[x*nb*2+y*2];
meanB = means[x*nb*2+y*2+1];
sdA = sds[x*nb*2+y*2];
sdB = sds[x*nb*2+y*2+1];
n = numPairs[x*nb+y];
}
__syncthreads();

threadSums[tx] = 0.f;
for(offset = tx; offset < dim; offset += NUMTHREADS) {
a = vectsa[x * dim + offset];
b = vectsb[y * dim + offset];
if(!(isnan(a) || isnan(b))) {
scoreA = (a - meanA) / sdA;
scoreB = (b - meanB) / sdB;
threadSums[tx] += scoreA * scoreB;
}
}
__syncthreads();

for(stride = NUMTHREADS >> 1; stride > 0; stride >>= 1) {
if(tx < stride) threadSums[tx] += threadSums[tx + stride];
__syncthreads();
}
if(tx == 0) correlations[x*nb+y] = threadSums[0] / (n - 1.f);
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
    const float *d_vectsa = NULL;
    const float *d_vectsb = NULL;
    const float *d_numPairs = NULL;
    const float *d_means = NULL;
    const float *d_sds = NULL;
    float *d_correlations = NULL;
    cudaMalloc(&d_vectsa, na * dim * sizeof(float));
    cudaMalloc(&d_vectsb, nb * dim * sizeof(float));
    cudaMalloc(&d_numPairs, na * nb * sizeof(float));
    cudaMalloc(&d_means, na * nb * 2 * sizeof(float));
    cudaMalloc(&d_sds, na * nb * 2 * sizeof(float));
    cudaMalloc(&d_correlations, na * nb * sizeof(float));
    
    // Warmup
    cudaFree(0);
    gpuPMCC<<<gridBlock, threadBlock>>>(d_vectsa, na, d_vectsb, nb, dim, d_numPairs, d_means, d_sds, d_correlations);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        gpuPMCC<<<gridBlock, threadBlock>>>(d_vectsa, na, d_vectsb, nb, dim, d_numPairs, d_means, d_sds, d_correlations);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        gpuPMCC<<<gridBlock, threadBlock>>>(d_vectsa, na, d_vectsb, nb, dim, d_numPairs, d_means, d_sds, d_correlations);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_vectsa);
    cudaFree(d_vectsb);
    cudaFree(d_numPairs);
    cudaFree(d_means);
    cudaFree(d_sds);
    cudaFree(d_correlations);
    
    return 0;
}
