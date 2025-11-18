#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define NUM_THREADS 32

__global__ void euclidean_kernel_same(const float * vg_a, size_t pitch_a, size_t n_a, const float * vg_b, size_t pitch_b, size_t n_b, size_t k, float * d, size_t pitch_d, float p)
{
size_t
x = blockIdx.x, y = blockIdx.y;

if((x == y) && (x < n_a) && (threadIdx.x == 0))
d[y * pitch_d + x] = 0.0;

// If all element is to be computed
if(y < n_a && x < y) {
__shared__ float temp[NUM_THREADS];

temp[threadIdx.x] = 0.0;

for(size_t offset = threadIdx.x; offset < k; offset += NUM_THREADS) {
float t = vg_a[x * pitch_a + offset] - vg_a[y * pitch_a + offset];
temp[threadIdx.x] += (t * t);
}

// Sync with other threads
__syncthreads();

// Reduce
for(size_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
if(threadIdx.x < stride)
temp[threadIdx.x] += temp[threadIdx.x + stride];
__syncthreads();
}

// Write to global memory
if(threadIdx.x == 0) {
float s = sqrt(temp[0]);
d[y * pitch_d + x] = s;
d[x * pitch_d + y] = s;
}
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    size_t size = 5 * 32 * 5 * 32;
    float *d_vg_a = NULL;
    float *d_vg_b = NULL;
    float *d_d = NULL;
    cudaMalloc(&d_vg_a, size * sizeof(float));
    cudaMalloc(&d_vg_b, size * sizeof(float));
    cudaMalloc(&d_d, size * sizeof(float));
    
    size_t pitch_a = 5 * 32;
    size_t n_a = 5 * 32;
    size_t pitch_b = 5 * 32;
    size_t n_b = 5 * 32;
    size_t k = 100;
    size_t pitch_d = 5 * 32;
    float p = 2.0f;
    
    // Warmup
    cudaFree(0);
    euclidean_kernel_same<<<gridBlock, threadBlock>>>(d_vg_a, pitch_a, n_a, d_vg_b, pitch_b, n_b, k, d_d, pitch_d, p);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        euclidean_kernel_same<<<gridBlock, threadBlock>>>(d_vg_a, pitch_a, n_a, d_vg_b, pitch_b, n_b, k, d_d, pitch_d, p);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        euclidean_kernel_same<<<gridBlock, threadBlock>>>(d_vg_a, pitch_a, n_a, d_vg_b, pitch_b, n_b, k, d_d, pitch_d, p);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_vg_a);
    cudaFree(d_vg_b);
    cudaFree(d_d);
    
    return 0;
}
