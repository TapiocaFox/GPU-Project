#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 1024
#define HALF_BLOCK_SIZE 512

__global__ void scan(float* in, float* out, float* post, int len) {
__shared__ float scan_array[HALF_BLOCK_SIZE];
unsigned int t = threadIdx.x;
unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;
int index;

if (start + t < len) scan_array[t] = in[start + t];
else scan_array[t] = 0;

if (start + BLOCK_SIZE + t < len) scan_array[BLOCK_SIZE + t] = in[start + BLOCK_SIZE + t];
else scan_array[BLOCK_SIZE + t] = 0;
__syncthreads();

for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
index = (t + 1) * stride * 2 - 1;
if (index < 2 * BLOCK_SIZE) scan_array[index] += scan_array[index - stride];
__syncthreads();
}

for (unsigned int stride = BLOCK_SIZE >> 1; stride; stride >>= 1) {
index = (t + 1) * stride * 2 - 1;
if (index + stride < 2 * BLOCK_SIZE) scan_array[index + stride] += scan_array[index];
__syncthreads();
}

if (start + t < len) out[start + t] = scan_array[t];
if (start + BLOCK_SIZE + t < len) out[start + BLOCK_SIZE + t] = scan_array[BLOCK_SIZE + t];

if (post && t == 0) post[blockIdx.x] = scan_array[2 * BLOCK_SIZE - 1];
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int len = 5 * 32 * 5 * 32;
    float *d_in = NULL;
    float *d_out = NULL;
    float *d_post = NULL;
    cudaMalloc(&d_in, len * sizeof(float));
    cudaMalloc(&d_out, len * sizeof(float));
    cudaMalloc(&d_post, 5 * 5 * sizeof(float));
    
    // Warmup
    cudaFree(0);
    scan<<<gridBlock, threadBlock>>>(d_in, d_out, d_post, len);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        scan<<<gridBlock, threadBlock>>>(d_in, d_out, d_post, len);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        scan<<<gridBlock, threadBlock>>>(d_in, d_out, d_post, len);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_post);
    
    return 0;
}
