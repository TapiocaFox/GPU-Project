#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void scale(float knot_max, int nx, int nsamples, float * x, int pitch_x)
{
int
col_idx = blockDim.x * blockIdx.x + threadIdx.x;

if(col_idx >= nx) return;

float
min, max,
* col = x + col_idx * pitch_x;

// find the min and the max
min = max = col[0];
for(int i = 1; i < nsamples; i++) {
if(col[i] < min) min = col[i];
if(col[i] > max) max = col[i];
}

float delta = max - min;
for(int i = 0; i < nsamples; i++)
col[i] = (knot_max * (col[i] - min)) / delta;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    float knot_max = 10.0f;
    int nx = 5 * 32 * 5 * 32;
    int nsamples = 100;
    int pitch_x = nsamples;
    float *d_x = NULL;
    cudaMalloc(&d_x, nx * pitch_x * sizeof(float));
    
    // Warmup
    cudaFree(0);
    scale<<<gridBlock, threadBlock>>>(knot_max, nx, nsamples, d_x, pitch_x);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        scale<<<gridBlock, threadBlock>>>(knot_max, nx, nsamples, d_x, pitch_x);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        scale<<<gridBlock, threadBlock>>>(knot_max, nx, nsamples, d_x, pitch_x);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_x);
    
    return 0;
}
