#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define NTHREADS 512

__global__ void makeHVector(int rows, float * input, float * output)
{
int
i, j;
float
elt, sum;
__shared__ float
beta, sums[NTHREADS];

if(threadIdx.x >= rows)
return;

sum = 0.f;
for(i = threadIdx.x ; i < rows; i += NTHREADS) {
if((threadIdx.x == 0) && (i == 0))
continue;
elt = input[i];
output[i] = elt;
sum += elt * elt;
}
sums[threadIdx.x] = sum;
__syncthreads();

for(i = blockDim.x >> 1; i > 0 ; i >>= 1) {
j = i+threadIdx.x;
if((threadIdx.x < i) && (j < rows))
sums[threadIdx.x] += sums[j];
__syncthreads();
}

if(threadIdx.x == 0) {
elt = input[0];
float norm = sqrtf(elt * elt + sums[0]);

if(elt > 0)
elt += norm;
else
elt -= norm;

output[0] = elt;

norm = elt * elt + sums[0];
beta = sqrtf(2.f / norm);
}
__syncthreads();

for(i = threadIdx.x; i < rows; i += NTHREADS)
output[i] *= beta;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int rows = 5 * 32 * 5 * 32;
    float *d_input = NULL;
    float *d_output = NULL;
    cudaMalloc(&d_input, rows * sizeof(float));
    cudaMalloc(&d_output, rows * sizeof(float));
    
    // Warmup
    cudaFree(0);
    makeHVector<<<gridBlock, threadBlock>>>(rows, d_input, d_output);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        makeHVector<<<gridBlock, threadBlock>>>(rows, d_input, d_output);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        makeHVector<<<gridBlock, threadBlock>>>(rows, d_input, d_output);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
