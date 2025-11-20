#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void shmem_reduce_kernel(float * d_out, const float * d_in)
{
// sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
extern __shared__ float sdata[];

int myId = threadIdx.x + blockDim.x * blockIdx.x;
int tid  = threadIdx.x;

// load shared mem from global mem
sdata[tid] = d_in[myId];
__syncthreads();            // make sure entire block is loaded!

// do reduction in shared mem
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
{
if (tid < s)
{
sdata[tid] += sdata[tid + s];
}
__syncthreads();        // make sure all adds at one stage are done!
}

// only thread 0 writes result for this block back to global mem
if (tid == 0)
{
d_out[blockIdx.x] = sdata[0];
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int size = 5 * 32 * 5 * 32;
    float *d_in = NULL;
    float *d_out = NULL;
    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, 5 * 5 * sizeof(float));
    
    // Warmup
    cudaFree(0);
    shmem_reduce_kernel<<<gridBlock, threadBlock, 32 * 32 * sizeof(float)>>>(d_out, d_in);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        shmem_reduce_kernel<<<gridBlock, threadBlock, 32 * 32 * sizeof(float)>>>(d_out, d_in);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        shmem_reduce_kernel<<<gridBlock, threadBlock, 32 * 32 * sizeof(float)>>>(d_out, d_in);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    return 0;
}
