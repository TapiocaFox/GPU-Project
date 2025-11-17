#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void deltaCalcOutput(float *OutActivation, float *Outputdelta, float *targets){
int n = blockIdx.x*blockDim.x + threadIdx.x;
Outputdelta[n] = (targets[n] - OutActivation[n]) * (1 / (1 + exp(-OutActivation[n]))*(1 - 1 / (1 + exp(-OutActivation[n]))));
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int size = 5 * 32 * 5 * 32;
    float *d_OutActivation = NULL;
    float *d_Outputdelta = NULL;
    float *d_targets = NULL;
    cudaMalloc(&d_OutActivation, size * sizeof(float));
    cudaMalloc(&d_Outputdelta, size * sizeof(float));
    cudaMalloc(&d_targets, size * sizeof(float));
    
    // Warmup
    cudaFree(0);
    deltaCalcOutput<<<gridBlock, threadBlock>>>(d_OutActivation, d_Outputdelta, d_targets);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        deltaCalcOutput<<<gridBlock, threadBlock>>>(d_OutActivation, d_Outputdelta, d_targets);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        deltaCalcOutput<<<gridBlock, threadBlock>>>(d_OutActivation, d_Outputdelta, d_targets);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_OutActivation);
    cudaFree(d_Outputdelta);
    cudaFree(d_targets);
    
    return 0;
}
