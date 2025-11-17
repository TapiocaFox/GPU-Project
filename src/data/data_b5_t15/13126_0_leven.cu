#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

extern "C"

__global__ void leven(char* a, char* b, char* costs, int size) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i > 0 && i < size) {

costs[0] = i;
int nw = i - 1;
for(int j = 1; j <= size; j++) {
int firstMin = costs[j] < costs[j-1] ? costs[j] : costs[j-1];
// This line is hard to read due to the lack of min() function
int secondMin = 1 + firstMin < a[i - 1] == b[j - 1] ? nw : nw + 1 ? 1 + firstMin : a[i - 1] == b[j - 1] ? nw : nw + 1;
int cj = secondMin;
nw = costs[j];
costs[j] = cj;
}
}

}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int size = 100;
    char *d_a = NULL;
    char *d_b = NULL;
    char *d_costs = NULL;
    cudaMalloc(&d_a, size * sizeof(char));
    cudaMalloc(&d_b, size * sizeof(char));
    cudaMalloc(&d_costs, (size + 1) * sizeof(char));
    
    // Warmup
    cudaFree(0);
    leven<<<gridBlock, threadBlock>>>(d_a, d_b, d_costs, size);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        leven<<<gridBlock, threadBlock>>>(d_a, d_b, d_costs, size);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        leven<<<gridBlock, threadBlock>>>(d_a, d_b, d_costs, size);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_costs);
    
    return 0;
}
