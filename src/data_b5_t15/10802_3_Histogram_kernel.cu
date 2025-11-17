#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void Histogram_kernel(int size, int bins, int cpu_bins, unsigned int *data, unsigned int *histo) {

extern __shared__ unsigned int l_mem[];
unsigned int* l_histo = l_mem;

// Block and thread index
const int bx = blockIdx.x;
const int tx = threadIdx.x;
const int bD = blockDim.x;
const int gD = gridDim.x;

// Output partition
int bins_per_wg   = (bins - cpu_bins) / gD;
int my_bins_start = bx * bins_per_wg + cpu_bins;
int my_bins_end   = my_bins_start + bins_per_wg;

// Constants for read access
const int begin = tx;
const int end   = size;
const int step  = bD;

// Sub-histograms initialization
for(int pos = tx; pos < bins_per_wg; pos += bD) {
l_histo[pos] = 0;
}

__syncthreads(); // Intra-block synchronization

// Main loop
for(int i = begin; i < end; i += step) {
// Global memory read
unsigned int d = ((data[i] * bins) >> 12);

if(d >= my_bins_start && d < my_bins_end) {
// Atomic vote in shared memory
atomicAdd(&l_histo[d - my_bins_start], 1);
}
}

__syncthreads(); // Intra-block synchronization

// Merge per-block histograms and write to global memory
for(int pos = tx; pos < bins_per_wg; pos += bD) {
unsigned int sum = 0;
for(int base = 0; base < (bins_per_wg); base += (bins_per_wg))
sum += l_histo[base + pos];
// Atomic addition in global memory
histo[pos + my_bins_start] += sum;
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int size = 5 * 32 * 5 * 32;
    int bins = 256;
    int cpu_bins = 0;
    int bins_per_wg = (bins - cpu_bins) / (5 * 5);
    unsigned int *d_data = NULL;
    unsigned int *d_histo = NULL;
    cudaMalloc(&d_data, size * sizeof(unsigned int));
    cudaMalloc(&d_histo, bins * sizeof(unsigned int));
    
    // Warmup
    cudaFree(0);
    Histogram_kernel<<<gridBlock, threadBlock, bins_per_wg * sizeof(unsigned int)>>>(size, bins, cpu_bins, d_data, d_histo);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        Histogram_kernel<<<gridBlock, threadBlock, bins_per_wg * sizeof(unsigned int)>>>(size, bins, cpu_bins, d_data, d_histo);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        Histogram_kernel<<<gridBlock, threadBlock, bins_per_wg * sizeof(unsigned int)>>>(size, bins, cpu_bins, d_data, d_histo);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_data);
    cudaFree(d_histo);
    
    return 0;
}
