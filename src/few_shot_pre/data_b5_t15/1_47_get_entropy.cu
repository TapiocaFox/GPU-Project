#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void get_entropy(int nbins, int nsamples, int nx, float * bin_scores, int pitch_bin_scores, float * entropies)
{
int
col_x = blockDim.x * blockIdx.x + threadIdx.x;

if(col_x >= nx)
return;

float
* in_col = bin_scores + col_x * pitch_bin_scores,
entropy = 0.f, prob, logp;

for(int i = 0; i < nbins; i++) {
prob = 0.f;
for(int j = 0; j < nsamples; j++)
prob += in_col[j * nbins + i];
prob /= (double) nsamples;

if(prob <= 0.f)
logp = 0.f;
else
logp = __log2f(prob);

entropy += prob * logp;
}
entropies[col_x] = -entropy;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int nbins = 10;
    int nsamples = 100;
    int nx = 5 * 32 * 5 * 32;
    int pitch_bin_scores = nbins * nsamples;
    float *d_bin_scores = NULL;
    float *d_entropies = NULL;
    cudaMalloc(&d_bin_scores, nx * pitch_bin_scores * sizeof(float));
    cudaMalloc(&d_entropies, nx * sizeof(float));
    
    // Warmup
    cudaFree(0);
    get_entropy<<<gridBlock, threadBlock>>>(nbins, nsamples, nx, d_bin_scores, pitch_bin_scores, d_entropies);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        get_entropy<<<gridBlock, threadBlock>>>(nbins, nsamples, nx, d_bin_scores, pitch_bin_scores, d_entropies);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        get_entropy<<<gridBlock, threadBlock>>>(nbins, nsamples, nx, d_bin_scores, pitch_bin_scores, d_entropies);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_bin_scores);
    cudaFree(d_entropies);
    
    return 0;
}
