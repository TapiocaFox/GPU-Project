#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void get_mi(int nbins, int nsamples, int nx, float * x_bin_scores, int pitch_x_bin_scores, float * entropies_x, int ny, float * y_bin_scores, int pitch_y_bin_scores, float * entropies_y, float * mis, int pitch_mis)
{
int
col_x = blockDim.x * blockIdx.x + threadIdx.x,
col_y = blockDim.y * blockIdx.y + threadIdx.y;

if((col_x >= nx) || (col_y >= ny))
return;

float
prob, logp, mi = 0.f,
* x_bins = x_bin_scores + col_x * pitch_x_bin_scores,
* y_bins = y_bin_scores + col_y * pitch_y_bin_scores;

// calculate joint entropy
for(int i = 0; i < nbins; i++) {
for(int j = 0; j < nbins; j++) {
prob = 0.f;
for(int k = 0; k < nsamples; k++)
prob += x_bins[k * nbins + i] * y_bins[k * nbins + j];
prob /= (float)nsamples;

if(prob <= 0.f)
logp = 0.f;
else
logp = __log2f(prob);

mi += prob * logp;
}
}

// calculate mi from entropies
mi += entropies_x[col_x] + entropies_y[col_y];
(mis + col_y * pitch_mis)[col_x] = mi;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int nbins = 10;
    int nsamples = 100;
    int nx = 5 * 32;
    int ny = 5 * 32;
    int pitch_x_bin_scores = nbins * nsamples;
    int pitch_y_bin_scores = nbins * nsamples;
    int pitch_mis = nx;
    float *d_x_bin_scores = NULL;
    float *d_entropies_x = NULL;
    float *d_y_bin_scores = NULL;
    float *d_entropies_y = NULL;
    float *d_mis = NULL;
    cudaMalloc(&d_x_bin_scores, nx * pitch_x_bin_scores * sizeof(float));
    cudaMalloc(&d_entropies_x, nx * sizeof(float));
    cudaMalloc(&d_y_bin_scores, ny * pitch_y_bin_scores * sizeof(float));
    cudaMalloc(&d_entropies_y, ny * sizeof(float));
    cudaMalloc(&d_mis, ny * pitch_mis * sizeof(float));
    
    // Warmup
    cudaFree(0);
    get_mi<<<gridBlock, threadBlock>>>(nbins, nsamples, nx, d_x_bin_scores, pitch_x_bin_scores, d_entropies_x, ny, d_y_bin_scores, pitch_y_bin_scores, d_entropies_y, d_mis, pitch_mis);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        get_mi<<<gridBlock, threadBlock>>>(nbins, nsamples, nx, d_x_bin_scores, pitch_x_bin_scores, d_entropies_x, ny, d_y_bin_scores, pitch_y_bin_scores, d_entropies_y, d_mis, pitch_mis);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        get_mi<<<gridBlock, threadBlock>>>(nbins, nsamples, nx, d_x_bin_scores, pitch_x_bin_scores, d_entropies_x, ny, d_y_bin_scores, pitch_y_bin_scores, d_entropies_y, d_mis, pitch_mis);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_x_bin_scores);
    cudaFree(d_entropies_x);
    cudaFree(d_y_bin_scores);
    cudaFree(d_entropies_y);
    cudaFree(d_mis);
    
    return 0;
}
