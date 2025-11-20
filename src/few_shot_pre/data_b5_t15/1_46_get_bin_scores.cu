#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__device__ float do_fraction(float numer, float denom) {
float result = 0.f;

if((numer == denom) && (numer != 0.f))
result = 1.f;
else if(denom != 0.f)
result = numer / denom;

return result;
}
__global__ void get_bin_scores(int nbins, int order, int nknots, float * knots, int nsamples, int nx, float * x, int pitch_x, float * bins, int pitch_bins)
{
int
col_x = blockDim.x * blockIdx.x + threadIdx.x;

if(col_x >= nx)
return;

float
ld, rd, z,
term1, term2,
* in_col = x + col_x * pitch_x,
* bin_col = bins + col_x * pitch_bins;
int i0;

for(int k = 0; k < nsamples; k++, bin_col += nbins) {
z = in_col[k];
i0 = (int)floorf(z) + order - 1;
if(i0 >= nbins)
i0 = nbins - 1;

bin_col[i0] = 1.f;
for(int i = 2; i <= order; i++) {
for(int j = i0 - i + 1; j <= i0; j++) {
rd = do_fraction(knots[j + i] - z, knots[j + i] - knots[j + 1]);

if((j < 0) || (j >= nbins) || (j >= nknots) || (j + i - 1 < 0) || (j > nknots))
term1 = 0.f;
else {
ld = do_fraction(z - knots[j],
knots[j + i - 1] - knots[j]);
term1 = ld * bin_col[j];
}

if((j + 1 < 0) || (j + 1 >= nbins) || (j + 1 >= nknots) || (j + i < 0) || (j + i >= nknots))
term2 = 0.f;
else {
rd = do_fraction(knots[j + i] - z,
knots[j + i] - knots[j + 1]);
term2 = rd * bin_col[j + 1];
}
bin_col[j] = term1 + term2;
}
}
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int nbins = 10;
    int order = 3;
    int nknots = 20;
    int nsamples = 100;
    int nx = 5 * 32 * 5 * 32;
    int pitch_x = nsamples;
    int pitch_bins = nbins * nsamples;
    float *d_knots = NULL;
    float *d_x = NULL;
    float *d_bins = NULL;
    cudaMalloc(&d_knots, nknots * sizeof(float));
    cudaMalloc(&d_x, nx * pitch_x * sizeof(float));
    cudaMalloc(&d_bins, nx * pitch_bins * sizeof(float));
    
    // Warmup
    cudaFree(0);
    get_bin_scores<<<gridBlock, threadBlock>>>(nbins, order, nknots, d_knots, nsamples, nx, d_x, pitch_x, d_bins, pitch_bins);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        get_bin_scores<<<gridBlock, threadBlock>>>(nbins, order, nknots, d_knots, nsamples, nx, d_x, pitch_x, d_bins, pitch_bins);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        get_bin_scores<<<gridBlock, threadBlock>>>(nbins, order, nknots, d_knots, nsamples, nx, d_x, pitch_x, d_bins, pitch_bins);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_knots);
    cudaFree(d_x);
    cudaFree(d_bins);
    
    return 0;
}
