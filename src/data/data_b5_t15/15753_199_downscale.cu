#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__device__ int translate_idx_inv( int ii, int d1, int d2, int d3, int scale_factor, int off_x, int off_y) {
int x, y, z, w;
w = ii % d3;
ii = ii/d3;
z = ii % d2;
ii = ii/d2;
y = ii % d1;
ii = ii/d1;
x = ii;
w = w*scale_factor+off_x;
z = z*scale_factor+off_y;
d2 *= scale_factor;
d3 *= scale_factor;
return (((x*d1+y)*d2)+z)*d3+w;
}
__device__ int translate_idx(int ii, int d1, int d2, int d3, int scale_factor) {
int x, y, z, w;
w = ii % d3;
ii = ii/d3;
z = ii % d2;
ii = ii/d2;
y = ii % d1;
ii = ii/d1;
x = ii;
w = w/scale_factor;
z = z/scale_factor;
d2 /= scale_factor;
d3 /= scale_factor;
return (((x*d1+y)*d2)+z)*d3+w;
}
__device__ __forceinline__ size_t idx(const size_t nc, const size_t height, const size_t width, const size_t y, const size_t x) {
return (nc * height + y) * width + x;
}
__global__ void downscale(float *gradInput_data, const float *gradOutput_data, long no_elements, int scale_factor, int d1, int d2, int d3) {
long ii = threadIdx.x + blockDim.x * blockIdx.x;
ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
if (ii >= no_elements) return;
for (int i=0; i < scale_factor; i++){
for(int j=0; j < scale_factor; j++){
int ipidx = translate_idx_inv(ii, d1, d2, d3, scale_factor, i, j);
gradInput_data[ii] += gradOutput_data[ipidx];
}
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int scale_factor = 2;
    int d1 = 3;
    int d2 = 80;
    int d3 = 80;
    long no_elements = d1 * d2 * d3;
    float *d_gradInput_data = NULL;
    const float *d_gradOutput_data = NULL;
    cudaMalloc(&d_gradInput_data, no_elements * sizeof(float));
    cudaMalloc(&d_gradOutput_data, d1 * (d2 * scale_factor) * (d3 * scale_factor) * sizeof(float));
    
    // Warmup
    cudaFree(0);
    downscale<<<gridBlock, threadBlock>>>(d_gradInput_data, d_gradOutput_data, no_elements, scale_factor, d1, d2, d3);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        downscale<<<gridBlock, threadBlock>>>(d_gradInput_data, d_gradOutput_data, no_elements, scale_factor, d1, d2, d3);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        downscale<<<gridBlock, threadBlock>>>(d_gradInput_data, d_gradOutput_data, no_elements, scale_factor, d1, d2, d3);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_gradInput_data);
    cudaFree(d_gradOutput_data);
    
    return 0;
}
