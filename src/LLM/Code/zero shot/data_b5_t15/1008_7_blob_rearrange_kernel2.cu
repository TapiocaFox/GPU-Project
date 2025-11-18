#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void blob_rearrange_kernel2(const float *in, float *out, int num, int channels, int width, int height, int widthheight, int padding, int pwidthheight)
{
int xy = blockIdx.x*blockDim.x + threadIdx.x;
if(xy>=widthheight)
return;

int ch = blockIdx.y;
int n  = blockIdx.z;


float value=in[(n*channels+ch)*widthheight+xy];

__syncthreads();

int xpad  = (xy % width + padding);
int ypad  = (xy / width + padding);
int xypad = ypad * (width+2*padding) + xpad;

out[(n*pwidthheight+xypad)*channels + ch] = value;
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5, 1);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int num = 1;
    int channels = 3;
    int width = 160;
    int height = 160;
    int widthheight = width * height;
    int padding = 1;
    int pwidthheight = (width + 2*padding) * (height + 2*padding);
    float *d_in = NULL;
    float *d_out = NULL;
    cudaMalloc(&d_in, num * channels * widthheight * sizeof(float));
    cudaMalloc(&d_out, num * channels * pwidthheight * sizeof(float));
    
    // Warmup
    cudaFree(0);
    blob_rearrange_kernel2<<<gridBlock, threadBlock>>>(d_in, d_out, num, channels, width, height, widthheight, padding, pwidthheight);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        blob_rearrange_kernel2<<<gridBlock, threadBlock>>>(d_in, d_out, num, channels, width, height, widthheight, padding, pwidthheight);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        blob_rearrange_kernel2<<<gridBlock, threadBlock>>>(d_in, d_out, num, channels, width, height, widthheight, padding, pwidthheight);
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
