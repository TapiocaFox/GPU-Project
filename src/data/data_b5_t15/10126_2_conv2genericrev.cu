#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

#define CUDA_SHARED_MEM_SIZE 1024

__global__ void conv2genericrev(float *input, float *kernel, float *output, int input_n, int input_h, int input_w, int kernel_n, int kernel_h, int kernel_w, float alpha, int stride_h, int stride_w)
{
// output dimensions
int output_h = input_h - (kernel_h - 1) * stride_h;
int output_w = input_w - (kernel_w - 1) * stride_w;

// this thread only processes one output, defined by the block Ids
int kk = blockIdx.x;
int ii = blockIdx.y;

// batch id
int batch = threadIdx.z;

// kernel id
int kid = threadIdx.x;
int nkids = blockDim.x;

// thread ID
int tid = kid + batch*blockDim.x;
int nthreads = blockDim.x * blockDim.z;

// one thread only sees one output
output = output + (kk * input_n + ii) * output_h*output_w;

// put the output in shared memory
__shared__ float shared_output[CUDA_SHARED_MEM_SIZE];

// generate tid outputs in shared memory
float *output_s = shared_output + tid*output_w*output_h;

// convolution loop
int xx, yy, kx, ky;
yy = threadIdx.y;
float *output_p = output_s + yy * output_w;
for(xx=0; xx<output_w; xx++) {
// Dot product in two dimensions... (between input image and kernel)
float *input_p = input + (ii + batch*input_n)*input_h*input_w + yy*stride_h*input_w + xx*stride_w;
float *kernel_p = kernel + (kk + batch*kernel_n)*kernel_w*kernel_h;
float sum = 0;
for(ky=0; ky<kernel_h; ky++) {
for(kx=kid; kx<kernel_w; kx+=nkids) {
sum += input_p[kx]*kernel_p[kx];
}
input_p += input_w;
kernel_p += kernel_w;
}
*(output_p++) = sum;
}
__syncthreads();

// reduce and write back
if (yy == 0) {
// reduce outputs
for (int k=1; k<nthreads; k++) {
for (int i=tid; i<output_w*output_h; i+=nthreads) {
shared_output[i] += shared_output[k*output_h*output_w + i];
}
}
__syncthreads();

// add existing output, and write back
for (int i=tid; i<output_w*output_h; i+=nthreads) {
output[i] += alpha*shared_output[i];
}
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32, 1);
    
    // Allocate device memory
    int input_n = 3;
    int input_h = 160;
    int input_w = 160;
    int kernel_n = 3;
    int kernel_h = 3;
    int kernel_w = 3;
    int output_h = input_h - (kernel_h - 1);
    int output_w = input_w - (kernel_w - 1);
    float alpha = 1.0f;
    int stride_h = 1;
    int stride_w = 1;
    float *d_input = NULL;
    float *d_kernel = NULL;
    float *d_output = NULL;
    cudaMalloc(&d_input, input_n * input_h * input_w * sizeof(float));
    cudaMalloc(&d_kernel, kernel_n * kernel_h * kernel_w * sizeof(float));
    cudaMalloc(&d_output, kernel_n * input_n * output_h * output_w * sizeof(float));
    
    // Warmup
    cudaFree(0);
    conv2genericrev<<<gridBlock, threadBlock>>>(d_input, d_kernel, d_output, input_n, input_h, input_w, kernel_n, kernel_h, kernel_w, alpha, stride_h, stride_w);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        conv2genericrev<<<gridBlock, threadBlock>>>(d_input, d_kernel, d_output, input_n, input_h, input_w, kernel_n, kernel_h, kernel_w, alpha, stride_h, stride_w);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        conv2genericrev<<<gridBlock, threadBlock>>>(d_input, d_kernel, d_output, input_n, input_h, input_w, kernel_n, kernel_h, kernel_w, alpha, stride_h, stride_w);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    
    return 0;
}
