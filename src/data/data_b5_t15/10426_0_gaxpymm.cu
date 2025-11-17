#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;


__global__ void gaxpymm(double *y, double *a, double *b, int m, int n, int p){
int bid = blockIdx.x;
int tid = threadIdx.x;
extern __shared__ double dots_s[];
if(bid<m)
if(tid<n){
for(int c=0;c<p;c++)
dots_s[bid*n*p+tid*p+c] = a[bid*n+tid] * *(b+(tid*p+c));
__syncthreads();
if(tid == 0){
for(int c=0;c<p;c++)
for(int i=1;i<n;i++){
dots_s[bid*n*p+c] +=dots_s[bid*n*p+i*p+c];
}
for(int c=0;c<p;c++)
*(y+(bid*p+c))=dots_s[bid*n*p+c];
}
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int m = 5;
    int n = 32;
    int p = 32;
    double *d_y = NULL;
    double *d_a = NULL;
    double *d_b = NULL;
    cudaMalloc(&d_y, m * p * sizeof(double));
    cudaMalloc(&d_a, m * n * sizeof(double));
    cudaMalloc(&d_b, n * p * sizeof(double));
    
    // Warmup
    cudaFree(0);
    gaxpymm<<<gridBlock, threadBlock, m * n * p * sizeof(double)>>>(d_y, d_a, d_b, m, n, p);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        gaxpymm<<<gridBlock, threadBlock, m * n * p * sizeof(double)>>>(d_y, d_a, d_b, m, n, p);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        gaxpymm<<<gridBlock, threadBlock, m * n * p * sizeof(double)>>>(d_y, d_a, d_b, m, n, p);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_y);
    cudaFree(d_a);
    cudaFree(d_b);
    
    return 0;
}
