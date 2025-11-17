#include "includes.h"

#define BLOCK_SIZE 1024

#ifndef THREADS
# define THREADS 1024
#endif

__global__ void total(float * input, float * output, unsigned int len) {
__shared__ float sum[2*BLOCK_SIZE];
unsigned int i = threadIdx.x;
unsigned int j = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

float localSum = (i < len) ? input[j] : 0;
if (j + blockDim.x < len) localSum += input[j + blockDim.x];

sum[i] = localSum;
__syncthreads();

for (unsigned int step = blockDim.x / 2; step >= 1; step >>= 1) {
if (i < step) sum[i] = localSum = localSum + sum[i + step];
__syncthreads();
}

if(i == 0) output[blockIdx.x] = sum[0];
}