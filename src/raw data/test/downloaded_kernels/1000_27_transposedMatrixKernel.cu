#include "includes.h"
__global__ void transposedMatrixKernel(int* d_a, int* d_b) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;

d_b[i * N + j] = d_a[j * N + i];
}