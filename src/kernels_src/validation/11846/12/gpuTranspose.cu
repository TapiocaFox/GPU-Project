#include "includes.h"
__global__ void gpuTranspose(float *a, float *b, int m, int n) {
uint i = blockDim.x * blockIdx.x + threadIdx.x;
uint j = blockDim.y * blockIdx.y + threadIdx.y;
if (i < m && j < n) {
b[j * m + i] = a[i * n + j];
}
}