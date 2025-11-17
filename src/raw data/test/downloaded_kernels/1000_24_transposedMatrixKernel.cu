#include "includes.h"
__global__ void transposedMatrixKernel(int* d_a, int* d_b) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;

while (i < N) {
j = threadIdx.y + blockDim.y * blockIdx.y;
while (j < N) {
d_b[i * N + j] = d_a[j * N + i];
j += blockDim.y * gridDim.y;
}
i += blockDim.x * gridDim.x;
}
}