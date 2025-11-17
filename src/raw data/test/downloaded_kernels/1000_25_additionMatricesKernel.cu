#include "includes.h"
__global__ void additionMatricesKernel(int* d_a, int* d_b, int* d_c) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
int j = threadIdx.y + blockIdx.y * blockDim.y;

while (i < N) {
j = threadIdx.y + blockIdx.y * blockDim.y;
while (j < N) {
d_c[i * N + j] = d_a[i * N + j] + d_b[i * N + j];
j += blockDim.y * gridDim.y;
}
i += blockDim.x * gridDim.x;
}
}