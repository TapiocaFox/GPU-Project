#include "includes.h"
__global__ void Matriz_GPU_Mult(int *a, int *b, int *c) {
int k, sum = 0;
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
if (i < N && j < N) {
for (k = 0; k < N; k++) {
sum += a[j * N + k] * b[k * N + i];
}
c[j * N + i] = sum;
}
}