#include "includes.h"

static unsigned int GRID_SIZE_N;
static unsigned int GRID_SIZE_4N;
static unsigned int MAX_STATE_VALUE;

__global__ static void cudaIIGammaKernel(double *extEV, double *x1, double *x2, double *x3, double *left, double *right) {
__shared__ volatile double al[64], ar[64], v[64], x1px2[16];
const int tid = (threadIdx.z * 16) + (threadIdx.y * 4) + threadIdx.x;
const int offset = 16 * blockIdx.x + 4 * threadIdx.z;
al[tid] = x1[offset + threadIdx.x] * left[tid];
ar[tid] = x2[offset + threadIdx.x] * right[tid];
__syncthreads();
if (threadIdx.x <= 1) {
al[tid] += al[tid + 2];
ar[tid] += ar[tid + 2];
}
__syncthreads();
if (threadIdx.x == 0) {
al[tid] += al[tid + 1];
ar[tid] += ar[tid + 1];
x1px2[(threadIdx.z * 4) + threadIdx.y] = al[tid] * ar[tid];
}
__syncthreads();
v[tid] = x1px2[threadIdx.y + (threadIdx.z * 4)] *
extEV[threadIdx.y * 4 + threadIdx.x];
__syncthreads();
if (threadIdx.y <= 1) {
v[tid] += v[tid + 8];
}
__syncthreads();
if (threadIdx.y == 0) {
v[tid] += v[tid + 4];
x3[offset + threadIdx.x] = v[tid];
}
}