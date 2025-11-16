#include "includes.h"

static unsigned int GRID_SIZE_N;
static unsigned int GRID_SIZE_4N;
static unsigned int MAX_STATE_VALUE;

__global__ static void cudaTIGammaKernel(double *extEV, double *x2, double *x3, unsigned char *tipX1, unsigned char *tipX2, double *r, double *uX1, double *uX2) {
__shared__ volatile double ump[64], x1px2[16], v[64];
const int tid = (threadIdx.z * 16) + (threadIdx.y * 4) + threadIdx.x;
const int offset = 16 * blockIdx.x + threadIdx.z * 4;
const int squareId = threadIdx.z * 4 + threadIdx.y;
uX1 += 16 * tipX1[blockIdx.x];
ump[tid] = x2[offset + threadIdx.x] * r[tid];
__syncthreads();
if (threadIdx.x <= 1) {
ump[tid] += ump[tid + 2];
}
__syncthreads();
if (threadIdx.x == 0) {
ump[tid] += ump[tid + 1];
uX2[4 * blockIdx.x + threadIdx.y] = ump[tid];
x1px2[squareId] = uX1[squareId] * ump[tid];
}
__syncthreads();
v[tid] = x1px2[squareId] * extEV[threadIdx.y * 4 + threadIdx.x];
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