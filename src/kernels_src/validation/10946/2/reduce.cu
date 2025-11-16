#include "includes.h"
__global__ static void reduce(int *g_idata, int l1, int l2) {
extern __shared__ unsigned int sdata[];
unsigned int tid = threadIdx.x;

if (tid < l1) {
sdata[tid] = g_idata[tid];
} else {
sdata[tid] = 0;
}
__syncthreads();

// Parallel Reduction (l2 must be power of 2)
for (unsigned int s = l2 / 2; s > 0; s >>= 1) {
if (tid < s)     {
sdata[tid] += sdata[tid + s];
}
__syncthreads();
}

if (tid == 0) {
g_idata[0] = sdata[0];
}
}