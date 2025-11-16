#include "includes.h"

static unsigned int GRID_SIZE_N;
static unsigned int GRID_SIZE_4N;
static unsigned int MAX_STATE_VALUE;

__global__ static void cudaPreTIGammaKernel(double *tipVector, double *l, double *ump)
{
__shared__ volatile double sump[64];
const int tid = threadIdx.y * 4 + threadIdx.x;
sump[tid] = tipVector[4 * blockIdx.x + threadIdx.x] * l[tid];
__syncthreads();
if (threadIdx.x <= 1)
{
sump[tid] += sump[tid + 2];
}
__syncthreads();
if (threadIdx.x == 0)
{
sump[tid] += sump[tid + 1];
ump[blockIdx.x * 16 + threadIdx.y] = sump[tid];
}
}