#include "includes.h"

// Save wave state as binary file.
__global__ void fillSpaceTSteps(int N, int T, float c, float dt, float dd, float *waveSpace, float *waveSpaceTMin1, float *waveSpaceTMin2)
{

int i = blockIdx.y * blockDim.y + threadIdx.y;
int j = blockIdx.x * blockDim.x + threadIdx.x;

waveSpace[N * i + j] = 2 * waveSpaceTMin1[N * i + j] - waveSpaceTMin2[N * i + j] + (c * c) * (dt/dd * dt/dd) * (waveSpaceTMin1[N * (i + 1) + j] + waveSpaceTMin1[N * (i - 1) + j] + waveSpaceTMin1[N * i + (j - 1)] + waveSpaceTMin1[N * i + (j + 1)] - 4 * waveSpaceTMin1[N * i + j]);

__syncthreads();
}