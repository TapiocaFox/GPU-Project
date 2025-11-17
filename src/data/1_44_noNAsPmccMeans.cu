#include "includes.h"
#define FALSE 0
#define TRUE !FALSE

#define NUMTHREADS 16
#define THREADWORK 32








__global__ void noNAsPmccMeans(int nRows, int nCols, float * a, float * means)
{
int
col = blockDim.x * blockIdx.x + threadIdx.x,
inOffset = col * nRows,
outOffset = threadIdx.x * blockDim.y,
j = outOffset + threadIdx.y;
float sum = 0.f;

if(col >= nCols) return;

__shared__ float threadSums[NUMTHREADS*NUMTHREADS];

for(int i = threadIdx.y; i < nRows; i += blockDim.y)
sum += a[inOffset + i];

threadSums[j] = sum;
__syncthreads();

for(int i = blockDim.y >> 1; i > 0; i >>= 1) {
if(threadIdx.y < i) {
threadSums[outOffset+threadIdx.y]
+= threadSums[outOffset+threadIdx.y + i];
}
__syncthreads();
}
if(threadIdx.y == 0)
means[col] = threadSums[outOffset] / (float)nRows;
}