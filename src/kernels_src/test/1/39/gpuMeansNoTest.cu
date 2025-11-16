#include "includes.h"
#define FALSE 0
#define TRUE !FALSE

#define NUMTHREADS 16
#define THREADWORK 32








__global__ void gpuMeansNoTest(const float * vectsA, size_t na, const float * vectsB, size_t nb, size_t dim, float * means, float * numPairs)
{
size_t
offset, stride,
bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x;
float a, b;

__shared__ float
threadSumsA[NUMTHREADS], threadSumsB[NUMTHREADS],
count[NUMTHREADS];

if((bx >= na) || (by >= nb))
return;

threadSumsA[tx] = 0.f;
threadSumsB[tx] = 0.f;
count[tx] = 0.f;

for(offset = tx; offset < dim; offset += NUMTHREADS) {
a = vectsA[bx * dim + offset];
b = vectsB[by * dim + offset];

threadSumsA[tx] += a;
threadSumsB[tx] += b;
count[tx] += 1.f;
}
__syncthreads();

for(stride = NUMTHREADS >> 1; stride > 0; stride >>= 1) {
if(tx < stride) {
threadSumsA[tx] += threadSumsA[tx + stride];
threadSumsB[tx] += threadSumsB[tx + stride];
count[tx] += count[tx+stride];
}
__syncthreads();
}
if(tx == 0) {
means[bx*nb*2+by*2] = threadSumsA[0] / count[0];
means[bx*nb*2+by*2+1] = threadSumsB[0] / count[0];
numPairs[bx*nb+by] = count[0];
}
}