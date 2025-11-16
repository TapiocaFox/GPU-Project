#include "includes.h"
#define FALSE 0
#define TRUE !FALSE

#define NUMTHREADS 16
#define THREADWORK 32








__global__ void gpuSignif(const float * gpuNumPairs, const float * gpuCorrelations, size_t n, float * gpuTScores)
{
size_t
i, start,
bx = blockIdx.x, tx = threadIdx.x;
float
radicand, cor, npairs;

start = bx * NUMTHREADS * THREADWORK + tx * THREADWORK;
for(i = 0; i < THREADWORK; i++) {
if(start+i >= n)
break;

npairs = gpuNumPairs[start+i];
cor = gpuCorrelations[start+i];
radicand = (npairs - 2.f) / (1.f - cor * cor);
gpuTScores[start+i] = cor * sqrtf(radicand);
}
}