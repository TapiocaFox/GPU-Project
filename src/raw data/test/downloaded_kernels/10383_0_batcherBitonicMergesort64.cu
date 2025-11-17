#include "includes.h"

// http://en.wikipedia.org/wiki/Bitonic_sort

__global__ void batcherBitonicMergesort64(float * d_out, const float * d_in)
{
// you are guaranteed this is called with <<<1, 64, 64*4>>>
extern __shared__ float sdata[];
int tid  = threadIdx.x;
sdata[tid] = d_in[tid];
__syncthreads();

for (int stage = 0; stage <= 5; stage++)
{
for (int substage = stage; substage >= 0; substage--)
{
// TODO
}
}

d_out[tid] = sdata[tid];
}