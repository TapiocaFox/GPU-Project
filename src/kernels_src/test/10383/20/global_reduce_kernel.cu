#include "includes.h"
__global__ void global_reduce_kernel(float * d_out, float * d_in)
{
int myId = threadIdx.x + blockDim.x * blockIdx.x;
int tid  = threadIdx.x;

// do reduction in global mem
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
{
if (tid < s)
{
d_in[myId] += d_in[myId + s];
}
__syncthreads();        // make sure all adds at one stage are done!
}

// only thread 0 writes result for this block back to global mem
if (tid == 0)
{
d_out[blockIdx.x] = d_in[myId];
}
}