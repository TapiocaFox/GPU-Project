#include "includes.h"
__global__ void simple_reduction(int *shared_var, int *input_values, int N, int iters)
{
__shared__ int local_mem[256];
int iter, i;
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int local_tid = threadIdx.x;
int local_dim = blockDim.x;
int minThreadInThisBlock = blockIdx.x * blockDim.x;
int maxThreadInThisBlock = minThreadInThisBlock + (blockDim.x - 1);

if (maxThreadInThisBlock >= N)
{
local_dim = N - minThreadInThisBlock;
}

for (iter = 0; iter < iters; iter++)
{
if (tid < N)
{
local_mem[local_tid] = input_values[tid];
}

// Required for correctness
// __syncthreads();

/*
* Perform the local reduction across values written to shared memory
* by threads in this thread block.
*/
if (local_tid == 0)
{
int sum = 0;

for (i = 0; i < local_dim; i++)
{
sum = sum + local_mem[i];
}

atomicAdd(shared_var, sum);
}

// Required for correctness
// __syncthreads();
}
}