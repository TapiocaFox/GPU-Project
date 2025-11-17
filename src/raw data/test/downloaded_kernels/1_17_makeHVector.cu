#include "includes.h"
#define NTHREADS 512





// Updates the column norms by subtracting the Hadamard-square of the
// Householder vector.
//
// N.B.:  Overflow incurred in computing the square should already have
// been detected in the original norm construction.

__global__ void makeHVector(int rows, float * input, float * output)
{
int
i, j;
float
elt, sum;
__shared__ float
beta, sums[NTHREADS];

if(threadIdx.x >= rows)
return;

sum = 0.f;
for(i = threadIdx.x ; i < rows; i += NTHREADS) {
if((threadIdx.x == 0) && (i == 0))
continue;
elt = input[i];
output[i] = elt;
sum += elt * elt;
}
sums[threadIdx.x] = sum;
__syncthreads();

for(i = blockDim.x >> 1; i > 0 ; i >>= 1) {
j = i+threadIdx.x;
if((threadIdx.x < i) && (j < rows))
sums[threadIdx.x] += sums[j];
__syncthreads();
}

if(threadIdx.x == 0) {
elt = input[0];
float norm = sqrtf(elt * elt + sums[0]);

if(elt > 0)
elt += norm;
else
elt -= norm;

output[0] = elt;

norm = elt * elt + sums[0];
beta = sqrtf(2.f / norm);
}
__syncthreads();

for(i = threadIdx.x; i < rows; i += NTHREADS)
output[i] *= beta;
}