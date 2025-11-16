#include "includes.h"
#define NTHREADS 512





// Updates the column norms by subtracting the Hadamard-square of the
// Householder vector.
//
// N.B.:  Overflow incurred in computing the square should already have
// been detected in the original norm construction.

__global__ void gpuFindMax(int n, float * data, int threadWorkLoad, int * maxIndex)
{
int
j, k,
start = threadWorkLoad * threadIdx.x,
end = start + threadWorkLoad;
__shared__ int maxIndicies[NTHREADS];

maxIndicies[threadIdx.x] = -1;

if(start >= n)
return;

int localMaxIndex = start;
for(int i = start+1; i < end; i++) {
if(i >= n)
break;
if(data[i] > data[localMaxIndex])
localMaxIndex = i;
}
maxIndicies[threadIdx.x] = localMaxIndex;
__syncthreads();

for(int i = blockDim.x >> 1; i > 0; i >>= 1) {
if(threadIdx.x < i) {
j = maxIndicies[threadIdx.x];
k = maxIndicies[i + threadIdx.x];
if((j != -1) && (k != -1) && (data[j] < data[k]))
maxIndicies[threadIdx.x] = k;
}
__syncthreads();
}
if(threadIdx.x == 0) {
*maxIndex = maxIndicies[0];
// debug printing
// printf("max index: %d\n", *maxIndex);
// printf("max norm: %f\n", data[*maxIndex]);
// end debug printing
}
}