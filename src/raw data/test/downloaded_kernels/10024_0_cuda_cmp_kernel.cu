#include "includes.h"
__global__ void cuda_cmp_kernel(std::size_t n, int* aptr, int* bptr, int* rptr) {
int i = threadIdx.x+blockIdx.x*blockDim.x;
int cmp = i<n? aptr[i]<bptr[i]: 0;
if (__syncthreads_or(cmp)) *rptr=1;
}