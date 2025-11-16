#include "includes.h"
__device__ unsigned int shared_reduce(unsigned int p, volatile unsigned int * s) {
// Assumes values in 'p' are either 1 or 0
// Assumes s[0:31] are allocated
// Sums p across warp, returning the result. Suggest you put
// result in s[0] and return it
// You may change any value in s
// You should execute no more than 5 + operations (if you're doing
// 31, you're doing it wrong)
//
// TODO: Fill in the rest of this function

return s[0];
}
__global__ void reduce(unsigned int * d_out_shared, const unsigned int * d_in)
{
extern __shared__ unsigned int s[];
int t = threadIdx.x;
int p = d_in[t];
unsigned int sr = shared_reduce(p, s);
if (t == 0)
{
*d_out_shared = sr;
}
}