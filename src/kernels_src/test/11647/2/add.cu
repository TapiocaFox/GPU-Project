#include "includes.h"
__global__ void add(int* in, int* out, int offset, int n){

int gid = threadIdx.x + blockIdx.x * blockDim.x;
if(gid >= n) return ;

out[gid] = in[gid];
if(gid >= offset)
out[gid] += in[gid-offset];
}