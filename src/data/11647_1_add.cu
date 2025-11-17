#include "includes.h"
__global__ void add(int* in, int offset, int n){

int gid = threadIdx.x + blockIdx.x * blockDim.x;
if(gid >= n) return ;

extern __shared__ int temp[];

temp[threadIdx.x] = in[gid];

__syncthreads(); //can only control threads in a block.
if(threadIdx.x >= offset){
in[threadIdx.x] += temp[threadIdx.x-offset];
} else if(gid >= offset){
in[threadIdx.x] += in[gid-offset];
}
in[gid] = temp[threadIdx.x];
}