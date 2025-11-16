#include "includes.h"
__global__ void add(int* in, int* out, int n){

int gid = threadIdx.x + blockIdx.x * blockDim.x;
if(gid >= n) return ;

extern __shared__ int temp[];

temp[threadIdx.x] = in[gid];

for(int offset=1; offset<n; offset=(offset<<1)){
__syncthreads();
if(threadIdx.x >= offset){
temp[threadIdx.x] += temp[threadIdx.x-offset];
} else if(gid >= offset){
temp[threadIdx.x] += in[gid-offset];
}
__syncthreads(); //can only control threads in a block.
in[gid] = temp[threadIdx.x];
}
out[gid] = in[gid];
//out = in;
}