#include "includes.h"
__global__ void add(int* in, int d, int n){

int gid = threadIdx.x + blockIdx.x * blockDim.x;
if(gid >= n) return ;

int pre = (d==0) ? 1 : (2<<(d-1));

if(gid >= pre) {
in[gid] += in[gid-pre];
}
}