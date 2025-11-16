#include "includes.h"
__global__ void add(int* in, int* out, int n){

int gid = threadIdx.x + blockIdx.x * blockDim.x;
if(gid >= n) return ;

extern __shared__ int temp[];

int pout = 0, pin = 1;
temp[threadIdx.x + pout * n] = (threadIdx.x>0) ? in[threadIdx.x-1] : 0;
__syncthreads();

for(int offset=1; offset<n; offset=(offset<<1)){
int t = pout;
pout = pin;
pin = t;

if(threadIdx.x >= offset){
temp[threadIdx.x + pout*n] += temp[threadIdx.x + pin*n - offset];
} else {
temp[threadIdx.x+pout*n] = temp[threadIdx.x+pin*n];
}
__syncthreads();
}
out[threadIdx.x] = temp[threadIdx.x+pout*n];
}