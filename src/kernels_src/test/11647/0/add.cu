#include "includes.h"

#define N 10000000 //input data size: 10,000,000
#define BLOCKSIZE 1024

/* prefix sum */

using namespace std;

__global__ void add(double* in, double* out, int offset, int n){

int gid = threadIdx.x + blockIdx.x * blockDim.x;
if(gid >= n) return ;

out[gid] = in[gid];
if(gid >= offset)
out[gid] += in[gid-offset];
}