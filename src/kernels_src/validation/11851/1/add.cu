#include "includes.h"
__global__ void add(float *c, float* a, float *b, int values){
int blockD = blockDim.x;
int blockX = blockIdx.x;
int threadX = threadIdx.x;

int i = blockX * blockD + threadX;
if(i < values)
c[i] = a[i] + b[i];
//printf("Hello Im thread %d in block %d of %d threads\n", threadX, blockX, blockD);
}