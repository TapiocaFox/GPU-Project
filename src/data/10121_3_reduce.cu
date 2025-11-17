#include "includes.h"
__global__ void reduce(int *g_idata, int searchedNumber, int *ok) {

int i = blockIdx.x * blockDim.x + threadIdx.x;
//printf("%d ", i);

__syncthreads();
//printf("%d %d///", g_idata[i], searchedNumber);
if (g_idata[i] == searchedNumber) {
printf("Found %d on %d position %d", searchedNumber, i, *ok);
*ok = i;
}
}