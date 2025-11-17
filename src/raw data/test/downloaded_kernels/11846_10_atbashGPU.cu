#include "includes.h"
__global__ void atbashGPU(char const *in, char *out, int n) {
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < n) {
out[n - 1 - i] = in[i];
}
}