#include "includes.h"
__global__ void post_scan(float* in, float* add, int len) {
unsigned int t = threadIdx.x;
unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

if (blockIdx.x) {
if (start + t < len) in[start + t] += add[blockIdx.x - 1];
if (start + BLOCK_SIZE + t < len) in[start + BLOCK_SIZE + t] += add[blockIdx.x - 1];
}
}