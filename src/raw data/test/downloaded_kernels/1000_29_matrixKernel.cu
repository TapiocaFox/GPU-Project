#include "includes.h"
__global__ void matrixKernel(float* d_in, float* d_out) {
// Block index
int bx = blockIdx.x;
int by = blockIdx.y;

// Thread index (current coefficient)
int tx = threadIdx.x;
int ty = threadIdx.y;

float dividend =
d_in[(by * BLOCK_SIZE + 0) * STRIDE + (bx * BLOCK_SIZE + 0)];
float divisor =
d_in[(by * BLOCK_SIZE + ty) * STRIDE + (bx * BLOCK_SIZE + tx)];

d_out[(by * BLOCK_SIZE + ty) * STRIDE + (bx * BLOCK_SIZE + tx)] =
dividend / divisor;
}