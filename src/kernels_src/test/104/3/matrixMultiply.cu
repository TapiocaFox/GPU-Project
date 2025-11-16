#include "includes.h"
__global__ void matrixMultiply(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if ((row < numCRows) && (col < numCColumns)) {
float value = 0;
#pragma unroll
for (int k = 0; k < numAColumns; ++k)
value += A[row * numAColumns + k] * B[k * numBColumns + col];
C[row * numCColumns + col] = value;
}
}