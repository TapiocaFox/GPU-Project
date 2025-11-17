#include "includes.h"
__global__ void matrixMultiplyTiled(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
unsigned int tx = threadIdx.x;
unsigned int ty = threadIdx.y;
unsigned int col = blockIdx.x * TILE_WIDTH + tx;
unsigned int row = blockIdx.y * TILE_WIDTH + ty;
float acc = 0;

for (int t = 0; t < (numAColumns-1)/TILE_WIDTH + 1; ++t) {
unsigned int ATilePitch = t * TILE_WIDTH + tx;
unsigned int BTilePitch = t * TILE_WIDTH + ty;

if (row < numARows && ATilePitch < numAColumns)
ds_A[ty][tx] = A[row * numAColumns + ATilePitch];
else
ds_A[ty][tx] = 0;

if (col < numBColumns && BTilePitch < numBRows)
ds_B[ty][tx] = B[BTilePitch * numBColumns + col];
else
ds_B[ty][tx] = 0;

__syncthreads();
#pragma unroll
for (int k = 0; k < TILE_WIDTH; ++k) acc += ds_A[ty][k] * ds_B[k][tx];
__syncthreads();
}

if (row < numCRows && col < numCColumns) C[row * numCColumns + col] = acc;
}