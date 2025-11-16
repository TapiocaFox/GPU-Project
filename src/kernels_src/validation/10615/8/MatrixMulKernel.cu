#include "includes.h"
__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P,int width){
__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

int bx = blockIdx.x; int by = blockIdx.y;
int tx = threadIdx.x; int ty = threadIdx.y;

int row = by * TILE_WIDTH + ty;
int col = bx * TILE_WIDTH + tx;

float Pvalue = 0;
//printf("%f\n", width/TILE_WIDTH );
for (int i = 0; i < width/TILE_WIDTH; ++i){
//printf("%d\n", i );

Mds[ty][tx] = d_M[row*width + i*TILE_WIDTH + tx];
Nds[ty][tx] =  d_N[(i*TILE_WIDTH + ty)*width + col];
__syncthreads();

for (int j = 0; j < TILE_WIDTH; ++j){
Pvalue += Mds[ty][j] * Nds[j][tx];
}
__syncthreads();
}
d_P[row*width + col] = Pvalue;
}