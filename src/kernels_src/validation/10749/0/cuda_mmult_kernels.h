#define TILE_SIZE 32
#define BLOCK_WIDTH 16
#include<stdio.h>
__global__ void matrixMultKernel_global(float* Ad, float* Bd, float* Cd, int n);
__global__ void matrixMultKernel_tiled(float* Ad, float* Bd, float* Cd, int n);
__global__ void matrixMultKernel_overlap(float* Ad, float* Bd, float* Cd, int n);
