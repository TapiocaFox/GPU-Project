#include "includes.h"
extern "C"
__global__ void mul(double* A, double* B, double* C, int size) {
int i = blockIdx.x * blockDim.x + threadIdx.x;

if(i < size) {
// compute a column
for(int j=0; j < size; j++) {
double sum = 0.0;
for(int k=0; k < size; k++) {
sum += A[ (i*size)+k ] * B[ (k*size)+j ];
}
C[ (i*size)+j ] = sum;
}
// end of column computing
}
}