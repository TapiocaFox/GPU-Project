#include "includes.h"
extern "C" {





}
__global__ void vmul(const float *a, const float *b, float *c)
{
int i = blockIdx.x *blockDim.x + threadIdx.x;
c[i] = a[i] * b[i];
}