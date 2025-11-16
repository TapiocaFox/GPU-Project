#include "includes.h"
__global__ void relu_gpu_backward(float *ingrad, float *outgrad, float *indata, int64_t N) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < N)
ingrad[tid] = indata[tid] > 0 ? 1 * outgrad[tid] : 0;
}