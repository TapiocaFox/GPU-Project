#include "includes.h"
__global__ void softmax_gradient_kernel( const int dim, const float* Y, const float* dY, float* dX) {
Y += blockIdx.x * dim;
dY += blockIdx.x * dim;
dX += blockIdx.x * dim;
const int idx = threadIdx.x;
__shared__ float reduction_buffer[SOFTMAX_NUM_THREADS];
float tmp;

// A two-level reduction to compute the inner products.
tmp = 0;
for (int i = idx; i < dim; i += blockDim.x) {
tmp += dY[i] * Y[i];
}
reduction_buffer[idx] = tmp;
__syncthreads();
if (idx == 0) {
tmp = reduction_buffer[0];
for (int i = 1; i < blockDim.x; ++i)
tmp += reduction_buffer[i];
reduction_buffer[0] = tmp;
}
__syncthreads();
// Compute gradient.
tmp = reduction_buffer[0];
for (int i = idx; i < dim; i += blockDim.x) {
dX[i] = Y[i] * (dY[i] - tmp);
}
}