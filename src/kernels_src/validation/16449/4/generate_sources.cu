#include "includes.h"
__global__ void generate_sources(curandState *state, int n, uint32_t *verts) {
int first = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;

curandState local_state = state[first];
for (int id = first ; id < n ; id += stride) {
verts[id] = curand(&local_state);
}

state[first] = local_state;
}