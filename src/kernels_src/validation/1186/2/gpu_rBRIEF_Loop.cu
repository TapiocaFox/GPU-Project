#include "includes.h"
__global__ void gpu_rBRIEF_Loop(int N, float4* patches, int4* pattern)
{
// // 1) Shared memory management
// extern __shared__ float4 shared[];
// int4* sharedPattern = (int4*) shared;
// float4* sharedPatches0 = (float4*) &shared[256];
// float4* sharedPatches1 = (float4*) &shared[N*blockDim.x*24 + 256];
// float4* thisPatches;
// float4* nextPatches;
// float4* tmp;
//
// // 2) Load pattern into shared memory (static part of kernel)
// int id = threadIdx.x;
// int stride = blockDim.x;
// for (int i = id; i < 256; i+= stride) {
//   sharedPattern[i] = pattern[i];
// }
//
// // 3) Preload patches 0 into shared memory
// int start = blockIdx.x * (N*24) + id;
// int end   = blockIdx.x * (N*24) + N*24;
// for (int i = start; i < end; i+=stride) {
//   sharedPatches0[i] = patches[i];
// }
// thisPatches = sharedPatches0;

// Kernel Loop begin:
//for (int i = blockIdx.x; i < (P - 1) * N * blockDim.x*24; i+= )

};