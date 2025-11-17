#include "includes.h"
// Cuckarood Cycle, a memory-hard proof-of-work by John Tromp and team Grin
// Copyright (c) 2018 Jiri Photon Vadura and John Tromp
// This GGM miner file is covered by the FAIR MINING license

//Includes for IntelliSense
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif





typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef u32 node_t;
typedef u64 nonce_t;


#define DUCK_SIZE_A 134LL
#define DUCK_SIZE_B 86LL

#define DUCK_A_EDGES (DUCK_SIZE_A * 1024LL)
#define DUCK_A_EDGES_64 (DUCK_A_EDGES * 64LL)

#define DUCK_B_EDGES (DUCK_SIZE_B * 1024LL)
#define DUCK_B_EDGES_64 (DUCK_B_EDGES * 64LL)

#define EDGE_BLOCK_SIZE (64)
#define EDGE_BLOCK_MASK (EDGE_BLOCK_SIZE - 1)

#define EDGEBITS 29
#define NEDGES2 ((node_t)1 << EDGEBITS)
#define NEDGES1 (NEDGES2/2)
#define NNODES1 NEDGES1
#define NNODES2 NEDGES2

#define EDGEMASK (NEDGES2 - 1)
#define NODE1MASK (NNODES1 - 1)

#define CTHREADS 1024
#define CTHREADS512 512
#define BKTMASK4K (4096-1)
#define BKTGRAN 64

#define EDGECNT 562036736
#define BUKETS 4096
#define BUKET_MASK (BUKETS-1)
#define BUKET_SIZE (EDGECNT/BUKETS)

#define XBITS 6
const u32 NX = 1 << XBITS;
const u32 NX2 = NX * NX;
const u32 XMASK = NX - 1;
const u32 YBITS = XBITS;
const u32 NY = 1 << YBITS;
const u32 YZBITS = EDGEBITS - XBITS;
const u32 ZBITS = YZBITS - YBITS;
const u32 NZ = 1 << ZBITS;
const u32 ZMASK = NZ - 1;

#define ROTL(x,b) ( ((x) << (b)) | ( (x) >> (64 - (b))) )
#define SIPROUND \
{ \
v0 += v1; v2 += v3; v1 = ROTL(v1,13); \
v3 = ROTL(v3,16); v1 ^= v0; v3 ^= v2; \
v0 = ROTL(v0,32); v2 += v1; v0 += v3; \
v1 = ROTL(v1,17);   v3 = ROTL(v3,25); \
v1 ^= v2; v3 ^= v0; v2 = ROTL(v2,32); \
}
__global__  void FluffyTail(const uint2 * source, uint2 * destination, const int * sourceIndexes, int * destinationIndexes)
{
const int lid = threadIdx.x;
const int group = blockIdx.x;

int myEdges = sourceIndexes[group];
__shared__ int destIdx;

if (lid == 0)
destIdx = atomicAdd(destinationIndexes, myEdges);

__syncthreads();

if (lid < myEdges)
{
destination[destIdx + lid] = source[group * DUCK_B_EDGES / 4 + lid];
}
}