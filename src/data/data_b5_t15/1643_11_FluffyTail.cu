#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

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

struct uint2 {
    unsigned int x, y;
};

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

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    const uint2 *d_source = NULL;
    uint2 *d_destination = NULL;
    const int *d_sourceIndexes = NULL;
    int *d_destinationIndexes = NULL;
    cudaMalloc(&d_source, 5 * 5 * DUCK_B_EDGES / 4 * sizeof(uint2));
    cudaMalloc(&d_destination, 5 * 5 * DUCK_B_EDGES / 4 * sizeof(uint2));
    cudaMalloc(&d_sourceIndexes, 5 * 5 * sizeof(int));
    cudaMalloc(&d_destinationIndexes, sizeof(int));
    
    // Warmup
    cudaFree(0);
    FluffyTail<<<gridBlock, threadBlock>>>(d_source, d_destination, d_sourceIndexes, d_destinationIndexes);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        FluffyTail<<<gridBlock, threadBlock>>>(d_source, d_destination, d_sourceIndexes, d_destinationIndexes);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        FluffyTail<<<gridBlock, threadBlock>>>(d_source, d_destination, d_sourceIndexes, d_destinationIndexes);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_source);
    cudaFree(d_destination);
    cudaFree(d_sourceIndexes);
    cudaFree(d_destinationIndexes);
    
    return 0;
}
