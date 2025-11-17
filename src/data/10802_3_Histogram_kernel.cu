#include "includes.h"
__global__ void Histogram_kernel(int size, int bins, int cpu_bins, unsigned int *data, unsigned int *histo) {

extern __shared__ unsigned int l_mem[];
unsigned int* l_histo = l_mem;

// Block and thread index
const int bx = blockIdx.x;
const int tx = threadIdx.x;
const int bD = blockDim.x;
const int gD = gridDim.x;

// Output partition
int bins_per_wg   = (bins - cpu_bins) / gD;
int my_bins_start = bx * bins_per_wg + cpu_bins;
int my_bins_end   = my_bins_start + bins_per_wg;

// Constants for read access
const int begin = tx;
const int end   = size;
const int step  = bD;

// Sub-histograms initialization
for(int pos = tx; pos < bins_per_wg; pos += bD) {
l_histo[pos] = 0;
}

__syncthreads(); // Intra-block synchronization

// Main loop
for(int i = begin; i < end; i += step) {
// Global memory read
unsigned int d = ((data[i] * bins) >> 12);

if(d >= my_bins_start && d < my_bins_end) {
// Atomic vote in shared memory
atomicAdd(&l_histo[d - my_bins_start], 1);
}
}

__syncthreads(); // Intra-block synchronization

// Merge per-block histograms and write to global memory
for(int pos = tx; pos < bins_per_wg; pos += bD) {
unsigned int sum = 0;
for(int base = 0; base < (bins_per_wg); base += (bins_per_wg))
sum += l_histo[base + pos];
// Atomic addition in global memory
histo[pos + my_bins_start] += sum;
}
}