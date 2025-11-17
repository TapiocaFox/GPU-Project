#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

extern "C" {
}
#define ROTATE_DOWN(val,MAX) ((val-1==-1)?MAX-1:val-1)
#define ROTATE_UP(val,MAX) ((val+1)%MAX)
/**
* GPU Device kernel for the for 2D stencil
* First attempt during hackaton
* M = Rows, N = Cols INCLUDING HALOS
* In this version now we replace the size of the shared memory to be just 3 rows (actually 1+HALO*2) rows
*/

__global__ void gpu_stencil2D_4pt_hack5_cp_rows(double * dst, double * shared_cols, double *shared_rows,int tile_y,int M, int N){


int base_global_row = (tile_y  * blockIdx.y );
int base_global_col = blockDim.x*blockIdx.x;
int base_global_idx = N*base_global_row + base_global_col ;
int nextRow = base_global_row+1;
bool legalNextRow = (nextRow<M)?1:0;
int t = threadIdx.x;
bool legalCurCol = (base_global_col + t)<N;
int idx = (base_global_row/tile_y)*2*N + t+base_global_col;
int idx_nextrow = idx + N;
if(legalCurCol){
shared_rows[idx] = dst[base_global_idx + t];
}
if(legalNextRow&&legalCurCol){
shared_rows[idx_nextrow] = dst[base_global_idx + N+t];
}
__syncthreads();


}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int tile_y = 32;
    int M = 160;
    int N = 160;
    double *d_dst = NULL;
    double *d_shared_cols = NULL;
    double *d_shared_rows = NULL;
    cudaMalloc(&d_dst, M * N * sizeof(double));
    cudaMalloc(&d_shared_cols, M * N * sizeof(double));
    cudaMalloc(&d_shared_rows, (M / tile_y) * 2 * N * sizeof(double));
    
    // Warmup
    cudaFree(0);
    gpu_stencil2D_4pt_hack5_cp_rows<<<gridBlock, threadBlock>>>(d_dst, d_shared_cols, d_shared_rows, tile_y, M, N);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        gpu_stencil2D_4pt_hack5_cp_rows<<<gridBlock, threadBlock>>>(d_dst, d_shared_cols, d_shared_rows, tile_y, M, N);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        gpu_stencil2D_4pt_hack5_cp_rows<<<gridBlock, threadBlock>>>(d_dst, d_shared_cols, d_shared_rows, tile_y, M, N);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_dst);
    cudaFree(d_shared_cols);
    cudaFree(d_shared_rows);
    
    return 0;
}
