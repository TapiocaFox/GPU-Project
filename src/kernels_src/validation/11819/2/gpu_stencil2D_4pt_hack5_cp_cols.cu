#include "includes.h"
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

__global__ void gpu_stencil2D_4pt_hack5_cp_cols(double * dst, double * shared_cols, double *shared_rows,int tile_x,int tile_y, int M, int N){

#ifdef CUDA_DARTS_DEBUG
if((blockIdx.x==0)&&(blockIdx.y==0)&&(threadIdx.y==0)){
printf("copy cols begin!\n");
}
#endif

int base_global_row = tile_y  * blockIdx.y;
int base_global_col = tile_x  * blockIdx.x;
int base_global_idx = N*base_global_row + base_global_col ;
int nextCol = base_global_col+1;
bool legalNextCol = (nextCol<N);
int t = threadIdx.y;
int idx = 2*M*blockIdx.x + t + base_global_row;
int idx_nextCol = idx + M ;
bool legalCurRow = (base_global_row + t)<M;
if(legalCurRow){
shared_cols[idx] = dst[base_global_idx + t*N];
}
if(legalNextCol && legalCurRow){
shared_cols[idx_nextCol] = dst[base_global_idx + t*N+1];
}
__syncthreads();


#ifdef CUDA_CUDA_DEBUG
//	if(threadIdx.y==0){
//		printf("blockDimy = %d\n",blockDim.y);
//	}
if(blockIdx.x==1 && t<5){
printf("addr: %d ,%f,\n",idx_nextCol,shared_cols[idx_nextCol]);
}
#endif

#ifdef CUDA_DARTS_DEBUG
if((blockIdx.x==0)&&(blockIdx.y==0)&&(threadIdx.y==0)){
printf("copy cols finish!\n");
}
#endif
}