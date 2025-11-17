#include "includes.h"
__global__ void gaxpy(double *y, double *a, double *x, int m, int n){
int bid = blockIdx.x;
int tid = threadIdx.x;
extern __shared__ double dots_s[];
if(bid<m)
if(tid<n){

dots_s[bid*n+tid] = a[bid*n+tid] * *(x+tid);
__syncthreads();
if(tid == 0){
for(int i=1;i<n;i++){
dots_s[bid*n] +=dots_s[bid*n+i];
//			printf("y=%d, dots_s=%d, bid=%d, tid=%d, i=%d, n=%d\n",dots_s[bid*n], dots_s[bid*n+i],bid,tid,i,n);
}
*(y+bid)=dots_s[bid*n];
//		printf("y[%d]=%d, bid=%d, tid=%d\n",bid,y[bid],bid,tid);
}
}
}