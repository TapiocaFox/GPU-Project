#include "includes.h"


__global__ void gaxpymm(double *y, double *a, double *b, int m, int n, int p){
int bid = blockIdx.x;
int tid = threadIdx.x;
extern __shared__ double dots_s[];
if(bid<m)
if(tid<n){
for(int c=0;c<p;c++)
dots_s[bid*n*p+tid*p+c] = a[bid*n+tid] * *(b+(tid*p+c));
__syncthreads();
if(tid == 0){
for(int c=0;c<p;c++)
for(int i=1;i<n;i++){
dots_s[bid*n*p+c] +=dots_s[bid*n*p+i*p+c];
//			printf("y=%d, dots_s=%d, bid=%d, tid=%d, i=%d, n=%d\n",dots_s[bid*n], dots_s[bid*n+i],bid,tid,i,n);
}
for(int c=0;c<p;c++)
*(y+(bid*p+c))=dots_s[bid*n*p+c];
//		printf("y[%d]=%d, bid=%d, tid=%d\n",bid,y[bid],bid,tid);
}
}
}