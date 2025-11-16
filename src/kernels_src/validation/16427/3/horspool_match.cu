#include "includes.h"
__global__ void horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size, int num_chunks, int text_size, int pat_len , unsigned int* d_output) {
extern __shared__ int s[];


int count = 0;
int myId = threadIdx.x + blockDim.x * blockIdx.x;
if(myId > num_chunks){ //if thread is an invalid thread
return;
}

int text_length = (chunk_size * myId) + chunk_size + pat_len - 1;

// don't need to check first pattern_length - 1 characters
int i = (myId*chunk_size) + pat_len - 1;
int k = 0;
while(i < text_length) {
// reset matched character count
k = 0;

if (i >= text_size) {
// break out if i tries to step past text length
break;
}

while(k <= pat_len - 1 && pattern[pat_len - 1 - k] == text[i - k]) {
// increment matched character count
k++;
}
if(k == pat_len) {
// increment pattern count, text index
++count;
++i;

} else {
i = i + shift_table[text[i]];
}
}

// atomicAdd(num_matches, count);
s[threadIdx.x] = count;
__syncthreads();

// Add count to total matches atomically
if (threadIdx.x == 0 ){
int sum = 0;
for(int idx =0; idx < NUM_THREADS_PER_BLOCK; idx++){
sum += s[idx];
}
d_output[blockIdx.x] = sum;
}
}