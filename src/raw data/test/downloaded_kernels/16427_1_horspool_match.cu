#include "includes.h"
// #pragma once



using namespace std;

#define NUM_THREADS_PER_BLOCK 512

int* create_shifts (char* pattern);

int linear_horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size,
int num_chunks, int text_size, int pat_len, int myId);


/*
*  Driver function
*  argv[0] is target pattern string
*  argv[1] is text path
*/
__global__ void horspool_match (char* text, char* pattern, int* shift_table, unsigned int* num_matches, int chunk_size, int num_chunks, int text_size, int pat_len) {

const int TABLE_SIZ = 126;

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

if (text[i] >= TABLE_SIZ || text[i] < 0) {
// move to next char if unknown char (Unicode, etc.)
++i;
} else {
while(k <= pat_len - 1 && pattern[pat_len - 1 - k] == text[i - k]) {
// increment matched character count
k++;
}
if(k == pat_len) {
// increment pattern count, text index
++count;
++i;

} else {
// add on shift if known char
i = i + shift_table[text[i]];
}
}
}

atomicAdd(num_matches, count);
}