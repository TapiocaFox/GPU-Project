#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

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

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int chunk_size = 100;
    int num_chunks = 5 * 32 * 5 * 32;
    int text_size = num_chunks * chunk_size;
    int pat_len = 10;
    char *d_text = NULL;
    char *d_pattern = NULL;
    int *d_shift_table = NULL;
    unsigned int *d_num_matches = NULL;
    cudaMalloc(&d_text, text_size * sizeof(char));
    cudaMalloc(&d_pattern, pat_len * sizeof(char));
    cudaMalloc(&d_shift_table, 126 * sizeof(int));
    cudaMalloc(&d_num_matches, sizeof(unsigned int));
    
    // Warmup
    cudaFree(0);
    horspool_match<<<gridBlock, threadBlock>>>(d_text, d_pattern, d_shift_table, d_num_matches, chunk_size, num_chunks, text_size, pat_len);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        horspool_match<<<gridBlock, threadBlock>>>(d_text, d_pattern, d_shift_table, d_num_matches, chunk_size, num_chunks, text_size, pat_len);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        horspool_match<<<gridBlock, threadBlock>>>(d_text, d_pattern, d_shift_table, d_num_matches, chunk_size, num_chunks, text_size, pat_len);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_text);
    cudaFree(d_pattern);
    cudaFree(d_shift_table);
    cudaFree(d_num_matches);
    
    return 0;
}
