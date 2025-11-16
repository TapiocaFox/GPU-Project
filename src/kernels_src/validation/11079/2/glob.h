//
// Created by barboutara on 8/2/2017.
//

#ifndef THESIS_3SLOG_CUDA_H
#define THESIS_3SLOG_CUDA_H

#include <cuda_runtime.h>
#include <glob.h>

typedef unsigned char uint8;

uint8 *d_current;
uint8 *d_previous;
int *d_vectors_x;
int *d_vectors_y;

int *d_M_B, *d_N_B, *d_B, *d_N, *d_M;

int totalsize;
int cudaThreads;
int framesize;
int t_N_B;
int t_M_B;
int t_B;
int t_M;
int t_N;

__global__ void log_motion_estimation_cuda(uint8 *current, uint8 *previous, int *vectors_x, int *vectors_y,
                                           int *M_B, int *N_B, int *B, int *M, int *N);

__global__ void log_motion_estimation_cuda2(uint8 *current, uint8 *previous, int *vectors_x, int *vectors_y,
                                            int M_B, int N_B, int B, int M, int N);

void initKernelAndStartIt(uint8 *current, uint8 *previous, int *vectors_x, int *vectors_y);

void initValuesAndAllocateMemory(int M_B, int N_B, int B, int M, int N);

void freeMemory();

void *fixed_cudaMalloc(size_t len);

#endif //THESIS_3SLOG_CUDA_H
