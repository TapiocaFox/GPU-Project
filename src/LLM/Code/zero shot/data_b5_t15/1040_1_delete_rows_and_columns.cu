#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

__global__ void delete_rows_and_columns(int *dl_matrix, int *deleted_rows, int *deleted_cols, const int search_depth, const int selected_row_id, const int total_dl_matrix_row_num, const int total_dl_matrix_col_num) {
for (int i = threadIdx.x; i < total_dl_matrix_col_num; i = i + blockDim.x) {
if (dl_matrix[selected_row_id * total_dl_matrix_col_num + i] == 1 &&
deleted_cols[i] == 0) {
deleted_cols[i] = search_depth;
for (int j = 0; j < total_dl_matrix_row_num; j++) {
if (dl_matrix[j * total_dl_matrix_col_num + i] == 1 &&
deleted_rows[j] == 0) {
atomicExch(deleted_rows + j, search_depth);
}
}
}
}
}

int main(int argc, char **argv) {
    cudaSetDevice(0);
    
    // Grid: (5, 5), Block: (32, 32)
    dim3 gridBlock(5, 5);
    dim3 threadBlock(32, 32);
    
    // Allocate device memory
    int total_dl_matrix_row_num = 100;
    int total_dl_matrix_col_num = 100;
    int search_depth = 1;
    int selected_row_id = 0;
    int *d_dl_matrix = NULL;
    int *d_deleted_rows = NULL;
    int *d_deleted_cols = NULL;
    cudaMalloc(&d_dl_matrix, total_dl_matrix_row_num * total_dl_matrix_col_num * sizeof(int));
    cudaMalloc(&d_deleted_rows, total_dl_matrix_row_num * sizeof(int));
    cudaMalloc(&d_deleted_cols, total_dl_matrix_col_num * sizeof(int));
    
    // Warmup
    cudaFree(0);
    delete_rows_and_columns<<<gridBlock, threadBlock>>>(d_dl_matrix, d_deleted_rows, d_deleted_cols, search_depth, selected_row_id, total_dl_matrix_row_num, total_dl_matrix_col_num);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 10; ++i) {
        delete_rows_and_columns<<<gridBlock, threadBlock>>>(d_dl_matrix, d_deleted_rows, d_deleted_cols, search_depth, selected_row_id, total_dl_matrix_row_num, total_dl_matrix_col_num);
    }
    cudaDeviceSynchronize();
    
    // Measure execution time
    auto start = steady_clock::now();
    for (int i = 0; i < 1000; ++i) {
        delete_rows_and_columns<<<gridBlock, threadBlock>>>(d_dl_matrix, d_deleted_rows, d_deleted_cols, search_depth, selected_row_id, total_dl_matrix_row_num, total_dl_matrix_col_num);
    }
    cudaDeviceSynchronize();
    auto end = steady_clock::now();
    
    auto usecs = duration_cast<duration<float, microseconds::period>>(end - start);
    float avg_time_us = usecs.count() / 1000.0f;
    float avg_time_ms = avg_time_us / 1000.0f;
    
    cout << "[" << avg_time_us << "," << avg_time_ms << "]" << endl;
    
    cudaFree(d_dl_matrix);
    cudaFree(d_deleted_rows);
    cudaFree(d_deleted_cols);
    
    return 0;
}
