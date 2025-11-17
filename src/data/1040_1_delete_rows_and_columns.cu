#include "includes.h"
__device__ void delete_rows_and_columns( bool *dl_matrix, bool *transpose_dl_matrix, const int *next_row, int *next_col, short *deleted_rows, short *deleted_cols, const int search_depth, const int selected_row_id, const int total_dl_matrix_row_num, const int total_dl_matrix_col_num) {
bool *selected_row = dl_matrix + selected_row_id * total_dl_matrix_col_num;
///*
for (int i = threadIdx.x; i < total_dl_matrix_col_num;
// // The below line will have negative effect of the col number is small
//  i += (next_col[selected_row_idx + i] + blockDim.x - 1) / blockDim.x
i += blockDim.x) {
if (deleted_cols[i] == 0 && selected_row[i] == 1) {
deleted_cols[i] = search_depth;
// atomicInc(&tmp_deleted_cols_count)
const bool *transpose_dl_matrix_ptr =
transpose_dl_matrix + i * total_dl_matrix_row_num;
int nr = next_row[i * total_dl_matrix_row_num];
for (int j = 0; j < total_dl_matrix_row_num;
j += nr, transpose_dl_matrix_ptr += nr) {
nr = next_row[i * total_dl_matrix_row_num + j];
if (deleted_rows[j] == 0 && *transpose_dl_matrix_ptr == 1
// dl_matrix[j * total_dl_matrix_col_num + i] == 1
) {
deleted_rows[j] = search_depth;
}
}
}
}
//*/
/*
int * tmp_row;
int * tmp_next_col;
for (int i = threadIdx.x; i < total_dl_matrix_row_num; i += blockDim.x){
tmp_row = dl_matrix + i * total_dl_matrix_col_num;
tmp_next_col = next_col + i * total_dl_matrix_col_num;
for (int j = 0; j < total_dl_matrix_col_num; j += tmp_next_col[j]){
if (tmp_row[j] + selected_row[j] == 2 && deleted_cols[j] !=-1){
deleted_rows[i] = deleted_rows[i]==0?search_depth:deleted_rows[i];
deleted_cols[j] = deleted_cols[j]==0?search_depth:deleted_cols[j];
}
}
}
*/
}
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