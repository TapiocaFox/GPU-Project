#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <iostream>

//#include "cub/cub/cub.cuh"

namespace gpu_mg {

__global__ void init_vertex_group(int *row_group, bool *dl_matrix,
                                  int *vertex_num, int *t_cn, int *t_rn,
                                  int *offset_row, int *offset_matrix,
                                  int graph_count);

__global__ void
mc_solver(bool *dl_matrix, bool *transpose_dl_matrix, int *next_col,
          int *next_row, int *results, int *deleted_cols, int *deleted_rows,
          int *col_group, int *row_group, int *conflict_count, int *vertex_num,
          int *total_dl_matrix_row_num, int *total_dl_matrix_col_num,
          int *offset_col, int *offset_row, int *offset_matrix,
          int *search_depth, int *selected_row_id, int *current_conflict_count,
          int *conflict_node_id, int *conflict_col_id,
          int *existance_of_candidate_rows, int *conflict_edge, int *max,
          const int graph_count, const int hard_conflict_threshold,
          const int graph_per_block);
// void mc_solver(int* dl_matrix, int* results, int* deleted_cols, int*
// deleted_rows, int* col_group,int* row_group, int* conflict_count,	const
// int vertex_num, const int total_dl_matrix_row_num, const int
// total_dl_matrix_col_num);

} // namespace gpu_mg
