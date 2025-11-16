#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <iostream>

namespace gpu {
//#include "cub/cub/cub.cuh"

__global__ void delete_rows_and_columns(int *dl_matrix, int *deleted_rows,
                                        int *deleted_cols,
                                        const int search_depth,
                                        const int selected_row_id,
                                        const int total_dl_matrix_row_num,
                                        const int total_dl_matrix_col_num);

__global__ void init_vectors(int *vec, const int vec_length);

__global__ void get_largest_value(int *vec, 
    const int vec_length, int* max);

__global__ void find_index(int *vec, const int vec_length, int *value, int *index);

__global__ void init_vectors_reserved(int *vec, const int vec_length);

__global__ void check_existance_of_candidate_rows(
    int *deleted_rows, int *row_group, const int search_depth, int *token,
    int *selected_row_id, const int total_dl_matrix_row_num);

__global__ void get_vertex_row_group(int *row_group, int *dl_matrix,
                                     const int vertex_num,
                                     const int total_dl_matrix_row_num,
                                     const int total_dl_matrix_col_num);

__global__ void print_vec(int *vec, int vec_length);

//__global__ void select_row(int* deleted_rows, int* row_group, const int
//search_depth, const int total_dl_matrix_row_num, int* selected_row_id);

__global__ void recover_deleted_rows(int *deleted_rows, const int search_depth,
                                     const int total_dl_matrix_row_num);

__global__ void recover_deleted_cols(int *deleted_cols, const int search_depth,
                                     const int total_dl_matrix_col_num);

__global__ void recover_results(int *results, const int search_depth,
                                const int total_dl_matrix_row_num);

__global__ void get_conflict_node_id(int *deleted_rows, int *row_group,
                                     const int search_depth,
                                     int *conflict_node_id,
                                     const int total_dl_matrix_row_num);

__global__ void get_conflict_edge(int *dl_matrix, int *deleted_rows,
                                 int *deleted_cols, int *row_group,
                                 const int conflict_node_id,
                                 const int search_depth, int *conflict_edge,
                                 const int vertex_num,
                                 const int total_dl_matrix_row_num,
                                 const int total_dl_matrix_col_num);
                                 
__global__ void get_conflict_col_id(int *dl_matrix, int *deleted_cols, int* conflict_col_id,
                                    int *conflict_edge, int total_dl_matrix_col_num, 
                                    int vertex_num);


__global__ void remove_cols(int *deleted_cols, int *col_group,
                            const int conflict_col_id,
                            const int total_dl_matrix_col_num);

void mc_solver(int *dl_matrix, int *results, int *deleted_cols,
               int *deleted_rows, int *col_group, int *row_group,
               int *conflict_count, const int vertex_num,
               const int total_dl_matrix_row_num,
               const int total_dl_matrix_col_num);

} // namespace gpu
