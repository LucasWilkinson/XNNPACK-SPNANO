//
// Created by lwilkinson on 10/9/22.
//

#ifndef XNNPACK_SPNANO_WRAPPER_H
#define XNNPACK_SPNANO_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif
typedef void* spnano_coo_t;
typedef void* spnano_matmul_t;
typedef void* spnano_executor_t;

spnano_coo_t      allocate_coo_matrix_f32(int m, int n);
void              deallocate_coo_matrix_f32(spnano_coo_t m);
void              coo_matrix_add_nnz_f32(spnano_coo_t m, int i, int j, float val);

spnano_matmul_t   allocate_matmul_f32(spnano_coo_t m, int num_threads, int b_cols);

spnano_executor_t get_executor_f32(spnano_matmul_t m);
int               get_num_parallel_tiles_f32(spnano_executor_t e);

void              begin_threaded_f32(spnano_executor_t e, float* output, const float* input, const float* bias, float min, float max);
void              spnano_run_thread(spnano_executor_t e, int p_tile, int thread_id);

#ifdef __cplusplus
};
#endif

#endif  // XNNPACK_SPNANO_WRAPPER_H
