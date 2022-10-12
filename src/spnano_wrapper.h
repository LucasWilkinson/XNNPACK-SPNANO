//
// Created by lwilkinson on 10/9/22.
//

#ifndef XNNPACK_SPNANO_WRAPPER_H
#define XNNPACK_SPNANO_WRAPPER_H

#include <xnnpack/operator.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef void* spnano_coo_t;
typedef void* spnano_matmul_t;
typedef void* spnano_executor_t;

enum xnn_status xnn_delete_spnano_operator_f32(xnn_operator_t op);


spnano_coo_t      spnano_allocate_coo_matrix_f32(int rows, int cols);
void              spnano_coo_matrix_add_nnz_f32(spnano_coo_t m, int i, int j, float val);

spnano_matmul_t   spnano_allocate_matmul_f32(spnano_coo_t m, int num_threads, int b_cols);
void spnano_delete_matmul_f32(spnano_matmul_t m);

spnano_executor_t spnano_get_executor_f32(spnano_matmul_t m);
int               spnano_get_num_parallel_tiles_f32(spnano_executor_t e);

void              spnano_begin_threaded_f32(spnano_executor_t e, float* output, const float* input, const float* bias, float min, float max);
void              spnano_run_thread_f32(spnano_executor_t e, int p, int tid);

#ifdef __cplusplus
};
#endif

#endif  // XNNPACK_SPNANO_WRAPPER_H
