#ifndef MATGEN_PARALLEL_SCALE_OMP_H
#define MATGEN_PARALLEL_SCALE_OMP_H

#include "matgen/core/coo_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

matgen_coo_matrix_t* matgen_matrix_scale_nearest_omp(
    const matgen_coo_matrix_t* input, size_t new_rows, size_t new_cols,
    int num_threads);

matgen_coo_matrix_t* matgen_matrix_scale_bilinear_omp(
    const matgen_coo_matrix_t* input, size_t new_rows, size_t new_cols,
    double sparsity_threshold, int num_threads);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_PARALLEL_SCALE_OMP_H
