#ifndef MATGEN_BACKENDS_OMP_INTERNAL_DCT_OMP_H
#define MATGEN_BACKENDS_OMP_INTERNAL_DCT_OMP_H

/**
 * @file dct_omp.h
 * @brief Internal header for OpenMP DCT interpolation scaling
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/algorithms/scaling.h> instead.
 */

#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scale a CSR matrix using DCT interpolation (OpenMP)
 *
 * Parallel OpenMP CPU implementation of DCT sparse matrix scaling.
 * Processes non-zero blocks in the DCT domain to preserve sparsity and structure.
 *
 * @param source Source CSR matrix
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param result Output CSR matrix
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_scale_dct_omp(
    const matgen_csr_matrix_t* source,
    matgen_index_t new_rows,
    matgen_index_t new_cols,
    matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_BACKENDS_OMP_INTERNAL_DCT_OMP_H
