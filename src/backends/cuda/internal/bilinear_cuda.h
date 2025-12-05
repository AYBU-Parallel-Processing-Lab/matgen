#ifndef MATGEN_BACKENDS_CUDA_INTERNAL_BILINEAR_CUDA_H
#define MATGEN_BACKENDS_CUDA_INTERNAL_BILINEAR_CUDA_H

/**
 * @file bilinear_cuda.h
 * @brief Internal header for CUDA bilinear interpolation scaling
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/algorithms/scaling.h> instead.
 */

#ifdef MATGEN_HAS_CUDA

#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scale a CSR matrix using bilinear interpolation (CUDA)
 *
 * GPU-accelerated implementation that parallelizes entry generation.
 * Each CUDA thread processes one source entry and generates its weighted
 * contributions to the destination matrix.
 *
 * Algorithm:
 *   1. Launch kernel with one thread per source entry
 *   2. Each thread computes bilinear weights for neighborhood
 *   3. Atomically accumulate contributions to global COO buffer
 *   4. Sort and sum duplicates on GPU
 *   5. Convert to CSR format
 *
 * @param source Source CSR matrix
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param result Output CSR matrix
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_scale_bilinear_cuda(const matgen_csr_matrix_t* source,
                                          matgen_index_t new_rows,
                                          matgen_index_t new_cols,
                                          matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_CUDA

#endif  // MATGEN_BACKENDS_CUDA_INTERNAL_BILINEAR_CUDA_H
