#ifndef MATGEN_BACKENDS_CUDA_INTERNAL_NEAREST_NEIGHBOR_CUDA_H
#define MATGEN_BACKENDS_CUDA_INTERNAL_NEAREST_NEIGHBOR_CUDA_H

/**
 * @file nearest_neighbor_cuda.h
 * @brief Internal header for CUDA nearest neighbor scaling
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
 * @brief Scale a CSR matrix using nearest neighbor method (CUDA)
 *
 * GPU-accelerated implementation that parallelizes block expansion.
 * Each CUDA thread processes one source entry and generates its block
 * of destination entries.
 *
 * Algorithm:
 *   1. Launch kernel with one thread per source entry
 *   2. Each thread computes destination cell range (block)
 *   3. Atomically accumulate entries to global COO buffer
 *   4. Sort and handle duplicates on GPU
 *   5. Convert to CSR format
 *
 * @param source Source CSR matrix
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param collision_policy How to handle duplicate entries
 * @param result Output CSR matrix
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_scale_nearest_neighbor_cuda(
    const matgen_csr_matrix_t* source, matgen_index_t new_rows,
    matgen_index_t new_cols, matgen_collision_policy_t collision_policy,
    matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_CUDA

#endif  // MATGEN_BACKENDS_CUDA_INTERNAL_NEAREST_NEIGHBOR_CUDA_H
