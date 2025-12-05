#ifndef MATGEN_BACKENDS_CUDA_INTERNAL_COO_CUDA_H
#define MATGEN_BACKENDS_CUDA_INTERNAL_COO_CUDA_H

/**
 * @file coo_cuda.h
 * @brief Internal header for CUDA parallel COO matrix operations
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/core/matrix/coo.h> instead.
 */

#ifdef MATGEN_HAS_CUDA

#include "matgen/core/matrix/coo.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// CUDA COO Operations
// =============================================================================

/**
 * @brief Create a new COO matrix (CUDA)
 *
 * Allocates memory on host. Data will be transferred to device as needed.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param nnz_hint Expected number of non-zeros (for pre-allocation)
 * @return Pointer to new matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_coo_create_cuda(matgen_index_t rows,
                                            matgen_index_t cols,
                                            matgen_size_t nnz_hint);

/**
 * @brief Sort COO matrix entries by (row, col) order (CUDA)
 *
 * Uses GPU-accelerated sorting (thrust::sort or CUB radix sort).
 * After sorting, is_sorted flag is set to true.
 *
 * @param matrix Matrix to sort
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_sort_cuda(matgen_coo_matrix_t* matrix);

/**
 * @brief Sum duplicate entries in a sorted COO matrix (CUDA)
 *
 * Assumes matrix is already sorted. Uses parallel reduction to combine
 * entries with identical (row, col) by summing their values.
 *
 * @param matrix Matrix to process (must be sorted)
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_sum_duplicates_cuda(matgen_coo_matrix_t* matrix);

/**
 * @brief Merge duplicate entries using collision policy (CUDA)
 *
 * @param matrix Matrix to process (must be sorted)
 * @param policy Collision policy (SUM, AVG, MAX, MIN, LAST)
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_merge_duplicates_cuda(
    matgen_coo_matrix_t* matrix, matgen_collision_policy_t policy);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_CUDA

#endif  // MATGEN_BACKENDS_CUDA_INTERNAL_COO_CUDA_H
