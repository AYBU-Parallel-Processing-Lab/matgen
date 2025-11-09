#ifndef MATGEN_BACKENDS_SEQ_INTERNAL_COO_SEQ_H
#define MATGEN_BACKENDS_SEQ_INTERNAL_COO_SEQ_H

/**
 * @file coo_seq.h
 * @brief Internal header for sequential COO matrix operations
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/core/matrix/coo.h> instead.
 */

#include "matgen/core/matrix/coo.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Sequential COO Operations
// =============================================================================

/**
 * @brief Create a new COO matrix (sequential)
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param nnz_hint Expected number of non-zeros (for pre-allocation)
 * @return Pointer to new matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_coo_create_seq(matgen_index_t rows,
                                           matgen_index_t cols,
                                           matgen_size_t nnz_hint);

/**
 * @brief Sort COO matrix entries by (row, col) order (sequential)
 *
 * Uses qsort internally. After sorting, is_sorted flag is set to true.
 *
 * @param matrix Matrix to sort
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_sort_seq(matgen_coo_matrix_t* matrix);

/**
 * @brief Sum duplicate entries in a sorted COO matrix (sequential)
 *
 * Assumes matrix is already sorted. Combines entries with identical (row, col)
 * by summing their values. Modifies the matrix in-place, reducing nnz.
 *
 * @param matrix Matrix to process (must be sorted)
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_sum_duplicates_seq(matgen_coo_matrix_t* matrix);

/**
 * @brief Merge duplicate entries using collision policy (sequential)
 *
 * @param matrix Matrix to process (must be sorted)
 * @param policy Collision policy (SUM, AVG, MAX, MIN, LAST)
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_merge_duplicates_seq(
    matgen_coo_matrix_t* matrix, matgen_collision_policy_t policy);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_BACKENDS_SEQ_INTERNAL_COO_SEQ_H
