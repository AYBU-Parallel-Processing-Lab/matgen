#ifndef MATGEN_BACKENDS_CUDA_INTERNAL_CSR_CUDA_H
#define MATGEN_BACKENDS_CUDA_INTERNAL_CSR_CUDA_H

/**
 * @file csr_cuda.h
 * @brief Internal header for CUDA parallel CSR matrix operations
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/core/matrix/csr.h> instead.
 */

#ifdef MATGEN_HAS_CUDA

#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// CUDA CSR Operations
// =============================================================================

/**
 * @brief Create a new CSR matrix (CUDA)
 *
 * Allocates memory on host. Data will be transferred to device as needed.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param nnz Number of non-zero entries
 * @return Pointer to new matrix, or NULL on error
 */
matgen_csr_matrix_t* matgen_csr_create_cuda(matgen_index_t rows,
                                            matgen_index_t cols,
                                            matgen_size_t nnz);

/**
 * @brief Clone a CSR matrix (CUDA)
 *
 * Creates a deep copy of the matrix structure and data.
 *
 * @param src Source matrix to clone
 * @return Pointer to new matrix, or NULL on error
 */
matgen_csr_matrix_t* matgen_csr_clone_cuda(const matgen_csr_matrix_t* src);

/**
 * @brief Transpose a CSR matrix (CUDA)
 *
 * Computes CSC format (which is CSR of transpose) using GPU parallelism.
 * Uses atomic operations or segmented sort for efficient transposition.
 *
 * @param matrix Source matrix
 * @param result Pointer to store transposed matrix
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_csr_transpose_cuda(const matgen_csr_matrix_t* matrix,
                                         matgen_csr_matrix_t** result);

/**
 * @brief Get row from CSR matrix (CUDA)
 *
 * Extracts a single row. For GPU backend, this copies data from device if
 * needed.
 *
 * @param matrix Source matrix
 * @param row_idx Row index to extract
 * @param nnz_out Output: number of non-zeros in this row
 * @param col_indices_out Output: column indices (caller must free)
 * @param values_out Output: values (caller must free)
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_csr_get_row_cuda(const matgen_csr_matrix_t* matrix,
                                       matgen_index_t row_idx,
                                       matgen_size_t* nnz_out,
                                       matgen_index_t** col_indices_out,
                                       matgen_value_t** values_out);

/**
 * @brief Compute row statistics in parallel (CUDA)
 *
 * Computes min/max/avg non-zeros per row using parallel reduction.
 *
 * @param matrix Source matrix
 * @param min_nnz_out Output: minimum nnz per row
 * @param max_nnz_out Output: maximum nnz per row
 * @param avg_nnz_out Output: average nnz per row
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_csr_row_stats_cuda(const matgen_csr_matrix_t* matrix,
                                         matgen_size_t* min_nnz_out,
                                         matgen_size_t* max_nnz_out,
                                         double* avg_nnz_out);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_CUDA

#endif  // MATGEN_BACKENDS_CUDA_INTERNAL_CSR_CUDA_H
