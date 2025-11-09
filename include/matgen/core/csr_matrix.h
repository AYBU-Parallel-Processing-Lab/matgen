#ifndef MATGEN_CORE_CSR_MATRIX_H
#define MATGEN_CORE_CSR_MATRIX_H

/**
 * @file csr_matrix.h
 * @brief Compressed Sparse Row (CSR) matrix format
 *
 * CSR format stores sparse matrices efficiently using three arrays:
 * - row_ptr: Marks where each row starts in col_indices/values
 * - col_indices: Column indices of non-zeros
 * - values: Values of non-zeros
 *
 * Benefits:
 * - Memory efficient (less overhead than COO)
 * - Fast row access and row operations
 */

#include <stdio.h>

#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CSR (Compressed Sparse Row) sparse matrix structure
 *
 * Storage format:
 * - row_ptr[i] points to start of row i in col_indices/values
 * - row_ptr[i+1] points to end of row i
 * - Number of non-zeros in row i = row_ptr[i+1] - row_ptr[i]
 */
typedef struct {
  matgen_index_t rows;  // Number of rows
  matgen_index_t cols;  // Number of columns
  matgen_size_t nnz;    // Number of non-zeros

  matgen_size_t* row_ptr;       // Row pointer array [rows + 1]
  matgen_index_t* col_indices;  // Column indices array [nnz]
  matgen_value_t* values;       // Values array [nnz]
} matgen_csr_matrix_t;

// =============================================================================
// Creation and Destruction
// =============================================================================

/**
 * @brief Create a new CSR matrix
 *
 * Allocates memory for row_ptr, col_indices, and values.
 * Arrays must be filled by the caller.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param nnz Number of non-zeros
 * @return Pointer to new matrix, or NULL on error
 */
matgen_csr_matrix_t* matgen_csr_create(matgen_index_t rows, matgen_index_t cols,
                                       matgen_size_t nnz);

/**
 * @brief Destroy a CSR matrix and free all resources
 *
 * @param matrix Matrix to destroy (can be NULL)
 */
void matgen_csr_destroy(matgen_csr_matrix_t* matrix);

// =============================================================================
// Matrix Access
// =============================================================================

/**
 * @brief Get value at (row, col)
 *
 * Uses binary search within the row for efficiency.
 *
 * @param matrix Matrix to query
 * @param row Row index (0-based)
 * @param col Column index (0-based)
 * @param[out] value Pointer to store the value (can be NULL to just check
 * existence)
 * @return MATGEN_SUCCESS if found, MATGEN_ERROR_INVALID_ARGUMENT if not found
 * or error
 */
matgen_error_t matgen_csr_get(const matgen_csr_matrix_t* matrix,
                              matgen_index_t row, matgen_index_t col,
                              matgen_value_t* value);

/**
 * @brief Check if entry exists at (row, col)
 *
 * @param matrix Matrix to query
 * @param row Row index (0-based)
 * @param col Column index (0-based)
 * @return true if entry exists, false otherwise
 */
bool matgen_csr_has_entry(const matgen_csr_matrix_t* matrix, matgen_index_t row,
                          matgen_index_t col);

/**
 * @brief Get the number of non-zeros in a specific row
 *
 * @param matrix Matrix to query
 * @param row Row index (0-based)
 * @return Number of non-zeros in row, or 0 on error
 */
matgen_size_t matgen_csr_row_nnz(const matgen_csr_matrix_t* matrix,
                                 matgen_index_t row);

/**
 * @brief Get pointer to row data (col_indices and values)
 *
 * Allows efficient iteration over a row's non-zeros.
 *
 * @param matrix Matrix to query
 * @param row Row index (0-based)
 * @param[out] row_start Index where row starts in col_indices/values
 * @param[out] row_end Index where row ends in col_indices/values
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_csr_get_row_range(const matgen_csr_matrix_t* matrix,
                                        matgen_index_t row,
                                        matgen_size_t* row_start,
                                        matgen_size_t* row_end);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Print matrix information to stream
 *
 * @param matrix Matrix to print info about
 * @param stream Output stream (e.g., stdout, stderr)
 */
void matgen_csr_print_info(const matgen_csr_matrix_t* matrix, FILE* stream);

/**
 * @brief Calculate memory usage in bytes
 *
 * @param matrix Matrix to calculate memory for
 * @return Total memory usage in bytes, or 0 if matrix is NULL
 */
matgen_size_t matgen_csr_memory_usage(const matgen_csr_matrix_t* matrix);

/**
 * @brief Validate CSR structure integrity
 *
 * Checks:
 * - Matrix pointer is valid
 * - row_ptr is monotonically increasing
 * - row_ptr[0] == 0 and row_ptr[rows] == nnz
 * - Column indices are in valid range [0, cols)
 * - Column indices within each row are sorted
 *
 * @param matrix Matrix to validate
 * @return true if valid, false otherwise
 */
bool matgen_csr_validate(const matgen_csr_matrix_t* matrix);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_CORE_CSR_MATRIX_H
