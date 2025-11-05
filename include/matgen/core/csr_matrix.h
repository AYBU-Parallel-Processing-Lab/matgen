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
 * - Memory efficient
 * - Fast row access and row operations
 * - Standard format for sparse BLAS operations
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

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
  size_t rows;  // Number of rows
  size_t cols;  // Number of columns
  size_t nnz;   // Number of non-zeros

  size_t* row_ptr;      // Row pointer array [rows + 1]
  size_t* col_indices;  // Column indices array [nnz]
  double* values;       // Values array [nnz]
} matgen_csr_matrix_t;

// =============================================================================
// Creation and Destruction
// =============================================================================

/**
 * @brief Create a new CSR matrix
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param nnz Number of non-zeros
 * @return Pointer to new matrix, or NULL on error
 */
matgen_csr_matrix_t* matgen_csr_create(size_t rows, size_t cols, size_t nnz);

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
 * @return Value at (row, col), or 0.0 if not present or on error
 */
double matgen_csr_get(const matgen_csr_matrix_t* matrix, size_t row,
                      size_t col);

/**
 * @brief Get the number of non-zeros in a specific row
 *
 * @param matrix Matrix to query
 * @param row Row index (0-based)
 * @return Number of non-zeros in row, or 0 on error
 */
size_t matgen_csr_row_nnz(const matgen_csr_matrix_t* matrix, size_t row);

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
 * @return Total memory usage in bytes
 */
size_t matgen_csr_memory_usage(const matgen_csr_matrix_t* matrix);

/**
 * @brief Validate CSR structure integrity
 *
 * Checks:
 * - row_ptr is monotonically increasing
 * - column indices are in valid range
 * - column indices within each row are sorted
 *
 * @param matrix Matrix to validate
 * @return true if valid, false otherwise
 */
bool matgen_csr_validate(const matgen_csr_matrix_t* matrix);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_CORE_CSR_MATRIX_H
