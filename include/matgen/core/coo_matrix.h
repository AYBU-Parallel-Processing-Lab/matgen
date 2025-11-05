#ifndef MATGEN_CORE_COO_MATRIX_H
#define MATGEN_CORE_COO_MATRIX_H

/**
 * @file coo_matrix.h
 * @brief Coordinate (COO) sparse matrix format.
 *
 * COO format stores each non-zero as a triplet (row, col, value).
 * This is the simplest format and is ideal for:
 * - Building matrices incrementally.
 * - Converting to other formats.
 * - Simple operations that don't require fast access.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief COO (Coordinate) sparse matrix structure.
 *
 * Stores matrix as arrays of (row, col, value) triplets.
 */
typedef struct {
  size_t rows;      // Number of rows in the matrix
  size_t cols;      // Number of columns in the matrix
  size_t nnz;       // Number of non-zero entries
  size_t capacity;  // Allocated capacity for non-zero entries

  size_t* row_indices;  // Array of row indices for non-zero entries
  size_t* col_indices;  // Array of column indices for non-zero entries
  double* values;       // Array of values for non-zero entries

  bool is_sorted;  // Indicates if the entries are sorted by (row, col)
} matgen_coo_matrix_t;

// =============================================================================
// Creation and Destruction
// =============================================================================

/**
 * @brief Create a new COO matrix.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param nnz_hint Expected number of non-zeros (for pre-allocation)
 * @return Pointer to new matrix, or NULL on error.
 */
matgen_coo_matrix_t* matgen_coo_create(size_t rows, size_t cols,
                                       size_t nnz_hint);

/**
 * @brief Destroy a COO matrix and free its resources.
 *
 * @param matrix Pointer to the matrix to destroy.
 */
void matgen_coo_destroy(matgen_coo_matrix_t* matrix);

// =============================================================================
// Building the Matrix
// =============================================================================

/**
 * @brief Add a single entry to the COO matrix
 *
 * Automatically grows the arrays if needed.
 *
 * @param matrix Matrix to modify
 * @param row Row index (0-based)
 * @param col Column index (0-based)
 * @param value Value to add
 * @return 0 on success, non-zero on error
 */
int matgen_coo_add_entry(matgen_coo_matrix_t* matrix, size_t row, size_t col,
                         double value);

/**
 * @brief Sort entries by (row, col) order
 *
 * @param matrix Matrix to sort
 * @return 0 on success, non-zero on error
 */
int matgen_coo_sort(matgen_coo_matrix_t* matrix);

// =============================================================================
// Matrix Access
// =============================================================================

/**
 * @brief Get value at (row, col)
 *
 * Linear search through entries. Returns 0.0 if not found.
 * For better performance with multiple queries, convert to CSR/CSC format.
 *
 * @param matrix Matrix to query
 * @param row Row index (0-based)
 * @param col Column index (0-based)
 * @return Value at (row, col), or 0.0 if not present or on error
 */
double matgen_coo_get(const matgen_coo_matrix_t* matrix, size_t row,
                      size_t col);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Reserve capacity for entries
 *
 * Pre-allocates memory to avoid repeated reallocations.
 *
 * @param matrix Matrix to reserve capacity for
 * @param capacity Desired capacity
 * @return 0 on success, non-zero on error
 */
int matgen_coo_reserve(matgen_coo_matrix_t* matrix, size_t capacity);

/**
 * @brief Clear all entries from the matrix
 *
 * Does not free memory, just resets nnz to 0.
 *
 * @param matrix Matrix to clear
 */
void matgen_coo_clear(matgen_coo_matrix_t* matrix);

/**
 * @brief Print matrix information to stream
 *
 * @param matrix Matrix to print info about
 * @param stream Output stream (e.g., stdout, stderr)
 */
void matgen_coo_print_info(const matgen_coo_matrix_t* matrix, FILE* stream);

/**
 * @brief Calculate memory usage in bytes
 *
 * @param matrix Matrix to calculate memory for
 * @return Total memory usage in bytes
 */
size_t matgen_coo_memory_usage(const matgen_coo_matrix_t* matrix);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_CORE_COO_MATRIX_H
