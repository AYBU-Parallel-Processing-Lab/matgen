#ifndef MATGEN_CORE_MATRIX_COO_H
#define MATGEN_CORE_MATRIX_COO_H

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

#include <stdio.h>

#include "matgen/core/execution/policy.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief COO (Coordinate) sparse matrix structure.
 *
 * Stores matrix as arrays of (row, col, value) triplets.
 */
typedef struct {
  matgen_index_t rows;     // Number of rows in the matrix
  matgen_index_t cols;     // Number of columns in the matrix
  matgen_size_t nnz;       // Number of non-zero entries
  matgen_size_t capacity;  // Allocated capacity for non-zero entries

  matgen_index_t* row_indices;  // Array of row indices [nnz]
  matgen_index_t* col_indices;  // Array of column indices [nnz]
  matgen_value_t* values;       // Array of values [nnz]

  bool is_sorted;  // Indicates if entries are sorted by (row, col)
} matgen_coo_matrix_t;

// =============================================================================
// Creation and Destruction
// =============================================================================

/**
 * @brief Create a new COO matrix.
 *
 * Uses automatic backend selection based on available hardware.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param nnz_hint Expected number of non-zeros (for pre-allocation)
 * @return Pointer to new matrix, or NULL on error.
 */
matgen_coo_matrix_t* matgen_coo_create(matgen_index_t rows, matgen_index_t cols,
                                       matgen_size_t nnz_hint);

/**
 * @brief Destroy a COO matrix and free its resources.
 *
 * @param matrix Pointer to the matrix to destroy (can be NULL).
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
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_add_entry(matgen_coo_matrix_t* matrix,
                                    matgen_index_t row, matgen_index_t col,
                                    matgen_value_t value);

/**
 * @brief Sort entries by (row, col) order
 *
 * Uses automatic backend selection. After sorting, is_sorted flag is set to
 * true.
 *
 * @param matrix Matrix to sort
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_sort(matgen_coo_matrix_t* matrix);

/**
 * @brief Sort entries with explicit execution policy
 *
 * @param matrix Matrix to sort
 * @param policy Execution policy (SEQ, PAR, AUTO, etc.)
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_sort_with_policy(matgen_coo_matrix_t* matrix,
                                           matgen_exec_policy_t policy);

/**
 * @brief Sum duplicate entries in a sorted COO matrix
 *
 * Assumes matrix is already sorted. Combines entries with identical (row, col)
 * by summing their values. Modifies the matrix in-place, reducing nnz.
 *
 * @param matrix Matrix to process (must be sorted)
 * @return MATGEN_SUCCESS on success, error code on failure
 *
 * @note Matrix must be sorted first using matgen_coo_sort()
 */
matgen_error_t matgen_coo_sum_duplicates(matgen_coo_matrix_t* matrix);

/**
 * @brief Sum duplicate entries with explicit execution policy
 *
 * @param matrix Matrix to process (must be sorted)
 * @param policy Execution policy (SEQ, PAR, AUTO, etc.)
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_sum_duplicates_with_policy(
    matgen_coo_matrix_t* matrix, matgen_exec_policy_t policy);

/**
 * @brief Merge duplicate entries in a sorted COO matrix using a collision
 * policy
 *
 * Assumes matrix is already sorted. Combines entries with identical (row, col)
 * according to the specified policy. Modifies the matrix in-place, reducing
 * nnz.
 *
 * @param matrix Matrix to process (must be sorted)
 * @param policy Collision policy (SUM, AVG, MAX)
 * @return MATGEN_SUCCESS on success, error code on failure
 *
 * @note Matrix must be sorted first using matgen_coo_sort()
 * @note For SUM policy, use matgen_coo_sum_duplicates() for better performance
 */
matgen_error_t matgen_coo_merge_duplicates(matgen_coo_matrix_t* matrix,
                                           matgen_collision_policy_t policy);

/**
 * @brief Merge duplicate entries with explicit execution policy
 *
 * @param matrix Matrix to process (must be sorted)
 * @param collision_policy Collision policy (SUM, AVG, MAX, MIN, LAST)
 * @param exec_policy Execution policy (SEQ, PAR, AUTO, etc.)
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_merge_duplicates_with_policy(
    matgen_coo_matrix_t* matrix, matgen_collision_policy_t collision_policy,
    matgen_exec_policy_t exec_policy);

// =============================================================================
// Matrix Access
// =============================================================================

/**
 * @brief Get value at (row, col)
 *
 * Linear search through entries. Returns 0.0 if not found.
 * For better performance with multiple queries, convert to CSR format.
 *
 * @param matrix Matrix to query
 * @param row Row index (0-based)
 * @param col Column index (0-based)
 * @param[out] value Pointer to store the value (can be NULL to just check
 * existence)
 * @return MATGEN_SUCCESS if found, MATGEN_ERROR_INVALID_ARGUMENT if not found
 * or error
 */
matgen_error_t matgen_coo_get(const matgen_coo_matrix_t* matrix,
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
bool matgen_coo_has_entry(const matgen_coo_matrix_t* matrix, matgen_index_t row,
                          matgen_index_t col);

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
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_coo_reserve(matgen_coo_matrix_t* matrix,
                                  matgen_size_t capacity);

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
 * @return Total memory usage in bytes, or 0 if matrix is NULL
 */
matgen_size_t matgen_coo_memory_usage(const matgen_coo_matrix_t* matrix);

/**
 * @brief Validate COO matrix integrity
 *
 * Checks:
 * - Matrix pointer is valid
 * - Indices are within bounds
 * - Arrays are allocated if nnz > 0
 *
 * @param matrix Matrix to validate
 * @return true if valid, false otherwise
 */
bool matgen_coo_validate(const matgen_coo_matrix_t* matrix);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_CORE_MATRIX_COO_H
