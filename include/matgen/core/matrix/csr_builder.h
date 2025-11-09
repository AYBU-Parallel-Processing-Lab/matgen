#ifndef MATGEN_CORE_MATRIX_CSR_BUILDER_H
#define MATGEN_CORE_MATRIX_CSR_BUILDER_H

/**
 * @file csr_builder.h
 * @brief Fast hash-based CSR matrix builder
 *
 * Provides efficient construction of CSR matrices with automatic duplicate
 * handling using hash tables. Eliminates need for COO intermediate format.
 */

#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// CSR Builder Type
// =============================================================================

/**
 * @brief Opaque CSR builder structure
 *
 * Internal structure for efficient CSR construction with hash-based
 * duplicate detection and thread-local buffering.
 */
typedef struct matgen_csr_builder matgen_csr_builder_t;

// =============================================================================
// Builder Creation and Destruction
// =============================================================================

/**
 * @brief Create a new CSR builder
 *
 * @param rows Number of rows in target matrix
 * @param cols Number of columns in target matrix
 * @param est_nnz Estimated number of non-zeros (for memory allocation)
 * @return Pointer to builder, or NULL on failure
 */
matgen_csr_builder_t* matgen_csr_builder_create(matgen_index_t rows,
                                                matgen_index_t cols,
                                                matgen_size_t est_nnz);

/**
 * @brief Create a CSR builder with execution policy
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param est_nnz Estimated non-zeros
 * @param policy Execution policy for parallel construction
 * @return Pointer to builder, or NULL on failure
 */
matgen_csr_builder_t* matgen_csr_builder_create_with_policy(
    matgen_index_t rows, matgen_index_t cols, matgen_size_t est_nnz,
    matgen_exec_policy_t policy);

/**
 * @brief Destroy CSR builder and free resources
 *
 * @param builder Builder to destroy
 */
void matgen_csr_builder_destroy(matgen_csr_builder_t* builder);

// =============================================================================
// Entry Addition
// =============================================================================

/**
 * @brief Add entry to builder (thread-safe, accumulates duplicates)
 *
 * If an entry at (row, col) already exists, values are summed.
 * This function is thread-safe when using OpenMP backend.
 *
 * @param builder CSR builder
 * @param row Row index
 * @param col Column index
 * @param value Value to add
 * @return MATGEN_SUCCESS or error code
 */
matgen_error_t matgen_csr_builder_add(matgen_csr_builder_t* builder,
                                      matgen_index_t row, matgen_index_t col,
                                      matgen_value_t value);

/**
 * @brief Add entry with collision policy
 *
 * @param builder CSR builder
 * @param row Row index
 * @param col Column index
 * @param value Value to add
 * @param policy How to handle duplicates (SUM, AVG, MAX, MIN, LAST)
 * @return MATGEN_SUCCESS or error code
 */
matgen_error_t matgen_csr_builder_add_with_policy(
    matgen_csr_builder_t* builder, matgen_index_t row, matgen_index_t col,
    matgen_value_t value, matgen_collision_policy_t policy);

/**
 * @brief Add multiple entries from arrays (bulk insert)
 *
 * More efficient than calling matgen_csr_builder_add repeatedly.
 * Thread-safe within the same row range.
 *
 * @param builder CSR builder
 * @param count Number of entries
 * @param rows Array of row indices
 * @param cols Array of column indices
 * @param values Array of values
 * @return MATGEN_SUCCESS or error code
 */
matgen_error_t matgen_csr_builder_add_batch(matgen_csr_builder_t* builder,
                                            matgen_size_t count,
                                            const matgen_index_t* rows,
                                            const matgen_index_t* cols,
                                            const matgen_value_t* values);

// =============================================================================
// Finalization
// =============================================================================

/**
 * @brief Finalize builder and create CSR matrix
 *
 * Merges all entries, sorts rows, and creates final CSR matrix.
 * Builder is consumed and cannot be used after this call.
 *
 * @param builder CSR builder (will be destroyed)
 * @return CSR matrix, or NULL on failure
 */
matgen_csr_matrix_t* matgen_csr_builder_finalize(matgen_csr_builder_t* builder);

// =============================================================================
// Query Functions
// =============================================================================

/**
 * @brief Get current number of entries in builder
 *
 * @param builder CSR builder
 * @return Number of entries (may include duplicates before finalization)
 */
matgen_size_t matgen_csr_builder_get_nnz(const matgen_csr_builder_t* builder);

/**
 * @brief Get matrix dimensions
 *
 * @param builder CSR builder
 * @param rows Output: number of rows
 * @param cols Output: number of columns
 */
void matgen_csr_builder_get_dims(const matgen_csr_builder_t* builder,
                                 matgen_index_t* rows, matgen_index_t* cols);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_CORE_MATRIX_CSR_BUILDER_H
