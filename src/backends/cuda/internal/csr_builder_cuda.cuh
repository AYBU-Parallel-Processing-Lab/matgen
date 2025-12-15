#ifndef MATGEN_BACKENDS_CUDA_INTERNAL_CSR_BUILDER_CUDA_H
#define MATGEN_BACKENDS_CUDA_INTERNAL_CSR_BUILDER_CUDA_H

/**
 * @file csr_builder_cuda.h
 * @brief Internal header for CUDA parallel CSR matrix builder
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/core/matrix/csr_builder.h>
 * instead.
 */

#ifdef MATGEN_HAS_CUDA

#include "matgen/core/matrix/csr_builder.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// CUDA CSR Builder Operations
// =============================================================================

/**
 * @brief Create a new CSR builder (CUDA)
 *
 * Allocates a builder that uses GPU-accelerated operations for construction.
 * The builder accumulates entries in COO format on the host, then performs
 * sorting and conversion on the GPU during finalization.
 *
 * @param rows Number of rows in target matrix
 * @param cols Number of columns in target matrix
 * @param est_nnz Estimated number of non-zeros (for pre-allocation)
 * @return Pointer to new builder, or NULL on error
 */
matgen_csr_builder_t* matgen_csr_builder_create_cuda(matgen_index_t rows,
                                                     matgen_index_t cols,
                                                     matgen_size_t est_nnz);

/**
 * @brief Destroy a CSR builder (CUDA)
 *
 * Frees all resources associated with the builder.
 *
 * @param builder Builder to destroy
 */
void matgen_csr_builder_destroy_cuda(matgen_csr_builder_t* builder);

/**
 * @brief Add an entry to the builder (CUDA)
 *
 * Accumulates entries in COO format. Multiple entries with the same (row, col)
 * will be summed during finalization.
 *
 * Thread-safe: Can be called from multiple CPU threads concurrently (uses
 * atomics).
 *
 * @param builder Builder instance
 * @param row Row index
 * @param col Column index
 * @param value Entry value
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_csr_builder_add_cuda(matgen_csr_builder_t* builder,
                                           matgen_index_t row,
                                           matgen_index_t col,
                                           matgen_value_t value);

/**
 * @brief Get current number of entries in builder (CUDA)
 *
 * Returns the total number of entries added (may include duplicates).
 *
 * @param builder Builder instance
 * @return Number of entries
 */
matgen_size_t matgen_csr_builder_get_nnz_cuda(
    const matgen_csr_builder_t* builder);

/**
 * @brief Finalize builder and produce CSR matrix (CUDA)
 *
 * Performs GPU-accelerated operations:
 * 1. Transfers COO data to device
 * 2. Sorts entries by (row, col) using Thrust
 * 3. Sums duplicate entries using parallel reduction
 * 4. Converts to CSR format
 * 5. Returns result to host
 *
 * After finalization, the builder is destroyed.
 *
 * @param builder Builder instance (consumed)
 * @return CSR matrix, or NULL on error
 */
matgen_csr_matrix_t* matgen_csr_builder_finalize_cuda(
    matgen_csr_builder_t* builder);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_CUDA

#endif  // MATGEN_BACKENDS_CUDA_INTERNAL_CSR_BUILDER_CUDA_H
