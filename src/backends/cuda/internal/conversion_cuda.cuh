#ifndef MATGEN_BACKENDS_CUDA_INTERNAL_CONVERSION_CUDA_H
#define MATGEN_BACKENDS_CUDA_INTERNAL_CONVERSION_CUDA_H

/**
 * @file conversion_cuda.h
 * @brief Internal header for CUDA parallel matrix format conversions
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/core/matrix/conversion.h> instead.
 */

#ifdef MATGEN_HAS_CUDA

#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/csr.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// CUDA Matrix Format Conversions
// =============================================================================

/**
 * @brief Convert COO matrix to CSR format (CUDA)
 *
 * Uses GPU-accelerated sorting and parallel prefix sum for efficient
 * conversion. If input COO is not sorted, it will be sorted first (a copy is
 * made).
 *
 * Algorithm:
 *   1. Sort COO by (row, col) if needed (GPU sort)
 *   2. Parallel histogram to count entries per row
 *   3. Prefix sum to build row_ptr array
 *   4. Parallel copy of col_indices and values
 *
 * @param coo Source COO matrix
 * @return CSR matrix, or NULL on error
 */
matgen_csr_matrix_t* matgen_coo_to_csr_cuda(const matgen_coo_matrix_t* coo);

/**
 * @brief Convert CSR matrix to COO format (CUDA)
 *
 * Uses parallel kernel to expand row pointers into explicit row indices.
 *
 * Algorithm:
 *   1. Parallel kernel: each entry computes its row via binary search
 *   2. Copy col_indices and values directly
 *
 * @param csr Source CSR matrix
 * @return COO matrix (sorted), or NULL on error
 */
matgen_coo_matrix_t* matgen_csr_to_coo_cuda(const matgen_csr_matrix_t* csr);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_CUDA

#endif  // MATGEN_BACKENDS_CUDA_INTERNAL_CONVERSION_CUDA_H
