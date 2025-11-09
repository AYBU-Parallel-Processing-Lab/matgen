#ifndef MATGEN_BACKENDS_SEQ_INTERNAL_CSR_SEQ_H
#define MATGEN_BACKENDS_SEQ_INTERNAL_CSR_SEQ_H

/**
 * @file csr_seq.h
 * @brief Internal header for sequential CSR matrix operations
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/core/matrix/csr.h> instead.
 */

#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Sequential CSR Operations
// =============================================================================

/**
 * @brief Create a new CSR matrix (sequential)
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param nnz Number of non-zeros
 * @return Pointer to new matrix, or NULL on error
 */
matgen_csr_matrix_t* matgen_csr_create_seq(matgen_index_t rows,
                                           matgen_index_t cols,
                                           matgen_size_t nnz);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_BACKENDS_SEQ_INTERNAL_CSR_SEQ_H
