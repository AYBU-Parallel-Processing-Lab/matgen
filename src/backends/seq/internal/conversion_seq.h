#ifndef MATGEN_BACKENDS_SEQ_INTERNAL_CONVERSION_SEQ_H
#define MATGEN_BACKENDS_SEQ_INTERNAL_CONVERSION_SEQ_H

/**
 * @file conversion_seq.h
 * @brief Internal header for sequential matrix format conversion
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/core/matrix/conversion.h> instead.
 */

#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/csr.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Sequential Conversion Operations
// =============================================================================

/**
 * @brief Convert COO matrix to CSR format (sequential)
 *
 * Assumes COO matrix is sorted by (row, col).
 *
 * @param coo Source COO matrix
 * @return New CSR matrix, or NULL on error
 */
matgen_csr_matrix_t* matgen_coo_to_csr_seq(const matgen_coo_matrix_t* coo);

/**
 * @brief Convert CSR matrix to COO format (sequential)
 *
 * @param csr Source CSR matrix
 * @return New COO matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_csr_to_coo_seq(const matgen_csr_matrix_t* csr);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_BACKENDS_SEQ_INTERNAL_CONVERSION_SEQ_H
