#ifndef MATGEN_CORE_MATRIX_CONVERT_H
#define MATGEN_CORE_MATRIX_CONVERT_H

/**
 * @file matrix_convert.h
 * @brief Format conversion functions for sparse matrices
 */

#include "matgen/core/coo_matrix.h"
#include "matgen/core/csr_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convert COO matrix to CSR format
 *
 * The COO matrix should be sorted by (row, col) for best performance.
 * If not sorted, it will be sorted during conversion.
 *
 * @param coo Input COO matrix
 * @return New CSR matrix, or NULL on error
 */
matgen_csr_matrix_t* matgen_coo_to_csr(matgen_coo_matrix_t* coo);

/**
 * @brief Convert CSR matrix to COO format
 *
 * @param csr Input CSR matrix
 * @return New COO matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_csr_to_coo(const matgen_csr_matrix_t* csr);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_CORE_MATRIX_CONVERT_H
