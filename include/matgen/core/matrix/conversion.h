#ifndef MATGEN_CORE_MATRIX_CONVERSION_H
#define MATGEN_CORE_MATRIX_CONVERSION_H

/**
 * @file conversion.h
 * @brief Format conversion functions for sparse matrices
 *
 * Converts between COO (Coordinate) and CSR (Compressed Sparse Row) formats.
 */

#include "matgen/core/execution/policy.h"
#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/csr.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convert COO matrix to CSR format
 *
 * The COO matrix will be sorted during conversion if not already sorted.
 * The original COO matrix is not modified (sorting is done on a copy if
 * needed). Uses automatic backend selection.
 *
 * @param coo Input COO matrix (const, not modified)
 * @return New CSR matrix, or NULL on error
 */
matgen_csr_matrix_t* matgen_coo_to_csr(const matgen_coo_matrix_t* coo);

/**
 * @brief Convert COO matrix to CSR format with explicit execution policy
 *
 * @param coo Input COO matrix (const, not modified)
 * @param policy Execution policy (SEQ, PAR, AUTO, etc.)
 * @return New CSR matrix, or NULL on error
 */
matgen_csr_matrix_t* matgen_coo_to_csr_with_policy(
    const matgen_coo_matrix_t* coo, matgen_exec_policy_t policy);

/**
 * @brief Convert CSR matrix to COO format
 *
 * Uses automatic backend selection.
 *
 * @param csr Input CSR matrix
 * @return New COO matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_csr_to_coo(const matgen_csr_matrix_t* csr);

/**
 * @brief Convert CSR matrix to COO format with explicit execution policy
 *
 * @param csr Input CSR matrix
 * @param policy Execution policy (SEQ, PAR, AUTO, etc.)
 * @return New COO matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_csr_to_coo_with_policy(
    const matgen_csr_matrix_t* csr, matgen_exec_policy_t policy);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_CORE_MATRIX_CONVERSION_H
