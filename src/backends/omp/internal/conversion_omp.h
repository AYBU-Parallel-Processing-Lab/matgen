#ifndef MATGEN_BACKENDS_OMP_INTERNAL_CONVERSION_OMP_H
#define MATGEN_BACKENDS_OMP_INTERNAL_CONVERSION_OMP_H

/**
 * @file conversion_omp.h
 * @brief Internal header for OpenMP parallel matrix format conversion
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/core/matrix/conversion.h> instead.
 */

#ifdef MATGEN_HAS_OPENMP

#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/csr.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// OpenMP Parallel Conversion Operations
// =============================================================================

/**
 * @brief Convert COO matrix to CSR format (OpenMP parallel)
 *
 * Assumes COO matrix is sorted by (row, col).
 * Uses parallel algorithms for conversion.
 *
 * @param coo Source COO matrix
 * @return New CSR matrix, or NULL on error
 */
matgen_csr_matrix_t* matgen_coo_to_csr_omp(const matgen_coo_matrix_t* coo);

/**
 * @brief Convert CSR matrix to COO format (OpenMP parallel)
 *
 * @param csr Source CSR matrix
 * @return New COO matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_csr_to_coo_omp(const matgen_csr_matrix_t* csr);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_OPENMP

#endif  // MATGEN_BACKENDS_OMP_INTERNAL_CONVERSION_OMP_H
