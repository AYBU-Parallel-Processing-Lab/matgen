#ifndef MATGEN_BACKENDS_OMP_INTERNAL_CSR_OMP_H
#define MATGEN_BACKENDS_OMP_INTERNAL_CSR_OMP_H

/**
 * @file csr_omp.h
 * @brief Internal header for OpenMP parallel CSR matrix operations
 *
 * This is an internal header used only by the library implementation.
 * Users should use the public API in <matgen/core/matrix/csr.h> instead.
 */

#ifdef MATGEN_HAS_OPENMP

#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// OpenMP Parallel CSR Operations
// =============================================================================

/**
 * @brief Create a new CSR matrix (OpenMP parallel allocation)
 *
 * Uses parallel memory initialization where beneficial.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param nnz Number of non-zeros
 * @return Pointer to new matrix, or NULL on error
 */
matgen_csr_matrix_t* matgen_csr_create_omp(matgen_index_t rows,
                                           matgen_index_t cols,
                                           matgen_size_t nnz);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_OPENMP

#endif  // MATGEN_BACKENDS_OMP_INTERNAL_CSR_OMP_H
