#ifndef MATGEN_ALGORITHMS_SCALING_BILINEAR_H
#define MATGEN_ALGORITHMS_SCALING_BILINEAR_H

/**
 * @file bilinear.h
 * @brief Bilinear interpolation for sparse matrix scaling (sequential)
 */

#include "matgen/core/csr_matrix.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scale sparse matrix using bilinear interpolation
 *
 * Distributes each source entry's value to 4 neighboring target cells
 * using bilinear weights. Values are accumulated (summed) at each target cell.
 *
 * @param source Source matrix (CSR format)
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param result Output: scaled matrix (CSR format)
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @note Sequential implementation (baseline)
 */
matgen_error_t matgen_scale_bilinear(const matgen_csr_matrix_t* source,
                                     matgen_index_t new_rows,
                                     matgen_index_t new_cols,
                                     matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_ALGORITHMS_SCALING_BILINEAR_H
