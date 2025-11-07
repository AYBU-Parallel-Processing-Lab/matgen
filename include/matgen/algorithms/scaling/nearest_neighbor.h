#ifndef MATGEN_ALGORITHMS_SCALING_NEAREST_NEIGHBOR_H
#define MATGEN_ALGORITHMS_SCALING_NEAREST_NEIGHBOR_H

/**
 * @file nearest_neighbor.h
 * @brief Nearest neighbor interpolation for sparse matrix scaling (sequential)
 */

#include "matgen/algorithms/scaling/scaling_types.h"
#include "matgen/core/csr_matrix.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scale sparse matrix using nearest neighbor interpolation
 *
 * Maps each source entry to the nearest target cell. When multiple source
 * entries map to the same target cell, values are combined according to
 * the collision policy.
 *
 * @param source Source matrix (CSR format)
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param collision_policy How to handle collisions
 * @param result Output: scaled matrix (CSR format)
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @note Sequential implementation (baseline)
 */
matgen_error_t matgen_scale_nearest_neighbor(
    const matgen_csr_matrix_t* source, matgen_index_t new_rows,
    matgen_index_t new_cols, matgen_collision_policy_t collision_policy,
    matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_ALGORITHMS_SCALING_NEAREST_NEIGHBOR_H
