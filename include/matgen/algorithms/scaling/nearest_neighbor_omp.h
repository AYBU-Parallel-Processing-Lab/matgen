#ifndef MATGEN_ALGORITHMS_SCALING_NEAREST_NEIGHBOR_OMP_H
#define MATGEN_ALGORITHMS_SCALING_NEAREST_NEIGHBOR_OMP_H

/**
 * @file nearest_neighbor_omp.h
 * @brief Nearest neighbor interpolation for sparse matrix scaling (OpenMP
 * parallel)
 */

#ifdef MATGEN_HAS_OPENMP

#include "matgen/algorithms/scaling/scaling_types.h"
#include "matgen/core/csr_matrix.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scale sparse matrix using nearest neighbor interpolation (OpenMP
 * parallel)
 *
 * Maps each source entry to a block of target cells, distributing the value
 * uniformly across the block to preserve mass. When multiple source entries
 * map to the same target cell, values are combined according to the collision
 * policy.
 *
 * This is a parallel implementation using OpenMP. Each thread processes a
 * subset of source rows using thread-local triplet buffers, which are then
 * merged at the end.
 *
 * @param source Source matrix (CSR format)
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param collision_policy How to handle collisions (SUM, AVG, MAX)
 * @param result Output: scaled matrix (CSR format)
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @note OpenMP parallel implementation
 * @note The number of threads can be controlled via OMP_NUM_THREADS environment
 *       variable or omp_set_num_threads()
 * @note For small matrices, may fall back to single-threaded execution
 */
matgen_error_t matgen_scale_nearest_neighbor_omp(
    const matgen_csr_matrix_t* source, matgen_index_t new_rows,
    matgen_index_t new_cols, matgen_collision_policy_t collision_policy,
    matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_OPENMP

#endif  // MATGEN_ALGORITHMS_SCALING_NEAREST_NEIGHBOR_OMP_H
