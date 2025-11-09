#ifndef MATGEN_ALGORITHMS_SCALING_NEAREST_NEIGHBOR_CUDA_H
#define MATGEN_ALGORITHMS_SCALING_NEAREST_NEIGHBOR_CUDA_H

/**
 * @file nearest_neighbor_cuda.h
 * @brief Nearest neighbor interpolation for sparse matrix scaling (CUDA GPU)
 */

#ifdef MATGEN_HAS_CUDA

#include "matgen/algorithms/scaling/scaling_types.h"
#include "matgen/core/csr_matrix.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scale sparse matrix using nearest neighbor interpolation (CUDA GPU)
 *
 * Maps each source entry to a block of target cells, distributing the value
 * uniformly across the block to preserve mass. When multiple source entries
 * map to the same target cell, values are combined according to the collision
 * policy.
 *
 * This is a CUDA GPU implementation. The source matrix is copied to device
 * memory, processed on the GPU using thread blocks, and the result is copied
 * back to host memory.
 *
 * @param source Source matrix (CSR format, on host)
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param collision_policy How to handle collisions (SUM, AVG, MAX)
 * @param result Output: scaled matrix (CSR format, on host)
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @note CUDA GPU implementation
 * @note Requires CUDA-capable GPU
 * @note For small matrices, may be slower than CPU versions due to transfer
 * overhead
 */
matgen_error_t matgen_scale_nearest_neighbor_cuda(
    const matgen_csr_matrix_t* source, matgen_index_t new_rows,
    matgen_index_t new_cols, matgen_collision_policy_t collision_policy,
    matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_CUDA

#endif  // MATGEN_ALGORITHMS_SCALING_NEAREST_NEIGHBOR_CUDA_H
