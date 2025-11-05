#ifndef MATGEN_CORE_MATRIX_SCALE_H
#define MATGEN_CORE_MATRIX_SCALE_H

/**
 * @file matrix_scale.h
 * @brief Sparse matrix scaling/resampling operations
 *
 * Provides methods to scale (resize) sparse matrices using:
 * - Nearest neighbor interpolation
 * - Bilinear interpolation
 *
 * Available in multiple parallel implementations:
 * - Sequential (baseline)
 * - OpenMP (shared memory parallel)
 * - MPI (distributed memory parallel)
 * - CUDA (GPU parallel)
 */

#include <stdbool.h>
#include <stddef.h>

#include "matgen/core/coo_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scaling method type
 */
typedef enum {
  MATGEN_SCALE_NEAREST,  // Nearest neighbor (fast, preserves sparsity)
  MATGEN_SCALE_BILINEAR  // Bilinear interpolation (smoother, may densify)
} matgen_scale_method_t;

/**
 * @brief Configuration for matrix scaling
 */
typedef struct {
  size_t new_rows;  // Target number of rows
  size_t new_cols;  // Target number of columns

  matgen_scale_method_t method;  // Scaling method

  double sparsity_threshold;  // For bilinear: drop values below this (0.0 =
                              // keep all)
  bool preserve_zeros;        // Whether to explicitly preserve zero entries

  // Parallel configuration
  int num_threads;    // OpenMP: number of threads (0 = auto)
  int num_processes;  // MPI: number of processes (auto-detected)
  int device_id;      // CUDA: GPU device ID
} matgen_scale_config_t;

/**
 * @brief Initialize scaling config with defaults
 */
void matgen_scale_config_init(matgen_scale_config_t* config, size_t new_rows,
                              size_t new_cols);

// =============================================================================
// Sequential Versions
// =============================================================================

/**
 * @brief Scale sparse matrix (sequential version)
 *
 * @param input Input COO matrix
 * @param config Scaling configuration
 * @return New scaled matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_matrix_scale(const matgen_coo_matrix_t* input,
                                         const matgen_scale_config_t* config);

/**
 * @brief Scale using nearest neighbor (sequential)
 */
matgen_coo_matrix_t* matgen_matrix_scale_nearest(
    const matgen_coo_matrix_t* input, size_t new_rows, size_t new_cols);

/**
 * @brief Scale using bilinear interpolation (sequential)
 */
matgen_coo_matrix_t* matgen_matrix_scale_bilinear(
    const matgen_coo_matrix_t* input, size_t new_rows, size_t new_cols,
    double sparsity_threshold);

// =============================================================================
// OpenMP Versions
// =============================================================================

#ifdef MATGEN_HAS_OPENMP

/**
 * @brief Scale sparse matrix (OpenMP parallel version)
 */
matgen_coo_matrix_t* matgen_matrix_scale_omp(
    const matgen_coo_matrix_t* input, const matgen_scale_config_t* config);

#endif /* MATGEN_HAS_OPENMP */

// =============================================================================
// MPI Versions
// =============================================================================

#ifdef MATGEN_HAS_MPI

/**
 * @brief Scale sparse matrix (MPI distributed version)
 *
 * Input matrix is distributed across MPI ranks.
 * Output is also distributed.
 */
matgen_coo_matrix_t* matgen_matrix_scale_mpi(
    const matgen_coo_matrix_t* input, const matgen_scale_config_t* config);

#endif /* MATGEN_HAS_MPI */

// =============================================================================
// CUDA Versions
// =============================================================================

#ifdef MATGEN_HAS_CUDA

/**
 * @brief Scale sparse matrix (CUDA GPU version)
 */
matgen_coo_matrix_t* matgen_matrix_scale_cuda(
    const matgen_coo_matrix_t* input, const matgen_scale_config_t* config);

#endif /* MATGEN_HAS_CUDA */

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_CORE_MATRIX_SCALE_H
