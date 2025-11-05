#ifndef MATGEN_CORE_MATRIX_GENERATE_H
#define MATGEN_CORE_MATRIX_GENERATE_H

/**
 * @file matrix_generate.h
 * @brief Random sparse matrix generation
 *
 * Provides functions for generating random sparse matrices with various
 * distributions and patterns.
 */

#include <stdbool.h>
#include <stddef.h>

#include "matgen/core/coo_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Value distribution types
 */
typedef enum {
  MATGEN_DIST_UNIFORM,  // Uniform distribution in [min, max]
  MATGEN_DIST_NORMAL,   // Normal/Gaussian distribution
  MATGEN_DIST_CONSTANT  // All values equal to constant
} matgen_distribution_t;

/**
 * @brief Configuration for random matrix generation
 */
typedef struct {
  size_t rows;     // Number of rows
  size_t cols;     // Number of columns
  size_t nnz;      // Number of non-zeros to generate
  double density;  // Alternative: density (0.0-1.0), overrides nnz if > 0

  matgen_distribution_t distribution;  // Value distribution type

  // Uniform distribution parameters
  double min_value;  // Minimum value (uniform)
  double max_value;  // Maximum value (uniform)

  // Normal distribution parameters
  double mean;    // Mean (normal)
  double stddev;  // Standard deviation (normal)

  // Constant distribution
  double constant_value;  // Constant value

  unsigned int seed;  // Random seed (0 = use time-based seed)

  bool allow_duplicates;  // Allow duplicate (row,col) pairs?
  bool sorted;            // Sort output by (row, col)?
} matgen_random_config_t;

/**
 * @brief Initialize random config with default values
 *
 * Defaults:
 * - distribution: MATGEN_DIST_UNIFORM
 * - min_value: 0.0, max_value: 1.0
 * - mean: 0.0, stddev: 1.0
 * - constant_value: 1.0
 * - seed: 0 (time-based)
 * - allow_duplicates: false
 * - sorted: true
 *
 * @param config Config to initialize
 * @param rows Number of rows
 * @param cols Number of columns
 * @param nnz Number of non-zeros
 */
void matgen_random_config_init(matgen_random_config_t* config, size_t rows,
                               size_t cols, size_t nnz);

/**
 * @brief Generate a random sparse COO matrix
 *
 * Generates a random sparse matrix according to the configuration.
 * If allow_duplicates is false and it's not possible to generate
 * the requested number of unique entries, returns NULL.
 *
 * @param config Configuration for generation
 * @return New COO matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_random_coo_create(
    const matgen_random_config_t* config);

/**
 * @brief Generate a random diagonal matrix
 *
 * Creates a diagonal matrix with random values on the diagonal.
 * Size will be min(rows, cols) non-zeros.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param distribution Value distribution
 * @param min_value Minimum value (uniform) or mean (normal)
 * @param max_value Maximum value (uniform) or stddev (normal)
 * @param seed Random seed (0 = time-based)
 * @return New COO matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_random_diagonal(size_t rows, size_t cols,
                                            matgen_distribution_t distribution,
                                            double min_value, double max_value,
                                            unsigned int seed);

/**
 * @brief Generate a random tridiagonal matrix
 *
 * Creates a tridiagonal matrix (main diagonal + upper + lower).
 *
 * @param size Matrix size (square matrix)
 * @param distribution Value distribution
 * @param min_value Minimum value (uniform) or mean (normal)
 * @param max_value Maximum value (uniform) or stddev (normal)
 * @param seed Random seed (0 = time-based)
 * @return New COO matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_random_tridiagonal(
    size_t size, matgen_distribution_t distribution, double min_value,
    double max_value, unsigned int seed);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_CORE_MATRIX_GENERATE_H
