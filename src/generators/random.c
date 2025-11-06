#include "matgen/generators/random.h"

#include <math.h>
#include <string.h>
#include <time.h>

#include "matgen/math/constants.h"
#include "matgen/utils/log.h"

// =============================================================================
// Simple RNG for reproducibility
// =============================================================================

typedef struct {
  u32 state;
} rng_state_t;

static void rng_init(rng_state_t* rng, u32 seed) {
  if (seed == 0) {
    rng->state = (u32)time(NULL);
  } else {
    rng->state = seed;
  }
  MATGEN_LOG_DEBUG("RNG initialized with seed: %u", rng->state);
}

// Generate random u32 (LCG from Numerical Recipes)
static u32 rng_next(rng_state_t* rng) {
  rng->state = (1664525U * rng->state) + 1013904223U;
  return rng->state;
}

// Generate random f64 in [0, 1)
static f64 rng_uniform_01(rng_state_t* rng) {
  return (f64)rng_next(rng) / (f64)0xFFFFFFFFU;
}

// Generate random f64 in [min, max]
static f64 rng_uniform(rng_state_t* rng, f64 min, f64 max) {
  return min + (rng_uniform_01(rng) * (max - min));
}

// Generate random f64 from normal distribution (Box-Muller)
static f64 rng_normal(rng_state_t* rng, f64 mean, f64 stddev) {
  f64 u1 = rng_uniform_01(rng);
  f64 u2 = rng_uniform_01(rng);

  // Avoid log(0)
  if (u1 < 1e-10) {
    u1 = 1e-10;
  }

  f64 z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * MATGEN_PI * u2);
  return mean + (stddev * z0);
}

// Generate value based on distribution
static f64 generate_value(rng_state_t* rng,
                          const matgen_random_config_t* config) {
  switch (config->distribution) {
    case MATGEN_DIST_UNIFORM:
      return rng_uniform(rng, config->min_value, config->max_value);

    case MATGEN_DIST_NORMAL:
      return rng_normal(rng, config->mean, config->stddev);

    case MATGEN_DIST_CONSTANT:
      return config->constant_value;

    default:
      return 0.0;
  }
}

// =============================================================================
// Configuration
// =============================================================================

void matgen_random_config_init(matgen_random_config_t* config,
                               matgen_index_t rows, matgen_index_t cols,
                               matgen_size_t nnz) {
  if (!config) {
    return;
  }

  memset(config, 0, sizeof(matgen_random_config_t));

  config->rows = rows;
  config->cols = cols;
  config->nnz = nnz;
  config->density = 0.0;

  config->distribution = MATGEN_DIST_UNIFORM;
  config->min_value = 0.0;
  config->max_value = 1.0;
  config->mean = 0.0;
  config->stddev = 1.0;
  config->constant_value = 1.0;

  config->seed = 0;  // Time-based
  config->allow_duplicates = false;
  config->sorted = true;
}

// =============================================================================
// Random Generation
// =============================================================================

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
matgen_coo_matrix_t* matgen_random_generate(
    const matgen_random_config_t* config) {
  if (!config) {
    MATGEN_LOG_ERROR("NULL config pointer");
    return NULL;
  }

  if (config->rows == 0 || config->cols == 0) {
    MATGEN_LOG_ERROR("Invalid dimensions: %llu x %llu",
                     (unsigned long long)config->rows,
                     (unsigned long long)config->cols);
    return NULL;
  }

  // Determine nnz from density if specified
  matgen_size_t nnz = config->nnz;
  if (config->density > 0.0) {
    u64 total_elements = (u64)config->rows * (u64)config->cols;
    nnz = (matgen_size_t)(config->density * (f64)total_elements);
    MATGEN_LOG_DEBUG("Using density %.4f -> nnz = %zu", config->density, nnz);
  }

  if (nnz == 0) {
    MATGEN_LOG_WARN("Generating empty matrix (nnz=0)");
  }

  // Check if nnz is achievable without duplicates
  u64 max_possible = (u64)config->rows * (u64)config->cols;
  if (!config->allow_duplicates && nnz > max_possible) {
    MATGEN_LOG_ERROR("Cannot generate %zu unique entries in %llu x %llu matrix",
                     nnz, (unsigned long long)config->rows,
                     (unsigned long long)config->cols);
    return NULL;
  }

  MATGEN_LOG_DEBUG("Generating random matrix %llu x %llu with %zu non-zeros",
                   (unsigned long long)config->rows,
                   (unsigned long long)config->cols, nnz);

  // Create matrix
  matgen_coo_matrix_t* matrix =
      matgen_coo_create(config->rows, config->cols, nnz);
  if (!matrix) {
    return NULL;
  }

  // Initialize RNG
  rng_state_t rng;
  rng_init(&rng, config->seed);

  if (config->allow_duplicates) {
    // Simple: generate random (row, col) pairs, allow duplicates
    for (matgen_size_t i = 0; i < nnz; i++) {
      matgen_index_t row = (matgen_index_t)(rng_next(&rng) % config->rows);
      matgen_index_t col = (matgen_index_t)(rng_next(&rng) % config->cols);
      f64 value = generate_value(&rng, config);

      if (matgen_coo_add_entry(matrix, row, col, value) != MATGEN_SUCCESS) {
        MATGEN_LOG_ERROR("Failed to add entry at (%llu, %llu)",
                         (unsigned long long)row, (unsigned long long)col);
        matgen_coo_destroy(matrix);
        return NULL;
      }
    }
  } else {
    // No duplicates: use rejection sampling or shuffle method
    f64 fill_ratio = (f64)nnz / (f64)max_possible;

    if (fill_ratio < 0.5) {
      // Rejection sampling: good for sparse matrices
      matgen_size_t added = 0;
      matgen_size_t attempts = 0;
      matgen_size_t max_attempts = nnz * 100;  // Avoid infinite loops

      while (added < nnz && attempts < max_attempts) {
        matgen_index_t row = (matgen_index_t)(rng_next(&rng) % config->rows);
        matgen_index_t col = (matgen_index_t)(rng_next(&rng) % config->cols);

        // Check if already exists
        if (!matgen_coo_has_entry(matrix, row, col)) {
          f64 value = generate_value(&rng, config);
          if (matgen_coo_add_entry(matrix, row, col, value) == MATGEN_SUCCESS) {
            added++;
          }
        }
        attempts++;
      }

      if (added < nnz) {
        MATGEN_LOG_ERROR(
            "Could only generate %zu/%zu unique entries after %zu attempts",
            added, nnz, attempts);
        matgen_coo_destroy(matrix);
        return NULL;
      }
    } else {
      // Dense case: reservoir sampling / shuffle approach
      MATGEN_LOG_DEBUG(
          "Using shuffle method for dense generation (fill_ratio=%.2f)",
          fill_ratio);

      // For dense matrices, generate all positions and randomly select
      // This is more efficient than rejection sampling when fill_ratio is high

      // We'll use a simpler approach: generate nnz unique random positions
      // by shuffling a virtual array of indices

      for (matgen_size_t i = 0; i < nnz; i++) {
        matgen_index_t row;
        matgen_index_t col;
        bool found_unique = false;
        matgen_size_t attempts = 0;
        matgen_size_t max_attempts = 1000;

        // Use rejection sampling but with better success rate for dense case
        while (!found_unique && attempts < max_attempts) {
          row = (matgen_index_t)(rng_next(&rng) % config->rows);
          col = (matgen_index_t)(rng_next(&rng) % config->cols);

          if (!matgen_coo_has_entry(matrix, row, col)) {
            found_unique = true;
          }
          attempts++;
        }

        if (!found_unique) {
          MATGEN_LOG_ERROR("Failed to find unique position after %zu attempts",
                           attempts);
          matgen_coo_destroy(matrix);
          return NULL;
        }

        f64 value = generate_value(&rng, config);
        if (matgen_coo_add_entry(matrix, row, col, value) != MATGEN_SUCCESS) {
          MATGEN_LOG_ERROR("Failed to add entry");
          matgen_coo_destroy(matrix);
          return NULL;
        }
      }
    }
  }

  // Sort if requested
  if (config->sorted) {
    if (matgen_coo_sort(matrix) != MATGEN_SUCCESS) {
      MATGEN_LOG_ERROR("Failed to sort matrix");
      matgen_coo_destroy(matrix);
      return NULL;
    }
  }

  MATGEN_LOG_DEBUG("Generated random matrix successfully");
  return matrix;
}

// =============================================================================
// Special Patterns
// =============================================================================

matgen_coo_matrix_t* matgen_random_diagonal(matgen_index_t rows,
                                            matgen_index_t cols,
                                            matgen_distribution_t distribution,
                                            f64 min_value, f64 max_value,
                                            u32 seed) {
  matgen_index_t diag_size = MATGEN_MIN(rows, cols);

  MATGEN_LOG_DEBUG("Generating %llu x %llu diagonal matrix",
                   (unsigned long long)rows, (unsigned long long)cols);

  matgen_coo_matrix_t* matrix = matgen_coo_create(rows, cols, diag_size);
  if (!matrix) {
    return NULL;
  }

  rng_state_t rng;
  rng_init(&rng, seed);

  matgen_random_config_t config;
  matgen_random_config_init(&config, rows, cols, diag_size);
  config.distribution = distribution;

  // Set distribution parameters based on type
  if (distribution == MATGEN_DIST_CONSTANT) {
    config.constant_value = min_value;
  } else if (distribution == MATGEN_DIST_UNIFORM) {
    config.min_value = min_value;
    config.max_value = max_value;
  } else if (distribution == MATGEN_DIST_NORMAL) {
    config.mean = min_value;
    config.stddev = max_value;
  }

  for (matgen_index_t i = 0; i < diag_size; i++) {
    f64 value = generate_value(&rng, &config);
    matgen_coo_add_entry(matrix, i, i, value);
  }

  matrix->is_sorted = true;
  return matrix;
}

matgen_coo_matrix_t* matgen_random_tridiagonal(
    matgen_index_t size, matgen_distribution_t distribution, f64 min_value,
    f64 max_value, u32 seed) {
  if (size == 0) {
    MATGEN_LOG_ERROR("Invalid size: 0");
    return NULL;
  }

  MATGEN_LOG_DEBUG("Generating %llu x %llu tridiagonal matrix",
                   (unsigned long long)size, (unsigned long long)size);

  // nnz = main diagonal + upper + lower
  // = size + (size-1) + (size-1) = 3*size - 2
  matgen_size_t nnz = (size == 1) ? 1 : ((3 * size) - 2);

  matgen_coo_matrix_t* matrix = matgen_coo_create(size, size, nnz);
  if (!matrix) {
    return NULL;
  }

  rng_state_t rng;
  rng_init(&rng, seed);

  matgen_random_config_t config;
  matgen_random_config_init(&config, size, size, nnz);
  config.distribution = distribution;

  // Set distribution parameters based on type
  if (distribution == MATGEN_DIST_CONSTANT) {
    config.constant_value = min_value;
  } else if (distribution == MATGEN_DIST_UNIFORM) {
    config.min_value = min_value;
    config.max_value = max_value;
  } else if (distribution == MATGEN_DIST_NORMAL) {
    config.mean = min_value;
    config.stddev = max_value;
  }

  // Main diagonal
  for (matgen_index_t i = 0; i < size; i++) {
    f64 value = generate_value(&rng, &config);
    matgen_coo_add_entry(matrix, i, i, value);
  }

  // Upper diagonal
  for (matgen_index_t i = 0; i < size - 1; i++) {
    f64 value = generate_value(&rng, &config);
    matgen_coo_add_entry(matrix, i, i + 1, value);
  }

  // Lower diagonal
  for (matgen_index_t i = 0; i < size - 1; i++) {
    f64 value = generate_value(&rng, &config);
    matgen_coo_add_entry(matrix, i + 1, i, value);
  }

  matgen_coo_sort(matrix);
  return matrix;
}

matgen_coo_matrix_t* matgen_random_uniform(matgen_index_t rows,
                                           matgen_index_t cols,
                                           matgen_size_t nnz, u32 seed) {
  matgen_random_config_t config;
  matgen_random_config_init(&config, rows, cols, nnz);
  config.seed = seed;

  return matgen_random_generate(&config);
}
