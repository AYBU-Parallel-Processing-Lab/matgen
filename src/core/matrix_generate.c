#include "matgen/core/matrix_generate.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Simple LCG random number generator for reproducibility
typedef struct {
  unsigned int state;
} rng_state_t;

static void rng_init(rng_state_t* rng, unsigned int seed) {
  if (seed == 0) {
    rng->state = (unsigned int)time(NULL);
  } else {
    rng->state = seed;
  }
}

// Generate random unsigned int
static unsigned int rng_next(rng_state_t* rng) {
  // LCG parameters from Numerical Recipes
  rng->state = (1664525U * rng->state) + 1013904223U;
  return rng->state;
}

// Generate random double in [0, 1)
static double rng_uniform_01(rng_state_t* rng) {
  return (double)rng_next(rng) / (double)0xFFFFFFFFU;
}

// Generate random double in [min, max]
static double rng_uniform(rng_state_t* rng, double min, double max) {
  return min + (rng_uniform_01(rng) * (max - min));
}

// Generate random double from normal distribution (Box-Muller)
static double rng_normal(rng_state_t* rng, double mean, double stddev) {
  // Box-Muller transform
  double u1 = rng_uniform_01(rng);
  double u2 = rng_uniform_01(rng);

  // Avoid log(0)
  if (u1 < 1e-10) {
    u1 = 1e-10;
  }

  double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
  return mean + (stddev * z0);
}

// Generate a random value according to distribution
static double generate_value(rng_state_t* rng,
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

// Hash function for position checking (simple but effective)
typedef struct {
  size_t* positions;  // Array of (row * cols + col)
  size_t count;
  size_t capacity;
} position_set_t;

static position_set_t* position_set_create(size_t capacity) {
  position_set_t* set = (position_set_t*)malloc(sizeof(position_set_t));
  if (!set) {
    return NULL;
  }
  set->positions = (size_t*)malloc(capacity * sizeof(size_t));
  if (!set->positions) {
    free(set);
    return NULL;
  }
  set->count = 0;
  set->capacity = capacity;
  return set;
}

static void position_set_destroy(position_set_t* set) {
  if (set) {
    free(set->positions);
    free(set);
  }
}

static bool position_set_contains(const position_set_t* set, size_t row,
                                  size_t col, size_t cols) {
  size_t pos = (row * cols) + col;
  for (size_t i = 0; i < set->count; i++) {
    if (set->positions[i] == pos) {
      return true;
    }
  }
  return false;
}

static bool position_set_add(position_set_t* set, size_t row, size_t col,
                             size_t cols) {
  if (set->count >= set->capacity) {
    return false;
  }
  size_t pos = (row * cols) + col;
  set->positions[set->count++] = pos;
  return true;
}

void matgen_random_config_init(matgen_random_config_t* config, size_t rows,
                               size_t cols, size_t nnz) {
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

  config->seed = 0;

  config->allow_duplicates = false;
  config->sorted = true;
}

matgen_coo_matrix_t* matgen_random_coo_create(
    const matgen_random_config_t* config) {
  if (!config || config->rows == 0 || config->cols == 0) {
    return NULL;
  }

  // Calculate nnz from density if specified
  size_t nnz = config->nnz;
  if (config->density > 0.0) {
    nnz =
        (size_t)(config->density * (double)config->rows * (double)config->cols);
    if (nnz == 0) {
      nnz = 1;
    }
  }

  // Check if requested nnz is possible without duplicates
  size_t max_nnz = config->rows * config->cols;
  if (!config->allow_duplicates && nnz > max_nnz) {
    return NULL;
  }

  // Initialize RNG
  rng_state_t rng;
  rng_init(&rng, config->seed);

  // Create matrix
  matgen_coo_matrix_t* matrix =
      matgen_coo_create(config->rows, config->cols, nnz);
  if (!matrix) {
    return NULL;
  }

  if (config->allow_duplicates) {
    // Simple case: allow duplicates
    for (size_t i = 0; i < nnz; i++) {
      size_t row = (size_t)(rng_uniform_01(&rng) * (double)config->rows);
      size_t col = (size_t)(rng_uniform_01(&rng) * (double)config->cols);
      double value = generate_value(&rng, config);

      if (matgen_coo_add_entry(matrix, row, col, value) != 0) {
        matgen_coo_destroy(matrix);
        return NULL;
      }
    }
  } else {
    // No duplicates: track positions
    position_set_t* positions = position_set_create(nnz);
    if (!positions) {
      matgen_coo_destroy(matrix);
      return NULL;
    }

    size_t attempts = 0;
    size_t max_attempts = nnz * 100;  // Avoid infinite loop

    while (matrix->nnz < nnz && attempts < max_attempts) {
      size_t row = (size_t)(rng_uniform_01(&rng) * (double)config->rows);
      size_t col = (size_t)(rng_uniform_01(&rng) * (double)config->cols);

      if (!position_set_contains(positions, row, col, config->cols)) {
        double value = generate_value(&rng, config);

        if (matgen_coo_add_entry(matrix, row, col, value) != 0) {
          position_set_destroy(positions);
          matgen_coo_destroy(matrix);
          return NULL;
        }

        position_set_add(positions, row, col, config->cols);
      }

      attempts++;
    }

    position_set_destroy(positions);

    // Check if we generated enough entries
    if (matrix->nnz < nnz) {
      matgen_coo_destroy(matrix);
      return NULL;
    }
  }

  // Sort if requested
  if (config->sorted) {
    matgen_coo_sort(matrix);
  }

  return matrix;
}

matgen_coo_matrix_t* matgen_random_diagonal(size_t rows, size_t cols,
                                            matgen_distribution_t distribution,
                                            double min_value, double max_value,
                                            unsigned int seed) {
  size_t diag_size = rows < cols ? rows : cols;

  matgen_random_config_t config;
  matgen_random_config_init(&config, rows, cols, diag_size);

  config.distribution = distribution;
  config.min_value = min_value;
  config.max_value = max_value;
  config.mean = min_value;
  config.stddev = max_value;
  config.seed = seed;
  config.allow_duplicates = false;
  config.sorted = true;

  // Initialize RNG
  rng_state_t rng;
  rng_init(&rng, seed);

  // Create matrix
  matgen_coo_matrix_t* matrix = matgen_coo_create(rows, cols, diag_size);
  if (!matrix) {
    return NULL;
  }

  // Add diagonal entries
  for (size_t i = 0; i < diag_size; i++) {
    double value = generate_value(&rng, &config);
    if (matgen_coo_add_entry(matrix, i, i, value) != 0) {
      matgen_coo_destroy(matrix);
      return NULL;
    }
  }

  matgen_coo_sort(matrix);
  return matrix;
}

matgen_coo_matrix_t* matgen_random_tridiagonal(
    size_t size, matgen_distribution_t distribution, double min_value,
    double max_value, unsigned int seed) {
  if (size == 0) {
    return NULL;
  }

  size_t nnz = size;  // Main diagonal
  if (size > 1) {
    nnz += 2 * (size - 1);  // Upper and lower diagonals
  }

  matgen_random_config_t config;
  matgen_random_config_init(&config, size, size, nnz);

  config.distribution = distribution;
  config.min_value = min_value;
  config.max_value = max_value;
  config.mean = min_value;
  config.stddev = max_value;
  config.seed = seed;

  // Initialize RNG
  rng_state_t rng;
  rng_init(&rng, seed);

  // Create matrix
  matgen_coo_matrix_t* matrix = matgen_coo_create(size, size, nnz);
  if (!matrix) {
    return NULL;
  }

  // Add main diagonal
  for (size_t i = 0; i < size; i++) {
    double value = generate_value(&rng, &config);
    if (matgen_coo_add_entry(matrix, i, i, value) != 0) {
      matgen_coo_destroy(matrix);
      return NULL;
    }
  }

  // Add upper diagonal
  for (size_t i = 0; i < size - 1; i++) {
    double value = generate_value(&rng, &config);
    if (matgen_coo_add_entry(matrix, i, i + 1, value) != 0) {
      matgen_coo_destroy(matrix);
      return NULL;
    }
  }

  // Add lower diagonal
  for (size_t i = 0; i < size - 1; i++) {
    double value = generate_value(&rng, &config);
    if (matgen_coo_add_entry(matrix, i + 1, i, value) != 0) {
      matgen_coo_destroy(matrix);
      return NULL;
    }
  }

  matgen_coo_sort(matrix);
  return matrix;
}
