#include "scale_omp.h"

#include "matgen/core/matrix_scale.h"

#ifdef MATGEN_HAS_OPENMP

// Main dispatcher for OpenMP scaling
matgen_coo_matrix_t* matgen_matrix_scale_omp(
    const matgen_coo_matrix_t* input, const matgen_scale_config_t* config) {
  if (!input || !config) {
    return NULL;
  }

  switch (config->method) {
    case MATGEN_SCALE_NEAREST:
      return matgen_matrix_scale_nearest_omp(
          input, config->new_rows, config->new_cols, config->num_threads);

    case MATGEN_SCALE_BILINEAR:
      return matgen_matrix_scale_bilinear_omp(
          input, config->new_rows, config->new_cols, config->sparsity_threshold,
          config->num_threads);

    default:
      return NULL;
  }
}

#endif  // MATGEN_HAS_OPENMP
