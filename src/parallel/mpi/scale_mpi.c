#include "scale_mpi.h"

#include "matgen/core/matrix_scale.h"

#ifdef MATGEN_HAS_MPI

// Main dispatcher for MPI scaling
matgen_coo_matrix_t* matgen_matrix_scale_mpi(
    const matgen_coo_matrix_t* input, const matgen_scale_config_t* config) {
  if (!input || !config) {
    return NULL;
  }

  switch (config->method) {
    case MATGEN_SCALE_NEAREST:
      return matgen_matrix_scale_nearest_mpi(input, config->new_rows,
                                             config->new_cols, MPI_COMM_WORLD);

    case MATGEN_SCALE_BILINEAR:
      return matgen_matrix_scale_bilinear_mpi(
          input, config->new_rows, config->new_cols, config->sparsity_threshold,
          MPI_COMM_WORLD);

    default:
      return NULL;
  }
}

#endif  // MATGEN_HAS_MPI
