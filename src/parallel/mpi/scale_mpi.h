#ifndef MATGEN_PARALLEL_MPI_SCALE_H
#define MATGEN_PARALLEL_MPI_SCALE_H

#include <mpi.h>

#include "matgen/core/coo_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

matgen_coo_matrix_t* matgen_matrix_scale_bilinear_mpi(
    const matgen_coo_matrix_t* input, size_t new_rows, size_t new_cols,
    double sparsity_threshold, MPI_Comm comm);

matgen_coo_matrix_t* matgen_matrix_scale_nearest_mpi(
    const matgen_coo_matrix_t* input, size_t new_rows, size_t new_cols,
    MPI_Comm comm);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_PARALLEL_MPI_SCALE_H
