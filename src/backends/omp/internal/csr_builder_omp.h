#ifndef MATGEN_BACKENDS_OMP_INTERNAL_CSR_BUILDER_OMP_H
#define MATGEN_BACKENDS_OMP_INTERNAL_CSR_BUILDER_OMP_H

#include "matgen/core/matrix/csr_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

// OpenMP implementation functions
matgen_csr_builder_t* matgen_csr_builder_create_omp(matgen_index_t rows,
                                                    matgen_index_t cols,
                                                    matgen_size_t est_nnz);

void matgen_csr_builder_destroy_omp(matgen_csr_builder_t* builder);

matgen_error_t matgen_csr_builder_add_omp(matgen_csr_builder_t* builder,
                                          matgen_index_t row,
                                          matgen_index_t col,
                                          matgen_value_t value);

matgen_size_t matgen_csr_builder_get_nnz_omp(
    const matgen_csr_builder_t* builder);

matgen_csr_matrix_t* matgen_csr_builder_finalize_omp(
    matgen_csr_builder_t* builder);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_BACKENDS_OMP_INTERNAL_CSR_BUILDER_OMP_H
