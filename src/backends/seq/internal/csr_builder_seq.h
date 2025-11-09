#ifndef MATGEN_BACKENDS_SEQ_INTERNAL_CSR_BUILDER_SEQ_H
#define MATGEN_BACKENDS_SEQ_INTERNAL_CSR_BUILDER_SEQ_H

#include "matgen/core/matrix/csr_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sequential implementation functions
matgen_csr_builder_t* matgen_csr_builder_create_seq(matgen_index_t rows,
                                                    matgen_index_t cols,
                                                    matgen_size_t est_nnz);

void matgen_csr_builder_destroy_seq(matgen_csr_builder_t* builder);

matgen_error_t matgen_csr_builder_add_seq(matgen_csr_builder_t* builder,
                                          matgen_index_t row,
                                          matgen_index_t col,
                                          matgen_value_t value);

matgen_size_t matgen_csr_builder_get_nnz_seq(
    const matgen_csr_builder_t* builder);

matgen_csr_matrix_t* matgen_csr_builder_finalize_seq(
    matgen_csr_builder_t* builder);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_BACKENDS_SEQ_INTERNAL_CSR_BUILDER_SEQ_H
