#include "backends/omp/internal/csr_omp.h"

#include <omp.h>
#include <stdlib.h>

#include "matgen/core/matrix/csr.h"
#include "matgen/utils/log.h"

// =============================================================================
// OpenMP Parallel CSR Operations
// =============================================================================

matgen_csr_matrix_t* matgen_csr_create_omp(matgen_index_t rows,
                                           matgen_index_t cols,
                                           matgen_size_t nnz) {
  if (rows == 0 || cols == 0) {
    MATGEN_LOG_ERROR("Invalid matrix dimensions: %llu x %llu",
                     (unsigned long long)rows, (unsigned long long)cols);
    return NULL;
  }

  matgen_csr_matrix_t* matrix =
      (matgen_csr_matrix_t*)malloc(sizeof(matgen_csr_matrix_t));
  if (!matrix) {
    MATGEN_LOG_ERROR("Failed to allocate CSR matrix structure");
    return NULL;
  }

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = nnz;

  // Allocate arrays
  matrix->row_ptr = (matgen_size_t*)calloc(rows + 1, sizeof(matgen_size_t));
  matrix->col_indices = (matgen_index_t*)malloc(nnz * sizeof(matgen_index_t));
  matrix->values = (matgen_value_t*)malloc(nnz * sizeof(matgen_value_t));

  if (!matrix->row_ptr ||
      (nnz > 0 && (!matrix->col_indices || !matrix->values))) {
    MATGEN_LOG_ERROR("Failed to allocate CSR matrix arrays");
    matgen_csr_destroy(matrix);
    return NULL;
  }

  int i;

  // Optionally: parallel initialization of large arrays
  if (nnz > 100000) {
#pragma omp parallel for
    for (i = 0; i < nnz; i++) {
      matrix->col_indices[i] = 0;
      matrix->values[i] = (matgen_value_t)0.0;
    }
  }

  MATGEN_LOG_DEBUG("Created CSR matrix (OMP) %llu x %llu with %zu non-zeros",
                   (unsigned long long)rows, (unsigned long long)cols, nnz);

  return matrix;
}
