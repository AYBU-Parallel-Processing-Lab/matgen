#include <stdlib.h>
#include <string.h>

#include "matgen/core/matrix_convert.h"
#include "matgen/util/log.h"

matgen_csr_matrix_t* matgen_coo_to_csr(matgen_coo_matrix_t* coo) {
  if (!coo) {
    MATGEN_LOG_ERROR("NULL COO matrix pointer");
    return NULL;
  }

  MATGEN_LOG_DEBUG("Converting COO to CSR (%zu x %zu, nnz: %zu)", coo->rows,
                   coo->cols, coo->nnz);

  // Sort COO if not already sorted
  if (!coo->is_sorted) {
    MATGEN_LOG_DEBUG("Sorting COO matrix before conversion");
    if (matgen_coo_sort(coo) != 0) {
      MATGEN_LOG_ERROR("Failed to sort COO matrix");
      return NULL;
    }
  }

  // Create CSR matrix
  matgen_csr_matrix_t* csr = matgen_csr_create(coo->rows, coo->cols, coo->nnz);
  if (!csr) {
    return NULL;
  }

  // Build row_ptr by counting entries in each row
  for (size_t i = 0; i < coo->nnz; i++) {
    csr->row_ptr[coo->row_indices[i] + 1]++;
  }

  // Convert counts to cumulative sums
  for (size_t i = 0; i < coo->rows; i++) {
    csr->row_ptr[i + 1] += csr->row_ptr[i];
  }

  // Copy column indices and values
  memcpy(csr->col_indices, coo->col_indices, coo->nnz * sizeof(size_t));
  memcpy(csr->values, coo->values, coo->nnz * sizeof(double));

  MATGEN_LOG_DEBUG("COO to CSR conversion complete");

  return csr;
}

matgen_coo_matrix_t* matgen_csr_to_coo(const matgen_csr_matrix_t* csr) {
  if (!csr) {
    MATGEN_LOG_ERROR("NULL CSR matrix pointer");
    return NULL;
  }

  MATGEN_LOG_DEBUG("Converting CSR to COO (%zu x %zu, nnz: %zu)", csr->rows,
                   csr->cols, csr->nnz);

  // Create COO matrix
  matgen_coo_matrix_t* coo = matgen_coo_create(csr->rows, csr->cols, csr->nnz);
  if (!coo) {
    return NULL;
  }

  // Convert CSR to COO
  size_t idx = 0;
  for (size_t i = 0; i < csr->rows; i++) {
    for (size_t j = csr->row_ptr[i]; j < csr->row_ptr[i + 1]; j++) {
      coo->row_indices[idx] = i;
      coo->col_indices[idx] = csr->col_indices[j];
      coo->values[idx] = csr->values[j];
      idx++;
    }
  }

  coo->nnz = csr->nnz;
  coo->is_sorted = true;  // CSR is always sorted

  MATGEN_LOG_DEBUG("CSR to COO conversion complete");

  return coo;
}
