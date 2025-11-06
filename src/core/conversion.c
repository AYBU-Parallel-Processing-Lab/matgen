#include "matgen/core/conversion.h"

#include <string.h>

#include "matgen/utils/log.h"

// =============================================================================
// COO to CSR Conversion
// =============================================================================

matgen_csr_matrix_t* matgen_coo_to_csr(const matgen_coo_matrix_t* coo) {
  if (!coo) {
    MATGEN_LOG_ERROR("NULL COO matrix pointer");
    return NULL;
  }

  if (!matgen_coo_validate(coo)) {
    MATGEN_LOG_ERROR("Invalid COO matrix");
    return NULL;
  }

  MATGEN_LOG_DEBUG("Converting COO (%llu x %llu, nnz=%zu) to CSR",
                   (unsigned long long)coo->rows, (unsigned long long)coo->cols,
                   coo->nnz);

  // Create CSR matrix
  matgen_csr_matrix_t* csr = matgen_csr_create(coo->rows, coo->cols, coo->nnz);
  if (!csr) {
    return NULL;
  }

  // Handle empty matrix
  if (coo->nnz == 0) {
    MATGEN_LOG_DEBUG("Empty matrix, conversion trivial");
    return csr;
  }

  // If COO is not sorted, we need to sort it first
  // Make a copy to avoid modifying the input
  matgen_coo_matrix_t* coo_sorted = NULL;
  const matgen_coo_matrix_t* coo_to_use = coo;

  if (!coo->is_sorted) {
    MATGEN_LOG_DEBUG("COO matrix not sorted, creating sorted copy");
    coo_sorted = matgen_coo_create(coo->rows, coo->cols, coo->nnz);
    if (!coo_sorted) {
      matgen_csr_destroy(csr);
      return NULL;
    }

    // Copy data
    memcpy(coo_sorted->row_indices, coo->row_indices,
           coo->nnz * sizeof(matgen_index_t));
    memcpy(coo_sorted->col_indices, coo->col_indices,
           coo->nnz * sizeof(matgen_index_t));
    memcpy(coo_sorted->values, coo->values, coo->nnz * sizeof(matgen_value_t));
    coo_sorted->nnz = coo->nnz;

    // Sort the copy
    if (matgen_coo_sort(coo_sorted) != MATGEN_SUCCESS) {
      MATGEN_LOG_ERROR("Failed to sort COO matrix");
      matgen_coo_destroy(coo_sorted);
      matgen_csr_destroy(csr);
      return NULL;
    }

    coo_to_use = coo_sorted;
  }

  // Count non-zeros per row
  for (matgen_size_t i = 0; i < coo_to_use->nnz; i++) {
    matgen_index_t row = coo_to_use->row_indices[i];
    csr->row_ptr[row + 1]++;
  }

  // Convert counts to cumulative sum (prefix sum)
  for (matgen_index_t i = 0; i < coo_to_use->rows; i++) {
    csr->row_ptr[i + 1] += csr->row_ptr[i];
  }

  // Fill in column indices and values
  for (matgen_size_t i = 0; i < coo_to_use->nnz; i++) {
    matgen_index_t row = coo_to_use->row_indices[i];
    matgen_size_t dest = csr->row_ptr[row];

    csr->col_indices[dest] = coo_to_use->col_indices[i];
    csr->values[dest] = coo_to_use->values[i];

    csr->row_ptr[row]++;
  }

  // Restore row_ptr (shift back)
  for (matgen_index_t i = coo_to_use->rows; i > 0; i--) {
    csr->row_ptr[i] = csr->row_ptr[i - 1];
  }
  csr->row_ptr[0] = 0;

  // Clean up sorted copy if we made one
  if (coo_sorted) {
    matgen_coo_destroy(coo_sorted);
  }

  MATGEN_LOG_DEBUG("COO to CSR conversion complete");

  return csr;
}

// =============================================================================
// CSR to COO Conversion
// =============================================================================

matgen_coo_matrix_t* matgen_csr_to_coo(const matgen_csr_matrix_t* csr) {
  if (!csr) {
    MATGEN_LOG_ERROR("NULL CSR matrix pointer");
    return NULL;
  }

  if (!matgen_csr_validate(csr)) {
    MATGEN_LOG_ERROR("Invalid CSR matrix");
    return NULL;
  }

  MATGEN_LOG_DEBUG("Converting CSR (%llu x %llu, nnz=%zu) to COO",
                   (unsigned long long)csr->rows, (unsigned long long)csr->cols,
                   csr->nnz);

  // Create COO matrix
  matgen_coo_matrix_t* coo = matgen_coo_create(csr->rows, csr->cols, csr->nnz);
  if (!coo) {
    return NULL;
  }

  // Handle empty matrix
  if (csr->nnz == 0) {
    MATGEN_LOG_DEBUG("Empty matrix, conversion trivial");
    return coo;
  }

  // Convert: iterate through each row
  matgen_size_t coo_idx = 0;
  for (matgen_index_t row = 0; row < csr->rows; row++) {
    matgen_size_t row_start = csr->row_ptr[row];
    matgen_size_t row_end = csr->row_ptr[row + 1];

    for (matgen_size_t j = row_start; j < row_end; j++) {
      coo->row_indices[coo_idx] = row;
      coo->col_indices[coo_idx] = csr->col_indices[j];
      coo->values[coo_idx] = csr->values[j];
      coo_idx++;
    }
  }

  coo->nnz = csr->nnz;
  coo->is_sorted = true;  // CSR is always sorted

  MATGEN_LOG_DEBUG("CSR to COO conversion complete");

  return coo;
}
