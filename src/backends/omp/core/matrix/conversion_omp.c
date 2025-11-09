#include "backends/omp/internal/conversion_omp.h"

#include <omp.h>
#include <string.h>

#include "matgen/core/execution/policy.h"
#include "matgen/utils/log.h"

// =============================================================================
// COO to CSR Conversion (OpenMP Parallel)
// =============================================================================

matgen_csr_matrix_t* matgen_coo_to_csr_omp(const matgen_coo_matrix_t* coo) {
  if (!coo) {
    MATGEN_LOG_ERROR("NULL COO matrix pointer");
    return NULL;
  }

  if (!matgen_coo_validate(coo)) {
    MATGEN_LOG_ERROR("Invalid COO matrix");
    return NULL;
  }

  MATGEN_LOG_DEBUG("Converting COO (%llu x %llu, nnz=%zu) to CSR (OMP)",
                   (unsigned long long)coo->rows, (unsigned long long)coo->cols,
                   coo->nnz);

  // Create CSR matrix using OMP backend
  matgen_csr_matrix_t* csr = matgen_csr_create_with_policy(
      coo->rows, coo->cols, coo->nnz, MATGEN_EXEC_PAR);
  if (!csr) {
    return NULL;
  }

  if (coo->nnz == 0) {
    MATGEN_LOG_DEBUG("Empty matrix, conversion trivial");
    return csr;
  }

  // If COO is not sorted, sort it first
  matgen_coo_matrix_t* coo_sorted = NULL;
  const matgen_coo_matrix_t* coo_to_use = coo;

  if (!coo->is_sorted) {
    MATGEN_LOG_DEBUG("COO matrix not sorted, creating sorted copy");
    coo_sorted = matgen_coo_create(coo->rows, coo->cols, coo->nnz);
    if (!coo_sorted) {
      matgen_csr_destroy(csr);
      return NULL;
    }

    memcpy(coo_sorted->row_indices, coo->row_indices,
           coo->nnz * sizeof(matgen_index_t));
    memcpy(coo_sorted->col_indices, coo->col_indices,
           coo->nnz * sizeof(matgen_index_t));
    memcpy(coo_sorted->values, coo->values, coo->nnz * sizeof(matgen_value_t));
    coo_sorted->nnz = coo->nnz;
    coo_sorted->is_sorted = false;

    if (matgen_coo_sort_with_policy(coo_sorted, MATGEN_EXEC_PAR) !=
        MATGEN_SUCCESS) {
      MATGEN_LOG_ERROR("Failed to sort COO matrix");
      matgen_coo_destroy(coo_sorted);
      matgen_csr_destroy(csr);
      return NULL;
    }

    coo_to_use = coo_sorted;
  }

  int i;

// Parallel histogram (count nnz per row)
#pragma omp parallel for
  for (i = 0; i <= coo_to_use->rows; i++) {
    csr->row_ptr[i] = 0;
  }

#pragma omp parallel for
  for (i = 0; i < coo_to_use->nnz; i++) {
    matgen_index_t row = coo_to_use->row_indices[i];
#pragma omp atomic
    csr->row_ptr[row + 1]++;
  }

  // Parallel prefix sum (exclusive scan)
  // Sequential for now (parallel scan is complex)
  for (matgen_index_t i = 0; i < coo_to_use->rows; i++) {
    csr->row_ptr[i + 1] += csr->row_ptr[i];
  }

  // Create temporary array for write positions
  matgen_size_t* write_pos =
      (matgen_size_t*)malloc((coo_to_use->rows + 1) * sizeof(matgen_size_t));
  if (!write_pos) {
    MATGEN_LOG_ERROR("Failed to allocate write position array");
    matgen_coo_destroy(coo_sorted);
    matgen_csr_destroy(csr);
    return NULL;
  }

  memcpy(write_pos, csr->row_ptr,
         (coo_to_use->rows + 1) * sizeof(matgen_size_t));

  // Fill CSR arrays - must be sequential due to write_pos updates
  // Or use atomic operations for parallel
  for (matgen_size_t i = 0; i < coo_to_use->nnz; i++) {
    matgen_index_t row = coo_to_use->row_indices[i];
    matgen_size_t dest = write_pos[row]++;

    csr->col_indices[dest] = coo_to_use->col_indices[i];
    csr->values[dest] = coo_to_use->values[i];
  }

  free(write_pos);

  if (coo_sorted) {
    matgen_coo_destroy(coo_sorted);
  }

  MATGEN_LOG_DEBUG("COO to CSR conversion complete (OMP)");

  return csr;
}

// =============================================================================
// CSR to COO Conversion (OpenMP Parallel)
// =============================================================================

matgen_coo_matrix_t* matgen_csr_to_coo_omp(const matgen_csr_matrix_t* csr) {
  if (!csr) {
    MATGEN_LOG_ERROR("NULL CSR matrix pointer");
    return NULL;
  }

  if (!matgen_csr_validate(csr)) {
    MATGEN_LOG_ERROR("Invalid CSR matrix");
    return NULL;
  }

  MATGEN_LOG_DEBUG("Converting CSR (%llu x %llu, nnz=%zu) to COO (OMP)",
                   (unsigned long long)csr->rows, (unsigned long long)csr->cols,
                   csr->nnz);

  matgen_coo_matrix_t* coo = matgen_coo_create(csr->rows, csr->cols, csr->nnz);
  if (!coo) {
    return NULL;
  }

  if (csr->nnz == 0) {
    MATGEN_LOG_DEBUG("Empty matrix, conversion trivial");
    return coo;
  }

  int row;

// Parallel conversion: each thread handles a subset of rows
#pragma omp parallel for schedule(dynamic, 256)
  for (row = 0; row < csr->rows; row++) {
    matgen_size_t row_start = csr->row_ptr[row];
    matgen_size_t row_end = csr->row_ptr[row + 1];

    for (matgen_size_t j = row_start; j < row_end; j++) {
      coo->row_indices[j] = row;
      coo->col_indices[j] = csr->col_indices[j];
      coo->values[j] = csr->values[j];
    }
  }

  coo->nnz = csr->nnz;
  coo->is_sorted = true;  // CSR is always sorted

  MATGEN_LOG_DEBUG("CSR to COO conversion complete (OMP)");

  return coo;
}
