#include "matgen/core/csr_matrix.h"

#include <stdio.h>
#include <stdlib.h>

#include "matgen/utils/log.h"

// =============================================================================
// Internal Helper Functions
// =============================================================================

// Binary search for column index within a row
static matgen_size_t binary_search_col(const matgen_index_t* col_indices,
                                       matgen_size_t start, matgen_size_t end,
                                       matgen_index_t target_col) {
  while (start < end) {
    matgen_size_t mid = start + ((end - start) / 2);
    if (col_indices[mid] == target_col) {
      return mid;  // Found
    }

    if (col_indices[mid] < target_col) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return (matgen_size_t)-1;  // Not found
}

// =============================================================================
// Creation and Destruction
// =============================================================================

matgen_csr_matrix_t* matgen_csr_create(matgen_index_t rows, matgen_index_t cols,
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
  // row_ptr has rows+1 elements (last element points past the end)
  matrix->row_ptr = (matgen_size_t*)calloc(rows + 1, sizeof(matgen_size_t));
  matrix->col_indices = (matgen_index_t*)malloc(nnz * sizeof(matgen_index_t));
  matrix->values = (matgen_value_t*)malloc(nnz * sizeof(matgen_value_t));

  if (!matrix->row_ptr ||
      (nnz > 0 && (!matrix->col_indices || !matrix->values))) {
    MATGEN_LOG_ERROR("Failed to allocate CSR matrix arrays");
    matgen_csr_destroy(matrix);
    return NULL;
  }

  MATGEN_LOG_DEBUG("Created CSR matrix %llu x %llu with %zu non-zeros",
                   (unsigned long long)rows, (unsigned long long)cols, nnz);

  return matrix;
}

void matgen_csr_destroy(matgen_csr_matrix_t* matrix) {
  if (!matrix) {
    return;
  }

  MATGEN_LOG_DEBUG("Destroying CSR matrix %llu x %llu (nnz: %zu)",
                   (unsigned long long)matrix->rows,
                   (unsigned long long)matrix->cols, matrix->nnz);

  free(matrix->row_ptr);
  free(matrix->col_indices);
  free(matrix->values);
  free(matrix);
}

// =============================================================================
// Matrix Access
// =============================================================================

matgen_error_t matgen_csr_get(const matgen_csr_matrix_t* matrix,
                              matgen_index_t row, matgen_index_t col,
                              matgen_value_t* value) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (row >= matrix->rows || col >= matrix->cols) {
    MATGEN_LOG_ERROR("Index out of bounds: (%llu, %llu) for %llu x %llu matrix",
                     (unsigned long long)row, (unsigned long long)col,
                     (unsigned long long)matrix->rows,
                     (unsigned long long)matrix->cols);
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Get row bounds
  matgen_size_t row_start = matrix->row_ptr[row];
  matgen_size_t row_end = matrix->row_ptr[row + 1];

  if (row_start == row_end) {
    // Empty row
    if (value) {
      *value = 0.0;
    }

    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Binary search for column within row
  matgen_size_t idx =
      binary_search_col(matrix->col_indices, row_start, row_end, col);

  if (idx == (matgen_size_t)-1) {
    // Not found
    if (value) {
      *value = 0.0;
    }

    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Found
  if (value) {
    *value = matrix->values[idx];
  }
  return MATGEN_SUCCESS;
}

bool matgen_csr_has_entry(const matgen_csr_matrix_t* matrix, matgen_index_t row,
                          matgen_index_t col) {
  if (!matrix || row >= matrix->rows || col >= matrix->cols) {
    return false;
  }

  matgen_size_t row_start = matrix->row_ptr[row];
  matgen_size_t row_end = matrix->row_ptr[row + 1];

  if (row_start == row_end) {
    return false;  // Empty row
  }

  matgen_size_t idx =
      binary_search_col(matrix->col_indices, row_start, row_end, col);
  return (idx != (matgen_size_t)-1);
}

matgen_size_t matgen_csr_row_nnz(const matgen_csr_matrix_t* matrix,
                                 matgen_index_t row) {
  if (!matrix || row >= matrix->rows) {
    MATGEN_LOG_ERROR("Invalid matrix or row index");
    return 0;
  }

  return matrix->row_ptr[row + 1] - matrix->row_ptr[row];
}

matgen_error_t matgen_csr_get_row_range(const matgen_csr_matrix_t* matrix,
                                        matgen_index_t row,
                                        matgen_size_t* row_start,
                                        matgen_size_t* row_end) {
  if (!matrix || !row_start || !row_end) {
    MATGEN_LOG_ERROR("NULL pointer argument");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (row >= matrix->rows) {
    MATGEN_LOG_ERROR("Row index %llu out of bounds (rows: %llu)",
                     (unsigned long long)row, (unsigned long long)matrix->rows);
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  *row_start = matrix->row_ptr[row];
  *row_end = matrix->row_ptr[row + 1];

  return MATGEN_SUCCESS;
}

// =============================================================================
// Utility Functions
// =============================================================================

void matgen_csr_print_info(const matgen_csr_matrix_t* matrix, FILE* stream) {
  if (!matrix || !stream) {
    return;
  }

  matgen_value_t sparsity = (matgen_value_t)0.0;
  if (matrix->rows > 0 && matrix->cols > 0) {
    u64 total_elements = (u64)matrix->rows * (u64)matrix->cols;
    sparsity = ((matgen_value_t)100.0 * (matgen_value_t)matrix->nnz) /
               (matgen_value_t)total_elements;
  }

  fprintf(stream, "CSR Matrix Information:\n");
  fprintf(stream, "  Dimensions: %llu x %llu\n",
          (unsigned long long)matrix->rows, (unsigned long long)matrix->cols);
  fprintf(stream, "  Non-zeros:  %zu\n", matrix->nnz);
  fprintf(stream, "  Sparsity:   %.4f%%\n", sparsity);
  fprintf(stream, "  Memory:     %zu bytes\n", matgen_csr_memory_usage(matrix));
}

matgen_size_t matgen_csr_memory_usage(const matgen_csr_matrix_t* matrix) {
  if (!matrix) {
    return 0;
  }

  matgen_size_t memory = sizeof(matgen_csr_matrix_t);
  memory += (matrix->rows + 1) * sizeof(matgen_size_t);  // row_ptr
  memory += matrix->nnz * sizeof(matgen_index_t);        // col_indices
  memory += matrix->nnz * sizeof(matgen_value_t);        // values

  return memory;
}

bool matgen_csr_validate(const matgen_csr_matrix_t* matrix) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return false;
  }

  // Check basic properties
  if (matrix->rows == 0 || matrix->cols == 0) {
    MATGEN_LOG_ERROR("Invalid dimensions: %llu x %llu",
                     (unsigned long long)matrix->rows,
                     (unsigned long long)matrix->cols);
    return false;
  }

  if (!matrix->row_ptr) {
    MATGEN_LOG_ERROR("NULL row_ptr array");
    return false;
  }

  if (matrix->nnz > 0 && (!matrix->col_indices || !matrix->values)) {
    MATGEN_LOG_ERROR("NULL arrays with nnz = %zu", matrix->nnz);
    return false;
  }

  // Check row_ptr[0] == 0
  if (matrix->row_ptr[0] != 0) {
    MATGEN_LOG_ERROR("row_ptr[0] = %zu, expected 0", matrix->row_ptr[0]);
    return false;
  }

  // Check row_ptr[rows] == nnz
  if (matrix->row_ptr[matrix->rows] != matrix->nnz) {
    MATGEN_LOG_ERROR("row_ptr[%llu] = %zu, expected %zu",
                     (unsigned long long)matrix->rows,
                     matrix->row_ptr[matrix->rows], matrix->nnz);
    return false;
  }

  // Check row_ptr is monotonically increasing
  for (matgen_index_t i = 0; i < matrix->rows; i++) {
    if (matrix->row_ptr[i] > matrix->row_ptr[i + 1]) {
      MATGEN_LOG_ERROR("row_ptr not monotonic at row %llu: %zu > %zu",
                       (unsigned long long)i, matrix->row_ptr[i],
                       matrix->row_ptr[i + 1]);
      return false;
    }
  }

  // Check all column indices are in bounds and sorted within rows
  for (matgen_index_t i = 0; i < matrix->rows; i++) {
    matgen_size_t row_start = matrix->row_ptr[i];
    matgen_size_t row_end = matrix->row_ptr[i + 1];

    for (matgen_size_t j = row_start; j < row_end; j++) {
      // Check column in bounds
      if (matrix->col_indices[j] >= matrix->cols) {
        MATGEN_LOG_ERROR("Column index %llu out of bounds in row %llu",
                         (unsigned long long)matrix->col_indices[j],
                         (unsigned long long)i);
        return false;
      }

      // Check sorted within row
      if (j > row_start &&
          matrix->col_indices[j] <= matrix->col_indices[j - 1]) {
        MATGEN_LOG_ERROR("Column indices not sorted in row %llu",
                         (unsigned long long)i);
        return false;
      }
    }
  }

  return true;
}
