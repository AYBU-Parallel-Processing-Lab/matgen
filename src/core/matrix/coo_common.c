#include <stdio.h>
#include <stdlib.h>

#include "matgen/core/matrix/coo.h"
#include "matgen/utils/log.h"

// Growth factor when reallocating
#define GROWTH_FACTOR 1.5

// =============================================================================
// Internal Helper Functions
// =============================================================================

// Resize internal arrays
static matgen_error_t coo_resize(matgen_coo_matrix_t* matrix,
                                 matgen_size_t new_capacity) {
  MATGEN_LOG_DEBUG("Resizing COO matrix from capacity %zu to %zu",
                   matrix->capacity, new_capacity);

  matgen_index_t* new_rows = (matgen_index_t*)realloc(
      matrix->row_indices, new_capacity * sizeof(matgen_index_t));
  matgen_index_t* new_cols = (matgen_index_t*)realloc(
      matrix->col_indices, new_capacity * sizeof(matgen_index_t));
  matgen_value_t* new_vals = (matgen_value_t*)realloc(
      matrix->values, new_capacity * sizeof(matgen_value_t));

  if (!new_rows || !new_cols || !new_vals) {
    MATGEN_LOG_ERROR("Failed to resize COO matrix to capacity %zu",
                     new_capacity);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  matrix->row_indices = new_rows;
  matrix->col_indices = new_cols;
  matrix->values = new_vals;
  matrix->capacity = new_capacity;

  return MATGEN_SUCCESS;
}

// =============================================================================
// Destruction
// =============================================================================

void matgen_coo_destroy(matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    return;
  }

  MATGEN_LOG_DEBUG("Destroying COO matrix %llu x %llu (nnz: %zu)",
                   (unsigned long long)matrix->rows,
                   (unsigned long long)matrix->cols, matrix->nnz);

  free(matrix->row_indices);
  free(matrix->col_indices);
  free(matrix->values);
  free(matrix);
}

// =============================================================================
// Building the Matrix
// =============================================================================

matgen_error_t matgen_coo_add_entry(matgen_coo_matrix_t* matrix,
                                    matgen_index_t row, matgen_index_t col,
                                    matgen_value_t value) {
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

  // Grow array if needed
  if (matrix->nnz >= matrix->capacity) {
    matgen_size_t new_capacity =
        (matgen_size_t)((matgen_value_t)matrix->capacity * GROWTH_FACTOR) + 1;
    matgen_error_t err = coo_resize(matrix, new_capacity);
    if (err != MATGEN_SUCCESS) {
      return err;
    }
  }

  // Add entry
  matrix->row_indices[matrix->nnz] = row;
  matrix->col_indices[matrix->nnz] = col;
  matrix->values[matrix->nnz] = value;
  matrix->nnz++;

  // Matrix is no longer sorted after adding
  matrix->is_sorted = false;

  MATGEN_LOG_TRACE("Added entry at (%llu, %llu) = %f, nnz now: %zu",
                   (unsigned long long)row, (unsigned long long)col, value,
                   matrix->nnz);

  return MATGEN_SUCCESS;
}

// =============================================================================
// Matrix Access
// =============================================================================

matgen_error_t matgen_coo_get(const matgen_coo_matrix_t* matrix,
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

  // Linear search (inefficient for large matrices - use CSR instead)
  if (matrix->is_sorted) {
    // Binary search for sorted matrix
    matgen_size_t left = 0;
    matgen_size_t right = matrix->nnz;

    while (left < right) {
      matgen_size_t mid = left + ((right - left) / 2);
      matgen_index_t mid_row = matrix->row_indices[mid];
      matgen_index_t mid_col = matrix->col_indices[mid];

      if (mid_row < row || (mid_row == row && mid_col < col)) {
        left = mid + 1;
      } else if (mid_row > row || (mid_row == row && mid_col > col)) {
        right = mid;
      } else {
        // Found
        if (value) {
          *value = matrix->values[mid];
        }
        return MATGEN_SUCCESS;
      }
    }
  } else {
    // Linear search for unsorted matrix
    for (matgen_size_t i = 0; i < matrix->nnz; i++) {
      if (matrix->row_indices[i] == row && matrix->col_indices[i] == col) {
        if (value) {
          *value = matrix->values[i];
        }
        return MATGEN_SUCCESS;
      }
    }
  }

  // Not found
  if (value) {
    *value = (matgen_value_t)0.0;
  }
  return MATGEN_ERROR_INVALID_ARGUMENT;
}

bool matgen_coo_has_entry(const matgen_coo_matrix_t* matrix, matgen_index_t row,
                          matgen_index_t col) {
  return matgen_coo_get(matrix, row, col, NULL) == MATGEN_SUCCESS;
}

// =============================================================================
// Utility Functions
// =============================================================================

matgen_error_t matgen_coo_reserve(matgen_coo_matrix_t* matrix,
                                  matgen_size_t capacity) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (capacity <= matrix->capacity) {
    return MATGEN_SUCCESS;  // Already have enough capacity
  }

  return coo_resize(matrix, capacity);
}

void matgen_coo_clear(matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    return;
  }

  matrix->nnz = 0;
  matrix->is_sorted = true;  // Empty matrix is trivially sorted

  MATGEN_LOG_DEBUG("Cleared COO matrix (capacity remains %zu)",
                   matrix->capacity);
}

void matgen_coo_print_info(const matgen_coo_matrix_t* matrix, FILE* stream) {
  if (!matrix || !stream) {
    return;
  }

  matgen_value_t density = (matgen_value_t)matrix->nnz /
                           (matgen_value_t)(matrix->rows * matrix->cols);
  matgen_value_t sparsity = (matgen_value_t)1.0 - density;

  fprintf(stream, "COO Matrix Information:\n");
  fprintf(stream, "  Dimensions: %llu x %llu\n",
          (unsigned long long)matrix->rows, (unsigned long long)matrix->cols);
  fprintf(stream, "  Non-zeros:  %zu (capacity: %zu)\n", matrix->nnz,
          matrix->capacity);
  fprintf(stream, "  Density:    %.4f%%\n", density * 100.0);
  fprintf(stream, "  Sparsity:   %.4f%%\n", sparsity * 100.0);
  fprintf(stream, "  Sorted:     %s\n", matrix->is_sorted ? "yes" : "no");
  fprintf(stream, "  Memory:     %zu bytes\n", matgen_coo_memory_usage(matrix));
}

matgen_size_t matgen_coo_memory_usage(const matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    return 0;
  }

  matgen_size_t memory = sizeof(matgen_coo_matrix_t);
  memory += matrix->capacity * sizeof(matgen_index_t);  // row_indices
  memory += matrix->capacity * sizeof(matgen_index_t);  // col_indices
  memory += matrix->capacity * sizeof(matgen_value_t);  // values

  return memory;
}

bool matgen_coo_validate(const matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return false;
  }

  if (matrix->rows == 0 || matrix->cols == 0) {
    MATGEN_LOG_ERROR("Invalid dimensions: %llu x %llu",
                     (unsigned long long)matrix->rows,
                     (unsigned long long)matrix->cols);
    return false;
  }

  if (matrix->nnz > matrix->capacity) {
    MATGEN_LOG_ERROR("nnz (%zu) exceeds capacity (%zu)", matrix->nnz,
                     matrix->capacity);
    return false;
  }

  if (matrix->capacity > 0 &&
      (!matrix->row_indices || !matrix->col_indices || !matrix->values)) {
    MATGEN_LOG_ERROR("NULL arrays with capacity = %zu", matrix->capacity);
    return false;
  }

  // Validate indices are in bounds
  for (matgen_size_t i = 0; i < matrix->nnz; i++) {
    if (matrix->row_indices[i] >= matrix->rows) {
      MATGEN_LOG_ERROR("Row index %llu out of bounds at position %zu",
                       (unsigned long long)matrix->row_indices[i], i);
      return false;
    }
    if (matrix->col_indices[i] >= matrix->cols) {
      MATGEN_LOG_ERROR("Column index %llu out of bounds at position %zu",
                       (unsigned long long)matrix->col_indices[i], i);
      return false;
    }
  }

  // If marked as sorted, verify it
  if (matrix->is_sorted) {
    for (matgen_size_t i = 1; i < matrix->nnz; i++) {
      matgen_index_t prev_row = matrix->row_indices[i - 1];
      matgen_index_t prev_col = matrix->col_indices[i - 1];
      matgen_index_t curr_row = matrix->row_indices[i];
      matgen_index_t curr_col = matrix->col_indices[i];

      if (prev_row > curr_row ||
          (prev_row == curr_row && prev_col > curr_col)) {
        MATGEN_LOG_ERROR(
            "Matrix marked as sorted but not sorted at position %zu", i);
        return false;
      }
    }
  }

  return true;
}
