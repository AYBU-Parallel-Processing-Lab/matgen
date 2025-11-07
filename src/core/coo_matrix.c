#include "matgen/core/coo_matrix.h"

#include <stdio.h>
#include <stdlib.h>

#include "matgen/utils/log.h"

// =============================================================================
// Configuration
// =============================================================================

// Initial capacity if no hint provided
#define DEFAULT_INITIAL_CAPACITY 1024

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
    // Note: realloc failure leaves original pointers valid
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  matrix->row_indices = new_rows;
  matrix->col_indices = new_cols;
  matrix->values = new_vals;
  matrix->capacity = new_capacity;

  return MATGEN_SUCCESS;
}

// Partition function for quicksort
static matgen_size_t partition(matgen_coo_matrix_t* matrix, matgen_size_t low,
                               matgen_size_t high) {
  matgen_index_t pivot_row = matrix->row_indices[high];
  matgen_index_t pivot_col = matrix->col_indices[high];

  matgen_size_t i = low;  // Position for next element <= pivot

  for (matgen_size_t j = low; j < high; j++) {
    if (matrix->row_indices[j] < pivot_row ||
        (matrix->row_indices[j] == pivot_row &&
         matrix->col_indices[j] < pivot_col)) {
      // Swap elements at i and j
      if (i != j) {
        matgen_index_t temp_row = matrix->row_indices[i];
        matgen_index_t temp_col = matrix->col_indices[i];
        matgen_value_t temp_val = matrix->values[i];

        matrix->row_indices[i] = matrix->row_indices[j];
        matrix->col_indices[i] = matrix->col_indices[j];
        matrix->values[i] = matrix->values[j];

        matrix->row_indices[j] = temp_row;
        matrix->col_indices[j] = temp_col;
        matrix->values[j] = temp_val;
      }
      i++;
    }
  }

  // Place pivot in correct position
  if (i != high) {
    matgen_index_t temp_row = matrix->row_indices[i];
    matgen_index_t temp_col = matrix->col_indices[i];
    matgen_value_t temp_val = matrix->values[i];

    matrix->row_indices[i] = matrix->row_indices[high];
    matrix->col_indices[i] = matrix->col_indices[high];
    matrix->values[i] = matrix->values[high];

    matrix->row_indices[high] = temp_row;
    matrix->col_indices[high] = temp_col;
    matrix->values[high] = temp_val;
  }

  return i;
}

// Quicksort implementation for COO matrix (without recursion)
static void quicksort_coo(matgen_coo_matrix_t* matrix, matgen_size_t low,
                          matgen_size_t high) {
  if (low >= high) {
    return;
  }

  // Allocate stack for subarray bounds
  struct {
    matgen_size_t low;
    matgen_size_t high;
  }* stack = malloc((high - low + 1) * sizeof(*stack));

  if (!stack) {
    return;  // Allocation failed
  }

  matgen_size_t stack_top = 0;

  // Push initial range
  stack[stack_top].low = low;
  stack[stack_top].high = high;
  stack_top++;

  while (stack_top > 0) {
    // Pop the next range
    stack_top--;
    matgen_size_t curr_low = stack[stack_top].low;
    matgen_size_t curr_high = stack[stack_top].high;

    if (curr_low < curr_high) {
      matgen_size_t pi = partition(matrix, curr_low, curr_high);

      // Push larger subarray first to minimize stack depth
      matgen_size_t left_size = pi - curr_low;
      matgen_size_t right_size = curr_high - pi;

      if (left_size > right_size) {
        // Push left first (larger)
        if (pi > 0) {
          stack[stack_top].low = curr_low;
          stack[stack_top].high = pi - 1;
          stack_top++;
        }
        // Then push right
        stack[stack_top].low = pi + 1;
        stack[stack_top].high = curr_high;
        stack_top++;
      } else {
        // Push right first (larger or equal)
        stack[stack_top].low = pi + 1;
        stack[stack_top].high = curr_high;
        stack_top++;
        // Then push left
        if (pi > 0) {
          stack[stack_top].low = curr_low;
          stack[stack_top].high = pi - 1;
          stack_top++;
        }
      }
    }
  }

  free(stack);
}

// =============================================================================
// Creation and Destruction
// =============================================================================

matgen_coo_matrix_t* matgen_coo_create(matgen_index_t rows, matgen_index_t cols,
                                       matgen_size_t nnz_hint) {
  if (rows == 0 || cols == 0) {
    MATGEN_LOG_ERROR("Invalid matrix dimensions: %llu x %llu",
                     (unsigned long long)rows, (unsigned long long)cols);
    return NULL;
  }

  matgen_coo_matrix_t* matrix =
      (matgen_coo_matrix_t*)malloc(sizeof(matgen_coo_matrix_t));
  if (!matrix) {
    MATGEN_LOG_ERROR("Failed to allocate COO matrix structure");
    return NULL;
  }

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = 0;
  matrix->capacity = (nnz_hint > 0) ? nnz_hint : DEFAULT_INITIAL_CAPACITY;
  matrix->is_sorted = true;  // Empty matrix is trivially sorted

  // Allocate arrays
  matrix->row_indices =
      (matgen_index_t*)malloc(matrix->capacity * sizeof(matgen_index_t));
  matrix->col_indices =
      (matgen_index_t*)malloc(matrix->capacity * sizeof(matgen_index_t));
  matrix->values =
      (matgen_value_t*)malloc(matrix->capacity * sizeof(matgen_value_t));

  if (!matrix->row_indices || !matrix->col_indices || !matrix->values) {
    MATGEN_LOG_ERROR("Failed to allocate COO matrix arrays");
    matgen_coo_destroy(matrix);
    return NULL;
  }

  MATGEN_LOG_DEBUG("Created COO matrix %llu x %llu with capacity %zu",
                   (unsigned long long)rows, (unsigned long long)cols,
                   matrix->capacity);

  return matrix;
}

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
        (matgen_size_t)((double)matrix->capacity * GROWTH_FACTOR) + 1;
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

  // Matrix is no longer sorted after adding (unless we can prove otherwise)
  matrix->is_sorted = false;

  MATGEN_LOG_TRACE("Added entry at (%llu, %llu) = %f, nnz now: %zu",
                   (unsigned long long)row, (unsigned long long)col, value,
                   matrix->nnz);

  return MATGEN_SUCCESS;
}

matgen_error_t matgen_coo_sort(matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (matrix->is_sorted || matrix->nnz <= 1) {
    MATGEN_LOG_DEBUG("Matrix already sorted or trivial (nnz: %zu)",
                     matrix->nnz);
    matrix->is_sorted = true;
    return MATGEN_SUCCESS;
  }

  MATGEN_LOG_DEBUG("Sorting COO matrix with %zu entries", matrix->nnz);

  quicksort_coo(matrix, 0, matrix->nnz - 1);

  matrix->is_sorted = true;

  MATGEN_LOG_DEBUG("Matrix sorted successfully");

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

  // Use binary search if matrix is sorted
  if (matrix->is_sorted && matrix->nnz > 0) {
    matgen_size_t left = 0;
    matgen_size_t right = matrix->nnz;

    while (left < right) {
      matgen_size_t mid = left + ((right - left) / 2);
      matgen_index_t mid_row = matrix->row_indices[mid];
      matgen_index_t mid_col = matrix->col_indices[mid];

      // Lexicographic comparison: compare row first, then column
      if (mid_row < row || (mid_row == row && mid_col < col)) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }

    // Check if element was found at position 'left'
    if (left < matrix->nnz && matrix->row_indices[left] == row &&
        matrix->col_indices[left] == col) {
      if (value) {
        *value = matrix->values[left];
      }
      return MATGEN_SUCCESS;
    }
  } else {
    // Fallback to linear search for unsorted matrices
    for (matgen_size_t i = 0; i < matrix->nnz; i++) {
      if (matrix->row_indices[i] == row && matrix->col_indices[i] == col) {
        if (value) {
          *value = matrix->values[i];
        }
        return MATGEN_SUCCESS;
      }
    }
  }

  // Element not found - return 0.0
  if (value) {
    *value = 0.0;
  }
  return MATGEN_ERROR_INVALID_ARGUMENT;
}

bool matgen_coo_has_entry(const matgen_coo_matrix_t* matrix, matgen_index_t row,
                          matgen_index_t col) {
  if (!matrix || row >= matrix->rows || col >= matrix->cols) {
    return false;
  }

  for (matgen_size_t i = 0; i < matrix->nnz; i++) {
    if (matrix->row_indices[i] == row && matrix->col_indices[i] == col) {
      return true;
    }
  }

  return false;
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
    // Already have enough capacity
    return MATGEN_SUCCESS;
  }

  return coo_resize(matrix, capacity);
}

void matgen_coo_clear(matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    return;
  }

  MATGEN_LOG_DEBUG("Clearing COO matrix (was: %zu entries)", matrix->nnz);

  matrix->nnz = 0;
  matrix->is_sorted = true;  // Empty matrix is sorted
}

void matgen_coo_print_info(const matgen_coo_matrix_t* matrix, FILE* stream) {
  if (!matrix || !stream) {
    return;
  }

  double sparsity = 0.0;
  if (matrix->rows > 0 && matrix->cols > 0) {
    u64 total_elements = (u64)matrix->rows * (u64)matrix->cols;
    sparsity = (100.0 * (double)matrix->nnz) / (double)total_elements;
  }

  fprintf(stream, "COO Matrix Information:\n");
  fprintf(stream, "  Dimensions: %llu x %llu\n",
          (unsigned long long)matrix->rows, (unsigned long long)matrix->cols);
  fprintf(stream, "  Non-zeros:  %zu\n", matrix->nnz);
  fprintf(stream, "  Capacity:   %zu\n", matrix->capacity);
  fprintf(stream, "  Sparsity:   %.4f%%\n", sparsity);
  fprintf(stream, "  Sorted:     %s\n", matrix->is_sorted ? "yes" : "no");
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

  // Check basic properties
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

  // Check arrays are allocated if nnz > 0
  if (matrix->nnz > 0) {
    if (!matrix->row_indices || !matrix->col_indices || !matrix->values) {
      MATGEN_LOG_ERROR("NULL arrays with nnz = %zu", matrix->nnz);
      return false;
    }
  }

  // Check all indices are in bounds
  for (matgen_size_t i = 0; i < matrix->nnz; i++) {
    if (matrix->row_indices[i] >= matrix->rows) {
      MATGEN_LOG_ERROR("Row index %llu out of bounds at entry %zu",
                       (unsigned long long)matrix->row_indices[i], i);
      return false;
    }
    if (matrix->col_indices[i] >= matrix->cols) {
      MATGEN_LOG_ERROR("Column index %llu out of bounds at entry %zu",
                       (unsigned long long)matrix->col_indices[i], i);
      return false;
    }
  }

  return true;
}
