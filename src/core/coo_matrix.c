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

// =============================================================================
// Optimized Sorting for COO Matrix (In-Place with Index Array)
// =============================================================================

// Structure to hold index and comparison key
typedef struct {
  matgen_size_t idx;  // Original index
  matgen_index_t row;
  matgen_index_t col;
} coo_sort_key_t;

// Comparison function for index array sorting
static int compare_coo_keys(const void* a, const void* b) {
  const coo_sort_key_t* key_a = (const coo_sort_key_t*)a;
  const coo_sort_key_t* key_b = (const coo_sort_key_t*)b;

  // Compare rows first
  if (key_a->row < key_b->row) {
    return -1;
  }

  if (key_a->row > key_b->row) {
    return 1;
  }

  // Rows equal, compare columns
  if (key_a->col < key_b->col) {
    return -1;
  }

  if (key_a->col > key_b->col) {
    return 1;
  }

  return 0;
}

// Apply permutation in-place using cycle-following algorithm
static void apply_permutation(matgen_index_t* row_indices,
                              matgen_index_t* col_indices,
                              matgen_value_t* values,
                              const coo_sort_key_t* keys, matgen_size_t n) {
  // Allocate visited array
  bool* visited = (bool*)calloc(n, sizeof(bool));
  if (!visited) {
    MATGEN_LOG_ERROR("Failed to allocate visited array for permutation");
    return;
  }

  // Apply permutation using cycle-following
  for (matgen_size_t i = 0; i < n; i++) {
    if (visited[i]) {
      continue;
    }

    matgen_size_t current = i;
    matgen_size_t next = keys[current].idx;

    if (next == current) {
      visited[current] = true;
      continue;
    }

    // Save the element being displaced
    matgen_index_t temp_row = row_indices[current];
    matgen_index_t temp_col = col_indices[current];
    matgen_value_t temp_val = values[current];

    // Follow the cycle
    while (next != i) {
      row_indices[current] = row_indices[next];
      col_indices[current] = col_indices[next];
      values[current] = values[next];

      visited[current] = true;
      current = next;
      next = keys[next].idx;
    }

    // Complete the cycle
    row_indices[current] = temp_row;
    col_indices[current] = temp_col;
    values[current] = temp_val;
    visited[current] = true;
  }

  free(visited);
}

// Fast in-place sorting using index array
static void sort_coo_inplace(matgen_coo_matrix_t* matrix) {
  if (matrix->nnz <= 1) {
    return;
  }

  // Allocate index array (much smaller than full entries)
  coo_sort_key_t* keys =
      (coo_sort_key_t*)malloc(matrix->nnz * sizeof(coo_sort_key_t));
  if (!keys) {
    MATGEN_LOG_ERROR("Failed to allocate sort keys");
    return;
  }

  // Build index array
  for (matgen_size_t i = 0; i < matrix->nnz; i++) {
    keys[i].idx = i;
    keys[i].row = matrix->row_indices[i];
    keys[i].col = matrix->col_indices[i];
  }

  // Sort the index array
  qsort(keys, matrix->nnz, sizeof(coo_sort_key_t), compare_coo_keys);

  // Apply permutation in-place
  apply_permutation(matrix->row_indices, matrix->col_indices, matrix->values,
                    keys, matrix->nnz);

  free(keys);
}

// =============================================================================
// Alternative: Radix Sort for Integer Keys (Even Faster for Large Matrices)
// =============================================================================

#define RADIX_BITS 8
#define RADIX_BUCKETS (1 << RADIX_BITS)
#define RADIX_MASK (RADIX_BUCKETS - 1)

// Encode row and column into 64-bit key for radix sort
static inline uint64_t encode_key(matgen_index_t row, matgen_index_t col) {
  return ((uint64_t)row << 32) | (uint64_t)col;
}

// Radix sort implementation (stable, O(n) for fixed key size)
static void radix_sort_coo(matgen_coo_matrix_t* matrix) {
  if (matrix->nnz <= 1) {
    return;
  }

  matgen_size_t n = matrix->nnz;

  // Allocate temporary buffers
  matgen_index_t* temp_rows =
      (matgen_index_t*)malloc(n * sizeof(matgen_index_t));
  matgen_index_t* temp_cols =
      (matgen_index_t*)malloc(n * sizeof(matgen_index_t));
  matgen_value_t* temp_vals =
      (matgen_value_t*)malloc(n * sizeof(matgen_value_t));
  uint64_t* keys = (uint64_t*)malloc(n * sizeof(uint64_t));
  uint64_t* temp_keys = (uint64_t*)malloc(n * sizeof(uint64_t));

  if (!temp_rows || !temp_cols || !temp_vals || !keys || !temp_keys) {
    MATGEN_LOG_ERROR("Failed to allocate temporary buffers for radix sort");
    free(temp_rows);
    free(temp_cols);
    free(temp_vals);
    free(keys);
    free(temp_keys);
    return;
  }

  // Encode keys
  for (matgen_size_t i = 0; i < n; i++) {
    keys[i] = encode_key(matrix->row_indices[i], matrix->col_indices[i]);
  }

  // Radix sort (process 8 bits at a time)
  matgen_size_t count[RADIX_BUCKETS];

  for (int shift = 0; shift < 64; shift += RADIX_BITS) {
    // Clear counts
    for (int i = 0; i < RADIX_BUCKETS; i++) {
      count[i] = 0;
    }

    // Count occurrences
    for (matgen_size_t i = 0; i < n; i++) {
      int bucket = (int)(keys[i] >> shift) & RADIX_MASK;
      count[bucket]++;
    }

    // Compute prefix sum
    for (int i = 1; i < RADIX_BUCKETS; i++) {
      count[i] += count[i - 1];
    }

    // Distribute elements (backwards for stability)
    for (matgen_size_t i = n; i > 0; i--) {
      matgen_size_t idx = i - 1;
      int bucket = (int)(keys[idx] >> shift) & RADIX_MASK;
      matgen_size_t dest = --count[bucket];

      temp_keys[dest] = keys[idx];
      temp_rows[dest] = matrix->row_indices[idx];
      temp_cols[dest] = matrix->col_indices[idx];
      temp_vals[dest] = matrix->values[idx];
    }

    // Swap buffers
    uint64_t* swap_keys = keys;
    keys = temp_keys;
    temp_keys = swap_keys;

    matgen_index_t* swap_rows = matrix->row_indices;
    matrix->row_indices = temp_rows;
    temp_rows = swap_rows;

    matgen_index_t* swap_cols = matrix->col_indices;
    matrix->col_indices = temp_cols;
    temp_cols = swap_cols;

    matgen_value_t* swap_vals = matrix->values;
    matrix->values = temp_vals;
    temp_vals = swap_vals;
  }

  // Free temporary buffers
  free(temp_rows);
  free(temp_cols);
  free(temp_vals);
  free(keys);
  free(temp_keys);
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

  // Choose sorting algorithm based on matrix size
  // Radix sort is O(n) but has overhead; quicksort is O(n log n)
  // Crossover point is typically around 100K-1M entries
  if (matrix->nnz > 100000) {
    MATGEN_LOG_DEBUG("Using radix sort for large matrix");
    radix_sort_coo(matrix);
  } else {
    MATGEN_LOG_DEBUG("Using index-based quicksort for small/medium matrix");
    sort_coo_inplace(matrix);
  }

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
