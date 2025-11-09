#include "backends/seq/internal/coo_seq.h"

#include <stdlib.h>

#include "matgen/core/matrix/coo.h"
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
      continue;  // Already processed in a previous cycle
    }

    // Start of a new cycle
    matgen_size_t current = i;
    matgen_index_t temp_row = row_indices[i];
    matgen_index_t temp_col = col_indices[i];
    matgen_value_t temp_val = values[i];

    while (!visited[current]) {
      visited[current] = true;
      matgen_size_t next = keys[current].idx;

      if (next == i) {
        // End of cycle
        row_indices[current] = temp_row;
        col_indices[current] = temp_col;
        values[current] = temp_val;
        break;
      }

      // Swap current and next
      row_indices[current] = row_indices[next];
      col_indices[current] = col_indices[next];
      values[current] = values[next];

      current = next;
    }
  }

  free(visited);
}

// Index-based in-place sorting (for small to medium matrices)
static void sort_coo_inplace(matgen_coo_matrix_t* matrix) {
  // Build index array with keys
  coo_sort_key_t* keys =
      (coo_sort_key_t*)malloc(matrix->nnz * sizeof(coo_sort_key_t));
  if (!keys) {
    MATGEN_LOG_ERROR("Failed to allocate keys for COO sorting");
    return;
  }

  for (matgen_size_t i = 0; i < matrix->nnz; i++) {
    keys[i].idx = i;
    keys[i].row = matrix->row_indices[i];
    keys[i].col = matrix->col_indices[i];
  }

  // Sort keys
  qsort(keys, matrix->nnz, sizeof(coo_sort_key_t), compare_coo_keys);

  // Apply permutation
  apply_permutation(matrix->row_indices, matrix->col_indices, matrix->values,
                    keys, matrix->nnz);

  free(keys);
}

// =============================================================================
// Radix Sort (for large matrices)
// =============================================================================

#define RADIX_BITS 8
#define RADIX_BUCKETS (1 << RADIX_BITS)
#define RADIX_MASK ((1 << RADIX_BITS) - 1)

static uint64_t encode_key(matgen_index_t row, matgen_index_t col) {
  return ((uint64_t)row << 32) | (uint64_t)col;
}

// Radix sort implementation
static void radix_sort_coo(matgen_coo_matrix_t* matrix) {
  matgen_size_t n = matrix->nnz;

  // Allocate temporary buffers
  matgen_index_t* tmp_rows =
      (matgen_index_t*)malloc(n * sizeof(matgen_index_t));
  matgen_index_t* tmp_cols =
      (matgen_index_t*)malloc(n * sizeof(matgen_index_t));
  matgen_value_t* tmp_vals =
      (matgen_value_t*)malloc(n * sizeof(matgen_value_t));

  if (!tmp_rows || !tmp_cols || !tmp_vals) {
    MATGEN_LOG_ERROR("Failed to allocate buffers for radix sort");
    free(tmp_rows);
    free(tmp_cols);
    free(tmp_vals);
    return;
  }

  // Encode keys
  uint64_t* keys = (uint64_t*)malloc(n * sizeof(uint64_t));
  uint64_t* tmp_keys = (uint64_t*)malloc(n * sizeof(uint64_t));

  if (!keys || !tmp_keys) {
    MATGEN_LOG_ERROR("Failed to allocate key buffers for radix sort");
    free(tmp_rows);
    free(tmp_cols);
    free(tmp_vals);
    free(keys);
    free(tmp_keys);
    return;
  }

  for (matgen_size_t i = 0; i < n; i++) {
    keys[i] = encode_key(matrix->row_indices[i], matrix->col_indices[i]);
  }

  // Radix sort passes (8 passes for 64-bit keys with 8-bit radix)
  for (int pass = 0; pass < 8; pass++) {
    int shift = pass * RADIX_BITS;

    // Count occurrences
    matgen_size_t count[RADIX_BUCKETS] = {0};
    for (matgen_size_t i = 0; i < n; i++) {
      int bucket = (int)(keys[i] >> shift) & RADIX_MASK;
      count[bucket]++;
    }

    // Compute prefix sum
    matgen_size_t offset[RADIX_BUCKETS];
    offset[0] = 0;
    for (int i = 1; i < RADIX_BUCKETS; i++) {
      offset[i] = offset[i - 1] + count[i - 1];
    }

    // Scatter elements to tmp arrays
    for (matgen_size_t i = 0; i < n; i++) {
      int bucket = (int)(keys[i] >> shift) & RADIX_MASK;
      matgen_size_t dest = offset[bucket]++;

      tmp_keys[dest] = keys[i];
      tmp_rows[dest] = matrix->row_indices[i];
      tmp_cols[dest] = matrix->col_indices[i];
      tmp_vals[dest] = matrix->values[i];
    }

    // Swap pointers
    uint64_t* swap_keys = keys;
    keys = tmp_keys;
    tmp_keys = swap_keys;

    matgen_index_t* swap_rows = matrix->row_indices;
    matrix->row_indices = tmp_rows;
    tmp_rows = swap_rows;

    matgen_index_t* swap_cols = matrix->col_indices;
    matrix->col_indices = tmp_cols;
    tmp_cols = swap_cols;

    matgen_value_t* swap_vals = matrix->values;
    matrix->values = tmp_vals;
    tmp_vals = swap_vals;
  }

  // Free temporary buffers
  free(tmp_rows);
  free(tmp_cols);
  free(tmp_vals);
  free(keys);
  free(tmp_keys);
}

// =============================================================================
// Sequential Backend Implementation for COO Matrix
// =============================================================================

matgen_coo_matrix_t* matgen_coo_create_seq(matgen_index_t rows,
                                           matgen_index_t cols,
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

  MATGEN_LOG_DEBUG("Created COO matrix (SEQ) %llu x %llu with capacity %zu",
                   (unsigned long long)rows, (unsigned long long)cols,
                   matrix->capacity);

  return matrix;
}

matgen_error_t matgen_coo_sort_seq(matgen_coo_matrix_t* matrix) {
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

  MATGEN_LOG_DEBUG("Sorting COO matrix (SEQ) with %zu entries", matrix->nnz);

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

  MATGEN_LOG_DEBUG("Matrix sorted successfully (SEQ)");

  return MATGEN_SUCCESS;
}

matgen_error_t matgen_coo_sum_duplicates_seq(matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (matrix->nnz <= 1) {
    MATGEN_LOG_DEBUG("Matrix has %zu entries, no duplicates possible",
                     matrix->nnz);
    return MATGEN_SUCCESS;
  }

  if (!matrix->is_sorted) {
    MATGEN_LOG_ERROR("Matrix must be sorted before calling sum_duplicates");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  MATGEN_LOG_DEBUG("Summing duplicates in COO matrix (SEQ) with %zu entries",
                   matrix->nnz);

  matgen_size_t write_idx = 0;

  for (matgen_size_t read_idx = 0; read_idx < matrix->nnz; read_idx++) {
    // Copy current entry to write position
    matrix->row_indices[write_idx] = matrix->row_indices[read_idx];
    matrix->col_indices[write_idx] = matrix->col_indices[read_idx];
    matrix->values[write_idx] = matrix->values[read_idx];

    // Sum all duplicates
    while (
        read_idx + 1 < matrix->nnz &&
        matrix->row_indices[read_idx + 1] == matrix->row_indices[write_idx] &&
        matrix->col_indices[read_idx + 1] == matrix->col_indices[write_idx]) {
      read_idx++;
      matrix->values[write_idx] += matrix->values[read_idx];
    }

    write_idx++;
  }

  matgen_size_t old_nnz = matrix->nnz;
  matrix->nnz = write_idx;

  MATGEN_LOG_DEBUG("Reduced nnz from %zu to %zu (removed %zu duplicates)",
                   old_nnz, matrix->nnz, old_nnz - matrix->nnz);

  return MATGEN_SUCCESS;
}

matgen_error_t matgen_coo_merge_duplicates_seq(
    matgen_coo_matrix_t* matrix, matgen_collision_policy_t policy) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (matrix->nnz <= 1) {
    MATGEN_LOG_DEBUG("Matrix has %zu entries, no duplicates possible",
                     matrix->nnz);
    return MATGEN_SUCCESS;
  }

  if (!matrix->is_sorted) {
    MATGEN_LOG_ERROR("Matrix must be sorted before calling merge_duplicates");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  MATGEN_LOG_DEBUG(
      "Merging duplicates in COO matrix (SEQ) with %zu entries (policy: %d)",
      matrix->nnz, policy);

  matgen_size_t write_idx = 0;

  for (matgen_size_t read_idx = 0; read_idx < matrix->nnz; read_idx++) {
    // Copy current entry to write position
    matrix->row_indices[write_idx] = matrix->row_indices[read_idx];
    matrix->col_indices[write_idx] = matrix->col_indices[read_idx];
    matrix->values[write_idx] = matrix->values[read_idx];

    // Count duplicates and merge according to policy
    matgen_size_t dup_count = 1;

    while (
        read_idx + 1 < matrix->nnz &&
        matrix->row_indices[read_idx + 1] == matrix->row_indices[write_idx] &&
        matrix->col_indices[read_idx + 1] == matrix->col_indices[write_idx]) {
      read_idx++;
      dup_count++;

      matgen_value_t current_val = matrix->values[write_idx];
      matgen_value_t new_val = matrix->values[read_idx];

      switch (policy) {
        case MATGEN_COLLISION_SUM:
          matrix->values[write_idx] = current_val + new_val;
          break;
        case MATGEN_COLLISION_AVG:
          // Incremental average: avg_n = avg_{n-1} + (x_n - avg_{n-1})/n
          matrix->values[write_idx] = current_val + ((new_val - current_val) /
                                                     (matgen_value_t)dup_count);
          break;
        case MATGEN_COLLISION_MAX:
          if (new_val > current_val) {
            matrix->values[write_idx] = new_val;
          }
          break;
        case MATGEN_COLLISION_MIN:
          if (new_val < current_val) {
            matrix->values[write_idx] = new_val;
          }
          break;
        case MATGEN_COLLISION_LAST:
          matrix->values[write_idx] = new_val;
          break;
        default:
          MATGEN_LOG_ERROR("Unknown collision policy: %d", policy);
          return MATGEN_ERROR_INVALID_ARGUMENT;
      }
    }

    write_idx++;
  }

  matgen_size_t old_nnz = matrix->nnz;
  matrix->nnz = write_idx;

  MATGEN_LOG_DEBUG("Reduced nnz from %zu to %zu (removed %zu duplicates)",
                   old_nnz, matrix->nnz, old_nnz - matrix->nnz);

  return MATGEN_SUCCESS;
}
