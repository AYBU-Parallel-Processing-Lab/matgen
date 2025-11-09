#include "backends/omp/internal/coo_omp.h"

#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/core/matrix/coo.h"
#include "matgen/utils/log.h"

// =============================================================================
// OpenMP Parallel COO Operations
// =============================================================================

matgen_coo_matrix_t* matgen_coo_create_omp(matgen_index_t rows,
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
  matrix->capacity = (nnz_hint > 0) ? nnz_hint : 1024;
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

  MATGEN_LOG_DEBUG("Created COO matrix (OMP) %llu x %llu with capacity %zu",
                   (unsigned long long)rows, (unsigned long long)cols,
                   matrix->capacity);

  return matrix;
}

// =============================================================================
// Parallel Sorting
// =============================================================================

// Comparison function for qsort
typedef struct {
  matgen_index_t row;
  matgen_index_t col;
  matgen_value_t val;
} triplet_t;

static int compare_triplets_qsort(const void* a, const void* b) {
  const triplet_t* t_a = (const triplet_t*)a;
  const triplet_t* t_b = (const triplet_t*)b;

  if (t_a->row < t_b->row) {
    return -1;
  }
  if (t_a->row > t_b->row) {
    return 1;
  }
  if (t_a->col < t_b->col) {
    return -1;
  }
  if (t_a->col > t_b->col) {
    return 1;
  }
  return 0;
}

// Helper for merge operation
static void merge_coo(const matgen_index_t* row_a, const matgen_index_t* col_a,
                      const matgen_value_t* val_a, matgen_size_t size_a,
                      const matgen_index_t* row_b, const matgen_index_t* col_b,
                      const matgen_value_t* val_b, matgen_size_t size_b,
                      matgen_index_t* row_out, matgen_index_t* col_out,
                      matgen_value_t* val_out) {
  matgen_size_t i = 0;
  matgen_size_t j = 0;
  matgen_size_t k = 0;

  while (i < size_a && j < size_b) {
    bool a_less_than_b = false;

    if (row_a[i] < row_b[j]) {
      a_less_than_b = true;
    } else if (row_a[i] == row_b[j]) {
      if (col_a[i] <= col_b[j]) {
        a_less_than_b = true;
      }
    }

    if (a_less_than_b) {
      row_out[k] = row_a[i];
      col_out[k] = col_a[i];
      val_out[k] = val_a[i];
      i++;
    } else {
      row_out[k] = row_b[j];
      col_out[k] = col_b[j];
      val_out[k] = val_b[j];
      j++;
    }
    k++;
  }

  // Copy remaining elements
  while (i < size_a) {
    row_out[k] = row_a[i];
    col_out[k] = col_a[i];
    val_out[k] = val_a[i];
    i++;
    k++;
  }

  while (j < size_b) {
    row_out[k] = row_b[j];
    col_out[k] = col_b[j];
    val_out[k] = val_b[j];
    j++;
    k++;
  }
}

matgen_error_t matgen_coo_sort_omp(matgen_coo_matrix_t* matrix) {
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

  MATGEN_LOG_DEBUG("Sorting COO matrix (OMP) with %zu entries", matrix->nnz);

  // For small matrices, use sequential qsort
  if (matrix->nnz < 100000) {
    // Pack into triplets for qsort
    triplet_t* triplets = malloc(matrix->nnz * sizeof(triplet_t));
    if (!triplets) {
      MATGEN_LOG_ERROR("Failed to allocate triplet buffer");
      return MATGEN_ERROR_OUT_OF_MEMORY;
    }

    for (matgen_size_t i = 0; i < matrix->nnz; i++) {
      triplets[i].row = matrix->row_indices[i];
      triplets[i].col = matrix->col_indices[i];
      triplets[i].val = matrix->values[i];
    }

    qsort(triplets, matrix->nnz, sizeof(triplet_t), compare_triplets_qsort);

    for (matgen_size_t i = 0; i < matrix->nnz; i++) {
      matrix->row_indices[i] = triplets[i].row;
      matrix->col_indices[i] = triplets[i].col;
      matrix->values[i] = triplets[i].val;
    }

    free(triplets);
    matrix->is_sorted = true;
    MATGEN_LOG_DEBUG("Matrix sorted successfully (sequential qsort)");
    return MATGEN_SUCCESS;
  }

  // Parallel merge sort for large matrices
  int num_threads = omp_get_max_threads();
  MATGEN_LOG_DEBUG("Using %d threads for parallel sort", num_threads);

  matgen_size_t chunk_size = (matrix->nnz + num_threads - 1) / num_threads;

  // Allocate temporary buffers
  matgen_index_t* tmp_rows = malloc(matrix->nnz * sizeof(matgen_index_t));
  matgen_index_t* tmp_cols = malloc(matrix->nnz * sizeof(matgen_index_t));
  matgen_value_t* tmp_vals = malloc(matrix->nnz * sizeof(matgen_value_t));

  if (!tmp_rows || !tmp_cols || !tmp_vals) {
    MATGEN_LOG_ERROR("Failed to allocate temporary buffers for sorting");
    free(tmp_rows);
    free(tmp_cols);
    free(tmp_vals);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

// Phase 1: Sort chunks in parallel using qsort
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    matgen_size_t start = tid * chunk_size;
    matgen_size_t end = start + chunk_size;

    if (end > matrix->nnz) {
      end = matrix->nnz;
    }

    if (start < matrix->nnz) {
      matgen_size_t len = end - start;

      // Pack chunk into triplets
      triplet_t* triplets = malloc(len * sizeof(triplet_t));
      if (triplets) {
        for (matgen_size_t i = 0; i < len; i++) {
          triplets[i].row = matrix->row_indices[start + i];
          triplets[i].col = matrix->col_indices[start + i];
          triplets[i].val = matrix->values[start + i];
        }

        // Sort chunk with qsort (much faster than insertion sort!)
        qsort(triplets, len, sizeof(triplet_t), compare_triplets_qsort);

        // Unpack back
        for (matgen_size_t i = 0; i < len; i++) {
          matrix->row_indices[start + i] = triplets[i].row;
          matrix->col_indices[start + i] = triplets[i].col;
          matrix->values[start + i] = triplets[i].val;
        }

        free(triplets);
      } else {
        MATGEN_LOG_WARN("Thread %d failed to allocate triplet buffer, skipping",
                        tid);
      }
    }
  }

  MATGEN_LOG_DEBUG("Phase 1 complete: chunks sorted");

  // Phase 2: Merge chunks iteratively
  matgen_size_t merge_width = chunk_size;
  bool use_tmp = false;

  while (merge_width < matrix->nnz) {
    MATGEN_LOG_DEBUG("Merge pass: width=%zu", merge_width);

    for (matgen_size_t start = 0; start < matrix->nnz;
         start += merge_width * 2) {
      matgen_size_t mid = start + merge_width;
      matgen_size_t end =
          (mid + merge_width < matrix->nnz) ? mid + merge_width : matrix->nnz;

      if (mid >= matrix->nnz) {
        // No second chunk to merge - just copy the first chunk to output
        if (use_tmp) {
          // Copy from tmp to matrix
          matgen_size_t len = matrix->nnz - start;
          memcpy(&matrix->row_indices[start], &tmp_rows[start],
                 len * sizeof(matgen_index_t));
          memcpy(&matrix->col_indices[start], &tmp_cols[start],
                 len * sizeof(matgen_index_t));
          memcpy(&matrix->values[start], &tmp_vals[start],
                 len * sizeof(matgen_value_t));
        } else {
          // Already in matrix, copy to tmp for next iteration
          matgen_size_t len = matrix->nnz - start;
          memcpy(&tmp_rows[start], &matrix->row_indices[start],
                 len * sizeof(matgen_index_t));
          memcpy(&tmp_cols[start], &matrix->col_indices[start],
                 len * sizeof(matgen_index_t));
          memcpy(&tmp_vals[start], &matrix->values[start],
                 len * sizeof(matgen_value_t));
        }
      } else {
        // Merge two chunks
        matgen_size_t size_a = mid - start;
        matgen_size_t size_b = end - mid;

        if (use_tmp) {
          // Read from tmp, write to matrix
          merge_coo(&tmp_rows[start], &tmp_cols[start], &tmp_vals[start],
                    size_a, &tmp_rows[mid], &tmp_cols[mid], &tmp_vals[mid],
                    size_b, &matrix->row_indices[start],
                    &matrix->col_indices[start], &matrix->values[start]);
        } else {
          // Read from matrix, write to tmp
          merge_coo(&matrix->row_indices[start], &matrix->col_indices[start],
                    &matrix->values[start], size_a, &matrix->row_indices[mid],
                    &matrix->col_indices[mid], &matrix->values[mid], size_b,
                    &tmp_rows[start], &tmp_cols[start], &tmp_vals[start]);
        }
      }
    }

    use_tmp = !use_tmp;
    merge_width *= 2;
  }

  // Copy back if final result is in tmp
  if (use_tmp) {
    memcpy(matrix->row_indices, tmp_rows, matrix->nnz * sizeof(matgen_index_t));
    memcpy(matrix->col_indices, tmp_cols, matrix->nnz * sizeof(matgen_index_t));
    memcpy(matrix->values, tmp_vals, matrix->nnz * sizeof(matgen_value_t));
  }

  free(tmp_rows);
  free(tmp_cols);
  free(tmp_vals);

  matrix->is_sorted = true;
  MATGEN_LOG_DEBUG("Matrix sorted successfully (OMP parallel merge sort)");

  return MATGEN_SUCCESS;
}

// =============================================================================
// Parallel Duplicate Handling
// =============================================================================

matgen_error_t matgen_coo_sum_duplicates_omp(matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (matrix->nnz <= 1) {
    return MATGEN_SUCCESS;
  }

  if (!matrix->is_sorted) {
    MATGEN_LOG_ERROR("Matrix must be sorted before calling sum_duplicates");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  MATGEN_LOG_DEBUG("Summing duplicates in COO matrix (OMP) with %zu entries",
                   matrix->nnz);

  // Three-phase parallel algorithm:
  // Phase 1: Mark boundaries (where (row,col) changes)
  // Phase 2: Prefix sum to find output positions
  // Phase 3: Parallel reduction within each group + compaction

  // Allocate boundary markers
  uint8_t* is_boundary = (uint8_t*)calloc(matrix->nnz, sizeof(uint8_t));
  if (!is_boundary) {
    MATGEN_LOG_ERROR("Failed to allocate boundary array");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Phase 1: Mark boundaries in parallel
  is_boundary[0] = 1;  // First entry is always a boundary

  int i;
#pragma omp parallel for schedule(static)
  for (i = 1; i < matrix->nnz; i++) {
    if (matrix->row_indices[i] != matrix->row_indices[i - 1] ||
        matrix->col_indices[i] != matrix->col_indices[i - 1]) {
      is_boundary[i] = 1;
    }
  }

  // Phase 2: Prefix sum to compute output indices (sequential for now)
  matgen_size_t* out_idx =
      (matgen_size_t*)malloc(matrix->nnz * sizeof(matgen_size_t));
  if (!out_idx) {
    free(is_boundary);
    MATGEN_LOG_ERROR("Failed to allocate output index array");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  out_idx[0] = 0;
  for (matgen_size_t j = 1; j < matrix->nnz; j++) {
    out_idx[j] = out_idx[j - 1] + is_boundary[j];
  }
  matgen_size_t new_nnz = out_idx[matrix->nnz - 1] + 1;

  // Allocate temporary output arrays
  matgen_index_t* out_rows =
      (matgen_index_t*)malloc(new_nnz * sizeof(matgen_index_t));
  matgen_index_t* out_cols =
      (matgen_index_t*)malloc(new_nnz * sizeof(matgen_index_t));
  matgen_value_t* out_vals =
      (matgen_value_t*)malloc(new_nnz * sizeof(matgen_value_t));

  if (!out_rows || !out_cols || !out_vals) {
    free(is_boundary);
    free(out_idx);
    free(out_rows);
    free(out_cols);
    free(out_vals);
    MATGEN_LOG_ERROR("Failed to allocate output arrays");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

// Initialize output arrays
#pragma omp parallel for schedule(static)
  for (i = 0; i < new_nnz; i++) {
    out_vals[i] = (matgen_value_t)0.0;
  }

// Phase 3: Parallel reduction and write
// Each thread processes a chunk and atomically adds to output
#pragma omp parallel for schedule(static)
  for (i = 0; i < matrix->nnz; i++) {
    matgen_size_t oidx = out_idx[i];

    if (is_boundary[i]) {
      // This is a boundary - write row/col
      out_rows[oidx] = matrix->row_indices[i];
      out_cols[oidx] = matrix->col_indices[i];
    }

// Atomically accumulate value
#pragma omp atomic
    out_vals[oidx] += matrix->values[i];
  }

  // Copy back to original matrix
  memcpy(matrix->row_indices, out_rows, new_nnz * sizeof(matgen_index_t));
  memcpy(matrix->col_indices, out_cols, new_nnz * sizeof(matgen_index_t));
  memcpy(matrix->values, out_vals, new_nnz * sizeof(matgen_value_t));

  free(is_boundary);
  free(out_idx);
  free(out_rows);
  free(out_cols);
  free(out_vals);

  matgen_size_t old_nnz = matrix->nnz;
  matrix->nnz = new_nnz;

  MATGEN_LOG_DEBUG("Reduced nnz from %zu to %zu (removed %zu duplicates)",
                   old_nnz, matrix->nnz, old_nnz - matrix->nnz);

  return MATGEN_SUCCESS;
}

matgen_error_t matgen_coo_merge_duplicates_omp(
    matgen_coo_matrix_t* matrix, matgen_collision_policy_t policy) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (matrix->nnz <= 1) {
    return MATGEN_SUCCESS;
  }

  if (!matrix->is_sorted) {
    MATGEN_LOG_ERROR("Matrix must be sorted before calling merge_duplicates");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  MATGEN_LOG_DEBUG(
      "Merging duplicates in COO matrix (OMP) with %zu entries (policy: %d)",
      matrix->nnz, policy);

  // Sequential implementation for now
  // TODO: Parallel version
  matgen_size_t write_idx = 0;

  for (matgen_size_t read_idx = 0; read_idx < matrix->nnz; read_idx++) {
    matrix->row_indices[write_idx] = matrix->row_indices[read_idx];
    matrix->col_indices[write_idx] = matrix->col_indices[read_idx];
    matrix->values[write_idx] = matrix->values[read_idx];

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
