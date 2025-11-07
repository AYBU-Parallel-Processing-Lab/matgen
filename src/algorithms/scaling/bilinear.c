#include "matgen/algorithms/scaling/bilinear.h"

#include <math.h>
#include <stdlib.h>

#include "matgen/algorithms/scaling/scaling_types.h"
#include "matgen/core/conversion.h"
#include "matgen/core/coo_matrix.h"
#include "matgen/utils/log.h"

// =============================================================================
// Helper: Accumulator using hash table
// =============================================================================

typedef struct {
  matgen_index_t row;
  matgen_index_t col;
  matgen_value_t value;
} accum_entry_t;

typedef struct {
  accum_entry_t* entries;
  size_t capacity;
  size_t size;
} accumulator_t;

static size_t hash_coord(matgen_index_t row, matgen_index_t col,
                         size_t capacity) {
  return ((size_t)row * 73856093 + (size_t)col * 19349663) % capacity;
}

static accumulator_t* accumulator_create(size_t capacity) {
  accumulator_t* acc = malloc(sizeof(accumulator_t));
  acc->entries = calloc(capacity, sizeof(accum_entry_t));
  acc->capacity = capacity;
  acc->size = 0;

  for (size_t i = 0; i < capacity; i++) {
    acc->entries[i].row = (matgen_index_t)-1;
  }

  return acc;
}

static void accumulator_destroy(accumulator_t* acc) {
  free(acc->entries);
  free(acc);
}

static void accumulator_add(accumulator_t* acc, matgen_index_t row,
                            matgen_index_t col, matgen_value_t value) {
  size_t idx = hash_coord(row, col, acc->capacity);

  // Linear probing
  while (acc->entries[idx].row != (matgen_index_t)-1) {
    if (acc->entries[idx].row == row && acc->entries[idx].col == col) {
      // Found existing entry - accumulate
      acc->entries[idx].value += value;
      return;
    }
    idx = (idx + 1) % acc->capacity;
  }

  // Insert new entry
  acc->entries[idx].row = row;
  acc->entries[idx].col = col;
  acc->entries[idx].value = value;
  acc->size++;
}

// =============================================================================
// Bilinear Interpolation Scaling with Value Conservation
// =============================================================================

matgen_error_t matgen_scale_bilinear(const matgen_csr_matrix_t* source,
                                     matgen_index_t new_rows,
                                     matgen_index_t new_cols,
                                     matgen_csr_matrix_t** result) {
  if (!source || !result) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (new_rows == 0 || new_cols == 0) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Calculate scale factors
  matgen_value_t row_scale =
      (matgen_value_t)new_rows / (matgen_value_t)source->rows;
  matgen_value_t col_scale =
      (matgen_value_t)new_cols / (matgen_value_t)source->cols;

  MATGEN_LOG_DEBUG(
      "Bilinear scaling: %llu×%llu -> %llu×%llu (scale: %.3fx%.3f)",
      (unsigned long long)source->rows, (unsigned long long)source->cols,
      (unsigned long long)new_rows, (unsigned long long)new_cols, row_scale,
      col_scale);

  // Estimate capacity
  size_t estimated_nnz =
      (size_t)((double)source->nnz * row_scale * col_scale * 1.5);
  size_t estimated_capacity = estimated_nnz * 2;

  MATGEN_LOG_DEBUG("Estimated output NNZ: %zu", estimated_nnz);

  accumulator_t* acc = accumulator_create(estimated_capacity);

  // Process each source entry
  for (matgen_index_t src_row = 0; src_row < source->rows; src_row++) {
    size_t row_start = source->row_ptr[src_row];
    size_t row_end = source->row_ptr[src_row + 1];

    for (size_t idx = row_start; idx < row_end; idx++) {
      matgen_index_t src_col = source->col_indices[idx];
      matgen_value_t src_val = source->values[idx];

      // Calculate the block this source entry maps to in target space
      matgen_index_t dst_row_start =
          (matgen_index_t)((double)src_row * row_scale);
      matgen_index_t dst_row_end =
          (matgen_index_t)((double)(src_row + 1) * row_scale);
      matgen_index_t dst_col_start =
          (matgen_index_t)((double)src_col * col_scale);
      matgen_index_t dst_col_end =
          (matgen_index_t)((double)(src_col + 1) * col_scale);

      // Clamp to valid range
      if (dst_row_end > new_rows) {
        dst_row_end = new_rows;
      }

      if (dst_col_end > new_cols) {
        dst_col_end = new_cols;
      }

      // Calculate block size
      matgen_index_t block_rows = dst_row_end - dst_row_start;
      matgen_index_t block_cols = dst_col_end - dst_col_start;
      matgen_size_t block_size =
          (matgen_size_t)block_rows * (matgen_size_t)block_cols;

      // If block is empty, skip
      if (block_size == 0) {
        continue;
      }

      // Value per cell to conserve total value
      // Each cell in the block gets an equal share
      matgen_value_t value_per_cell = src_val / (matgen_value_t)block_size;

      // Distribute value uniformly across the block
      for (matgen_index_t dst_row = dst_row_start; dst_row < dst_row_end;
           dst_row++) {
        for (matgen_index_t dst_col = dst_col_start; dst_col < dst_col_end;
             dst_col++) {
          accumulator_add(acc, dst_row, dst_col, value_per_cell);
        }
      }
    }
  }

  MATGEN_LOG_DEBUG("Accumulated %zu entries (estimated %zu)", acc->size,
                   estimated_nnz);

  // Convert accumulator to COO matrix
  matgen_coo_matrix_t* coo = matgen_coo_create(new_rows, new_cols, acc->size);
  if (!coo) {
    accumulator_destroy(acc);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  for (size_t i = 0; i < acc->capacity; i++) {
    if (acc->entries[i].row != (matgen_index_t)-1) {
      matgen_coo_add_entry(coo, acc->entries[i].row, acc->entries[i].col,
                           acc->entries[i].value);
    }
  }

  accumulator_destroy(acc);

  // Convert COO to CSR
  *result = matgen_coo_to_csr(coo);
  if (!(*result)) {
    matgen_coo_destroy(coo);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  matgen_coo_destroy(coo);

  MATGEN_LOG_DEBUG("Bilinear scaling completed: output NNZ = %zu",
                   (*result)->nnz);

  return MATGEN_SUCCESS;
}
