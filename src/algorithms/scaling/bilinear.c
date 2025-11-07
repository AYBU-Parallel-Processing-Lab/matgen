#include "matgen/algorithms/scaling/bilinear.h"

#include <math.h>
#include <stdlib.h>

#include "matgen/algorithms/scaling/scaling_types.h"
#include "matgen/core/conversion.h"
#include "matgen/core/coo_matrix.h"

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
// Bilinear Interpolation Scaling
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

  // Create coordinate mapper
  matgen_coordinate_mapper_t mapper = matgen_coordinate_mapper_create(
      source->rows, source->cols, new_rows, new_cols);

  // Create accumulator (bilinear can create up to 4x entries)
  size_t estimated_capacity = source->nnz * 8;
  accumulator_t* acc = accumulator_create(estimated_capacity);

  // Process each source entry
  for (matgen_index_t i = 0; i < source->rows; i++) {
    size_t row_start = source->row_ptr[i];
    size_t row_end = source->row_ptr[i + 1];

    for (size_t j = row_start; j < row_end; j++) {
      matgen_index_t src_col = source->col_indices[j];
      matgen_value_t src_val = source->values[j];

      // Map to fractional target coordinates
      matgen_fractional_coord_t coord =
          matgen_map_fractional(&mapper, i, src_col);

      // Compute bilinear weights
      matgen_value_t w_row = coord.row - (matgen_value_t)coord.row_floor;
      matgen_value_t w_col = coord.col - (matgen_value_t)coord.col_floor;

      // Distribute value to 4 neighbors
      matgen_value_t w00 = (1.0 - w_row) * (1.0 - w_col);
      matgen_value_t w01 = (1.0 - w_row) * w_col;
      matgen_value_t w10 = w_row * (1.0 - w_col);
      matgen_value_t w11 = w_row * w_col;

      // Add contributions (only if weight is significant)
      if (w00 > 1e-10) {
        accumulator_add(acc, coord.row_floor, coord.col_floor, src_val * w00);
      }
      if (w01 > 1e-10 && coord.col_ceil != coord.col_floor) {
        accumulator_add(acc, coord.row_floor, coord.col_ceil, src_val * w01);
      }
      if (w10 > 1e-10 && coord.row_ceil != coord.row_floor) {
        accumulator_add(acc, coord.row_ceil, coord.col_floor, src_val * w10);
      }
      if (w11 > 1e-10 && coord.row_ceil != coord.row_floor &&
          coord.col_ceil != coord.col_floor) {
        accumulator_add(acc, coord.row_ceil, coord.col_ceil, src_val * w11);
      }
    }
  }

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
  return MATGEN_SUCCESS;
}
