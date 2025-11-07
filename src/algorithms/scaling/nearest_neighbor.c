#include "matgen/algorithms/scaling/nearest_neighbor.h"

#include <stdlib.h>
#include <string.h>

#include "matgen/core/conversion.h"
#include "matgen/core/coo_matrix.h"
#include "matgen/utils/log.h"

// =============================================================================
// Helper: Hash table for accumulating values at same coordinates
// =============================================================================

typedef struct {
  matgen_index_t row;
  matgen_index_t col;
  matgen_value_t value;
  size_t count;  // For averaging collision policy
} hash_entry_t;

typedef struct {
  hash_entry_t* entries;
  size_t capacity;
  size_t size;
} hash_table_t;

static size_t hash_coord(matgen_index_t row, matgen_index_t col,
                         size_t capacity) {
  return ((size_t)row * 73856093 + (size_t)col * 19349663) % capacity;
}

static hash_table_t* hash_table_create(size_t capacity) {
  hash_table_t* table = malloc(sizeof(hash_table_t));
  table->entries = calloc(capacity, sizeof(hash_entry_t));
  table->capacity = capacity;
  table->size = 0;

  for (size_t i = 0; i < capacity; i++) {
    table->entries[i].row = (matgen_index_t)-1;
  }

  return table;
}

static void hash_table_destroy(hash_table_t* table) {
  free(table->entries);
  free(table);
}

static void hash_table_insert(hash_table_t* table, matgen_index_t row,
                              matgen_index_t col, matgen_value_t value,
                              matgen_collision_policy_t policy) {
  size_t idx = hash_coord(row, col, table->capacity);

  // Linear probing
  while (table->entries[idx].row != (matgen_index_t)-1) {
    if (table->entries[idx].row == row && table->entries[idx].col == col) {
      // Found existing entry - handle collision
      switch (policy) {
        case MATGEN_COLLISION_SUM:
          table->entries[idx].value += value;
          break;
        case MATGEN_COLLISION_AVG:
          table->entries[idx].value += value;
          table->entries[idx].count++;
          break;
        case MATGEN_COLLISION_MAX:
          if (value > table->entries[idx].value) {
            table->entries[idx].value = value;
          }
          break;
      }
      return;
    }
    idx = (idx + 1) % table->capacity;
  }

  // Insert new entry
  table->entries[idx].row = row;
  table->entries[idx].col = col;
  table->entries[idx].value = value;
  table->entries[idx].count = 1;
  table->size++;
}

// =============================================================================
// Nearest Neighbor Scaling with Proper Expansion
// =============================================================================

matgen_error_t matgen_scale_nearest_neighbor(
    const matgen_csr_matrix_t* source, matgen_index_t new_rows,
    matgen_index_t new_cols, matgen_collision_policy_t collision_policy,
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
      "Nearest neighbor scaling: %llu×%llu -> %llu×%llu (scale: %.3fx%.3f)",
      (unsigned long long)source->rows, (unsigned long long)source->cols,
      (unsigned long long)new_rows, (unsigned long long)new_cols, row_scale,
      col_scale);

  // Estimate capacity: for upscaling, NNZ grows by scale²
  size_t estimated_nnz =
      (size_t)((double)source->nnz * row_scale * col_scale * 1.2);
  size_t estimated_capacity = estimated_nnz * 2;

  MATGEN_LOG_DEBUG("Estimated output NNZ: %zu", estimated_nnz);

  hash_table_t* table = hash_table_create(estimated_capacity);

  // Process each source entry
  for (matgen_index_t src_row = 0; src_row < source->rows; src_row++) {
    size_t row_start = source->row_ptr[src_row];
    size_t row_end = source->row_ptr[src_row + 1];

    for (size_t idx = row_start; idx < row_end; idx++) {
      matgen_index_t src_col = source->col_indices[idx];
      matgen_value_t src_val = source->values[idx];

      // Determine the block this source entry maps to
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

      // Replicate value to all cells in the block (nearest neighbor)
      for (matgen_index_t dst_row = dst_row_start; dst_row < dst_row_end;
           dst_row++) {
        for (matgen_index_t dst_col = dst_col_start; dst_col < dst_col_end;
             dst_col++) {
          hash_table_insert(table, dst_row, dst_col, src_val, collision_policy);
        }
      }
    }
  }

  MATGEN_LOG_DEBUG("Accumulated %zu entries (estimated %zu)", table->size,
                   estimated_nnz);

  // Convert hash table to COO matrix
  matgen_coo_matrix_t* coo = matgen_coo_create(new_rows, new_cols, table->size);
  if (!coo) {
    hash_table_destroy(table);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  for (size_t i = 0; i < table->capacity; i++) {
    if (table->entries[i].row != (matgen_index_t)-1) {
      matgen_value_t value = table->entries[i].value;

      // Apply averaging if needed
      if (collision_policy == MATGEN_COLLISION_AVG &&
          table->entries[i].count > 1) {
        value /= (matgen_value_t)table->entries[i].count;
      }

      matgen_coo_add_entry(coo, table->entries[i].row, table->entries[i].col,
                           value);
    }
  }

  hash_table_destroy(table);

  // Convert COO to CSR
  *result = matgen_coo_to_csr(coo);
  if (!(*result)) {
    matgen_coo_destroy(coo);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  matgen_coo_destroy(coo);

  MATGEN_LOG_DEBUG("Nearest neighbor scaling completed: output NNZ = %zu",
                   (*result)->nnz);

  return MATGEN_SUCCESS;
}
