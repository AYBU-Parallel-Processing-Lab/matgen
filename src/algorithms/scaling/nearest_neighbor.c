#include "matgen/algorithms/scaling/nearest_neighbor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/core/conversion.h"
#include "matgen/core/coo_matrix.h"
#include "matgen/core/types.h"
#include "matgen/utils/accumulator.h"
#include "matgen/utils/log.h"

// Threshold for using stack vs heap allocation for weights
#define MATGEN_NN_STACK_THRESHOLD 64

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
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

  *result = NULL;

  double row_scale = (double)new_rows / (double)source->rows;
  double col_scale = (double)new_cols / (double)source->cols;

  MATGEN_LOG_DEBUG(
      "Nearest neighbor scaling: %llu×%llu -> %llu×%llu (scale: %.3fx%.3f)",
      (unsigned long long)source->rows, (unsigned long long)source->cols,
      (unsigned long long)new_rows, (unsigned long long)new_cols, row_scale,
      col_scale);

  // Estimate: each source entry expands to ~scale² target entries
  size_t estimated_nnz =
      (size_t)((double)source->nnz * row_scale * col_scale * 1.1);

  MATGEN_LOG_DEBUG("Estimated output NNZ: %zu", estimated_nnz);

  matgen_accumulator_t* acc =
      matgen_accumulator_create(estimated_nnz, collision_policy);
  if (!acc) {
    MATGEN_LOG_ERROR("Failed to create accumulator");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  matgen_error_t err = MATGEN_SUCCESS;

  for (matgen_index_t src_row = 0; src_row < source->rows; src_row++) {
    size_t row_start = source->row_ptr[src_row];
    size_t row_end = source->row_ptr[src_row + 1];

    for (size_t idx = row_start; idx < row_end; idx++) {
      matgen_index_t src_col = source->col_indices[idx];
      matgen_value_t src_val = source->values[idx];

      if (src_val == 0.0) {
        continue;
      }

      // Calculate target block boundaries
      matgen_index_t dst_row_start =
          (matgen_index_t)((double)src_row * row_scale);
      matgen_index_t dst_row_end =
          (matgen_index_t)((double)(src_row + 1) * row_scale);
      matgen_index_t dst_col_start =
          (matgen_index_t)((double)src_col * col_scale);
      matgen_index_t dst_col_end =
          (matgen_index_t)((double)(src_col + 1) * col_scale);

      // Clamp to valid range
      dst_row_end = MATGEN_CLAMP(dst_row_end, 0, new_rows);
      dst_col_end = MATGEN_CLAMP(dst_col_end, 0, new_cols);

      // Ensure at least one cell per block
      if (dst_row_end <= dst_row_start) {
        dst_row_end = dst_row_start + 1;
      }
      if (dst_col_end <= dst_col_start) {
        dst_col_end = dst_col_start + 1;
      }

      // Calculate block size
      matgen_index_t block_rows = dst_row_end - dst_row_start;
      matgen_index_t block_cols = dst_col_end - dst_col_start;
      double block_size = (double)block_rows * (double)block_cols;

      // Distribute value uniformly across block
      // Each cell gets: src_val / block_size
      // This preserves sum: block_size * (src_val / block_size) = src_val
      matgen_value_t cell_val = src_val / block_size;

      // Fill entire block
      for (matgen_index_t dr = dst_row_start; dr < dst_row_end; dr++) {
        for (matgen_index_t dc = dst_col_start; dc < dst_col_end; dc++) {
          err = matgen_accumulator_add(acc, dr, dc, cell_val);
          if (err != MATGEN_SUCCESS) {
            MATGEN_LOG_ERROR(
                "Failed to add entry to accumulator at (%llu, %llu)",
                (unsigned long long)dr, (unsigned long long)dc);
            matgen_accumulator_destroy(acc);
            return err;
          }
        }
      }
    }
  }

  size_t final_size = matgen_accumulator_size(acc);
  double load_factor = matgen_accumulator_load_factor(acc);

  MATGEN_LOG_DEBUG("Accumulated %zu entries (estimated %zu, load factor: %.2f)",
                   final_size, estimated_nnz, load_factor);

  matgen_coo_matrix_t* coo = matgen_accumulator_to_coo(acc, new_rows, new_cols);
  matgen_accumulator_destroy(acc);

  if (!coo) {
    MATGEN_LOG_ERROR("Failed to convert accumulator to COO matrix");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  *result = matgen_coo_to_csr(coo);
  matgen_coo_destroy(coo);

  if (!(*result)) {
    MATGEN_LOG_ERROR("Failed to convert COO to CSR matrix");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  MATGEN_LOG_DEBUG("Nearest neighbor scaling completed: output NNZ = %zu",
                   (*result)->nnz);

  return MATGEN_SUCCESS;
}
