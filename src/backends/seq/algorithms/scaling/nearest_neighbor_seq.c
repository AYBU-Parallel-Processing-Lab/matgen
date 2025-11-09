#include "backends/seq/internal/nearest_neighbor_seq.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/core/execution/policy.h"
#include "matgen/core/matrix/conversion.h"
#include "matgen/core/matrix/coo.h"
#include "matgen/core/types.h"
#include "matgen/utils/log.h"
#include "matgen/utils/triplet_buffer.h"

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
matgen_error_t matgen_scale_nearest_neighbor_seq(
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

  matgen_value_t row_scale =
      (matgen_value_t)new_rows / (matgen_value_t)source->rows;
  matgen_value_t col_scale =
      (matgen_value_t)new_cols / (matgen_value_t)source->cols;

  MATGEN_LOG_DEBUG(
      "Nearest neighbor scaling (SEQ): %llu×%llu -> %llu×%llu "
      "(scale: %.3fx%.3f)",
      (unsigned long long)source->rows, (unsigned long long)source->cols,
      (unsigned long long)new_rows, (unsigned long long)new_cols, row_scale,
      col_scale);

  // Estimate: each source entry expands to ~scale² target entries
  size_t estimated_nnz =
      (size_t)((matgen_value_t)source->nnz * row_scale * col_scale * 1.1);

  MATGEN_LOG_DEBUG("Estimated output NNZ: %zu", estimated_nnz);

  // Create triplet buffer
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(estimated_nnz);
  if (!buffer) {
    MATGEN_LOG_ERROR("Failed to create triplet buffer");
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
          (matgen_index_t)((matgen_value_t)src_row * row_scale);
      matgen_index_t dst_row_end =
          (matgen_index_t)((matgen_value_t)(src_row + 1) * row_scale);
      matgen_index_t dst_col_start =
          (matgen_index_t)((matgen_value_t)src_col * col_scale);
      matgen_index_t dst_col_end =
          (matgen_index_t)((matgen_value_t)(src_col + 1) * col_scale);

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
      matgen_value_t block_size =
          (matgen_value_t)block_rows * (matgen_value_t)block_cols;

      // Distribute value uniformly across block
      // Each cell gets: src_val / block_size
      // This preserves sum: block_size * (src_val / block_size) = src_val
      matgen_value_t cell_val = src_val / block_size;

      // Fill entire block
      for (matgen_index_t dr = dst_row_start; dr < dst_row_end; dr++) {
        for (matgen_index_t dc = dst_col_start; dc < dst_col_end; dc++) {
          err = matgen_triplet_buffer_add(buffer, dr, dc, cell_val);
          if (err != MATGEN_SUCCESS) {
            MATGEN_LOG_ERROR("Failed to add entry to buffer at (%llu, %llu)",
                             (unsigned long long)dr, (unsigned long long)dc);
            matgen_triplet_buffer_destroy(buffer);
            return err;
          }
        }
      }
    }
  }

  size_t total_triplets = matgen_triplet_buffer_size(buffer);
  MATGEN_LOG_DEBUG("Generated %zu triplets", total_triplets);

  // Create COO matrix (uses AUTO by default, which will resolve to SEQ within
  // this backend)
  matgen_coo_matrix_t* coo =
      matgen_coo_create(new_rows, new_cols, total_triplets);
  if (!coo) {
    MATGEN_LOG_ERROR("Failed to create COO matrix");
    matgen_triplet_buffer_destroy(buffer);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Copy triplets to COO matrix
  memcpy(coo->row_indices, buffer->rows,
         total_triplets * sizeof(matgen_index_t));
  memcpy(coo->col_indices, buffer->cols,
         total_triplets * sizeof(matgen_index_t));
  memcpy(coo->values, buffer->vals, total_triplets * sizeof(matgen_value_t));

  coo->nnz = total_triplets;
  coo->is_sorted = false;

  matgen_triplet_buffer_destroy(buffer);

  // Sort and handle duplicates according to collision policy using sequential
  // backend
  MATGEN_LOG_DEBUG("Sorting and handling duplicates (policy: %d)...",
                   collision_policy);

  err = matgen_coo_sort_with_policy(coo, MATGEN_EXEC_SEQ);
  if (err != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to sort COO matrix");
    matgen_coo_destroy(coo);
    return err;
  }

  // Handle duplicates based on collision policy
  if (collision_policy == MATGEN_COLLISION_SUM) {
    // Sum duplicates
    err = matgen_coo_sum_duplicates_with_policy(coo, MATGEN_EXEC_SEQ);
  } else {
    // Use merge for other policies (AVG, MAX, MIN, LAST)
    err = matgen_coo_merge_duplicates_with_policy(coo, collision_policy,
                                                  MATGEN_EXEC_SEQ);
  }

  if (err != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to handle duplicates");
    matgen_coo_destroy(coo);
    return err;
  }

  MATGEN_LOG_DEBUG("After deduplication: %zu entries", coo->nnz);

  // Convert to CSR using sequential policy
  *result = matgen_coo_to_csr_with_policy(coo, MATGEN_EXEC_SEQ);
  matgen_coo_destroy(coo);

  if (!(*result)) {
    MATGEN_LOG_ERROR("Failed to convert COO to CSR matrix");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  MATGEN_LOG_DEBUG("Nearest neighbor scaling (SEQ) completed: output NNZ = %zu",
                   (*result)->nnz);

  return MATGEN_SUCCESS;
}
