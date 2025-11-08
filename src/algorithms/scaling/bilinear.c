#include "matgen/algorithms/scaling/bilinear.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/core/conversion.h"
#include "matgen/core/coo_matrix.h"
#include "matgen/core/types.h"
#include "matgen/utils/log.h"
#include "matgen/utils/triplet_buffer.h"

// Threshold for using stack vs heap allocation for weights
// For blocks larger than this, we use heap allocation
#define MATGEN_BILINEAR_STACK_THRESHOLD 64

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
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

  *result = NULL;

  matgen_value_t row_scale =
      (matgen_value_t)new_rows / (matgen_value_t)source->rows;
  matgen_value_t col_scale =
      (matgen_value_t)new_cols / (matgen_value_t)source->cols;

  MATGEN_LOG_DEBUG(
      "Bilinear scaling (sequential): %zu×%zu -> %zu×%zu (scale: "
      "%.3fx%.3f)",
      source->rows, source->cols, new_rows, new_cols, row_scale, col_scale);

  // Estimate output NNZ
  matgen_value_t avg_contributions_per_source =
      max((matgen_value_t)1.0, row_scale * col_scale);
  size_t estimated_nnz = (size_t)((matgen_value_t)source->nnz *
                                  avg_contributions_per_source * 1.2);

  MATGEN_LOG_DEBUG("Estimated output NNZ: %zu", estimated_nnz);

  // Create triplet buffer
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(estimated_nnz);
  if (!buffer) {
    MATGEN_LOG_ERROR("Failed to create triplet buffer");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Stack-allocated weight buffer for small blocks
  matgen_value_t stack_weights[MATGEN_BILINEAR_STACK_THRESHOLD];
  matgen_value_t* heap_weights = NULL;

  // Process each source entry
  matgen_error_t err = MATGEN_SUCCESS;
  for (matgen_index_t src_row = 0; src_row < source->rows; src_row++) {
    size_t row_start = source->row_ptr[src_row];
    size_t row_end = source->row_ptr[src_row + 1];

    for (size_t idx = row_start; idx < row_end; idx++) {
      matgen_index_t src_col = source->col_indices[idx];
      matgen_value_t src_val = source->values[idx];

      // Skip zero values
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

      matgen_index_t block_rows = dst_row_end - dst_row_start;
      matgen_index_t block_cols = dst_col_end - dst_col_start;
      matgen_size_t block_size =
          (matgen_size_t)block_rows * (matgen_size_t)block_cols;

      // For 1x1 blocks, just use the original value
      if (block_size == 1) {
        err = matgen_triplet_buffer_add(buffer, dst_row_start, dst_col_start,
                                        src_val);
        if (err != MATGEN_SUCCESS) {
          MATGEN_LOG_ERROR("Failed to add entry to buffer");
          goto cleanup;
        }
        continue;
      }

      // Allocate weight buffer based on block size
      matgen_value_t* weights = NULL;
      if (block_size <= MATGEN_BILINEAR_STACK_THRESHOLD) {
        weights = stack_weights;
      } else {
        // Need heap allocation for large blocks
        free(heap_weights);
        heap_weights =
            (matgen_value_t*)malloc(block_size * sizeof(matgen_value_t));
        if (!heap_weights) {
          MATGEN_LOG_ERROR(
              "Failed to allocate weight buffer for block size %zu",
              block_size);
          err = MATGEN_ERROR_OUT_OF_MEMORY;
          goto cleanup;
        }
        weights = heap_weights;
      }

      // Calculate source cell center in target space
      matgen_value_t src_center_row =
          ((matgen_value_t)src_row + (matgen_value_t)0.5) * row_scale;
      matgen_value_t src_center_col =
          ((matgen_value_t)src_col + (matgen_value_t)0.5) * col_scale;

      // Pre-calculate normalization factors for bilinear weights
      matgen_value_t row_norm_factor =
          (matgen_value_t)block_rows / (matgen_value_t)2.0;
      matgen_value_t col_norm_factor =
          (matgen_value_t)block_cols / (matgen_value_t)2.0;

      // Calculate and store all weights
      matgen_value_t total_weight = (matgen_value_t)0.0;
      size_t weight_idx = 0;

      for (matgen_index_t dr = 0; dr < block_rows; dr++) {
        matgen_index_t dst_row = dst_row_start + dr;
        matgen_value_t dst_center_row =
            (matgen_value_t)dst_row + (matgen_value_t)0.5;
        matgen_value_t row_dist =
            (matgen_value_t)fabs(dst_center_row - src_center_row);
        matgen_value_t row_weight =
            (matgen_value_t)1.0 - (row_dist / row_norm_factor);
        row_weight =
            MATGEN_CLAMP(row_weight, (matgen_value_t)0.0, (matgen_value_t)1.0);

        for (matgen_index_t dc = 0; dc < block_cols; dc++) {
          matgen_index_t dst_col = dst_col_start + dc;
          matgen_value_t dst_center_col =
              (matgen_value_t)dst_col + (matgen_value_t)0.5;
          matgen_value_t col_dist =
              (matgen_value_t)fabs(dst_center_col - src_center_col);
          matgen_value_t col_weight =
              (matgen_value_t)1.0 - (col_dist / col_norm_factor);
          col_weight = MATGEN_CLAMP(col_weight, (matgen_value_t)0.0,
                                    (matgen_value_t)1.0);

          matgen_value_t weight = row_weight * col_weight;
          weights[weight_idx++] = weight;
          total_weight += weight;
        }
      }

      // Pre-normalize weights
      if (total_weight > (matgen_value_t)0.0) {
        for (size_t i = 0; i < block_size; i++) {
          weights[i] /= total_weight;
        }
      }

      // Distribute using pre-normalized weights
      weight_idx = 0;
      for (matgen_index_t dr = 0; dr < block_rows; dr++) {
        matgen_index_t dst_row = dst_row_start + dr;
        for (matgen_index_t dc = 0; dc < block_cols; dc++) {
          matgen_index_t dst_col = dst_col_start + dc;
          matgen_value_t normalized_weight = weights[weight_idx++];

          // Distribute (skip if effectively zero)
          if (normalized_weight > (matgen_value_t)1e-12) {
            matgen_value_t weighted_val = src_val * normalized_weight;
            err = matgen_triplet_buffer_add(buffer, dst_row, dst_col,
                                            weighted_val);
            if (err != MATGEN_SUCCESS) {
              MATGEN_LOG_ERROR("Failed to add entry to buffer");
              goto cleanup;
            }
          }
        }
      }
    }
  }

  size_t total_triplets = matgen_triplet_buffer_size(buffer);
  MATGEN_LOG_DEBUG("Generated %zu triplets", total_triplets);

  // Create COO matrix and transfer triplets
  matgen_coo_matrix_t* coo =
      matgen_coo_create(new_rows, new_cols, total_triplets);
  if (!coo) {
    MATGEN_LOG_ERROR("Failed to create COO matrix");
    err = MATGEN_ERROR_OUT_OF_MEMORY;
    goto cleanup;
  }

  // Copy triplets to COO matrix
  memcpy(coo->row_indices, buffer->rows,
         total_triplets * sizeof(matgen_index_t));
  memcpy(coo->col_indices, buffer->cols,
         total_triplets * sizeof(matgen_index_t));
  memcpy(coo->values, buffer->vals, total_triplets * sizeof(matgen_value_t));
  coo->nnz = total_triplets;

  // Sort and sum duplicates
  MATGEN_LOG_DEBUG("Sorting and summing duplicates...");
  matgen_coo_sort(coo);
  matgen_coo_sum_duplicates(coo);

  MATGEN_LOG_DEBUG("After deduplication: %zu entries", coo->nnz);

  // Convert to CSR
  *result = matgen_coo_to_csr(coo);
  matgen_coo_destroy(coo);

  if (!(*result)) {
    MATGEN_LOG_ERROR("Failed to convert COO to CSR matrix");
    err = MATGEN_ERROR_OUT_OF_MEMORY;
    goto cleanup;
  }

  MATGEN_LOG_DEBUG("Bilinear scaling (sequential) completed: output NNZ = %zu",
                   (*result)->nnz);

cleanup:
  free(heap_weights);
  matgen_triplet_buffer_destroy(buffer);

  return err;
}
