#include "matgen/algorithms/scaling/bilinear.h"

#include <math.h>
#include <stdlib.h>

#include "matgen/core/conversion.h"
#include "matgen/core/coo_matrix.h"
#include "matgen/core/types.h"
#include "matgen/utils/accumulator.h"
#include "matgen/utils/log.h"

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

  double row_scale = (double)new_rows / (double)source->rows;
  double col_scale = (double)new_cols / (double)source->cols;

  MATGEN_LOG_DEBUG(
      "Bilinear scaling: %llu×%llu -> %llu×%llu (scale: %.3fx%.3f)",
      (unsigned long long)source->rows, (unsigned long long)source->cols,
      (unsigned long long)new_rows, (unsigned long long)new_cols, row_scale,
      col_scale);

  // For upscaling, each source cell spreads to ~scale² target cells
  // For downscaling, multiple source cells contribute to each target
  double avg_contributions_per_source = fmax(1.0, row_scale * col_scale);
  size_t estimated_nnz =
      (size_t)((double)source->nnz * avg_contributions_per_source * 1.2);

  MATGEN_LOG_DEBUG("Estimated output NNZ: %zu", estimated_nnz);

  matgen_accumulator_t* acc =
      matgen_accumulator_create(estimated_nnz, MATGEN_COLLISION_SUM);
  if (!acc) {
    MATGEN_LOG_ERROR("Failed to create accumulator");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Stack-allocated weight buffer for small blocks
  double stack_weights[MATGEN_BILINEAR_STACK_THRESHOLD];
  double* heap_weights = NULL;
  double* weights = NULL;

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

      // Calculate target block boundaries (same as nearest neighbor)
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

      matgen_index_t block_rows = dst_row_end - dst_row_start;
      matgen_index_t block_cols = dst_col_end - dst_col_start;
      matgen_size_t block_size =
          (matgen_size_t)block_rows * (matgen_size_t)block_cols;

      // For 1x1 blocks, just use the original value
      if (block_size == 1) {
        err =
            matgen_accumulator_add(acc, dst_row_start, dst_col_start, src_val);
        if (err != MATGEN_SUCCESS) {
          MATGEN_LOG_ERROR("Failed to add entry to accumulator");
          goto cleanup;
        }
        continue;
      }

      // Allocate weight buffer based on block size
      if (block_size <= MATGEN_BILINEAR_STACK_THRESHOLD) {
        weights = stack_weights;
      } else {
        // Need heap allocation for large blocks
        // Reallocate only if current heap buffer is too small
        if (!heap_weights || block_size > MATGEN_BILINEAR_STACK_THRESHOLD) {
          free(heap_weights);
          heap_weights = (double*)malloc(block_size * sizeof(double));
          if (!heap_weights) {
            MATGEN_LOG_ERROR(
                "Failed to allocate weight buffer for block size %zu",
                block_size);
            err = MATGEN_ERROR_OUT_OF_MEMORY;
            goto cleanup;
          }
        }
        weights = heap_weights;
      }

      // Calculate source cell center in target space
      double src_center_row = ((double)src_row + 0.5) * row_scale;
      double src_center_col = ((double)src_col + 0.5) * col_scale;

      // Pre-calculate normalization factors for bilinear weights
      double row_norm_factor = (double)block_rows / 2.0;
      double col_norm_factor = (double)block_cols / 2.0;

      // Single pass: calculate and store all weights
      double total_weight = 0.0;
      size_t weight_idx = 0;

      for (matgen_index_t dr = 0; dr < block_rows; dr++) {
        matgen_index_t dst_row = dst_row_start + dr;
        double dst_center_row = (double)dst_row + 0.5;
        double row_dist = fabs(dst_center_row - src_center_row);
        double row_weight = 1.0 - (row_dist / row_norm_factor);

        // Clamp to [0, 1]
        row_weight = MATGEN_CLAMP(row_weight, 0.0, 1.0);

        for (matgen_index_t dc = 0; dc < block_cols; dc++) {
          matgen_index_t dst_col = dst_col_start + dc;
          double dst_center_col = (double)dst_col + 0.5;
          double col_dist = fabs(dst_center_col - src_center_col);
          double col_weight = 1.0 - (col_dist / col_norm_factor);

          // Clamp to [0, 1]
          col_weight = MATGEN_CLAMP(col_weight, 0.0, 1.0);

          double weight = row_weight * col_weight;
          weights[weight_idx++] = weight;
          total_weight += weight;
        }
      }

      // Second pass: normalize and distribute using cached weights
      weight_idx = 0;
      for (matgen_index_t dr = 0; dr < block_rows; dr++) {
        matgen_index_t dst_row = dst_row_start + dr;
        for (matgen_index_t dc = 0; dc < block_cols; dc++) {
          matgen_index_t dst_col = dst_col_start + dc;

          double weight = weights[weight_idx++];

          // Normalize and distribute
          if (weight > 0.0) {
            matgen_value_t weighted_val = src_val * (weight / total_weight);
            err = matgen_accumulator_add(acc, dst_row, dst_col, weighted_val);
            if (err != MATGEN_SUCCESS) {
              MATGEN_LOG_ERROR("Failed to add entry to accumulator");
              goto cleanup;
            }
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

  if (!coo) {
    MATGEN_LOG_ERROR("Failed to convert accumulator to COO matrix");
    err = MATGEN_ERROR_OUT_OF_MEMORY;
    goto cleanup;
  }

  *result = matgen_coo_to_csr(coo);
  matgen_coo_destroy(coo);

  if (!(*result)) {
    MATGEN_LOG_ERROR("Failed to convert COO to CSR matrix");
    err = MATGEN_ERROR_OUT_OF_MEMORY;
    goto cleanup;
  }

  MATGEN_LOG_DEBUG("Bilinear scaling completed: output NNZ = %zu",
                   (*result)->nnz);

cleanup:
  free(heap_weights);
  matgen_accumulator_destroy(acc);

  return err;
}
