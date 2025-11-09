#include "backends/omp/internal/bilinear_omp.h"

#include <math.h>
#include <omp.h>
#include <stdlib.h>

#include "backends/omp/internal/csr_builder_omp.h"
#include "matgen/core/types.h"
#include "matgen/utils/log.h"

#define MATGEN_BILINEAR_STACK_THRESHOLD 64

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
matgen_error_t matgen_scale_bilinear_omp(const matgen_csr_matrix_t* source,
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
      "Bilinear scaling (OMP): %zu×%zu -> %zu×%zu (scale: %.3fx%.3f)",
      source->rows, source->cols, new_rows, new_cols, row_scale, col_scale);

  int num_threads = omp_get_max_threads();
  MATGEN_LOG_DEBUG("Using %d OpenMP threads", num_threads);

  // Estimate output NNZ
  matgen_value_t avg_contributions_per_source =
      max((matgen_value_t)1.0, row_scale * col_scale);
  size_t estimated_nnz_total = (size_t)((matgen_value_t)source->nnz *
                                        avg_contributions_per_source * 1.2);

  MATGEN_LOG_DEBUG("Estimated output NNZ: %zu", estimated_nnz_total);

  // Create CSR builder (OpenMP backend)
  matgen_csr_builder_t* builder =
      matgen_csr_builder_create_omp(new_rows, new_cols, estimated_nnz_total);
  if (!builder) {
    MATGEN_LOG_ERROR("Failed to create CSR builder");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  matgen_error_t err = MATGEN_SUCCESS;

// Process rows in parallel
#pragma omp parallel
  {
    matgen_value_t stack_weights[MATGEN_BILINEAR_STACK_THRESHOLD];
    matgen_value_t* heap_weights = NULL;

    int src_row;

#pragma omp for schedule(dynamic, 16)
    for (src_row = 0; src_row < source->rows; src_row++) {
      if (err != MATGEN_SUCCESS) {
        continue;
      }

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

        dst_row_end = MATGEN_CLAMP(dst_row_end, 0, new_rows);
        dst_col_end = MATGEN_CLAMP(dst_col_end, 0, new_cols);

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

        if (block_size == 1) {
          matgen_error_t local_err = matgen_csr_builder_add_omp(
              builder, dst_row_start, dst_col_start, src_val);
          if (local_err != MATGEN_SUCCESS) {
#pragma omp atomic write
            err = local_err;
            continue;
          }
          continue;
        }

        // Allocate weight buffer
        matgen_value_t* weights = NULL;
        if (block_size <= MATGEN_BILINEAR_STACK_THRESHOLD) {
          weights = stack_weights;
        } else {
          free(heap_weights);
          heap_weights = malloc(block_size * sizeof(matgen_value_t));
          if (!heap_weights) {
#pragma omp atomic write
            err = MATGEN_ERROR_OUT_OF_MEMORY;
            continue;
          }
          weights = heap_weights;
        }

        // Calculate weights
        matgen_value_t src_center_row =
            ((matgen_value_t)src_row + (matgen_value_t)0.5) * row_scale;
        matgen_value_t src_center_col =
            ((matgen_value_t)src_col + (matgen_value_t)0.5) * col_scale;

        matgen_value_t row_norm_factor =
            (matgen_value_t)block_rows / (matgen_value_t)2.0;
        matgen_value_t col_norm_factor =
            (matgen_value_t)block_cols / (matgen_value_t)2.0;

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
          row_weight = MATGEN_CLAMP(row_weight, (matgen_value_t)0.0,
                                    (matgen_value_t)1.0);

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

        // Normalize weights
        if (total_weight > (matgen_value_t)0.0) {
          for (size_t i = 0; i < block_size; i++) {
            weights[i] /= total_weight;
          }
        }

        // Distribute
        weight_idx = 0;
        for (matgen_index_t dr = 0; dr < block_rows; dr++) {
          matgen_index_t dst_row = dst_row_start + dr;
          for (matgen_index_t dc = 0; dc < block_cols; dc++) {
            matgen_index_t dst_col = dst_col_start + dc;
            matgen_value_t normalized_weight = weights[weight_idx++];

            if (normalized_weight > (matgen_value_t)1e-12) {
              matgen_value_t weighted_val = src_val * normalized_weight;
              matgen_error_t local_err = matgen_csr_builder_add_omp(
                  builder, dst_row, dst_col, weighted_val);
              if (local_err != MATGEN_SUCCESS) {
#pragma omp atomic write
                err = local_err;
                break;
              }
            }
          }
        }
      }
    }

    free(heap_weights);
  }

  if (err != MATGEN_SUCCESS) {
    matgen_csr_builder_destroy(builder);
    return err;
  }

  MATGEN_LOG_DEBUG("Building CSR matrix from triplets...");

  // Finalize builder and get CSR matrix
  // The builder handles sorting, deduplication (summing), and CSR construction
  *result = matgen_csr_builder_finalize_omp(builder);

  if (!(*result)) {
    MATGEN_LOG_ERROR("Failed to finalize CSR builder");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  MATGEN_LOG_DEBUG("Bilinear scaling (OMP) completed: output NNZ = %zu",
                   (*result)->nnz);

  return MATGEN_SUCCESS;
}
