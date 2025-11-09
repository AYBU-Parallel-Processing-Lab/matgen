#include "backends/omp/internal/nearest_neighbor_omp.h"

#include <math.h>
#include <omp.h>
#include <stdlib.h>

#include "backends/omp/internal/csr_builder_omp.h"
#include "matgen/core/types.h"
#include "matgen/utils/log.h"

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
matgen_error_t matgen_scale_nearest_neighbor_omp(
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

  // Note: Current CSR builder only supports SUM collision policy
  if (collision_policy != MATGEN_COLLISION_SUM) {
    MATGEN_LOG_WARN(
        "CSR builder currently only supports SUM collision policy, "
        "using SUM instead of policy %d",
        collision_policy);
  }

  matgen_value_t row_scale =
      (matgen_value_t)new_rows / (matgen_value_t)source->rows;
  matgen_value_t col_scale =
      (matgen_value_t)new_cols / (matgen_value_t)source->cols;

  MATGEN_LOG_DEBUG(
      "Nearest neighbor scaling (OMP): %llu×%llu -> %llu×%llu "
      "(scale: %.3fx%.3f)",
      (unsigned long long)source->rows, (unsigned long long)source->cols,
      (unsigned long long)new_rows, (unsigned long long)new_cols, row_scale,
      col_scale);

  int num_threads = omp_get_max_threads();
  MATGEN_LOG_DEBUG("Using %d OpenMP threads", num_threads);

  // Estimate output NNZ
  size_t estimated_nnz_total =
      (size_t)((matgen_value_t)source->nnz * row_scale * col_scale * 1.1);

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
        matgen_value_t block_size =
            (matgen_value_t)block_rows * (matgen_value_t)block_cols;

        matgen_value_t cell_val = src_val / block_size;

        // Fill entire block
        for (matgen_index_t dr = dst_row_start; dr < dst_row_end; dr++) {
          for (matgen_index_t dc = dst_col_start; dc < dst_col_end; dc++) {
            matgen_error_t local_err =
                matgen_csr_builder_add_omp(builder, dr, dc, cell_val);
            if (local_err != MATGEN_SUCCESS) {
#pragma omp atomic write
              err = local_err;
              break;
            }
          }
          if (err != MATGEN_SUCCESS) {
            break;
          }
        }
      }
    }
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

  MATGEN_LOG_DEBUG("Nearest neighbor scaling (OMP) completed: output NNZ = %zu",
                   (*result)->nnz);

  return MATGEN_SUCCESS;
}
