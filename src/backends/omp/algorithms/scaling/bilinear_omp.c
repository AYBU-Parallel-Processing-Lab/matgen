#include "backends/omp/internal/bilinear_omp.h"

#include <math.h>
#include <omp.h>
#include <stdlib.h>

#include "backends/omp/internal/csr_builder_omp.h"
#include "matgen/core/types.h"
#include "matgen/utils/log.h"

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
      "Bilinear scaling (OMP): %llu×%llu -> %llu×%llu (scale: %.3fx%.3f)",
      (unsigned long long)source->rows, (unsigned long long)source->cols,
      (unsigned long long)new_rows, (unsigned long long)new_cols, row_scale,
      col_scale);

  int num_threads = omp_get_max_threads();
  MATGEN_LOG_DEBUG("Using %d OpenMP threads", num_threads);

  // Estimate output NNZ
  matgen_value_t max_row_contrib = ceilf((2.0F * row_scale) + 2.0F);
  matgen_value_t max_col_contrib = ceilf((2.0F * col_scale) + 2.0F);
  matgen_value_t max_contributions_per_source =
      max_row_contrib * max_col_contrib;

  size_t estimated_nnz_total = (size_t)((matgen_value_t)source->nnz *
                                        max_contributions_per_source * 1.5F);

  if (estimated_nnz_total < source->nnz * 4) {
    estimated_nnz_total = source->nnz * 4;
  }

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

    matgen_error_t local_err_check = MATGEN_SUCCESS;

#pragma omp for schedule(dynamic, 16)
    for (src_row = 0; src_row < source->rows; src_row++) {
#pragma omp atomic read
      local_err_check = err;
      if (local_err_check != MATGEN_SUCCESS) {
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

        // For bilinear interpolation, calculate destination range
        matgen_value_t dst_row_start_f = (matgen_value_t)fmax(
            0.0, ((matgen_value_t)src_row - 1.0) * row_scale);
        matgen_value_t dst_row_end_f =
            (matgen_value_t)((matgen_value_t)src_row + 1.0) * row_scale;
        matgen_value_t dst_col_start_f = (matgen_value_t)fmax(
            0.0, ((matgen_value_t)src_col - 1.0) * col_scale);
        matgen_value_t dst_col_end_f =
            (matgen_value_t)((matgen_value_t)src_col + 1.0) * col_scale;

        matgen_index_t dst_row_start = (matgen_index_t)ceil(dst_row_start_f);
        matgen_index_t dst_row_end = (matgen_index_t)ceil(dst_row_end_f);
        matgen_index_t dst_col_start = (matgen_index_t)ceil(dst_col_start_f);
        matgen_index_t dst_col_end = (matgen_index_t)ceil(dst_col_end_f);

        dst_row_start = MATGEN_CLAMP(dst_row_start, 0, new_rows);
        dst_row_end = MATGEN_CLAMP(dst_row_end, 0, new_rows);
        dst_col_start = MATGEN_CLAMP(dst_col_start, 0, new_cols);
        dst_col_end = MATGEN_CLAMP(dst_col_end, 0, new_cols);

        // For each destination cell, calculate bilinear weight
        for (matgen_index_t dst_row = dst_row_start; dst_row < dst_row_end;
             dst_row++) {
          for (matgen_index_t dst_col = dst_col_start; dst_col < dst_col_end;
               dst_col++) {
            // Map destination back to source coordinates (fractional)
            matgen_value_t src_y = (matgen_value_t)dst_row / row_scale;
            matgen_value_t src_x = (matgen_value_t)dst_col / col_scale;

            // Find the 4 bilinear neighbors
            matgen_index_t y0 = (matgen_index_t)floor(src_y);
            matgen_index_t y1 = (matgen_index_t)ceil(src_y);
            matgen_index_t x0 = (matgen_index_t)floor(src_x);
            matgen_index_t x1 = (matgen_index_t)ceil(src_x);

            // Clamp neighbors to valid source bounds
            y0 = MATGEN_CLAMP(y0, 0, source->rows - 1);
            y1 = MATGEN_CLAMP(y1, 0, source->rows - 1);
            x0 = MATGEN_CLAMP(x0, 0, source->cols - 1);
            x1 = MATGEN_CLAMP(x1, 0, source->cols - 1);

            // Calculate fractional parts for bilinear interpolation
            matgen_value_t dy = src_y - (matgen_value_t)y0;
            matgen_value_t dx = src_x - (matgen_value_t)x0;

            // Clamp fractional parts to [0, 1]
            dy = MATGEN_CLAMP(dy, 0.0, 1.0);
            dx = MATGEN_CLAMP(dx, 0.0, 1.0);

            // Determine which of the 4 neighbors we are and calculate
            // bilinear weight
            matgen_value_t weight = (matgen_value_t)0.0;

            if (src_row == y0 && src_col == x0) {
              // Bottom-left neighbor
              weight = (matgen_value_t)(1.0 - dy) * (matgen_value_t)(1.0 - dx);
            } else if (src_row == y0 && src_col == x1) {
              // Bottom-right neighbor
              weight = (matgen_value_t)(1.0 - dy) * dx;
            } else if (src_row == y1 && src_col == x0) {
              // Top-left neighbor
              weight = dy * (matgen_value_t)(1.0 - dx);
            } else if (src_row == y1 && src_col == x1) {
              // Top-right neighbor
              weight = dy * dx;
            }

            // Add weighted contribution if non-zero
            if (weight > 1e-12) {
              matgen_value_t contribution = src_val * weight;
              matgen_error_t local_err = matgen_csr_builder_add_omp(
                  builder, dst_row, dst_col, contribution);
              if (local_err != MATGEN_SUCCESS) {
#pragma omp atomic write
                err = local_err;
                break;
              }
            }

#pragma omp atomic read
            local_err_check = err;
            if (local_err_check != MATGEN_SUCCESS) {
              break;  // Break from dst_row loop
            }
          }

#pragma omp atomic read
          local_err_check = err;
          if (local_err_check != MATGEN_SUCCESS) {
            break;  // Break from idx loop
          }
        }
      }
    }
  }

  if (err != MATGEN_SUCCESS) {
    matgen_csr_builder_destroy(builder);
    return err;
  }

  MATGEN_LOG_DEBUG("Building CSR matrix...");

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
