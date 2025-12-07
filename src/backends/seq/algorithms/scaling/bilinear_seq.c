#include "backends/seq/internal/bilinear_seq.h"

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
matgen_error_t matgen_scale_bilinear_seq(const matgen_csr_matrix_t* source,
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
      "Bilinear scaling (SEQ): %zu×%zu -> %zu×%zu (scale: %.3fx%.3f)",
      source->rows, source->cols, new_rows, new_cols, row_scale, col_scale);

  // For bilinear, each source entry contributes to a neighborhood
  // Each source entry at (src_row, src_col) contributes to destination cells in
  // range:
  // - Row range: [(src_row - 1) * row_scale, (src_row + 1) * row_scale]
  // - Col range: [(src_col - 1) * col_scale, (src_col + 1) * col_scale]
  matgen_value_t max_row_contrib = ceilf((2.0F * row_scale) + 2.0F);
  matgen_value_t max_col_contrib = ceilf((2.0F * col_scale) + 2.0F);
  matgen_value_t max_contributions_per_source =
      max_row_contrib * max_col_contrib;

  // Use 1.5x safety factor for edge cases and rounding
  size_t estimated_nnz = (size_t)((matgen_value_t)source->nnz *
                                  max_contributions_per_source * 1.5F);

  // Ensure minimum buffer size
  if (estimated_nnz < source->nnz * 4) {
    estimated_nnz = source->nnz * 4;
  }

  MATGEN_LOG_DEBUG("Estimated output NNZ: %zu", estimated_nnz);

  // Create triplet buffer
  matgen_triplet_buffer_t* buffer = matgen_triplet_buffer_create(estimated_nnz);
  if (!buffer) {
    MATGEN_LOG_ERROR("Failed to create triplet buffer");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

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

      // For bilinear interpolation, a source cell at (src_row, src_col)
      // contributes to destination cells where it appears as one of the 4
      // bilinear neighbors.
      //
      // A destination cell (dst_row, dst_col) uses bilinear interpolation from:
      //   src_y = dst_row / row_scale (fractional)
      //   src_x = dst_col / col_scale (fractional)
      //   neighbors: floor(src_y), ceil(src_y), floor(src_x), ceil(src_x)
      //
      // So source (src_row, src_col) contributes when:
      //   floor(dst_row / row_scale) == src_row OR ceil(dst_row / row_scale) ==
      //   src_row floor(dst_col / col_scale) == src_col OR ceil(dst_col /
      //   col_scale) == src_col

      // Calculate destination range for rows
      // Combined range: (src_row-1) * row_scale < dst_row < (src_row+1) *
      // row_scale Use fmax to avoid negative values before casting to unsigned
      matgen_value_t dst_row_start_f = (matgen_value_t)fmax(
          0.0, ((matgen_value_t)src_row - 1.0) * row_scale);
      matgen_value_t dst_row_end_f =
          (matgen_value_t)((matgen_value_t)src_row + 1.0) * row_scale;
      matgen_value_t dst_col_start_f = (matgen_value_t)fmax(
          0.0, ((matgen_value_t)src_col - 1.0) * col_scale);
      matgen_value_t dst_col_end_f =
          (matgen_value_t)((matgen_value_t)src_col + 1.0) * col_scale;

      // Convert to integer indices
      matgen_index_t dst_row_start = (matgen_index_t)ceil(dst_row_start_f);
      matgen_index_t dst_row_end = (matgen_index_t)ceil(dst_row_end_f);
      matgen_index_t dst_col_start = (matgen_index_t)ceil(dst_col_start_f);
      matgen_index_t dst_col_end = (matgen_index_t)ceil(dst_col_end_f);

      // Clamp to valid range
      dst_row_start = MATGEN_CLAMP(dst_row_start, 0, new_rows);
      dst_row_end = MATGEN_CLAMP(dst_row_end, 0, new_rows);
      dst_col_start = MATGEN_CLAMP(dst_col_start, 0, new_cols);
      dst_col_end = MATGEN_CLAMP(dst_col_end, 0, new_cols);

      // For each destination cell in this neighborhood, calculate bilinear
      // weight
      for (matgen_index_t dst_row = dst_row_start; dst_row < dst_row_end;
           dst_row++) {
        for (matgen_index_t dst_col = dst_col_start; dst_col < dst_col_end;
             dst_col++) {
          // Map destination cell back to source coordinates (fractional)
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

          // Determine which of the 4 neighbors we are and calculate bilinear
          // weight
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
          // If none of the above, weight remains 0

          // Add weighted contribution if non-zero
          if (weight > 1e-12) {
            matgen_value_t contribution = src_val * weight;
            err = matgen_triplet_buffer_add(buffer, dst_row, dst_col,
                                            contribution);
            if (err != MATGEN_SUCCESS) {
              MATGEN_LOG_ERROR("Failed to add entry to buffer at (%llu, %llu)",
                               (unsigned long long)dst_row,
                               (unsigned long long)dst_col);
              matgen_triplet_buffer_destroy(buffer);
              return err;
            }
          }
        }
      }
    }
  }

  size_t total_triplets = matgen_triplet_buffer_size(buffer);
  MATGEN_LOG_DEBUG("Generated %zu triplets", total_triplets);

  // Create COO matrix using sequential backend explicitly
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

  // Sort and sum duplicates using sequential policy
  MATGEN_LOG_DEBUG("Sorting and summing duplicates...");
  err = matgen_coo_sort_with_policy(coo, MATGEN_EXEC_SEQ);
  if (err != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to sort COO matrix");
    matgen_coo_destroy(coo);
    return err;
  }

  err = matgen_coo_sum_duplicates_with_policy(coo, MATGEN_EXEC_SEQ);
  if (err != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to sum duplicates");
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

  MATGEN_LOG_DEBUG("Bilinear scaling (SEQ) completed: output NNZ = %zu",
                   (*result)->nnz);

  return MATGEN_SUCCESS;
}
