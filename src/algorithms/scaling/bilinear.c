#include "matgen/algorithms/scaling/bilinear.h"

#include <math.h>
#include <stdlib.h>

#include "matgen/core/conversion.h"
#include "matgen/core/coo_matrix.h"
#include "matgen/utils/accumulator.h"
#include "matgen/utils/log.h"

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

  // Calculate scale factors using double precision
  double row_scale = (double)new_rows / (double)source->rows;
  double col_scale = (double)new_cols / (double)source->cols;

  MATGEN_LOG_DEBUG(
      "Bilinear scaling: %llu×%llu -> %llu×%llu (scale: %.3fx%.3f)",
      (unsigned long long)source->rows, (unsigned long long)source->cols,
      (unsigned long long)new_rows, (unsigned long long)new_cols, row_scale,
      col_scale);

  // Bilinear creates up to 4 neighbors per source entry
  // Estimate conservatively
  size_t estimated_nnz = (size_t)((double)source->nnz * 4.0 * 1.5);

  MATGEN_LOG_DEBUG("Estimated output NNZ: %zu", estimated_nnz);

  // Create accumulator with SUM policy (bilinear weights are summed)
  matgen_accumulator_t* acc =
      matgen_accumulator_create(estimated_nnz, MATGEN_COLLISION_SUM);
  if (!acc) {
    MATGEN_LOG_ERROR("Failed to create accumulator");
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

      // Calculate continuous target position (center of source cell)
      // Source cell [src_row, src_row+1) maps to target range
      double src_row_center = (double)src_row + 0.5;
      double src_col_center = (double)src_col + 0.5;

      // Map to target space
      double dst_row_float = src_row_center * row_scale;
      double dst_col_float = src_col_center * col_scale;

      // Find the 4 surrounding target cells
      // We want the cells whose centers surround our mapped point
      matgen_index_t row0 = (matgen_index_t)floor(dst_row_float - 0.5);
      matgen_index_t row1 = row0 + 1;
      matgen_index_t col0 = (matgen_index_t)floor(dst_col_float - 0.5);
      matgen_index_t col1 = col0 + 1;

      // Clamp to valid range
      if (row0 >= new_rows) {
        row0 = new_rows - 1;
      }

      if (row1 >= new_rows) {
        row1 = new_rows - 1;
      }

      if (col0 >= new_cols) {
        col0 = new_cols - 1;
      }

      if (col1 >= new_cols) {
        col1 = new_cols - 1;
      }

      // Calculate fractional distances within the cell
      // Distance from row0's center to our point
      double row0_center = (double)row0 + 0.5;
      double col0_center = (double)col0 + 0.5;

      double row_frac = (dst_row_float - row0_center);
      double col_frac = (dst_col_float - col0_center);

      // Clamp fractions to [0, 1] range
      if (row_frac < 0.0) {
        row_frac = 0.0;
      }

      if (row_frac > 1.0) {
        row_frac = 1.0;
      }

      if (col_frac < 0.0) {
        col_frac = 0.0;
      }

      if (col_frac > 1.0) {
        col_frac = 1.0;
      }

      // Calculate bilinear interpolation weights
      // Weight decreases with distance from the point
      double w00 = (1.0 - row_frac) * (1.0 - col_frac);  // Top-left
      double w01 = (1.0 - row_frac) * col_frac;          // Top-right
      double w10 = row_frac * (1.0 - col_frac);          // Bottom-left
      double w11 = row_frac * col_frac;                  // Bottom-right

      // Normalize weights to conserve value (should sum to ~1.0, but ensure it)
      double weight_sum = w00 + w01 + w10 + w11;
      if (weight_sum > 1e-10) {
        w00 /= weight_sum;
        w01 /= weight_sum;
        w10 /= weight_sum;
        w11 /= weight_sum;
      } else {
        // Degenerate case: distribute equally
        w00 = w01 = w10 = w11 = 0.25;
      }

      // Distribute value to 4 neighbors with bilinear weights
      // Only add non-negligible contributions
      const double epsilon = 1e-12;

      if (w00 > epsilon) {
        err = matgen_accumulator_add(acc, row0, col0, src_val * w00);
        if (err != MATGEN_SUCCESS) {
          MATGEN_LOG_ERROR("Failed to add entry to accumulator");
          matgen_accumulator_destroy(acc);
          return err;
        }
      }

      if (w01 > epsilon && col1 != col0) {
        err = matgen_accumulator_add(acc, row0, col1, src_val * w01);
        if (err != MATGEN_SUCCESS) {
          MATGEN_LOG_ERROR("Failed to add entry to accumulator");
          matgen_accumulator_destroy(acc);
          return err;
        }
      }

      if (w10 > epsilon && row1 != row0) {
        err = matgen_accumulator_add(acc, row1, col0, src_val * w10);
        if (err != MATGEN_SUCCESS) {
          MATGEN_LOG_ERROR("Failed to add entry to accumulator");
          matgen_accumulator_destroy(acc);
          return err;
        }
      }

      if (w11 > epsilon && row1 != row0 && col1 != col0) {
        err = matgen_accumulator_add(acc, row1, col1, src_val * w11);
        if (err != MATGEN_SUCCESS) {
          MATGEN_LOG_ERROR("Failed to add entry to accumulator");
          matgen_accumulator_destroy(acc);
          return err;
        }
      }
    }
  }

  size_t final_size = matgen_accumulator_size(acc);
  double load_factor = matgen_accumulator_load_factor(acc);

  MATGEN_LOG_DEBUG("Accumulated %zu entries (estimated %zu, load factor: %.2f)",
                   final_size, estimated_nnz, load_factor);

  // Convert accumulator to COO matrix
  matgen_coo_matrix_t* coo = matgen_accumulator_to_coo(acc, new_rows, new_cols);
  matgen_accumulator_destroy(acc);

  if (!coo) {
    MATGEN_LOG_ERROR("Failed to convert accumulator to COO matrix");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Convert COO to CSR
  *result = matgen_coo_to_csr(coo);
  matgen_coo_destroy(coo);

  if (!(*result)) {
    MATGEN_LOG_ERROR("Failed to convert COO to CSR matrix");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  MATGEN_LOG_DEBUG("Bilinear scaling completed: output NNZ = %zu",
                   (*result)->nnz);

  return MATGEN_SUCCESS;
}
