#include "matgen/algorithms/scaling/bilinear_omp.h"

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/core/conversion.h"
#include "matgen/core/coo_matrix.h"
#include "matgen/core/types.h"
#include "matgen/utils/log.h"
#include "matgen/utils/triplet_buffer.h"

// Threshold for using stack vs heap allocation for weights
#define MATGEN_BILINEAR_STACK_THRESHOLD 64

// Minimum number of rows per thread to enable parallelization
#define MATGEN_MIN_ROWS_PER_THREAD 100

// Initial capacity for thread-local triplet buffers
#define MATGEN_INITIAL_TRIPLET_CAPACITY 1024

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

  int num_threads = omp_get_max_threads();
  MATGEN_LOG_DEBUG(
      "Bilinear scaling (OpenMP, %d threads): %zu×%zu -> %zu×%zu "
      "(scale: %.3fx%.3f)",
      num_threads, source->rows, source->cols, new_rows, new_cols, row_scale,
      col_scale);

  // Estimate output NNZ
  matgen_value_t avg_contributions_per_source =
      max((matgen_value_t)1.0, row_scale * col_scale);
  size_t estimated_nnz = (size_t)((matgen_value_t)source->nnz *
                                  avg_contributions_per_source * 1.2);

  MATGEN_LOG_DEBUG("Estimated output NNZ: %zu", estimated_nnz);

  // Check if parallelization is worthwhile
  int actual_threads = num_threads;
  if (source->rows < (matgen_size_t)MATGEN_MIN_ROWS_PER_THREAD * num_threads) {
    actual_threads = max(1, (int)(source->rows / MATGEN_MIN_ROWS_PER_THREAD));
  }

  if (actual_threads <= 1) {
    MATGEN_LOG_DEBUG(
        "Matrix too small for parallel processing, using 1 thread");
    actual_threads = 1;
  }

  // Create thread-local triplet buffers
  matgen_triplet_buffer_t** thread_buffers = (matgen_triplet_buffer_t**)calloc(
      actual_threads, sizeof(matgen_triplet_buffer_t*));
  if (!thread_buffers) {
    MATGEN_LOG_ERROR("Failed to allocate thread buffer array");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Initialize thread-local buffers
  size_t buffer_capacity = estimated_nnz / actual_threads;
  if (buffer_capacity < MATGEN_INITIAL_TRIPLET_CAPACITY) {
    buffer_capacity = MATGEN_INITIAL_TRIPLET_CAPACITY;
  }

  for (int t = 0; t < actual_threads; t++) {
    thread_buffers[t] = matgen_triplet_buffer_create(buffer_capacity);
    if (!thread_buffers[t]) {
      MATGEN_LOG_ERROR("Failed to create thread-local buffer %d", t);
      for (int i = 0; i < t; i++) {
        matgen_triplet_buffer_destroy(thread_buffers[i]);
      }
      free((void*)thread_buffers);
      return MATGEN_ERROR_OUT_OF_MEMORY;
    }
  }

  // Shared error flag
  int shared_error = 0;

// Parallel processing of source rows
#pragma omp parallel num_threads(actual_threads)
  {
    int thread_id = omp_get_thread_num();
    matgen_triplet_buffer_t* local_buffer = thread_buffers[thread_id];

    // Thread-local weight buffers
    matgen_value_t stack_weights[MATGEN_BILINEAR_STACK_THRESHOLD];
    matgen_value_t* heap_weights = NULL;

    int src_row;

#pragma omp for schedule(dynamic, 16)
    for (src_row = 0; src_row < (int)source->rows; src_row++) {
      // Check if another thread encountered an error
      if (shared_error) {
        continue;
      }

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
          matgen_error_t local_err = matgen_triplet_buffer_add(
              local_buffer, dst_row_start, dst_col_start, src_val);
          if (local_err != MATGEN_SUCCESS) {
#pragma omp atomic write
            shared_error = 1;
            break;
          }
          continue;
        }

        // Allocate weight buffer based on block size
        matgen_value_t* weights = NULL;
        if (block_size <= MATGEN_BILINEAR_STACK_THRESHOLD) {
          weights = stack_weights;
        } else {
          free(heap_weights);
          heap_weights =
              (matgen_value_t*)malloc(block_size * sizeof(matgen_value_t));
          if (!heap_weights) {
#pragma omp atomic write
            shared_error = 1;
            break;
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
              matgen_error_t local_err = matgen_triplet_buffer_add(
                  local_buffer, dst_row, dst_col, weighted_val);
              if (local_err != MATGEN_SUCCESS) {
#pragma omp atomic write
                shared_error = 1;
                break;
              }
            }
          }
          if (shared_error) {
            break;
          }
        }

        if (shared_error) {
          break;
        }
      }
    }

    // Cleanup thread-local heap weights
    free(heap_weights);
  }

  // Check for errors during parallel processing
  if (shared_error) {
    MATGEN_LOG_ERROR("Error occurred during parallel processing");
    for (int t = 0; t < actual_threads; t++) {
      matgen_triplet_buffer_destroy(thread_buffers[t]);
    }
    free((void*)thread_buffers);
    return MATGEN_ERROR_UNKNOWN;
  }

  // Calculate total size
  size_t total_triplets = 0;
  for (int t = 0; t < actual_threads; t++) {
    total_triplets += matgen_triplet_buffer_size(thread_buffers[t]);
  }

  MATGEN_LOG_DEBUG("Generated %zu triplets from %d threads", total_triplets,
                   actual_threads);

  // Create COO matrix and merge all triplets
  matgen_coo_matrix_t* coo =
      matgen_coo_create(new_rows, new_cols, total_triplets);
  if (!coo) {
    MATGEN_LOG_ERROR("Failed to create COO matrix");
    for (int t = 0; t < actual_threads; t++) {
      matgen_triplet_buffer_destroy(thread_buffers[t]);
    }
    free((void*)thread_buffers);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Concatenate all triplets
  size_t offset = 0;
  for (int t = 0; t < actual_threads; t++) {
    matgen_triplet_buffer_t* buf = thread_buffers[t];
    size_t buf_size = matgen_triplet_buffer_size(buf);

    memcpy(&coo->row_indices[offset], buf->rows,
           buf_size * sizeof(matgen_index_t));
    memcpy(&coo->col_indices[offset], buf->cols,
           buf_size * sizeof(matgen_index_t));
    memcpy(&coo->values[offset], buf->vals, buf_size * sizeof(matgen_value_t));

    offset += buf_size;
    matgen_triplet_buffer_destroy(buf);
  }
  free((void*)thread_buffers);

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
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  MATGEN_LOG_DEBUG("Bilinear scaling (OpenMP) completed: output NNZ = %zu",
                   (*result)->nnz);

  return MATGEN_SUCCESS;
}
