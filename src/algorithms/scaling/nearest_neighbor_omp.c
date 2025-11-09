#include "matgen/algorithms/scaling/nearest_neighbor_omp.h"

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/core/conversion.h"
#include "matgen/core/coo_matrix.h"
#include "matgen/core/types.h"
#include "matgen/utils/log.h"
#include "matgen/utils/triplet_buffer.h"

// Minimum number of rows per thread to enable parallelization
#define MATGEN_MIN_ROWS_PER_THREAD 100

// Initial capacity for thread-local triplet buffers
#define MATGEN_INITIAL_TRIPLET_CAPACITY 1024

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

  matgen_value_t row_scale =
      (matgen_value_t)new_rows / (matgen_value_t)source->rows;
  matgen_value_t col_scale =
      (matgen_value_t)new_cols / (matgen_value_t)source->cols;

  int num_threads = omp_get_max_threads();
  MATGEN_LOG_DEBUG(
      "Nearest neighbor scaling (OpenMP, %d threads): %llu×%llu -> %llu×%llu "
      "(scale: %.3fx%.3f)",
      num_threads, (unsigned long long)source->rows,
      (unsigned long long)source->cols, (unsigned long long)new_rows,
      (unsigned long long)new_cols, row_scale, col_scale);

  // Estimate output NNZ: each source entry expands to ~scale² target entries
  size_t estimated_nnz =
      (size_t)((matgen_value_t)source->nnz * row_scale * col_scale * 1.2);

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

        // Calculate block size
        matgen_index_t block_rows = dst_row_end - dst_row_start;
        matgen_index_t block_cols = dst_col_end - dst_col_start;
        matgen_value_t block_size =
            (matgen_value_t)block_rows * (matgen_value_t)block_cols;

        // Distribute value uniformly across block to preserve mass
        // Each cell gets: src_val / block_size
        matgen_value_t cell_val = src_val / block_size;

        // Fill entire block
        for (matgen_index_t dr = dst_row_start; dr < dst_row_end; dr++) {
          for (matgen_index_t dc = dst_col_start; dc < dst_col_end; dc++) {
            matgen_error_t local_err =
                matgen_triplet_buffer_add(local_buffer, dr, dc, cell_val);
            if (local_err != MATGEN_SUCCESS) {
#pragma omp atomic write
              shared_error = 1;
              break;
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

  // Sort and handle duplicates according to collision policy
  MATGEN_LOG_DEBUG("Sorting and handling duplicates (policy: %d)...",
                   collision_policy);
  matgen_coo_sort(coo);

  // Handle duplicates based on collision policy
  matgen_error_t err = MATGEN_SUCCESS;
  if (collision_policy == MATGEN_COLLISION_SUM) {
    // Sum duplicates (fast path)
    matgen_coo_sum_duplicates(coo);
  } else if (collision_policy == MATGEN_COLLISION_AVG ||
             collision_policy == MATGEN_COLLISION_MAX) {
    // General merge with specified policy
    err = matgen_coo_merge_duplicates(coo, collision_policy);
    if (err != MATGEN_SUCCESS) {
      MATGEN_LOG_ERROR("Failed to merge duplicates");
      matgen_coo_destroy(coo);
      return err;
    }
  }

  MATGEN_LOG_DEBUG("After deduplication: %zu entries", coo->nnz);

  // Convert to CSR
  *result = matgen_coo_to_csr(coo);
  matgen_coo_destroy(coo);

  if (!(*result)) {
    MATGEN_LOG_ERROR("Failed to convert COO to CSR matrix");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  MATGEN_LOG_DEBUG(
      "Nearest neighbor scaling (OpenMP) completed: output NNZ = %zu",
      (*result)->nnz);

  return MATGEN_SUCCESS;
}
