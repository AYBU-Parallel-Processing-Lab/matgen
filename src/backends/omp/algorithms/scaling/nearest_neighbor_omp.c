#include "../../internal/nearest_neighbor_omp.h"

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/core/execution/policy.h"
#include "matgen/core/matrix/conversion.h"
#include "matgen/core/matrix/coo.h"
#include "matgen/core/types.h"
#include "matgen/utils/log.h"

// Thread-local triplet buffer
typedef struct {
  matgen_size_t capacity;
  matgen_size_t count;
  matgen_index_t* rows;
  matgen_index_t* cols;
  matgen_value_t* vals;
} thread_local_buffer_t;

static thread_local_buffer_t* create_thread_buffer(matgen_size_t capacity) {
  thread_local_buffer_t* buf = malloc(sizeof(thread_local_buffer_t));
  if (!buf) {
    return NULL;
  }

  buf->capacity = capacity;
  buf->count = 0;
  buf->rows = malloc(capacity * sizeof(matgen_index_t));
  buf->cols = malloc(capacity * sizeof(matgen_index_t));
  buf->vals = malloc(capacity * sizeof(matgen_value_t));

  if (!buf->rows || !buf->cols || !buf->vals) {
    free(buf->rows);
    free(buf->cols);
    free(buf->vals);
    free(buf);
    return NULL;
  }

  return buf;
}

static void destroy_thread_buffer(thread_local_buffer_t* buf) {
  if (!buf) {
    return;
  }
  free(buf->rows);
  free(buf->cols);
  free(buf->vals);
  free(buf);
}

static matgen_error_t add_to_thread_buffer(thread_local_buffer_t* buf,
                                           matgen_index_t row,
                                           matgen_index_t col,
                                           matgen_value_t val) {
  if (buf->count >= buf->capacity) {
    matgen_size_t new_capacity = buf->capacity * 2;
    matgen_index_t* new_rows =
        realloc(buf->rows, new_capacity * sizeof(matgen_index_t));
    matgen_index_t* new_cols =
        realloc(buf->cols, new_capacity * sizeof(matgen_index_t));
    matgen_value_t* new_vals =
        realloc(buf->vals, new_capacity * sizeof(matgen_value_t));

    if (!new_rows || !new_cols || !new_vals) {
      free(new_rows);
      free(new_cols);
      free(new_vals);
      return MATGEN_ERROR_OUT_OF_MEMORY;
    }

    buf->rows = new_rows;
    buf->cols = new_cols;
    buf->vals = new_vals;
    buf->capacity = new_capacity;
  }

  buf->rows[buf->count] = row;
  buf->cols[buf->count] = col;
  buf->vals[buf->count] = val;
  buf->count++;

  return MATGEN_SUCCESS;
}

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

  MATGEN_LOG_DEBUG(
      "Nearest neighbor scaling (OMP): %llu×%llu -> %llu×%llu "
      "(scale: %.3fx%.3f)",
      (unsigned long long)source->rows, (unsigned long long)source->cols,
      (unsigned long long)new_rows, (unsigned long long)new_cols, row_scale,
      col_scale);

  int num_threads = omp_get_max_threads();
  MATGEN_LOG_DEBUG("Using %d OpenMP threads", num_threads);

  // Estimate output NNZ per thread
  size_t estimated_nnz_total =
      (size_t)((matgen_value_t)source->nnz * row_scale * col_scale * 1.1);
  size_t estimated_per_thread = (estimated_nnz_total / num_threads) + 1000;

  // Allocate thread-local buffers
  thread_local_buffer_t** thread_buffers = (thread_local_buffer_t**)malloc(
      num_threads * sizeof(thread_local_buffer_t*));
  if (!thread_buffers) {
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  for (int i = 0; i < num_threads; i++) {
    thread_buffers[i] = create_thread_buffer(estimated_per_thread);
    if (!thread_buffers[i]) {
      for (int j = 0; j < i; j++) {
        destroy_thread_buffer(thread_buffers[j]);
      }
      free((void*)thread_buffers);
      return MATGEN_ERROR_OUT_OF_MEMORY;
    }
  }

  matgen_error_t err = MATGEN_SUCCESS;

// Process rows in parallel
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    thread_local_buffer_t* my_buffer = thread_buffers[tid];

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
                add_to_thread_buffer(my_buffer, dr, dc, cell_val);
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
    for (int i = 0; i < num_threads; i++) {
      destroy_thread_buffer(thread_buffers[i]);
    }
    free((void*)thread_buffers);
    return err;
  }

  // Merge thread buffers
  matgen_size_t total_triplets = 0;
  for (int i = 0; i < num_threads; i++) {
    total_triplets += thread_buffers[i]->count;
  }

  MATGEN_LOG_DEBUG("Generated %zu triplets from %d threads", total_triplets,
                   num_threads);

  // Create COO matrix
  matgen_coo_matrix_t* coo =
      matgen_coo_create(new_rows, new_cols, total_triplets);
  if (!coo) {
    for (int i = 0; i < num_threads; i++) {
      destroy_thread_buffer(thread_buffers[i]);
    }
    free((void*)thread_buffers);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Copy from thread buffers to COO
  matgen_size_t offset = 0;
  for (int i = 0; i < num_threads; i++) {
    thread_local_buffer_t* buf = thread_buffers[i];
    memcpy(&coo->row_indices[offset], buf->rows,
           buf->count * sizeof(matgen_index_t));
    memcpy(&coo->col_indices[offset], buf->cols,
           buf->count * sizeof(matgen_index_t));
    memcpy(&coo->values[offset], buf->vals,
           buf->count * sizeof(matgen_value_t));
    offset += buf->count;
  }

  coo->nnz = total_triplets;
  coo->is_sorted = false;

  // Clean up thread buffers
  for (int i = 0; i < num_threads; i++) {
    destroy_thread_buffer(thread_buffers[i]);
  }
  free((void*)thread_buffers);

  // Sort and handle duplicates using OMP
  MATGEN_LOG_DEBUG("Sorting and handling duplicates (policy: %d)...",
                   collision_policy);

  err = matgen_coo_sort_with_policy(coo, MATGEN_EXEC_PAR);
  if (err != MATGEN_SUCCESS) {
    matgen_coo_destroy(coo);
    return err;
  }

  if (collision_policy == MATGEN_COLLISION_SUM) {
    err = matgen_coo_sum_duplicates_with_policy(coo, MATGEN_EXEC_PAR);
  } else {
    err = matgen_coo_merge_duplicates_with_policy(coo, collision_policy,
                                                  MATGEN_EXEC_PAR);
  }

  if (err != MATGEN_SUCCESS) {
    matgen_coo_destroy(coo);
    return err;
  }

  MATGEN_LOG_DEBUG("After deduplication: %zu entries", coo->nnz);

  // Convert to CSR using OMP
  *result = matgen_coo_to_csr_with_policy(coo, MATGEN_EXEC_PAR);
  matgen_coo_destroy(coo);

  if (!(*result)) {
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  MATGEN_LOG_DEBUG("Nearest neighbor scaling (OMP) completed: output NNZ = %zu",
                   (*result)->nnz);

  return MATGEN_SUCCESS;
}
