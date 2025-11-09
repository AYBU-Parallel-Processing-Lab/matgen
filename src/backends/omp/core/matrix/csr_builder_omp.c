#include "backends/omp/internal/csr_builder_omp.h"

#include <omp.h>
#include <stdlib.h>

#include "backends/omp/internal/csr_omp.h"
#include "core/matrix/csr_builder_internal.h"
#include "matgen/utils/log.h"

// =============================================================================
// Builder Creation and Destruction
// =============================================================================

matgen_csr_builder_t* matgen_csr_builder_create_omp(matgen_index_t rows,
                                                    matgen_index_t cols,
                                                    matgen_size_t est_nnz) {
  matgen_csr_builder_t* builder =
      (matgen_csr_builder_t*)malloc(sizeof(matgen_csr_builder_t));
  if (!builder) {
    return NULL;
  }

  builder->rows = rows;
  builder->cols = cols;
  builder->est_nnz = est_nnz;
  builder->policy = MATGEN_EXEC_PAR;
  builder->collision_policy = MATGEN_COLLISION_SUM;
  builder->finalized = false;

  // Get number of threads
  int num_threads;
#pragma omp parallel
  {
#pragma omp single
    num_threads = omp_get_num_threads();
  }
  builder->backend.omp.num_threads = num_threads;

  MATGEN_LOG_DEBUG("Creating CSR builder (OMP) with %d threads", num_threads);

  // Allocate thread-local builders
  builder->backend.omp.thread_builders =
      (csr_thread_builder_t*)calloc(num_threads, sizeof(csr_thread_builder_t));
  if (!builder->backend.omp.thread_builders) {
    free(builder);
    return NULL;
  }

  // Partition rows among threads
  matgen_index_t rows_per_thread = (rows + num_threads - 1) / num_threads;

  for (int tid = 0; tid < num_threads; tid++) {
    csr_thread_builder_t* tb = &builder->backend.omp.thread_builders[tid];

    tb->row_start = tid * rows_per_thread;
    tb->row_end = (tid + 1) * rows_per_thread;
    if (tb->row_end > rows) {
      tb->row_end = rows;
    }

    matgen_index_t num_rows = tb->row_end - tb->row_start;
    if (num_rows > 0) {
      tb->rows = (csr_row_buffer_t*)malloc(num_rows * sizeof(csr_row_buffer_t));
      if (!tb->rows) {
        // Cleanup and fail
        for (int i = 0; i < tid; i++) {
          free(builder->backend.omp.thread_builders[i].rows);
        }
        free(builder->backend.omp.thread_builders);
        free(builder);
        return NULL;
      }

      // Initialize row buffers
      for (matgen_index_t r = 0; r < num_rows; r++) {
        csr_builder_init_row_buffer(&tb->rows[r]);
      }
    } else {
      tb->rows = NULL;
    }

    tb->entry_count = 0;
  }

  MATGEN_LOG_DEBUG("CSR builder created: %llu x %llu, est_nnz=%zu",
                   (unsigned long long)rows, (unsigned long long)cols, est_nnz);

  return builder;
}

void matgen_csr_builder_destroy_omp(matgen_csr_builder_t* builder) {
  if (!builder) {
    return;
  }

  if (builder->backend.omp.thread_builders) {
    for (int tid = 0; tid < builder->backend.omp.num_threads; tid++) {
      csr_thread_builder_t* tb = &builder->backend.omp.thread_builders[tid];
      if (tb->rows) {
        for (matgen_index_t r = 0; r < (tb->row_end - tb->row_start); r++) {
          csr_builder_destroy_row_buffer(&tb->rows[r]);
        }
        free(tb->rows);
      }
    }
    free(builder->backend.omp.thread_builders);
  }

  free(builder);
}

// =============================================================================
// Entry Addition
// =============================================================================

matgen_error_t matgen_csr_builder_add_omp(matgen_csr_builder_t* builder,
                                          matgen_index_t row,
                                          matgen_index_t col,
                                          matgen_value_t value) {
  if (!builder || builder->finalized) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (row >= builder->rows || col >= builder->cols) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Find which thread owns this row
  int tid = 0;
  matgen_index_t rows_per_thread =
      (builder->rows + builder->backend.omp.num_threads - 1) /
      builder->backend.omp.num_threads;
  tid = (int)(row / rows_per_thread);
  if (tid >= builder->backend.omp.num_threads) {
    tid = builder->backend.omp.num_threads - 1;
  }

  csr_thread_builder_t* tb = &builder->backend.omp.thread_builders[tid];

  // Get local row index
  matgen_index_t local_row = row - tb->row_start;

  // Add to row buffer
  matgen_error_t err =
      csr_builder_add_to_row_buffer(&tb->rows[local_row], col, value);
  if (err == MATGEN_SUCCESS) {
    tb->entry_count++;
  }

  return err;
}

// =============================================================================
// Query Functions
// =============================================================================

matgen_size_t matgen_csr_builder_get_nnz_omp(
    const matgen_csr_builder_t* builder) {
  if (!builder) {
    return 0;
  }

  matgen_size_t total = 0;
  for (int tid = 0; tid < builder->backend.omp.num_threads; tid++) {
    total += builder->backend.omp.thread_builders[tid].entry_count;
  }
  return total;
}

// =============================================================================
// Finalization
// =============================================================================

matgen_csr_matrix_t* matgen_csr_builder_finalize_omp(
    matgen_csr_builder_t* builder) {
  if (!builder || builder->finalized) {
    return NULL;
  }

  MATGEN_LOG_DEBUG("Finalizing CSR builder (OMP)");

  builder->finalized = true;

  // Phase 1: Count entries per row (parallel)
  matgen_size_t* row_counts =
      (matgen_size_t*)calloc(builder->rows, sizeof(matgen_size_t));
  if (!row_counts) {
    return NULL;
  }

  int tid;

#pragma omp parallel for
  for (tid = 0; tid < builder->backend.omp.num_threads; tid++) {
    csr_thread_builder_t* tb = &builder->backend.omp.thread_builders[tid];

    for (matgen_index_t r = 0; r < (tb->row_end - tb->row_start); r++) {
      csr_row_buffer_t* row_buf = &tb->rows[r];
      matgen_size_t count = 0;

      // Count hash table entries
      for (int i = 0; i < MATGEN_CSR_BUILDER_HASH_SIZE; i++) {
        if (row_buf->hash_table[i].col != (matgen_index_t)-1) {
          count++;
        }
      }

      // Add overflow entries
      count += row_buf->overflow_count;

      row_counts[tb->row_start + r] = count;
    }
  }

  // Phase 2: Compute row_ptr (prefix sum - sequential)
  matgen_size_t total_nnz = 0;
  for (matgen_index_t r = 0; r < builder->rows; r++) {
    total_nnz += row_counts[r];
  }

  MATGEN_LOG_DEBUG("Total unique entries: %zu", total_nnz);

  // Create CSR matrix
  matgen_csr_matrix_t* csr =
      matgen_csr_create_omp(builder->rows, builder->cols, total_nnz);
  if (!csr) {
    free(row_counts);
    return NULL;
  }

  csr->row_ptr[0] = 0;
  for (matgen_index_t r = 0; r < builder->rows; r++) {
    csr->row_ptr[r + 1] = csr->row_ptr[r] + row_counts[r];
  }

  free(row_counts);

  int row_idx;

// Phase 3: Extract and sort entries (parallel)
#pragma omp parallel for schedule(dynamic, 128)
  for (row_idx = 0; row_idx < builder->rows; row_idx++) {
    // Find thread that owns this row
    int tid = 0;
    matgen_index_t rows_per_thread =
        (builder->rows + builder->backend.omp.num_threads - 1) /
        builder->backend.omp.num_threads;
    tid = row_idx / (int)rows_per_thread;
    if (tid >= builder->backend.omp.num_threads) {
      tid = builder->backend.omp.num_threads - 1;
    }

    csr_thread_builder_t* tb = &builder->backend.omp.thread_builders[tid];
    matgen_index_t local_row = row_idx - tb->row_start;

    if (local_row < 0 || local_row >= (tb->row_end - tb->row_start)) {
      continue;  // Row not in this thread's range
    }

    csr_row_buffer_t* row_buf = &tb->rows[local_row];

    matgen_size_t row_start = csr->row_ptr[row_idx];
    matgen_size_t row_nnz = csr->row_ptr[row_idx + 1] - row_start;

    if (row_nnz == 0) {
      continue;
    }

    // Collect entries from hash table and overflow
    csr_hash_entry_t* entries =
        (csr_hash_entry_t*)malloc(row_nnz * sizeof(csr_hash_entry_t));
    if (!entries) {
      continue;  // Error, but continue processing other rows
    }

    matgen_size_t idx = 0;

    // Collect from hash table
    for (int i = 0; i < MATGEN_CSR_BUILDER_HASH_SIZE; i++) {
      if (row_buf->hash_table[i].col != (matgen_index_t)-1) {
        entries[idx++] = row_buf->hash_table[i];
      }
    }

    // Collect from overflow
    for (matgen_size_t i = 0; i < row_buf->overflow_count; i++) {
      entries[idx++] = row_buf->overflow[i];
    }

    // Sort by column
    qsort(entries, row_nnz, sizeof(csr_hash_entry_t),
          csr_builder_compare_entries);

    // Write to CSR
    for (matgen_size_t i = 0; i < row_nnz; i++) {
      csr->col_indices[row_start + i] = entries[i].col;
      csr->values[row_start + i] = entries[i].val;
    }

    free(entries);
  }

  // Cleanup
  matgen_csr_builder_destroy_omp(builder);

  MATGEN_LOG_DEBUG("CSR builder finalized: %llu x %llu, nnz=%zu",
                   (unsigned long long)csr->rows, (unsigned long long)csr->cols,
                   csr->nnz);

  return csr;
}
