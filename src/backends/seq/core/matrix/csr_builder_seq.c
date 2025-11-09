#include "backends/seq/internal/csr_builder_seq.h"

#include <stdlib.h>

#include "backends/seq/internal/csr_seq.h"
#include "core/matrix/csr_builder_internal.h"
#include "matgen/utils/log.h"

// =============================================================================
// Builder Creation and Destruction
// =============================================================================

matgen_csr_builder_t* matgen_csr_builder_create_seq(matgen_index_t rows,
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
  builder->policy = MATGEN_EXEC_SEQ;
  builder->collision_policy = MATGEN_COLLISION_SUM;
  builder->finalized = false;
  builder->backend.seq.entry_count = 0;

  // Allocate row buffers
  builder->backend.seq.row_buffers =
      (csr_row_buffer_t*)malloc(rows * sizeof(csr_row_buffer_t));
  if (!builder->backend.seq.row_buffers) {
    free(builder);
    return NULL;
  }

  // Initialize all row buffers
  for (matgen_index_t r = 0; r < rows; r++) {
    csr_builder_init_row_buffer(&builder->backend.seq.row_buffers[r]);
  }

  MATGEN_LOG_DEBUG("Created CSR builder (SEQ) %llu x %llu, est_nnz=%zu",
                   (unsigned long long)rows, (unsigned long long)cols, est_nnz);

  return builder;
}

void matgen_csr_builder_destroy_seq(matgen_csr_builder_t* builder) {
  if (!builder) {
    return;
  }

  if (builder->backend.seq.row_buffers) {
    for (matgen_index_t r = 0; r < builder->rows; r++) {
      csr_builder_destroy_row_buffer(&builder->backend.seq.row_buffers[r]);
    }
    free(builder->backend.seq.row_buffers);
  }

  free(builder);
}

// =============================================================================
// Entry Addition
// =============================================================================

matgen_error_t matgen_csr_builder_add_seq(matgen_csr_builder_t* builder,
                                          matgen_index_t row,
                                          matgen_index_t col,
                                          matgen_value_t value) {
  if (!builder || builder->finalized) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (row >= builder->rows || col >= builder->cols) {
    MATGEN_LOG_ERROR("Index out of bounds: (%llu, %llu) in %llu x %llu matrix",
                     (unsigned long long)row, (unsigned long long)col,
                     (unsigned long long)builder->rows,
                     (unsigned long long)builder->cols);
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Add to row buffer
  matgen_error_t err = csr_builder_add_to_row_buffer(
      &builder->backend.seq.row_buffers[row], col, value);
  if (err == MATGEN_SUCCESS) {
    builder->backend.seq.entry_count++;
  }

  return err;
}

// =============================================================================
// Query Functions
// =============================================================================

matgen_size_t matgen_csr_builder_get_nnz_seq(
    const matgen_csr_builder_t* builder) {
  if (!builder) {
    return 0;
  }
  return builder->backend.seq.entry_count;
}

// =============================================================================
// Finalization
// =============================================================================

matgen_csr_matrix_t* matgen_csr_builder_finalize_seq(
    matgen_csr_builder_t* builder) {
  if (!builder || builder->finalized) {
    return NULL;
  }

  MATGEN_LOG_DEBUG("Finalizing CSR builder (SEQ)");

  builder->finalized = true;

  // Phase 1: Count entries per row
  matgen_size_t total_nnz = 0;
  matgen_size_t* row_counts =
      (matgen_size_t*)malloc(builder->rows * sizeof(matgen_size_t));
  if (!row_counts) {
    return NULL;
  }

  for (matgen_index_t r = 0; r < builder->rows; r++) {
    csr_row_buffer_t* row_buf = &builder->backend.seq.row_buffers[r];
    matgen_size_t count = 0;

    // Count hash table entries
    for (int i = 0; i < MATGEN_CSR_BUILDER_HASH_SIZE; i++) {
      if (row_buf->hash_table[i].col != (matgen_index_t)-1) {
        count++;
      }
    }

    // Add overflow entries
    count += row_buf->overflow_count;

    row_counts[r] = count;
    total_nnz += count;
  }

  MATGEN_LOG_DEBUG("Total unique entries: %zu (input entries: %zu)", total_nnz,
                   builder->backend.seq.entry_count);

  // Phase 2: Create CSR matrix and compute row_ptr
  matgen_csr_matrix_t* csr =
      matgen_csr_create_seq(builder->rows, builder->cols, total_nnz);
  if (!csr) {
    free(row_counts);
    return NULL;
  }

  csr->row_ptr[0] = 0;
  for (matgen_index_t r = 0; r < builder->rows; r++) {
    csr->row_ptr[r + 1] = csr->row_ptr[r] + row_counts[r];
  }

  free(row_counts);

  // Phase 3: Extract and sort entries for each row
  for (matgen_index_t r = 0; r < builder->rows; r++) {
    csr_row_buffer_t* row_buf = &builder->backend.seq.row_buffers[r];

    matgen_size_t row_start = csr->row_ptr[r];
    matgen_size_t row_nnz = csr->row_ptr[r + 1] - row_start;

    if (row_nnz == 0) {
      continue;
    }

    // Collect entries from hash table and overflow
    csr_hash_entry_t* entries =
        (csr_hash_entry_t*)malloc(row_nnz * sizeof(csr_hash_entry_t));
    if (!entries) {
      matgen_csr_destroy(csr);
      matgen_csr_builder_destroy_seq(builder);
      return NULL;
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
  matgen_csr_builder_destroy_seq(builder);

  MATGEN_LOG_DEBUG("CSR builder finalized: %llu x %llu, nnz=%zu",
                   (unsigned long long)csr->rows, (unsigned long long)csr->cols,
                   csr->nnz);

  return csr;
}
