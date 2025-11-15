#ifndef MATGEN_CORE_MATRIX_CSR_BUILDER_INTERNAL_H
#define MATGEN_CORE_MATRIX_CSR_BUILDER_INTERNAL_H

/**
 * @file csr_builder_internal.h
 * @brief Internal data structures for CSR builder (shared across backends)
 */

#include <stdlib.h>

#include "matgen/core/execution/policy.h"
#include "matgen/core/types.h"

#ifdef MATGEN_HAS_OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Hash Table Configuration
// =============================================================================

#define MATGEN_CSR_BUILDER_HASH_SIZE 64  // Per-row hash table size
#define MATGEN_CSR_BUILDER_MAX_PROBE 8   // Max linear probing distance

// =============================================================================
// Shared Data Structures
// =============================================================================

/**
 * @brief Hash table entry for column-value pair
 */
typedef struct {
  matgen_index_t col;  // Column index (-1 = empty slot)
  matgen_value_t val;  // Accumulated value
} csr_hash_entry_t;

/**
 * @brief Per-row hash buffer with overflow handling
 *
 * Uses a small fixed-size hash table with linear probing.
 * Overflow entries are stored in a dynamic array.
 */
typedef struct {
  csr_hash_entry_t hash_table[MATGEN_CSR_BUILDER_HASH_SIZE];
  matgen_size_t overflow_count;
  matgen_size_t overflow_capacity;
  csr_hash_entry_t* overflow;  // Dynamic overflow array
} csr_row_buffer_t;

// =============================================================================
// Backend-Specific Structures
// =============================================================================

/**
 * @brief Thread-local builder (OpenMP backend)
 */
typedef struct {
  matgen_index_t row_start;   // First row this thread handles
  matgen_index_t row_end;     // Last row (exclusive)
  csr_row_buffer_t* rows;     // Array of row buffers for this thread's rows
  matgen_size_t entry_count;  // Total entries added by this thread
} csr_thread_builder_t;

/**
 * @brief Unified CSR builder structure (supports all backends)
 */
struct matgen_csr_builder {
  // Common fields (all backends)
  matgen_index_t rows;
  matgen_index_t cols;
  matgen_size_t est_nnz;
  matgen_exec_policy_t policy;
  matgen_collision_policy_t collision_policy;
  bool finalized;

  // Backend-specific data (union)
  union {
    // Sequential backend
    struct {
      csr_row_buffer_t* row_buffers;  // Array of row buffers (one per row)
      matgen_size_t entry_count;
    } seq;

    // OpenMP backend
    struct {
      int num_threads;
      csr_thread_builder_t* thread_builders;  // Array of thread-local builders
      omp_lock_t* row_locks;  // Per-row locks to prevent race conditions
    } omp;
  } backend;
};

// =============================================================================
// Common Utility Functions
// =============================================================================

/**
 * @brief Hash function for column indices
 */
static inline matgen_size_t csr_builder_hash_col(matgen_index_t col) {
  return col % MATGEN_CSR_BUILDER_HASH_SIZE;
}

/**
 * @brief Initialize a row buffer
 */
static inline void csr_builder_init_row_buffer(csr_row_buffer_t* row) {
  for (int i = 0; i < MATGEN_CSR_BUILDER_HASH_SIZE; i++) {
    row->hash_table[i].col = (matgen_index_t)-1;  // Mark as empty
    row->hash_table[i].val = (matgen_value_t)0.0;
  }
  row->overflow_count = 0;
  row->overflow_capacity = 0;
  row->overflow = NULL;
}

/**
 * @brief Destroy a row buffer (free overflow array)
 */
static inline void csr_builder_destroy_row_buffer(csr_row_buffer_t* row) {
  free(row->overflow);
}

/**
 * @brief Add entry to row buffer with duplicate detection
 *
 * @param row Row buffer
 * @param col Column index
 * @param val Value to add/accumulate
 * @return MATGEN_SUCCESS or error code
 */
static inline matgen_error_t csr_builder_add_to_row_buffer(
    csr_row_buffer_t* row, matgen_index_t col, matgen_value_t val) {
  // Try hash table first (linear probing)
  matgen_size_t hash = csr_builder_hash_col(col);

  for (int probe = 0; probe < MATGEN_CSR_BUILDER_MAX_PROBE; probe++) {
    matgen_size_t idx = (hash + probe) % MATGEN_CSR_BUILDER_HASH_SIZE;

    if (row->hash_table[idx].col == (matgen_index_t)-1) {
      // Empty slot - insert
      row->hash_table[idx].col = col;
      row->hash_table[idx].val = val;
      return MATGEN_SUCCESS;
    }

    if (row->hash_table[idx].col == col) {
      // Found duplicate - accumulate
      row->hash_table[idx].val += val;
      return MATGEN_SUCCESS;
    }
  }

  // Hash table full/collision - use overflow
  for (matgen_size_t i = 0; i < row->overflow_count; i++) {
    if (row->overflow[i].col == col) {
      // Found in overflow - accumulate
      row->overflow[i].val += val;
      return MATGEN_SUCCESS;
    }
  }

  // Need to add to overflow
  if (row->overflow_count >= row->overflow_capacity) {
    matgen_size_t new_cap =
        row->overflow_capacity == 0 ? 16 : row->overflow_capacity * 2;
    csr_hash_entry_t* new_overflow = (csr_hash_entry_t*)realloc(
        row->overflow, new_cap * sizeof(csr_hash_entry_t));
    if (!new_overflow) {
      return MATGEN_ERROR_OUT_OF_MEMORY;
    }
    row->overflow = new_overflow;
    row->overflow_capacity = new_cap;
  }

  row->overflow[row->overflow_count].col = col;
  row->overflow[row->overflow_count].val = val;
  row->overflow_count++;

  return MATGEN_SUCCESS;
}

/**
 * @brief Comparison function for sorting entries by column
 */
static inline int csr_builder_compare_entries(const void* a, const void* b) {
  const csr_hash_entry_t* ea = (const csr_hash_entry_t*)a;
  const csr_hash_entry_t* eb = (const csr_hash_entry_t*)b;

  if (ea->col < eb->col) {
    return -1;
  }

  if (ea->col > eb->col) {
    return 1;
  }
  return 0;
}

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_CORE_MATRIX_CSR_BUILDER_INTERNAL_H
