#include "omp_utils.h"

#include <stdlib.h>

#ifdef MATGEN_HAS_OPENMP
#include <omp.h>

void matgen_omp_set_threads(int num_threads) {
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }
}

int matgen_omp_get_threads(void) { return omp_get_max_threads(); }

matgen_coo_matrix_t** matgen_omp_create_thread_matrices(
    size_t rows, size_t cols, size_t estimated_nnz_per_thread,
    int num_threads) {
  matgen_coo_matrix_t** thread_outputs =
      (matgen_coo_matrix_t**)calloc(num_threads, sizeof(matgen_coo_matrix_t*));

  if (!thread_outputs) {
    return NULL;
  }

  // Initialize per-thread matrices
  for (int t = 0; t < num_threads; t++) {
    thread_outputs[t] = matgen_coo_create(rows, cols, estimated_nnz_per_thread);
    if (!thread_outputs[t]) {
      // Cleanup on failure
      for (int i = 0; i < t; i++) {
        matgen_coo_destroy(thread_outputs[i]);
      }
      free((void*)thread_outputs);
      return NULL;
    }
  }

  return thread_outputs;
}

void matgen_omp_destroy_thread_matrices(matgen_coo_matrix_t** thread_outputs,
                                        int num_threads) {
  if (!thread_outputs) {
    return;
  }

  for (int t = 0; t < num_threads; t++) {
    matgen_coo_destroy(thread_outputs[t]);
  }

  free((void*)thread_outputs);
}

matgen_coo_matrix_t* matgen_omp_merge_coo(matgen_coo_matrix_t** thread_outputs,
                                          int num_threads) {
  if (!thread_outputs || num_threads <= 0) {
    return NULL;
  }

  // Calculate total nnz
  size_t total_nnz = 0;
  size_t rows = thread_outputs[0]->rows;
  size_t cols = thread_outputs[0]->cols;

  for (int t = 0; t < num_threads; t++) {
    total_nnz += thread_outputs[t]->nnz;
  }

  // Create merged matrix
  matgen_coo_matrix_t* merged = matgen_coo_create(rows, cols, total_nnz);
  if (!merged) {
    return NULL;
  }

  // Merge all thread-local matrices
  for (int t = 0; t < num_threads; t++) {
    matgen_coo_matrix_t* local = thread_outputs[t];
    for (size_t k = 0; k < local->nnz; k++) {
      if (matgen_coo_add_entry(merged, local->row_indices[k],
                               local->col_indices[k], local->values[k]) != 0) {
        matgen_coo_destroy(merged);
        return NULL;
      }
    }
  }

  return merged;
}

#endif  // MATGEN_HAS_OPENMP
