#ifndef MATGEN_OMP_UTILS_H
#define MATGEN_OMP_UTILS_H

#include <stddef.h>

#include "matgen/core/coo_matrix.h"

#ifdef MATGEN_HAS_OPENMP

#ifdef __cplusplus
extern "C" {
#endif

// Set number of OpenMP threads (0 = use default)
void matgen_omp_set_threads(int num_threads);

// Get current number of OpenMP threads
int matgen_omp_get_threads(void);

// Merge multiple thread-local COO matrices into one
// thread_outputs: array of matrices from each thread
// num_threads: number of thread-local matrices
// Returns: merged matrix, or NULL on error
matgen_coo_matrix_t* matgen_omp_merge_coo(matgen_coo_matrix_t** thread_outputs,
                                          int num_threads);

// Create per-thread COO matrix array
// rows, cols: dimensions for each matrix
// estimated_nnz_per_thread: capacity hint for each thread
// Returns: array of num_threads matrices, or NULL on error
matgen_coo_matrix_t** matgen_omp_create_thread_matrices(
    size_t rows, size_t cols, size_t estimated_nnz_per_thread, int num_threads);

// Destroy array of thread-local matrices
void matgen_omp_destroy_thread_matrices(matgen_coo_matrix_t** thread_outputs,
                                        int num_threads);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_OPENMP

#endif  // MATGEN_OMP_UTILS_H
