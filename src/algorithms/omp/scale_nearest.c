#include <math.h>
#include <stdlib.h>

#include "matgen/core/csr_matrix.h"
#include "matgen/utils/matrix_convert.h"
#include "omp_utils.h"
#include "scale_omp.h"

#ifdef MATGEN_HAS_OPENMP
#include <omp.h>

matgen_coo_matrix_t* matgen_matrix_scale_nearest_omp(
    const matgen_coo_matrix_t* input, size_t new_rows, size_t new_cols,
    int num_threads) {
  if (!input || new_rows == 0 || new_cols == 0) {
    return NULL;
  }

  // Set number of threads
  matgen_omp_set_threads(num_threads);
  int max_threads = matgen_omp_get_threads();

  // Convert to CSR for efficient random access
  matgen_csr_matrix_t* csr = matgen_coo_to_csr((matgen_coo_matrix_t*)input);
  if (!csr) {
    return NULL;
  }

  // Estimate output size
  double sparsity = (double)input->nnz / (double)(input->rows * input->cols);
  size_t estimated_nnz =
      (size_t)(sparsity * (double)new_rows * (double)new_cols);
  if (estimated_nnz < input->nnz) {
    estimated_nnz = input->nnz;
  }

  // Create per-thread output matrices
  size_t nnz_per_thread = (estimated_nnz / max_threads) + 100;
  matgen_coo_matrix_t** thread_outputs = matgen_omp_create_thread_matrices(
      new_rows, new_cols, nnz_per_thread, max_threads);

  if (!thread_outputs) {
    matgen_csr_destroy(csr);
    return NULL;
  }

  // Precompute scaling factors
  double row_scale = (double)input->rows / (double)new_rows;
  double col_scale = (double)input->cols / (double)new_cols;

// Parallel scaling - each thread processes a range of rows
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    matgen_coo_matrix_t* local_output = thread_outputs[tid];

    int i;

// Dynamic scheduling for load balancing (rows may have different sparsity)
#pragma omp for schedule(dynamic, 16)
    for (i = 0; i < (int)new_rows; i++) {
      for (size_t j = 0; j < new_cols; j++) {
        // Map to old coordinates using nearest neighbor
        size_t i_old = (size_t)round((double)i * row_scale);
        size_t j_old = (size_t)round((double)j * col_scale);

        // Clamp to valid range
        if (i_old >= input->rows) {
          i_old = input->rows - 1;
        }

        if (j_old >= input->cols) {
          j_old = input->cols - 1;
        }

        // Look up value (CSR reads are thread-safe)
        double value = matgen_csr_get(csr, i_old, j_old);

        // Add to thread-local output if non-zero
        if (value != 0.0) {
          matgen_coo_add_entry(local_output, i, j, value);
        }
      }
    }
  }

  // Merge thread-local outputs
  matgen_coo_matrix_t* output =
      matgen_omp_merge_coo(thread_outputs, max_threads);

  // Cleanup
  matgen_omp_destroy_thread_matrices(thread_outputs, max_threads);
  matgen_csr_destroy(csr);

  // Sort result
  if (output) {
    matgen_coo_sort(output);
  }

  return output;
}

#endif  // MATGEN_HAS_OPENMP
