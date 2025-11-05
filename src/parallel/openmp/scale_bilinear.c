#include <math.h>
#include <stdlib.h>

#include "matgen/core/csr_matrix.h"
#include "matgen/core/matrix_convert.h"
#include "omp_utils.h"
#include "scale_omp.h"

#ifdef MATGEN_HAS_OPENMP
#include <omp.h>

// Helper: Bilinear interpolation at floating-point coordinates
static inline double bilinear_interp(const matgen_csr_matrix_t* csr, double i_f,
                                     double j_f) {
  // Get integer parts
  size_t i0 = (size_t)floor(i_f);
  size_t j0 = (size_t)floor(j_f);
  size_t i1 = i0 + 1;
  size_t j1 = j0 + 1;

  // Clamp to bounds
  if (i1 >= csr->rows) {
    i1 = csr->rows - 1;
  }

  if (j1 >= csr->cols) {
    j1 = csr->cols - 1;
  }

  // Get fractional parts
  double di = i_f - (double)i0;
  double dj = j_f - (double)j0;

  // Get four corner values (reads are thread-safe)
  double v00 = matgen_csr_get(csr, i0, j0);
  double v01 = matgen_csr_get(csr, i0, j1);
  double v10 = matgen_csr_get(csr, i1, j0);
  double v11 = matgen_csr_get(csr, i1, j1);

  // Bilinear interpolation
  double v0 = (v00 * (1.0 - dj)) + (v01 * dj);
  double v1 = (v10 * (1.0 - dj)) + (v11 * dj);
  double value = (v0 * (1.0 - di)) + (v1 * di);

  return value;
}

matgen_coo_matrix_t* matgen_matrix_scale_bilinear_omp(
    const matgen_coo_matrix_t* input, size_t new_rows, size_t new_cols,
    double sparsity_threshold, int num_threads) {
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

  // Estimate output size (bilinear may add entries)
  size_t estimated_nnz = input->nnz * 4;

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

// Parallel bilinear interpolation
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    matgen_coo_matrix_t* local_output = thread_outputs[tid];

    int i;

// Dynamic scheduling for load balancing
#pragma omp for schedule(dynamic, 16)
    for (i = 0; i < (int)new_rows; i++) {
      for (size_t j = 0; j < new_cols; j++) {
        // Map to continuous coordinates in original matrix
        double i_f = (double)i * row_scale;
        double j_f = (double)j * col_scale;

        // Bilinear interpolation
        double value = bilinear_interp(csr, i_f, j_f);

        // Add to thread-local output if above threshold
        if (fabs(value) > sparsity_threshold) {
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
