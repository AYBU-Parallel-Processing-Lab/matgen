#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/core/csr_matrix.h"
#include "matgen/utils/matrix_convert.h"
#include "mpi_utils.h"
#include "scale_mpi.h"

#ifdef MATGEN_HAS_MPI

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

  // Get four corner values
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

matgen_coo_matrix_t* matgen_matrix_scale_bilinear_mpi(
    const matgen_coo_matrix_t* input, size_t new_rows, size_t new_cols,
    double sparsity_threshold, MPI_Comm comm) {
  int rank;
  int size;
  if (matgen_mpi_get_rank_size(&rank, &size, comm) != 0) {
    return NULL;
  }

  if (!input || new_rows == 0 || new_cols == 0) {
    return NULL;
  }

  // Make a copy of input on root for broadcasting
  matgen_coo_matrix_t* input_copy = NULL;
  if (rank == 0) {
    input_copy = matgen_coo_create(input->rows, input->cols, input->nnz);
    if (input_copy) {
      memcpy(input_copy->row_indices, input->row_indices,
             input->nnz * sizeof(size_t));
      memcpy(input_copy->col_indices, input->col_indices,
             input->nnz * sizeof(size_t));
      memcpy(input_copy->values, input->values, input->nnz * sizeof(double));
      input_copy->nnz = input->nnz;
      input_copy->is_sorted = input->is_sorted;
    }
  }

  // Broadcast input matrix to all processes
  if (matgen_mpi_broadcast_coo(&input_copy, 0, comm) != 0) {
    return NULL;
  }

  // Convert to CSR for efficient random access
  matgen_csr_matrix_t* csr = matgen_coo_to_csr(input_copy);
  if (!csr) {
    matgen_coo_destroy(input_copy);
    return NULL;
  }

  // Compute this rank's row range
  size_t start_row;
  size_t end_row;
  size_t local_rows;
  matgen_mpi_compute_row_range(rank, size, new_rows, &start_row, &end_row,
                               &local_rows);

  // Estimate local output size (bilinear may add entries)
  size_t estimated_local_nnz = input_copy->nnz * 4 / size;
  if (estimated_local_nnz == 0) {
    estimated_local_nnz = 100;
  }

  // Create local output matrix
  matgen_coo_matrix_t* local_output =
      matgen_coo_create(new_rows, new_cols, estimated_local_nnz);
  if (!local_output) {
    matgen_csr_destroy(csr);
    matgen_coo_destroy(input_copy);
    return NULL;
  }

  // Precompute scaling factors
  double row_scale = (double)input_copy->rows / (double)new_rows;
  double col_scale = (double)input_copy->cols / (double)new_cols;

  // Each rank computes its portion with bilinear interpolation
  for (size_t i = start_row; i < end_row; i++) {
    for (size_t j = 0; j < new_cols; j++) {
      // Map to continuous coordinates in original matrix
      double i_f = (double)i * row_scale;
      double j_f = (double)j * col_scale;

      // Bilinear interpolation
      double value = bilinear_interp(csr, i_f, j_f);

      // Add to local output if above threshold
      if (fabs(value) > sparsity_threshold) {
        if (matgen_coo_add_entry(local_output, i, j, value) != 0) {
          // Handle error but continue
        }
      }
    }
  }

  // Clean up temporary structures
  matgen_csr_destroy(csr);
  matgen_coo_destroy(input_copy);

  // Gather all local outputs to root
  matgen_coo_matrix_t* global_output =
      matgen_mpi_gather_coo(local_output, new_rows, new_cols, 0, comm);

  // Sort on root
  if (rank == 0 && global_output) {
    matgen_coo_sort(global_output);
  }

  // Clean up local output
  matgen_coo_destroy(local_output);

  return global_output;
}

#endif  // MATGEN_HAS_MPI
