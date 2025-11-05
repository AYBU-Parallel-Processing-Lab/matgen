#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/core/csr_matrix.h"
#include "matgen/core/matrix_convert.h"
#include "mpi_utils.h"
#include "scale_mpi.h"

#ifdef MATGEN_HAS_MPI

matgen_coo_matrix_t* matgen_matrix_scale_nearest_mpi(
    const matgen_coo_matrix_t* input, size_t new_rows, size_t new_cols,
    MPI_Comm comm) {
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

  // Compute this rank's row range in output matrix
  size_t start_row;
  size_t end_row;
  size_t local_rows;
  matgen_mpi_compute_row_range(rank, size, new_rows, &start_row, &end_row,
                               &local_rows);

  // Estimate local output size
  double sparsity =
      (double)input_copy->nnz / (double)(input_copy->rows * input_copy->cols);
  size_t estimated_local_nnz =
      (size_t)(sparsity * (double)local_rows * (double)new_cols);
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

  // Each rank computes its portion of the output independently
  for (size_t i = start_row; i < end_row; i++) {
    for (size_t j = 0; j < new_cols; j++) {
      // Map to old coordinates using nearest neighbor
      size_t i_old = (size_t)round((double)i * row_scale);
      size_t j_old = (size_t)round((double)j * col_scale);

      // Clamp to valid range
      if (i_old >= input_copy->rows) {
        i_old = input_copy->rows - 1;
      }
      if (j_old >= input_copy->cols) {
        j_old = input_copy->cols - 1;
      }

      // Look up value in CSR matrix
      double value = matgen_csr_get(csr, i_old, j_old);

      // Add to local output if non-zero
      if (value != 0.0) {
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
