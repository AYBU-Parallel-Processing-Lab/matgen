#ifndef MATGEN_MPI_UTILS_H
#define MATGEN_MPI_UTILS_H

// MPI utility functions for distributed parallel operations

#include <stddef.h>

#include "matgen/core/coo_matrix.h"
#include "matgen/core/csr_matrix.h"

#ifdef MATGEN_HAS_MPI
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

// Broadcast COO matrix from root to all processes
// matrix: pointer to matrix (NULL on non-root, will be allocated)
// root: rank that has the matrix
// comm: MPI communicator
// Returns: 0 on success, -1 on error
int matgen_mpi_broadcast_coo(matgen_coo_matrix_t** matrix, int root,
                             MPI_Comm comm);

// Broadcast CSR matrix from root to all processes
int matgen_mpi_broadcast_csr(matgen_csr_matrix_t** matrix, int root,
                             MPI_Comm comm);

// Compute row range for this rank (load balancing)
// Distributes rows evenly with remainder going to first ranks
void matgen_mpi_compute_row_range(int rank, int size, size_t total_rows,
                                  size_t* start_row, size_t* end_row,
                                  size_t* local_rows);

// Gather distributed COO matrices from all ranks to root
// local: this rank's portion of the matrix
// total_rows, total_cols: dimensions of the full matrix
// root: rank to gather to
// comm: MPI communicator
// Returns: full matrix on root, NULL on other ranks or error
matgen_coo_matrix_t* matgen_mpi_gather_coo(matgen_coo_matrix_t* local,
                                           size_t total_rows, size_t total_cols,
                                           int root, MPI_Comm comm);

// Scatter COO matrix rows from root to all processes
// global: full matrix on root (NULL on other ranks)
// root: rank that has the full matrix
// comm: MPI communicator
// Returns: local portion for this rank, or NULL on error
matgen_coo_matrix_t* matgen_mpi_scatter_coo(const matgen_coo_matrix_t* global,
                                            int root, MPI_Comm comm);

// Get MPI rank and size with error checking
// Returns: 0 on success, -1 on error
int matgen_mpi_get_rank_size(int* rank, int* size, MPI_Comm comm);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_MPI

#endif  // MATGEN_MPI_UTILS_H
