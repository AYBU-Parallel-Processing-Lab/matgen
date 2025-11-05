#include "mpi_utils.h"

#include <stdlib.h>
#include <string.h>

#ifdef MATGEN_HAS_MPI

int matgen_mpi_get_rank_size(int* rank, int* size, MPI_Comm comm) {
  if (MPI_Comm_rank(comm, rank) != MPI_SUCCESS) {
    return -1;
  }
  if (MPI_Comm_size(comm, size) != MPI_SUCCESS) {
    return -1;
  }
  return 0;
}

int matgen_mpi_broadcast_coo(matgen_coo_matrix_t** matrix, int root,
                             MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  // Broadcast dimensions and metadata
  size_t dims[4];  // rows, cols, nnz, capacity

  if (rank == root) {
    if (!*matrix) {
      return -1;
    }
    dims[0] = (*matrix)->rows;
    dims[1] = (*matrix)->cols;
    dims[2] = (*matrix)->nnz;
    dims[3] = (*matrix)->capacity;
  }

  if (MPI_Bcast(dims, 4, MPI_UNSIGNED_LONG, root, comm) != MPI_SUCCESS) {
    return -1;
  }

  // Non-root ranks allocate matrix
  if (rank != root) {
    *matrix = matgen_coo_create(dims[0], dims[1], dims[3]);
    if (!*matrix) {
      return -1;
    }
    (*matrix)->nnz = dims[2];
  }

  // Broadcast matrix data
  if (MPI_Bcast((*matrix)->row_indices, (int)dims[2], MPI_UNSIGNED_LONG, root,
                comm) != MPI_SUCCESS) {
    return -1;
  }
  if (MPI_Bcast((*matrix)->col_indices, (int)dims[2], MPI_UNSIGNED_LONG, root,
                comm) != MPI_SUCCESS) {
    return -1;
  }
  if (MPI_Bcast((*matrix)->values, (int)dims[2], MPI_DOUBLE, root, comm) !=
      MPI_SUCCESS) {
    return -1;
  }

  // Broadcast is_sorted flag
  int is_sorted = (*matrix)->is_sorted ? 1 : 0;
  if (MPI_Bcast(&is_sorted, 1, MPI_INT, root, comm) != MPI_SUCCESS) {
    return -1;
  }
  (*matrix)->is_sorted = (is_sorted != 0);

  return 0;
}

int matgen_mpi_broadcast_csr(matgen_csr_matrix_t** matrix, int root,
                             MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  // Broadcast dimensions
  size_t dims[3];  // rows, cols, nnz

  if (rank == root) {
    if (!*matrix) {
      return -1;
    }
    dims[0] = (*matrix)->rows;
    dims[1] = (*matrix)->cols;
    dims[2] = (*matrix)->nnz;
  }

  if (MPI_Bcast(dims, 3, MPI_UNSIGNED_LONG, root, comm) != MPI_SUCCESS) {
    return -1;
  }

  // Non-root ranks allocate matrix
  if (rank != root) {
    *matrix = matgen_csr_create(dims[0], dims[1], dims[2]);
    if (!*matrix) {
      return -1;
    }
  }

  // Broadcast matrix data
  if (MPI_Bcast((*matrix)->row_ptr, (int)dims[0] + 1, MPI_UNSIGNED_LONG, root,
                comm) != MPI_SUCCESS) {
    return -1;
  }
  if (MPI_Bcast((*matrix)->col_indices, (int)dims[2], MPI_UNSIGNED_LONG, root,
                comm) != MPI_SUCCESS) {
    return -1;
  }
  if (MPI_Bcast((*matrix)->values, (int)dims[2], MPI_DOUBLE, root, comm) !=
      MPI_SUCCESS) {
    return -1;
  }

  return 0;
}

void matgen_mpi_compute_row_range(int rank, int size, size_t total_rows,
                                  size_t* start_row, size_t* end_row,
                                  size_t* local_rows) {
  size_t rows_per_rank = total_rows / size;
  size_t remainder = total_rows % size;

  // Distribute remainder among first ranks for load balancing
  if ((size_t)rank < remainder) {
    *start_row = rank * (rows_per_rank + 1);
    *local_rows = rows_per_rank + 1;
  } else {
    *start_row = (rank * rows_per_rank) + remainder;
    *local_rows = rows_per_rank;
  }

  *end_row = *start_row + *local_rows;
}

matgen_coo_matrix_t* matgen_mpi_gather_coo(matgen_coo_matrix_t* local,
                                           size_t total_rows, size_t total_cols,
                                           int root, MPI_Comm comm) {
  int rank;
  int size;
  if (matgen_mpi_get_rank_size(&rank, &size, comm) != 0) {
    return NULL;
  }

  // Gather nnz counts from all ranks
  size_t local_nnz = local ? local->nnz : 0;
  size_t* all_nnz = NULL;

  if (rank == root) {
    all_nnz = (size_t*)malloc(size * sizeof(size_t));
    if (!all_nnz) {
      return NULL;
    }
  }

  if (MPI_Gather(&local_nnz, 1, MPI_UNSIGNED_LONG, all_nnz, 1,
                 MPI_UNSIGNED_LONG, root, comm) != MPI_SUCCESS) {
    if (rank == root) {
      free(all_nnz);
    }
    return NULL;
  }

  // Calculate total nnz and displacements
  size_t total_nnz = 0;
  int* displs = NULL;
  int* counts = NULL;

  if (rank == root) {
    displs = (int*)malloc(size * sizeof(int));
    counts = (int*)malloc(size * sizeof(int));

    if (!displs || !counts) {
      free(all_nnz);
      free(displs);
      free(counts);
      return NULL;
    }

    displs[0] = 0;
    counts[0] = (int)all_nnz[0];
    total_nnz = all_nnz[0];

    for (int i = 1; i < size; i++) {
      displs[i] = displs[i - 1] + counts[i - 1];
      counts[i] = (int)all_nnz[i];
      total_nnz += all_nnz[i];
    }
  }

  // Create global matrix on root
  matgen_coo_matrix_t* global = NULL;
  if (rank == root) {
    global = matgen_coo_create(total_rows, total_cols, total_nnz);
    if (!global) {
      free(all_nnz);
      free(displs);
      free(counts);
      return NULL;
    }
    global->nnz = total_nnz;
  }

  // Gather data from all ranks
  size_t* global_rows = (rank == root) ? global->row_indices : NULL;
  size_t* global_cols = (rank == root) ? global->col_indices : NULL;
  double* global_vals = (rank == root) ? global->values : NULL;

  size_t* local_rows = local ? local->row_indices : NULL;
  size_t* local_cols = local ? local->col_indices : NULL;
  double* local_vals = local ? local->values : NULL;

  if (MPI_Gatherv(local_rows, (int)local_nnz, MPI_UNSIGNED_LONG, global_rows,
                  counts, displs, MPI_UNSIGNED_LONG, root,
                  comm) != MPI_SUCCESS) {
    if (rank == root) {
      matgen_coo_destroy(global);
      free(all_nnz);
      free(displs);
      free(counts);
    }
    return NULL;
  }

  if (MPI_Gatherv(local_cols, (int)local_nnz, MPI_UNSIGNED_LONG, global_cols,
                  counts, displs, MPI_UNSIGNED_LONG, root,
                  comm) != MPI_SUCCESS) {
    if (rank == root) {
      matgen_coo_destroy(global);
      free(all_nnz);
      free(displs);
      free(counts);
    }
    return NULL;
  }

  if (MPI_Gatherv(local_vals, (int)local_nnz, MPI_DOUBLE, global_vals, counts,
                  displs, MPI_DOUBLE, root, comm) != MPI_SUCCESS) {
    if (rank == root) {
      matgen_coo_destroy(global);
      free(all_nnz);
      free(displs);
      free(counts);
    }
    return NULL;
  }

  // Cleanup
  if (rank == root) {
    free(all_nnz);
    free(displs);
    free(counts);
  }

  return global;
}

matgen_coo_matrix_t* matgen_mpi_scatter_coo(const matgen_coo_matrix_t* global,
                                            int root, MPI_Comm comm) {
  ((void)global);
  ((void)root);

  int rank;
  int size;
  if (matgen_mpi_get_rank_size(&rank, &size, comm) != 0) {
    return NULL;
  }

  // TODO: Implement scatter if needed

  return NULL;
}

#endif  // MATGEN_HAS_MPI
