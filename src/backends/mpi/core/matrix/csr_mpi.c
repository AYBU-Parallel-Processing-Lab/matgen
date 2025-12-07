#include "backends/mpi/internal/csr_mpi.h"

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/core/matrix/csr.h"
#include "matgen/utils/log.h"

// =============================================================================
// MPI Datatype Helpers
// =============================================================================

// Determine correct MPI datatype for size_t
#if SIZE_MAX == UINT64_MAX
#define MPI_SIZE_T MPI_UINT64_T
#elif SIZE_MAX == UINT32_MAX
#define MPI_SIZE_T MPI_UINT32_T
#else
#error "Unsupported size_t size"
#endif

// Determine correct MPI datatype for matgen_index_t (which is u64)
// Since matgen_index_t is typedef u64, it's always 64-bit
#define MPI_INDEX_T MPI_UINT64_T

// Determine correct MPI datatype for matgen_value_t (which is f32)
// Since matgen_value_t is typedef f32, it's always float
#define MPI_VALUE_T MPI_FLOAT

// =============================================================================
// Sequential CSR Creation (for MPI gather - NO dispatch!)
// =============================================================================

// Forward declare sequential creation to avoid dispatch overhead
extern matgen_csr_matrix_t* matgen_csr_create_seq(matgen_index_t rows,
                                                  matgen_index_t cols,
                                                  matgen_size_t nnz);

// =============================================================================
// MPI Backend Implementation for CSR Matrix
// =============================================================================

matgen_error_t matgen_csr_get_distribution(matgen_index_t global_rows,
                                           matgen_csr_mpi_dist_t* dist) {
  if (!dist) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &dist->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &dist->size);

  dist->global_rows = global_rows;

  matgen_index_t rows_per_rank = global_rows / dist->size;
  matgen_index_t remainder = global_rows % dist->size;

  if (dist->rank < (int)remainder) {
    dist->local_row_count = rows_per_rank + 1;
    dist->local_row_start = dist->rank * dist->local_row_count;
  } else {
    dist->local_row_count = rows_per_rank;
    dist->local_row_start = (remainder * (rows_per_rank + 1)) +
                            ((dist->rank - remainder) * rows_per_rank);
  }

  return MATGEN_SUCCESS;
}

matgen_csr_matrix_t* matgen_csr_create_mpi(matgen_index_t rows,
                                           matgen_index_t cols,
                                           matgen_size_t nnz) {
  if (rows == 0 || cols == 0) {
    MATGEN_LOG_ERROR("Invalid matrix dimensions: %llu x %llu",
                     (unsigned long long)rows, (unsigned long long)cols);
    return NULL;
  }

  matgen_csr_mpi_dist_t dist;
  if (matgen_csr_get_distribution(rows, &dist) != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to compute distribution");
    return NULL;
  }

  matgen_csr_matrix_t* matrix =
      (matgen_csr_matrix_t*)malloc(sizeof(matgen_csr_matrix_t));
  if (!matrix) {
    MATGEN_LOG_ERROR("Failed to allocate CSR matrix structure");
    return NULL;
  }

  matrix->rows = dist.local_row_count;
  matrix->cols = cols;
  matrix->nnz = nnz;

  matrix->row_ptr =
      (matgen_size_t*)calloc(matrix->rows + 1, sizeof(matgen_size_t));
  matrix->col_indices = (matgen_index_t*)malloc(nnz * sizeof(matgen_index_t));
  matrix->values = (matgen_value_t*)malloc(nnz * sizeof(matgen_value_t));

  if (!matrix->row_ptr) {
    MATGEN_LOG_ERROR("Failed to allocate CSR row_ptr");
    free(matrix);
    return NULL;
  }

  if (nnz > 0 && (!matrix->col_indices || !matrix->values)) {
    MATGEN_LOG_ERROR("Failed to allocate CSR matrix arrays");
    free(matrix->row_ptr);
    free(matrix->col_indices);
    free(matrix->values);
    free(matrix);
    return NULL;
  }

  MATGEN_LOG_DEBUG(
      "[Rank %d] Created local CSR matrix (MPI): global %llu x %llu, local "
      "rows [%llu, %llu), local nnz %zu",
      dist.rank, (unsigned long long)rows, (unsigned long long)cols,
      (unsigned long long)dist.local_row_start,
      (unsigned long long)(dist.local_row_start + dist.local_row_count), nnz);

  return matrix;
}

matgen_error_t matgen_csr_get_global_nnz(matgen_size_t local_nnz,
                                         matgen_size_t* global_nnz) {
  if (!global_nnz) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  MPI_Allreduce(&local_nnz, global_nnz, 1, MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);

  return MATGEN_SUCCESS;
}

matgen_csr_matrix_t* matgen_csr_gather(const matgen_csr_matrix_t* local_matrix,
                                       matgen_index_t global_rows,
                                       matgen_index_t global_cols) {
  if (!local_matrix) {
    MATGEN_LOG_ERROR("NULL local matrix pointer");
    return NULL;
  }

  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MATGEN_LOG_DEBUG("[Rank %d] Gathering CSR matrix to rank 0", rank);

  // Step 1: Gather local NNZ counts and row counts to rank 0
  matgen_size_t local_nnz = local_matrix->nnz;
  matgen_index_t local_rows = local_matrix->rows;

  matgen_size_t* all_nnz = NULL;
  matgen_index_t* all_row_counts = NULL;

  if (rank == 0) {
    all_nnz = (matgen_size_t*)malloc(size * sizeof(matgen_size_t));
    all_row_counts = (matgen_index_t*)malloc(size * sizeof(matgen_index_t));
    if (!all_nnz || !all_row_counts) {
      MATGEN_LOG_ERROR("[Rank 0] Failed to allocate gather buffers");
      free(all_nnz);
      free(all_row_counts);
      return NULL;
    }
  }

  MPI_Gather(&local_nnz, 1, MPI_SIZE_T, all_nnz, 1, MPI_SIZE_T, 0,
             MPI_COMM_WORLD);
  MPI_Gather(&local_rows, 1, MPI_INDEX_T, all_row_counts, 1, MPI_INDEX_T, 0,
             MPI_COMM_WORLD);

  // Step 2: Rank 0 creates global matrix
  matgen_csr_matrix_t* global_matrix = NULL;
  matgen_size_t total_nnz = 0;

  if (rank == 0) {
    // Compute total NNZ
    for (int i = 0; i < size; i++) {
      total_nnz += all_nnz[i];
      MATGEN_LOG_DEBUG("[Rank 0] Rank %d has %zu nnz, %llu rows", i, all_nnz[i],
                       (unsigned long long)all_row_counts[i]);
    }

    MATGEN_LOG_DEBUG(
        "[Rank 0] Creating global CSR matrix: %llu x %llu, nnz %zu",
        (unsigned long long)global_rows, (unsigned long long)global_cols,
        total_nnz);

    global_matrix = matgen_csr_create(global_rows, global_cols, total_nnz);
    if (!global_matrix) {
      MATGEN_LOG_ERROR("[Rank 0] Failed to create global CSR matrix");
      free(all_nnz);
      free(all_row_counts);
      return NULL;
    }
  }

  // Step 3: Gather col_indices and values using Gatherv
  int* recvcounts = NULL;
  int* displs = NULL;

  if (rank == 0) {
    recvcounts = (int*)malloc(size * sizeof(int));
    displs = (int*)malloc(size * sizeof(int));

    int offset = 0;
    for (int i = 0; i < size; i++) {
      recvcounts[i] = (int)all_nnz[i];
      displs[i] = offset;
      offset += recvcounts[i];
    }
  }

  MATGEN_LOG_DEBUG("[Rank %d] Gathering col_indices and values...", rank);

  MPI_Gatherv(local_matrix->col_indices, (int)local_nnz, MPI_INDEX_T,
              rank == 0 ? global_matrix->col_indices : NULL, recvcounts, displs,
              MPI_INDEX_T, 0, MPI_COMM_WORLD);

  MPI_Gatherv(local_matrix->values, (int)local_nnz, MPI_VALUE_T,
              rank == 0 ? global_matrix->values : NULL, recvcounts, displs,
              MPI_VALUE_T, 0, MPI_COMM_WORLD);

  MATGEN_LOG_DEBUG("[Rank %d] Finished gathering col_indices and values", rank);

  // Step 4: Exchange row_ptr data
  // All non-zero ranks send their row_ptr to rank 0
  // Rank 0 receives and reconstructs the global row_ptr

  if (rank != 0) {
    // Non-root ranks: send row_ptr to rank 0
    MATGEN_LOG_DEBUG("[Rank %d] Sending row_ptr (%llu elements) to rank 0",
                     rank, (unsigned long long)(local_matrix->rows + 1));

    MPI_Send(local_matrix->row_ptr, (int)(local_matrix->rows + 1), MPI_SIZE_T,
             0, 0, MPI_COMM_WORLD);

    MATGEN_LOG_DEBUG("[Rank %d] Sent row_ptr to rank 0", rank);
  } else {
    // Rank 0: reconstruct global row_ptr from all ranks
    MATGEN_LOG_DEBUG("[Rank 0] Reconstructing global row_ptr...");

    global_matrix->row_ptr[0] = 0;
    matgen_index_t current_row = 0;

    for (int i = 0; i < size; i++) {
      matgen_size_t* local_row_ptr = NULL;

      if (i == 0) {
        // Use our own local matrix's row_ptr
        local_row_ptr = local_matrix->row_ptr;
        MATGEN_LOG_DEBUG("[Rank 0] Using own row_ptr (%llu elements)",
                         (unsigned long long)(all_row_counts[i] + 1));
      } else {
        // Allocate buffer and receive from rank i
        local_row_ptr = (matgen_size_t*)malloc((all_row_counts[i] + 1) *
                                               sizeof(matgen_size_t));
        if (!local_row_ptr) {
          MATGEN_LOG_ERROR(
              "[Rank 0] Failed to allocate row_ptr buffer for rank %d", i);
          matgen_csr_destroy(global_matrix);
          free(all_nnz);
          free(all_row_counts);
          free(recvcounts);
          free(displs);
          return NULL;
        }

        MATGEN_LOG_DEBUG(
            "[Rank 0] Receiving row_ptr from rank %d (%llu elements)...", i,
            (unsigned long long)(all_row_counts[i] + 1));

        MPI_Recv(local_row_ptr, (int)(all_row_counts[i] + 1), MPI_SIZE_T, i, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MATGEN_LOG_DEBUG("[Rank 0] Received row_ptr from rank %d", i);
      }

      // Build global row_ptr from local row_ptr
      for (matgen_index_t r = 0; r < all_row_counts[i]; r++) {
        matgen_size_t row_nnz = local_row_ptr[r + 1] - local_row_ptr[r];
        global_matrix->row_ptr[current_row + 1] =
            global_matrix->row_ptr[current_row] + row_nnz;
        current_row++;
      }

      if (i > 0) {
        free(local_row_ptr);
      }
    }

    MATGEN_LOG_DEBUG("[Rank 0] Global row_ptr reconstruction complete");

    free(all_nnz);
    free(all_row_counts);
    free(recvcounts);
    free(displs);
  }

  MATGEN_LOG_DEBUG("[Rank %d] CSR gather complete", rank);

  return global_matrix;
}

matgen_csr_matrix_t* matgen_csr_scatter(
    const matgen_csr_matrix_t* global_matrix) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  matgen_index_t global_rows = 0;
  matgen_index_t global_cols = 0;

  if (rank == 0) {
    if (!global_matrix) {
      MATGEN_LOG_ERROR("NULL global matrix on rank 0");
      // Broadcast error to other ranks
      global_rows = 0;
      MPI_Bcast(&global_rows, 1, MPI_INDEX_T, 0, MPI_COMM_WORLD);
      return NULL;
    }
    global_rows = global_matrix->rows;
    global_cols = global_matrix->cols;
  }

  MPI_Bcast(&global_rows, 1, MPI_INDEX_T, 0, MPI_COMM_WORLD);

  // Check for error condition
  if (global_rows == 0) {
    return NULL;
  }

  MPI_Bcast(&global_cols, 1, MPI_INDEX_T, 0, MPI_COMM_WORLD);

  MATGEN_LOG_DEBUG("[Rank %d] Scattering CSR matrix from rank 0", rank);

  matgen_csr_mpi_dist_t dist;
  matgen_csr_get_distribution(global_rows, &dist);

  matgen_size_t local_nnz = 0;

  if (rank == 0) {
    local_nnz = global_matrix->row_ptr[dist.local_row_count] -
                global_matrix->row_ptr[0];
  }

  matgen_size_t* all_local_nnz = NULL;
  if (rank == 0) {
    all_local_nnz = (matgen_size_t*)malloc(size * sizeof(matgen_size_t));
    if (!all_local_nnz) {
      MATGEN_LOG_ERROR("Failed to allocate scatter buffer");
      return NULL;
    }

    matgen_index_t offset = 0;
    for (int i = 0; i < size; i++) {
      matgen_csr_mpi_dist_t i_dist;
      i_dist.rank = i;
      i_dist.size = size;
      i_dist.global_rows = global_rows;

      if (i < (int)(global_rows % size)) {
        i_dist.local_row_count = (global_rows / size) + 1;
      } else {
        i_dist.local_row_count = global_rows / size;
      }

      all_local_nnz[i] =
          global_matrix->row_ptr[offset + i_dist.local_row_count] -
          global_matrix->row_ptr[offset];
      offset += i_dist.local_row_count;
    }
  }

  MPI_Scatter(all_local_nnz, 1, MPI_SIZE_T, &local_nnz, 1, MPI_SIZE_T, 0,
              MPI_COMM_WORLD);

  matgen_csr_matrix_t* local_matrix =
      matgen_csr_create_mpi(global_rows, global_cols, local_nnz);
  if (!local_matrix) {
    MATGEN_LOG_ERROR("Failed to create local CSR matrix");
    if (rank == 0) {
      free(all_local_nnz);
    }
    return NULL;
  }

  // Scatter col_indices and values
  int* sendcounts = NULL;
  int* displs = NULL;

  if (rank == 0) {
    sendcounts = (int*)malloc(size * sizeof(int));
    displs = (int*)malloc(size * sizeof(int));

    if (!sendcounts || !displs) {
      MATGEN_LOG_ERROR("Failed to allocate scatter arrays");
      free(all_local_nnz);
      free(sendcounts);
      free(displs);
      matgen_csr_destroy(local_matrix);
      return NULL;
    }

    int offset = 0;
    for (int i = 0; i < size; i++) {
      sendcounts[i] = (int)all_local_nnz[i];
      displs[i] = offset;
      offset += sendcounts[i];
    }
  }

  MPI_Scatterv(rank == 0 ? global_matrix->col_indices : NULL, sendcounts,
               displs, MPI_INDEX_T, local_matrix->col_indices, (int)local_nnz,
               MPI_INDEX_T, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? global_matrix->values : NULL, sendcounts, displs,
               MPI_VALUE_T, local_matrix->values, (int)local_nnz, MPI_VALUE_T,
               0, MPI_COMM_WORLD);

  // Scatter row_ptr
  if (rank == 0) {
    matgen_index_t row_offset = 0;

    for (int i = 0; i < size; i++) {
      matgen_csr_mpi_dist_t i_dist;
      i_dist.rank = i;
      i_dist.size = size;
      i_dist.global_rows = global_rows;

      if (i < (int)(global_rows % size)) {
        i_dist.local_row_count = (global_rows / size) + 1;
      } else {
        i_dist.local_row_count = global_rows / size;
      }

      matgen_size_t* temp_row_ptr = (matgen_size_t*)malloc(
          (i_dist.local_row_count + 1) * sizeof(matgen_size_t));
      if (!temp_row_ptr) {
        MATGEN_LOG_ERROR("Failed to allocate temp row_ptr");
        free(all_local_nnz);
        free(sendcounts);
        free(displs);
        matgen_csr_destroy(local_matrix);
        return NULL;
      }

      matgen_size_t base = global_matrix->row_ptr[row_offset];
      for (matgen_index_t r = 0; r <= i_dist.local_row_count; r++) {
        temp_row_ptr[r] = global_matrix->row_ptr[row_offset + r] - base;
      }

      if (i == 0) {
        memcpy(local_matrix->row_ptr, temp_row_ptr,
               (i_dist.local_row_count + 1) * sizeof(matgen_size_t));
      } else {
        MPI_Send(temp_row_ptr, (int)(i_dist.local_row_count + 1), MPI_SIZE_T, i,
                 1, MPI_COMM_WORLD);
      }

      free(temp_row_ptr);
      row_offset += i_dist.local_row_count;
    }

    free(all_local_nnz);
    free(sendcounts);
    free(displs);
  } else {
    MPI_Recv(local_matrix->row_ptr, (int)(dist.local_row_count + 1), MPI_SIZE_T,
             0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  MATGEN_LOG_DEBUG("[Rank %d] CSR scatter complete (local nnz: %zu)", rank,
                   local_nnz);

  return local_matrix;
}
