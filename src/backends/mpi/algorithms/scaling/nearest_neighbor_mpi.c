#include "backends/mpi/internal/nearest_neighbor_mpi.h"

#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "backends/mpi/internal/conversion_mpi.h"
#include "backends/mpi/internal/coo_mpi.h"
#include "backends/mpi/internal/csr_mpi.h"
#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/csr.h"
#include "matgen/utils/log.h"

// =============================================================================
// Helper Structures
// =============================================================================

typedef struct {
  matgen_index_t row;
  matgen_index_t col;
  matgen_value_t val;
} triplet_t;

// =============================================================================
// MPI Nearest Neighbor Scaling Implementation
// =============================================================================

// NOLINTNEXTLINE
matgen_error_t matgen_scale_nearest_neighbor_mpi(
    const matgen_csr_matrix_t* source, matgen_index_t new_rows,
    matgen_index_t new_cols, matgen_collision_policy_t collision_policy,
    matgen_csr_matrix_t** result) {
  if (!source || !result) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (new_rows == 0 || new_cols == 0) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  *result = NULL;

  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Get source distribution info
  matgen_index_t global_source_rows = 0;
  matgen_index_t local_source_rows = source->rows;

#if defined(MATGEN_INDEX_64)
  MPI_Allreduce(&local_source_rows, &global_source_rows, 1, MPI_UINT64_T,
                MPI_SUM, MPI_COMM_WORLD);
#else
  MPI_Allreduce(&local_source_rows, &global_source_rows, 1, MPI_UINT32_T,
                MPI_SUM, MPI_COMM_WORLD);
#endif

  matgen_csr_mpi_dist_t source_dist;
  matgen_csr_get_distribution(global_source_rows, &source_dist);

  matgen_value_t row_scale =
      (matgen_value_t)new_rows / (matgen_value_t)global_source_rows;
  matgen_value_t col_scale =
      (matgen_value_t)new_cols / (matgen_value_t)source->cols;

  MATGEN_LOG_DEBUG(
      "[Rank %d] Nearest neighbor scaling (MPI): local %llu×%llu (global "
      "%llu×%llu) -> %llu×%llu (scale: %.3fx%.3f)",
      rank, (unsigned long long)source->rows, (unsigned long long)source->cols,
      (unsigned long long)global_source_rows, (unsigned long long)source->cols,
      (unsigned long long)new_rows, (unsigned long long)new_cols, row_scale,
      col_scale);

  // Step 1: Generate local triplets from source rows
  matgen_value_t max_block_size =
      ceilf(row_scale + 1.0F) * ceilf(col_scale + 1.0F);
  size_t estimated_local_triplets =
      (size_t)((matgen_value_t)source->nnz * max_block_size * 2.0);

  triplet_t* local_triplets =
      (triplet_t*)malloc(estimated_local_triplets * sizeof(triplet_t));
  if (!local_triplets) {
    MATGEN_LOG_ERROR("Failed to allocate triplet buffer");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  size_t local_triplet_count = 0;

  for (matgen_index_t local_src_row = 0; local_src_row < source->rows;
       local_src_row++) {
    matgen_index_t global_src_row = source_dist.local_row_start + local_src_row;
    size_t row_start = source->row_ptr[local_src_row];
    size_t row_end = source->row_ptr[local_src_row + 1];

    for (size_t idx = row_start; idx < row_end; idx++) {
      matgen_index_t src_col = source->col_indices[idx];
      matgen_value_t src_val = source->values[idx];

      if (src_val == 0.0) {
        continue;
      }

      // Calculate destination cell range (nearest neighbor block)
      matgen_value_t dst_row_start_f =
          ((matgen_value_t)global_src_row - 0.5F) * row_scale;
      matgen_value_t dst_row_end_f =
          ((matgen_value_t)global_src_row + 0.5F) * row_scale;
      matgen_value_t dst_col_start_f =
          ((matgen_value_t)src_col - 0.5F) * col_scale;
      matgen_value_t dst_col_end_f =
          ((matgen_value_t)src_col + 0.5F) * col_scale;

      matgen_index_t dst_row_start = (matgen_index_t)ceilf(dst_row_start_f);
      matgen_index_t dst_row_end = (matgen_index_t)floorf(dst_row_end_f);
      matgen_index_t dst_col_start = (matgen_index_t)ceilf(dst_col_start_f);
      matgen_index_t dst_col_end = (matgen_index_t)floorf(dst_col_end_f);

      // Clamp to valid range
      dst_row_start = MATGEN_CLAMP(dst_row_start, 0, new_rows);
      dst_row_end = MATGEN_CLAMP(dst_row_end, 0, new_rows);
      dst_col_start = MATGEN_CLAMP(dst_col_start, 0, new_cols);
      dst_col_end = MATGEN_CLAMP(dst_col_end, 0, new_cols);

      // Skip if no valid cells in range
      if (dst_row_end <= dst_row_start || dst_col_end <= dst_col_start) {
        continue;
      }

      // Generate block entries (each gets full source value)
      for (matgen_index_t dst_row = dst_row_start; dst_row < dst_row_end;
           dst_row++) {
        for (matgen_index_t dst_col = dst_col_start; dst_col < dst_col_end;
             dst_col++) {
          if (local_triplet_count >= estimated_local_triplets) {
            MATGEN_LOG_ERROR("[Rank %d] Triplet buffer overflow", rank);
            free(local_triplets);
            return MATGEN_ERROR_OUT_OF_MEMORY;
          }

          local_triplets[local_triplet_count].row = dst_row;
          local_triplets[local_triplet_count].col = dst_col;
          local_triplets[local_triplet_count].val = src_val;
          local_triplet_count++;
        }
      }
    }
  }

  MATGEN_LOG_DEBUG("[Rank %d] Generated %zu local triplets", rank,
                   local_triplet_count);

  // Step 2: Redistribute triplets by destination row
  // Get destination distribution
  matgen_csr_mpi_dist_t dest_dist;
  matgen_csr_get_distribution(new_rows, &dest_dist);

  // Count triplets per destination rank
  int* send_counts = (int*)calloc(size, sizeof(int));
  int* send_displs = (int*)malloc(size * sizeof(int));
  int* recv_counts = (int*)malloc(size * sizeof(int));
  int* recv_displs = (int*)malloc(size * sizeof(int));

  if (!send_counts || !send_displs || !recv_counts || !recv_displs) {
    free(local_triplets);
    free(send_counts);
    free(send_displs);
    free(recv_counts);
    free(recv_displs);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Count how many triplets go to each rank
  for (size_t i = 0; i < local_triplet_count; i++) {
    matgen_index_t dst_row = local_triplets[i].row;

    // Determine which rank owns this destination row
    int target_rank = 0;
    matgen_index_t rows_per_rank = new_rows / size;
    matgen_index_t remainder = new_rows % size;

    if (dst_row < remainder * (rows_per_rank + 1)) {
      target_rank = (int)(dst_row / (rows_per_rank + 1));
    } else {
      target_rank =
          (int)(remainder +
                ((dst_row - remainder * (rows_per_rank + 1)) / rows_per_rank));
    }

    send_counts[target_rank]++;
  }

  // Compute send displacements
  send_displs[0] = 0;
  for (int i = 1; i < size; i++) {
    send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
  }

  // Sort triplets by target rank
  triplet_t* sorted_triplets =
      (triplet_t*)malloc(local_triplet_count * sizeof(triplet_t));
  if (!sorted_triplets) {
    free(local_triplets);
    free(send_counts);
    free(send_displs);
    free(recv_counts);
    free(recv_displs);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  int* current_pos = (int*)calloc(size, sizeof(int));
  for (size_t i = 0; i < local_triplet_count; i++) {
    matgen_index_t dst_row = local_triplets[i].row;

    int target_rank = 0;
    matgen_index_t rows_per_rank = new_rows / size;
    matgen_index_t remainder = new_rows % size;

    if (dst_row < remainder * (rows_per_rank + 1)) {
      target_rank = (int)(dst_row / (rows_per_rank + 1));
    } else {
      target_rank =
          (int)(remainder +
                ((dst_row - remainder * (rows_per_rank + 1)) / rows_per_rank));
    }

    int pos = send_displs[target_rank] + current_pos[target_rank];
    sorted_triplets[pos] = local_triplets[i];
    current_pos[target_rank]++;
  }

  free(local_triplets);
  free(current_pos);

  // Exchange counts
  MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT,
               MPI_COMM_WORLD);

  // Compute receive displacements and total
  recv_displs[0] = 0;
  for (int i = 1; i < size; i++) {
    recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
  }
  int total_recv = recv_displs[size - 1] + recv_counts[size - 1];

  MATGEN_LOG_DEBUG("[Rank %d] Receiving %d triplets", rank, total_recv);

  // Allocate receive buffer
  triplet_t* recv_triplets = (triplet_t*)malloc(total_recv * sizeof(triplet_t));
  if (!recv_triplets) {
    free(sorted_triplets);
    free(send_counts);
    free(send_displs);
    free(recv_counts);
    free(recv_displs);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Create MPI datatype for triplet
  MPI_Datatype triplet_type;
  MPI_Type_contiguous(sizeof(triplet_t), MPI_BYTE, &triplet_type);
  MPI_Type_commit(&triplet_type);

  // All-to-all exchange
  MPI_Alltoallv(sorted_triplets, send_counts, send_displs, triplet_type,
                recv_triplets, recv_counts, recv_displs, triplet_type,
                MPI_COMM_WORLD);

  MPI_Type_free(&triplet_type);
  free(sorted_triplets);
  free(send_counts);
  free(send_displs);
  free(recv_counts);
  free(recv_displs);

  // Step 3: Build local COO from received triplets
  matgen_coo_matrix_t* coo =
      matgen_coo_create_mpi(new_rows, new_cols, total_recv);
  if (!coo) {
    free(recv_triplets);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  for (int i = 0; i < total_recv; i++) {
    coo->row_indices[i] = recv_triplets[i].row;
    coo->col_indices[i] = recv_triplets[i].col;
    coo->values[i] = recv_triplets[i].val;
  }
  coo->nnz = total_recv;
  coo->is_sorted = false;

  free(recv_triplets);

  // Step 4: Sort and handle duplicates with collision policy
  MATGEN_LOG_DEBUG("[Rank %d] Sorting and handling duplicates (policy: %d)...",
                   rank, collision_policy);

  matgen_error_t err = matgen_coo_sort_mpi(coo);
  if (err != MATGEN_SUCCESS) {
    matgen_coo_destroy(coo);
    return err;
  }

  // Handle duplicates based on collision policy
  if (collision_policy == MATGEN_COLLISION_SUM) {
    err = matgen_coo_sum_duplicates_mpi(coo);
  } else {
    err = matgen_coo_merge_duplicates_mpi(coo, collision_policy);
  }

  if (err != MATGEN_SUCCESS) {
    matgen_coo_destroy(coo);
    return err;
  }

  MATGEN_LOG_DEBUG("[Rank %d] After deduplication: %zu entries", rank,
                   coo->nnz);

  // Step 5: Convert to CSR
  *result = matgen_coo_to_csr_mpi(coo);
  matgen_coo_destroy(coo);

  if (!(*result)) {
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  matgen_size_t global_nnz;
  matgen_csr_get_global_nnz((*result)->nnz, &global_nnz);

  MATGEN_LOG_DEBUG(
      "[Rank %d] Nearest neighbor scaling (MPI) complete: local nnz=%zu, "
      "global nnz=%zu",
      rank, (*result)->nnz, global_nnz);

  return MATGEN_SUCCESS;
}
