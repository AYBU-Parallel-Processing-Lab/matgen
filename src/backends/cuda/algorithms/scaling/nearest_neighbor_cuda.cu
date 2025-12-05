// File: src/backends/cuda/algorithms/scaling/nearest_neighbor_cuda.cu
#include <cuda_runtime.h>
#include <math.h>

#include "backends/cuda/internal/conversion_cuda.h"
#include "backends/cuda/internal/coo_cuda.h"
#include "backends/cuda/internal/nearest_neighbor_cuda.h"
#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"
#include "matgen/utils/log.h"

// =============================================================================
// CUDA Error Checking
// =============================================================================

#define CUDA_CHECK(call)                                              \
  do {                                                                \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
      MATGEN_LOG_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, \
                       cudaGetErrorString(err));                      \
      return MATGEN_ERROR_CUDA;                                       \
    }                                                                 \
  } while (0)

#define CUDA_BLOCK_SIZE 256

// =============================================================================
// CUDA Kernels
// =============================================================================

/**
 * @brief Kernel to compute nearest neighbor block expansion for each source
 * entry
 *
 * Each thread processes one source entry and generates block of destination
 * entries. Uses atomic counter for output position tracking.
 */
__global__ void nearest_neighbor_scale_kernel(
    const matgen_size_t* src_row_ptr, const matgen_index_t* src_col_indices,
    const matgen_value_t* src_values, matgen_index_t src_rows,
    matgen_index_t src_cols, matgen_index_t dst_rows, matgen_index_t dst_cols,
    matgen_value_t row_scale, matgen_value_t col_scale,
    matgen_index_t* out_rows, matgen_index_t* out_cols,
    matgen_value_t* out_vals, matgen_size_t* out_count,
    matgen_size_t max_output_size) {
  // Each thread processes one source entry
  matgen_size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Find which source entry this thread handles
  matgen_index_t src_row = 0;
  matgen_size_t entry_idx = global_idx;

  // Binary search to find row
  matgen_index_t low = 0;
  matgen_index_t high = src_rows;
  while (low < high) {
    matgen_index_t mid = low + (high - low) / 2;
    if (src_row_ptr[mid] <= entry_idx) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  src_row = low - 1;

  // Check if this is a valid entry
  if (src_row >= src_rows || entry_idx >= src_row_ptr[src_rows]) {
    return;
  }

  matgen_index_t src_col = src_col_indices[entry_idx];
  matgen_value_t src_val = src_values[entry_idx];

  // Skip zero values
  if (src_val == 0.0) {
    return;
  }

  // Calculate destination cell range that maps to this source cell
  // A destination cell (dr, dc) maps to source (src_row, src_col) when:
  //   round(dr / row_scale) == src_row
  //   round(dc / col_scale) == src_col
  // This is equivalent to:
  //   (src_row - 0.5) * row_scale < dr < (src_row + 0.5) * row_scale

  matgen_value_t dst_row_start_f = ((matgen_value_t)src_row - 0.5f) * row_scale;
  matgen_value_t dst_row_end_f = ((matgen_value_t)src_row + 0.5f) * row_scale;
  matgen_value_t dst_col_start_f = ((matgen_value_t)src_col - 0.5f) * col_scale;
  matgen_value_t dst_col_end_f = ((matgen_value_t)src_col + 0.5f) * col_scale;

  // Convert to integer ranges (using ceil for start)
  matgen_index_t dst_row_start = (matgen_index_t)ceilf(dst_row_start_f);
  matgen_index_t dst_row_end = (matgen_index_t)ceilf(dst_row_end_f);
  matgen_index_t dst_col_start = (matgen_index_t)ceilf(dst_col_start_f);
  matgen_index_t dst_col_end = (matgen_index_t)ceilf(dst_col_end_f);

  // Clamp to valid range
  dst_row_start = min(max((matgen_index_t)0, dst_row_start), dst_rows);
  dst_row_end = min(max((matgen_index_t)0, dst_row_end), dst_rows);
  dst_col_start = min(max((matgen_index_t)0, dst_col_start), dst_cols);
  dst_col_end = min(max((matgen_index_t)0, dst_col_end), dst_cols);

  // Ensure at least one cell if this source cell should contribute
  if (dst_row_end <= dst_row_start) {
    dst_row_end = min(dst_row_start + 1, dst_rows);
  }
  if (dst_col_end <= dst_col_start) {
    dst_col_end = min(dst_col_start + 1, dst_cols);
  }

  // Generate entries for each destination cell in the block
  // Each cell gets the FULL source value (not divided)
  for (matgen_index_t dst_row = dst_row_start; dst_row < dst_row_end;
       dst_row++) {
    for (matgen_index_t dst_col = dst_col_start; dst_col < dst_col_end;
         dst_col++) {
      // Atomic increment to get output position
      matgen_size_t pos = atomicAdd((unsigned long long*)out_count, 1ULL);

      if (pos < max_output_size) {
        out_rows[pos] = dst_row;
        out_cols[pos] = dst_col;
        out_vals[pos] = src_val;
      }
    }
  }
}

// =============================================================================
// CUDA Backend Implementation
// =============================================================================

matgen_error_t matgen_scale_nearest_neighbor_cuda(
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

  matgen_value_t row_scale =
      (matgen_value_t)new_rows / (matgen_value_t)source->rows;
  matgen_value_t col_scale =
      (matgen_value_t)new_cols / (matgen_value_t)source->cols;

  MATGEN_LOG_DEBUG(
      "Nearest neighbor scaling (CUDA): %llu×%llu -> %llu×%llu (scale: "
      "%.3fx%.3f)",
      (unsigned long long)source->rows, (unsigned long long)source->cols,
      (unsigned long long)new_rows, (unsigned long long)new_cols, row_scale,
      col_scale);

  // Estimate output size: each source entry expands to ~ceil(scale)² entries
  // Use conservative 2.0x safety factor
  matgen_value_t max_block_size =
      ceilf(row_scale + 1.0f) * ceilf(col_scale + 1.0f);
  size_t estimated_nnz =
      (size_t)((matgen_value_t)source->nnz * max_block_size * 2.0);

  // Ensure minimum buffer size
  if (estimated_nnz < source->nnz) {
    estimated_nnz = source->nnz * 2;
  }

  MATGEN_LOG_DEBUG("Estimated output NNZ: %zu (max block size per entry: %.1f)",
                   estimated_nnz, max_block_size);

  // Allocate device memory for source CSR
  matgen_size_t* d_src_row_ptr = nullptr;
  matgen_index_t* d_src_col_indices = nullptr;
  matgen_value_t* d_src_values = nullptr;

  size_t size_row_ptr = (source->rows + 1) * sizeof(matgen_size_t);
  size_t size_col_indices = source->nnz * sizeof(matgen_index_t);
  size_t size_values = source->nnz * sizeof(matgen_value_t);

  CUDA_CHECK(cudaMalloc(&d_src_row_ptr, size_row_ptr));
  CUDA_CHECK(cudaMalloc(&d_src_col_indices, size_col_indices));
  CUDA_CHECK(cudaMalloc(&d_src_values, size_values));

  // Copy source to device
  CUDA_CHECK(cudaMemcpy(d_src_row_ptr, source->row_ptr, size_row_ptr,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_src_col_indices, source->col_indices,
                        size_col_indices, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_src_values, source->values, size_values,
                        cudaMemcpyHostToDevice));

  // Allocate device memory for output COO
  matgen_index_t* d_out_rows = nullptr;
  matgen_index_t* d_out_cols = nullptr;
  matgen_value_t* d_out_vals = nullptr;
  matgen_size_t* d_out_count = nullptr;

  size_t output_buffer_size = estimated_nnz;
  size_t size_out_indices = output_buffer_size * sizeof(matgen_index_t);
  size_t size_out_values = output_buffer_size * sizeof(matgen_value_t);

  CUDA_CHECK(cudaMalloc(&d_out_rows, size_out_indices));
  CUDA_CHECK(cudaMalloc(&d_out_cols, size_out_indices));
  CUDA_CHECK(cudaMalloc(&d_out_vals, size_out_values));
  CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(matgen_size_t)));
  CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(matgen_size_t)));

  // Launch kernel (one thread per source entry)
  int blocks = (source->nnz + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

  nearest_neighbor_scale_kernel<<<blocks, CUDA_BLOCK_SIZE>>>(
      d_src_row_ptr, d_src_col_indices, d_src_values, source->rows,
      source->cols, new_rows, new_cols, row_scale, col_scale, d_out_rows,
      d_out_cols, d_out_vals, d_out_count, output_buffer_size);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Get actual output count
  matgen_size_t actual_nnz;
  CUDA_CHECK(cudaMemcpy(&actual_nnz, d_out_count, sizeof(matgen_size_t),
                        cudaMemcpyDeviceToHost));

  MATGEN_LOG_DEBUG("Generated %zu triplets (buffer size: %zu)", actual_nnz,
                   output_buffer_size);

  if (actual_nnz > output_buffer_size) {
    MATGEN_LOG_ERROR(
        "Output buffer overflow: generated %zu entries, buffer size %zu. "
        "Increase estimation factor.",
        actual_nnz, output_buffer_size);
    cudaFree(d_src_row_ptr);
    cudaFree(d_src_col_indices);
    cudaFree(d_src_values);
    cudaFree(d_out_rows);
    cudaFree(d_out_cols);
    cudaFree(d_out_vals);
    cudaFree(d_out_count);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Create COO matrix and copy results
  matgen_coo_matrix_t* coo =
      matgen_coo_create_cuda(new_rows, new_cols, actual_nnz);
  if (!coo) {
    MATGEN_LOG_ERROR("Failed to create COO matrix");
    cudaFree(d_src_row_ptr);
    cudaFree(d_src_col_indices);
    cudaFree(d_src_values);
    cudaFree(d_out_rows);
    cudaFree(d_out_cols);
    cudaFree(d_out_vals);
    cudaFree(d_out_count);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Copy output from device to host
  CUDA_CHECK(cudaMemcpy(coo->row_indices, d_out_rows,
                        actual_nnz * sizeof(matgen_index_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(coo->col_indices, d_out_cols,
                        actual_nnz * sizeof(matgen_index_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(coo->values, d_out_vals,
                        actual_nnz * sizeof(matgen_value_t),
                        cudaMemcpyDeviceToHost));

  coo->nnz = actual_nnz;
  coo->is_sorted = false;

  // Cleanup device memory
  cudaFree(d_src_row_ptr);
  cudaFree(d_src_col_indices);
  cudaFree(d_src_values);
  cudaFree(d_out_rows);
  cudaFree(d_out_cols);
  cudaFree(d_out_vals);
  cudaFree(d_out_count);

  // Sort and handle duplicates using CUDA
  MATGEN_LOG_DEBUG("Sorting and handling duplicates (CUDA, policy: %d)...",
                   collision_policy);

  matgen_error_t err = matgen_coo_sort_cuda(coo);
  if (err != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to sort COO matrix");
    matgen_coo_destroy(coo);
    return err;
  }

  // Handle duplicates based on collision policy
  if (collision_policy == MATGEN_COLLISION_SUM) {
    err = matgen_coo_sum_duplicates_cuda(coo);
  } else {
    err = matgen_coo_merge_duplicates_cuda(coo, collision_policy);
  }

  if (err != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to handle duplicates");
    matgen_coo_destroy(coo);
    return err;
  }

  MATGEN_LOG_DEBUG("After deduplication: %zu entries", coo->nnz);

  // Convert to CSR using CUDA
  *result = matgen_coo_to_csr_cuda(coo);
  matgen_coo_destroy(coo);

  if (!(*result)) {
    MATGEN_LOG_ERROR("Failed to convert COO to CSR");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  MATGEN_LOG_DEBUG(
      "Nearest neighbor scaling (CUDA) completed: output NNZ = %zu",
      (*result)->nnz);

  return MATGEN_SUCCESS;
}
