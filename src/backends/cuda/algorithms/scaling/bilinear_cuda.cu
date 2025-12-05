#include <cuda_runtime.h>
#include <math.h>

#include "backends/cuda/internal/bilinear_cuda.h"
#include "backends/cuda/internal/conversion_cuda.h"
#include "backends/cuda/internal/coo_cuda.h"
#include "backends/cuda/internal/csr_builder_cuda.h"
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
#define MAX_CONTRIBUTIONS_PER_ENTRY \
  16  // Max neighborhood size per source entry

// =============================================================================
// CUDA Kernels
// =============================================================================

/**
 * @brief Kernel to compute bilinear contributions for each source entry
 *
 * Each thread processes one source entry and generates weighted contributions.
 * Outputs to a pre-allocated COO buffer with atomic position tracking.
 */
__global__ void bilinear_scale_kernel(
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

  // Calculate destination range for this source entry
  matgen_value_t dst_row_start_f =
      fmaxf(0.0f, ((matgen_value_t)src_row - 1.0f) * row_scale);
  matgen_value_t dst_row_end_f = ((matgen_value_t)src_row + 1.0f) * row_scale;
  matgen_value_t dst_col_start_f =
      fmaxf(0.0f, ((matgen_value_t)src_col - 1.0f) * col_scale);
  matgen_value_t dst_col_end_f = ((matgen_value_t)src_col + 1.0f) * col_scale;

  matgen_index_t dst_row_start = (matgen_index_t)ceilf(dst_row_start_f);
  matgen_index_t dst_row_end = (matgen_index_t)ceilf(dst_row_end_f);
  matgen_index_t dst_col_start = (matgen_index_t)ceilf(dst_col_start_f);
  matgen_index_t dst_col_end = (matgen_index_t)ceilf(dst_col_end_f);

  // Clamp to valid range
  dst_row_start = min(max((matgen_index_t)0, dst_row_start), dst_rows);
  dst_row_end = min(max((matgen_index_t)0, dst_row_end), dst_rows);
  dst_col_start = min(max((matgen_index_t)0, dst_col_start), dst_cols);
  dst_col_end = min(max((matgen_index_t)0, dst_col_end), dst_cols);

  // Generate contributions for each destination cell in neighborhood
  for (matgen_index_t dst_row = dst_row_start; dst_row < dst_row_end;
       dst_row++) {
    for (matgen_index_t dst_col = dst_col_start; dst_col < dst_col_end;
         dst_col++) {
      // Map destination cell back to source coordinates
      matgen_value_t src_y = (matgen_value_t)dst_row / row_scale;
      matgen_value_t src_x = (matgen_value_t)dst_col / col_scale;

      // Find bilinear neighbors
      matgen_index_t y0 = (matgen_index_t)floorf(src_y);
      matgen_index_t y1 = (matgen_index_t)ceilf(src_y);
      matgen_index_t x0 = (matgen_index_t)floorf(src_x);
      matgen_index_t x1 = (matgen_index_t)ceilf(src_x);

      // Clamp to source bounds
      y0 = min(max((matgen_index_t)0, y0), src_rows - 1);
      y1 = min(max((matgen_index_t)0, y1), src_rows - 1);
      x0 = min(max((matgen_index_t)0, x0), src_cols - 1);
      x1 = min(max((matgen_index_t)0, x1), src_cols - 1);

      // Calculate fractional parts
      matgen_value_t dy = src_y - (matgen_value_t)y0;
      matgen_value_t dx = src_x - (matgen_value_t)x0;

      // Clamp to [0, 1]
      dy = fminf(fmaxf(dy, 0.0f), 1.0f);
      dx = fminf(fmaxf(dx, 0.0f), 1.0f);

      // Determine weight based on which neighbor we are
      matgen_value_t weight = 0.0f;

      if (src_row == y0 && src_col == x0) {
        // Bottom-left
        weight = (1.0f - dy) * (1.0f - dx);
      } else if (src_row == y0 && src_col == x1) {
        // Bottom-right
        weight = (1.0f - dy) * dx;
      } else if (src_row == y1 && src_col == x0) {
        // Top-left
        weight = dy * (1.0f - dx);
      } else if (src_row == y1 && src_col == x1) {
        // Top-right
        weight = dy * dx;
      }

      // Add contribution if non-zero
      if (weight > 1e-12f) {
        matgen_value_t contribution = src_val * weight;

        // Atomic increment to get output position
        matgen_size_t pos = atomicAdd((unsigned long long*)out_count, 1ULL);

        if (pos < max_output_size) {
          out_rows[pos] = dst_row;
          out_cols[pos] = dst_col;
          out_vals[pos] = contribution;
        }
      }
    }
  }
}

// =============================================================================
// CUDA Backend Implementation
// =============================================================================

matgen_error_t matgen_scale_bilinear_cuda(const matgen_csr_matrix_t* source,
                                          matgen_index_t new_rows,
                                          matgen_index_t new_cols,
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
      "Bilinear scaling (CUDA): %llu×%llu -> %llu×%llu (scale: %.3fx%.3f)",
      (unsigned long long)source->rows, (unsigned long long)source->cols,
      (unsigned long long)new_rows, (unsigned long long)new_cols, row_scale,
      col_scale);

  // FIXED: More accurate estimation
  // Each source entry contributes to a neighborhood of at most:
  // - ceil(row_scale + 1) rows
  // - ceil(col_scale + 1) cols
  // But bilinear weights mean many contributions, so use higher safety factor
  matgen_value_t max_contributions_per_source =
      ceilf(row_scale + 1.0f) * ceilf(col_scale + 1.0f);

  // Use 2.0x safety factor to account for edge effects and overlaps
  size_t estimated_nnz = (size_t)((matgen_value_t)source->nnz *
                                  max_contributions_per_source * 2.0);

  // Ensure minimum buffer size
  if (estimated_nnz < source->nnz) {
    estimated_nnz = source->nnz * 2;
  }

  MATGEN_LOG_DEBUG(
      "Estimated output NNZ: %zu (max contributions per entry: %.1f)",
      estimated_nnz, max_contributions_per_source);

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

  bilinear_scale_kernel<<<blocks, CUDA_BLOCK_SIZE>>>(
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
        "Increase estimation factor or use adaptive allocation.",
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

  // Sort and sum duplicates using CUDA
  MATGEN_LOG_DEBUG("Sorting and summing duplicates (CUDA)...");

  matgen_error_t err = matgen_coo_sort_cuda(coo);
  if (err != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to sort COO matrix");
    matgen_coo_destroy(coo);
    return err;
  }

  err = matgen_coo_sum_duplicates_cuda(coo);
  if (err != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to sum duplicates");
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

  MATGEN_LOG_DEBUG("Bilinear scaling (CUDA) completed: output NNZ = %zu",
                   (*result)->nnz);

  return MATGEN_SUCCESS;
}
