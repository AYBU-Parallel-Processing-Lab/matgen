#include <cuda_runtime.h>
#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

#include <cstdlib>
#include <cstring>

#include "backends/cuda/internal/csr_cuda.cuh"
#include "matgen/core/matrix/csr.h"
#include "matgen/utils/log.h"

// =============================================================================
// CUDA Error Checking Macro
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

#define CUDA_CHECK_RETURN_NULL(call)                                  \
  do {                                                                \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
      MATGEN_LOG_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, \
                       cudaGetErrorString(err));                      \
      return NULL;                                                    \
    }                                                                 \
  } while (0)

// =============================================================================
// CUDA Kernels
// =============================================================================

/**
 * @brief Kernel to compute COO row indices from CSR row pointers
 *
 * Each thread processes one entry and looks up which row it belongs to.
 */
__global__ void csr_to_coo_rows_kernel(const matgen_size_t* row_ptr,
                                       matgen_index_t* coo_rows,
                                       matgen_index_t num_rows,
                                       matgen_size_t nnz) {
  matgen_size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= nnz) return;

  // Binary search to find which row this entry belongs to
  matgen_index_t low = 0;
  matgen_index_t high = num_rows;

  while (low < high) {
    matgen_index_t mid = low + (high - low) / 2;
    if (row_ptr[mid] <= idx) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  coo_rows[idx] = low - 1;
}

/**
 * @brief Kernel to build CSR row_ptr from COO row indices
 *
 * Uses atomic increment to count entries per row.
 */
__global__ void coo_to_csr_count_kernel(const matgen_index_t* coo_rows,
                                        matgen_size_t* row_counts,
                                        matgen_size_t nnz) {
  matgen_size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= nnz) return;

  atomicAdd((unsigned long long*)&row_counts[coo_rows[idx]], 1ULL);
}

/**
 * @brief Kernel to compute row statistics (prepare for reduction)
 */
__global__ void compute_row_lengths_kernel(const matgen_size_t* row_ptr,
                                           matgen_size_t* row_lengths,
                                           matgen_index_t num_rows) {
  matgen_index_t row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= num_rows) return;

  row_lengths[row] = row_ptr[row + 1] - row_ptr[row];
}

/**
 * @brief Kernel for parallel matrix transpose using scatter
 */
__global__ void transpose_scatter_kernel(const matgen_size_t* src_row_ptr,
                                         const matgen_index_t* src_col_indices,
                                         const matgen_value_t* src_values,
                                         const matgen_size_t* dst_row_scatter,
                                         matgen_index_t* dst_col_indices,
                                         matgen_value_t* dst_values,
                                         matgen_index_t num_rows) {
  matgen_index_t src_row = blockIdx.x * blockDim.x + threadIdx.x;

  if (src_row >= num_rows) return;

  matgen_size_t row_start = src_row_ptr[src_row];
  matgen_size_t row_end = src_row_ptr[src_row + 1];

  // For each entry in source row
  for (matgen_size_t i = row_start; i < row_end; i++) {
    matgen_index_t src_col = src_col_indices[i];
    matgen_value_t value = src_values[i];

    // Scatter to transposed position
    // Atomic increment to get position in destination row
    matgen_size_t dst_pos =
        atomicAdd((unsigned long long*)&dst_row_scatter[src_col], 1ULL);

    dst_col_indices[dst_pos] = src_row;
    dst_values[dst_pos] = value;
  }
}

// =============================================================================
// CUDA Backend Implementation
// =============================================================================

matgen_csr_matrix_t* matgen_csr_create_cuda(matgen_index_t rows,
                                            matgen_index_t cols,
                                            matgen_size_t nnz) {
  if (rows == 0 || cols == 0) {
    MATGEN_LOG_ERROR("Invalid matrix dimensions: %llu x %llu",
                     (unsigned long long)rows, (unsigned long long)cols);
    return NULL;
  }

  matgen_csr_matrix_t* matrix =
      (matgen_csr_matrix_t*)malloc(sizeof(matgen_csr_matrix_t));
  if (!matrix) {
    MATGEN_LOG_ERROR("Failed to allocate CSR matrix structure");
    return NULL;
  }

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = nnz;

  // Allocate host arrays
  matrix->row_ptr = (matgen_size_t*)calloc(rows + 1, sizeof(matgen_size_t));
  matrix->col_indices = nullptr;
  matrix->values = nullptr;

  if (nnz > 0) {
    matrix->col_indices = (matgen_index_t*)malloc(nnz * sizeof(matgen_index_t));
    matrix->values = (matgen_value_t*)malloc(nnz * sizeof(matgen_value_t));
  }

  if (!matrix->row_ptr ||
      (nnz > 0 && (!matrix->col_indices || !matrix->values))) {
    MATGEN_LOG_ERROR("Failed to allocate CSR matrix arrays");
    matgen_csr_destroy(matrix);
    return NULL;
  }

  MATGEN_LOG_DEBUG("Created CSR matrix (CUDA) %llu x %llu with %zu non-zeros",
                   (unsigned long long)rows, (unsigned long long)cols, nnz);

  return matrix;
}

matgen_csr_matrix_t* matgen_csr_clone_cuda(const matgen_csr_matrix_t* src) {
  if (!src) {
    MATGEN_LOG_ERROR("NULL source matrix pointer");
    return NULL;
  }

  matgen_csr_matrix_t* dst =
      matgen_csr_create_cuda(src->rows, src->cols, src->nnz);
  if (!dst) {
    return NULL;
  }

  // Copy row pointers
  memcpy(dst->row_ptr, src->row_ptr, (src->rows + 1) * sizeof(matgen_size_t));

  // Copy column indices and values
  if (src->nnz > 0) {
    memcpy(dst->col_indices, src->col_indices,
           src->nnz * sizeof(matgen_index_t));
    memcpy(dst->values, src->values, src->nnz * sizeof(matgen_value_t));
  }

  MATGEN_LOG_DEBUG("Cloned CSR matrix (CUDA) %llu x %llu with %zu non-zeros",
                   (unsigned long long)src->rows, (unsigned long long)src->cols,
                   src->nnz);

  return dst;
}

matgen_error_t matgen_csr_transpose_cuda(const matgen_csr_matrix_t* matrix,
                                         matgen_csr_matrix_t** result) {
  if (!matrix || !result) {
    MATGEN_LOG_ERROR("NULL pointer argument");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  MATGEN_LOG_DEBUG("Transposing CSR matrix (CUDA) %llu x %llu with %zu nnz",
                   (unsigned long long)matrix->rows,
                   (unsigned long long)matrix->cols, matrix->nnz);

  matgen_index_t trans_rows = matrix->cols;
  matgen_index_t trans_cols = matrix->rows;

  matgen_csr_matrix_t* trans =
      matgen_csr_create_cuda(trans_rows, trans_cols, matrix->nnz);
  if (!trans) {
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  if (matrix->nnz == 0) {
    *result = trans;
    return MATGEN_SUCCESS;
  }

  // Device pointers
  matgen_size_t* d_src_row_ptr = nullptr;
  matgen_index_t* d_src_col_indices = nullptr;
  matgen_value_t* d_src_values = nullptr;
  matgen_size_t* d_dst_row_counts =
      nullptr;  // histogram counts (length trans_rows)
  matgen_size_t* d_dst_row_ptr =
      nullptr;  // row_ptr of transposed (length trans_rows+1)
  matgen_size_t* d_dst_row_scatter = nullptr;
  matgen_index_t* d_dst_col_indices = nullptr;
  matgen_value_t* d_dst_values = nullptr;

  size_t size_row_ptr = (matrix->rows + 1) * sizeof(matgen_size_t);
  size_t size_trans_row_ptr = (trans_rows + 1) * sizeof(matgen_size_t);
  size_t size_col_indices = matrix->nnz * sizeof(matgen_index_t);
  size_t size_values = matrix->nnz * sizeof(matgen_value_t);

  // allocate device memory
  CUDA_CHECK(cudaMalloc(&d_src_row_ptr, size_row_ptr));
  CUDA_CHECK(cudaMalloc(&d_src_col_indices, size_col_indices));
  CUDA_CHECK(cudaMalloc(&d_src_values, size_values));
  CUDA_CHECK(cudaMalloc(&d_dst_row_counts, trans_rows * sizeof(matgen_size_t)));
  CUDA_CHECK(cudaMalloc(&d_dst_row_ptr, size_trans_row_ptr));
  CUDA_CHECK(cudaMalloc(&d_dst_row_scatter, size_trans_row_ptr));
  CUDA_CHECK(cudaMalloc(&d_dst_col_indices, size_col_indices));
  CUDA_CHECK(cudaMalloc(&d_dst_values, size_values));

  // copy source arrays to device
  CUDA_CHECK(cudaMemcpy(d_src_row_ptr, matrix->row_ptr, size_row_ptr,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_src_col_indices, matrix->col_indices,
                        size_col_indices, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_src_values, matrix->values, size_values,
                        cudaMemcpyHostToDevice));

  // zero initialize counts
  CUDA_CHECK(
      cudaMemset(d_dst_row_counts, 0, trans_rows * sizeof(matgen_size_t)));

  // Use Thrust on device to compute histogram efficiently:
  // Strategy: copy column indices to a temporary device_vector, sort them,
  // reduce_by_key to get unique column ids and counts, then scatter counts
  // into the d_dst_row_counts array (length = trans_rows). This leaves
  // original d_src_col_indices untouched.

  try {
    // Wrap raw device pointers
    thrust::device_ptr<matgen_index_t> dev_src_cols(d_src_col_indices);
    thrust::device_ptr<matgen_size_t> dev_dst_counts(d_dst_row_counts);
    thrust::device_ptr<matgen_size_t> dev_dst_row_ptr(d_dst_row_ptr);

    // Copy column indices into a temporary device_vector (we will sort this)
    thrust::device_vector<matgen_index_t> tmp_cols(matrix->nnz);
    thrust::copy(dev_src_cols, dev_src_cols + matrix->nnz, tmp_cols.begin());

    // sort tmp_cols
    thrust::sort(tmp_cols.begin(), tmp_cols.end());

    // reduce_by_key over sorted columns to get unique column values and counts
    thrust::device_vector<matgen_index_t> unique_cols(matrix->nnz);
    thrust::device_vector<matgen_size_t> counts_vec(matrix->nnz);

    auto new_end =
        thrust::reduce_by_key(tmp_cols.begin(), tmp_cols.end(),
                              thrust::make_constant_iterator((matgen_size_t)1),
                              unique_cols.begin(), counts_vec.begin());

    size_t unique_count = new_end.first - unique_cols.begin();

    // zero target counts array
    thrust::fill(dev_dst_counts, dev_dst_counts + trans_rows, (matgen_size_t)0);

    // scatter counts_vec[0:unique_count] into dev_dst_counts at indices
    // unique_cols Prepare device_ptrs for scatter
    thrust::device_ptr<matgen_size_t> dev_counts_ptr = counts_vec.data();
    thrust::device_ptr<matgen_index_t> dev_unique_ptr = unique_cols.data();

    // scatter: source, source_end, map_begin, target_begin
    thrust::scatter(dev_counts_ptr, dev_counts_ptr + unique_count,
                    dev_unique_ptr, dev_dst_counts);

    // exclusive scan to build row_ptr of transposed matrix
    // exclusive_scan over counts of length trans_rows -> yields first
    // trans_rows entries then set last element to total nnz
    thrust::exclusive_scan(dev_dst_counts, dev_dst_counts + trans_rows,
                           dev_dst_row_ptr);

    // compute total nnz (last element = row_ptr[trans_rows-1] +
    // counts[trans_rows-1]) read last prefix and last count from device and
    // write final element
    matgen_size_t last_prefix = 0;
    matgen_size_t last_count = 0;
    if (trans_rows > 0) {
      CUDA_CHECK(cudaMemcpy(&last_prefix, d_dst_row_ptr + (trans_rows - 1),
                            sizeof(matgen_size_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(&last_count, d_dst_row_counts + (trans_rows - 1),
                            sizeof(matgen_size_t), cudaMemcpyDeviceToHost));
    }
    matgen_size_t total_nnz = last_prefix + last_count;
    CUDA_CHECK(cudaMemcpy(d_dst_row_ptr + trans_rows, &total_nnz,
                          sizeof(matgen_size_t), cudaMemcpyHostToDevice));

    // Copy row_ptr into scatter buffer (initial positions)
    CUDA_CHECK(cudaMemcpy(d_dst_row_scatter, d_dst_row_ptr, size_trans_row_ptr,
                          cudaMemcpyDeviceToDevice));

    // Launch scatter kernel: one thread per source row (matrix->rows)
    int blocks = (matrix->rows + 255) / 256;
    transpose_scatter_kernel<<<blocks, 256>>>(
        d_src_row_ptr, d_src_col_indices, d_src_values, d_dst_row_scatter,
        d_dst_col_indices, d_dst_values, matrix->rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy results back to host trans structure
    CUDA_CHECK(cudaMemcpy(trans->row_ptr, d_dst_row_ptr, size_trans_row_ptr,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(trans->col_indices, d_dst_col_indices,
                          size_col_indices, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(trans->values, d_dst_values, size_values,
                          cudaMemcpyDeviceToHost));

    MATGEN_LOG_DEBUG("Transpose completed (CUDA)");
    *result = trans;

  } catch (const thrust::system_error& e) {
    MATGEN_LOG_ERROR("Thrust error during transpose: %s", e.what());
    matgen_csr_destroy(trans);
    // free device memory
    cudaFree(d_src_row_ptr);
    cudaFree(d_src_col_indices);
    cudaFree(d_src_values);
    cudaFree(d_dst_row_counts);
    cudaFree(d_dst_row_ptr);
    cudaFree(d_dst_row_scatter);
    cudaFree(d_dst_col_indices);
    cudaFree(d_dst_values);
    return MATGEN_ERROR_CUDA;
  }

  // cleanup device memory
  cudaFree(d_src_row_ptr);
  cudaFree(d_src_col_indices);
  cudaFree(d_src_values);
  cudaFree(d_dst_row_counts);
  cudaFree(d_dst_row_ptr);
  cudaFree(d_dst_row_scatter);
  cudaFree(d_dst_col_indices);
  cudaFree(d_dst_values);

  return MATGEN_SUCCESS;
}

matgen_error_t matgen_csr_get_row_cuda(const matgen_csr_matrix_t* matrix,
                                       matgen_index_t row_idx,
                                       matgen_size_t* nnz_out,
                                       matgen_index_t** col_indices_out,
                                       matgen_value_t** values_out) {
  if (!matrix || !nnz_out || !col_indices_out || !values_out) {
    MATGEN_LOG_ERROR("NULL pointer argument");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (row_idx >= matrix->rows) {
    MATGEN_LOG_ERROR("Row index %llu out of bounds (num_rows: %llu)",
                     (unsigned long long)row_idx,
                     (unsigned long long)matrix->rows);
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  matgen_size_t row_start = matrix->row_ptr[row_idx];
  matgen_size_t row_end = matrix->row_ptr[row_idx + 1];
  matgen_size_t row_nnz = row_end - row_start;

  *nnz_out = row_nnz;

  if (row_nnz == 0) {
    *col_indices_out = nullptr;
    *values_out = nullptr;
    return MATGEN_SUCCESS;
  }

  // Allocate and copy row data
  *col_indices_out = (matgen_index_t*)malloc(row_nnz * sizeof(matgen_index_t));
  *values_out = (matgen_value_t*)malloc(row_nnz * sizeof(matgen_value_t));

  if (!*col_indices_out || !*values_out) {
    MATGEN_LOG_ERROR("Failed to allocate row data");
    free(*col_indices_out);
    free(*values_out);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  memcpy(*col_indices_out, &matrix->col_indices[row_start],
         row_nnz * sizeof(matgen_index_t));
  memcpy(*values_out, &matrix->values[row_start],
         row_nnz * sizeof(matgen_value_t));

  return MATGEN_SUCCESS;
}

matgen_error_t matgen_csr_row_stats_cuda(const matgen_csr_matrix_t* matrix,
                                         matgen_size_t* min_nnz_out,
                                         matgen_size_t* max_nnz_out,
                                         double* avg_nnz_out) {
  if (!matrix || !min_nnz_out || !max_nnz_out || !avg_nnz_out) {
    MATGEN_LOG_ERROR("NULL pointer argument");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (matrix->rows == 0) {
    *min_nnz_out = 0;
    *max_nnz_out = 0;
    *avg_nnz_out = 0.0;
    return MATGEN_SUCCESS;
  }

  MATGEN_LOG_DEBUG("Computing row statistics (CUDA) for %llu rows",
                   (unsigned long long)matrix->rows);

  // Allocate device memory
  matgen_size_t* d_row_ptr = nullptr;
  matgen_size_t* d_row_lengths = nullptr;

  size_t size_row_ptr = (matrix->rows + 1) * sizeof(matgen_size_t);
  size_t size_row_lengths = matrix->rows * sizeof(matgen_size_t);

  CUDA_CHECK(cudaMalloc(&d_row_ptr, size_row_ptr));
  CUDA_CHECK(cudaMalloc(&d_row_lengths, size_row_lengths));

  // Copy row pointers to device
  CUDA_CHECK(cudaMemcpy(d_row_ptr, matrix->row_ptr, size_row_ptr,
                        cudaMemcpyHostToDevice));

  // Compute row lengths
  int blocks = (matrix->rows + 255) / 256;
  compute_row_lengths_kernel<<<blocks, 256>>>(d_row_ptr, d_row_lengths,
                                              matrix->rows);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  try {
    // Use thrust to compute min/max/sum
    thrust::device_ptr<matgen_size_t> thrust_lengths(d_row_lengths);

    auto minmax =
        thrust::minmax_element(thrust_lengths, thrust_lengths + matrix->rows);

    matgen_size_t min_nnz, max_nnz;
    CUDA_CHECK(cudaMemcpy(&min_nnz, thrust::raw_pointer_cast(minmax.first),
                          sizeof(matgen_size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&max_nnz, thrust::raw_pointer_cast(minmax.second),
                          sizeof(matgen_size_t), cudaMemcpyDeviceToHost));

    matgen_size_t sum_nnz =
        thrust::reduce(thrust_lengths, thrust_lengths + matrix->rows,
                       (matgen_size_t)0, thrust::plus<matgen_size_t>());

    *min_nnz_out = min_nnz;
    *max_nnz_out = max_nnz;
    *avg_nnz_out = (double)sum_nnz / (double)matrix->rows;

    MATGEN_LOG_DEBUG("Row stats: min=%zu, max=%zu, avg=%.2f", min_nnz, max_nnz,
                     *avg_nnz_out);

  } catch (const thrust::system_error& e) {
    MATGEN_LOG_ERROR("Thrust error: %s", e.what());
    cudaFree(d_row_ptr);
    cudaFree(d_row_lengths);
    return MATGEN_ERROR_CUDA;
  }

  cudaFree(d_row_ptr);
  cudaFree(d_row_lengths);

  return MATGEN_SUCCESS;
}
