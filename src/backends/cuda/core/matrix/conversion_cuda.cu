#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>

#include <cstdlib>
#include <cstring>

#include "backends/cuda/internal/conversion_cuda.cuh"
#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/csr.h"
#include "matgen/utils/log.h"

#ifndef MATGEN_HAS_CUDA
#warning \
    "MATGEN_HAS_CUDA not defined: conversion_cuda compiled but will not be used"
#endif

// Error macro (returns NULL for pointer-returning functions or error codes
// where appropriate)
#define CUDA_CHECK_RET_NULL(call)                                     \
  do {                                                                \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
      MATGEN_LOG_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, \
                       cudaGetErrorString(err));                      \
      return NULL;                                                    \
    }                                                                 \
  } while (0)

#define CUDA_CHECK_RETURN(call, retval)                               \
  do {                                                                \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
      MATGEN_LOG_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, \
                       cudaGetErrorString(err));                      \
      return retval;                                                  \
    }                                                                 \
  } while (0)

// -----------------------------------------------------------------------------
// Kernel to expand CSR->COO: one thread per row, write row index into positions
// -----------------------------------------------------------------------------
__global__ static void expand_csr_rows_kernel(const matgen_size_t* d_row_ptr,
                                              matgen_index_t* d_coo_rows,
                                              matgen_size_t nnz,
                                              matgen_index_t num_rows) {
  matgen_index_t row = (matgen_index_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;

  matgen_size_t start = d_row_ptr[row];
  matgen_size_t end = d_row_ptr[row + 1];

  for (matgen_size_t p = start; p < end; ++p) {
    d_coo_rows[p] = row;
  }
}

// -----------------------------------------------------------------------------
// COO -> CSR (CUDA)
// -----------------------------------------------------------------------------
matgen_csr_matrix_t* matgen_coo_to_csr_cuda(const matgen_coo_matrix_t* coo) {
  if (!coo) {
    MATGEN_LOG_ERROR("NULL COO matrix pointer");
    return NULL;
  }

  if (!matgen_coo_validate(coo)) {
    MATGEN_LOG_ERROR("Invalid COO matrix");
    return NULL;
  }

  MATGEN_LOG_DEBUG("Converting COO (%llu x %llu, nnz=%zu) to CSR (CUDA)",
                   (unsigned long long)coo->rows, (unsigned long long)coo->cols,
                   coo->nnz);

  matgen_csr_matrix_t* csr = matgen_csr_create(coo->rows, coo->cols, coo->nnz);
  if (!csr) return NULL;

  if (coo->nnz == 0) {
    MATGEN_LOG_DEBUG("Empty matrix, conversion trivial");
    return csr;
  }

  size_t nnz = coo->nnz;
  matgen_index_t nrows = coo->rows;

  try {
    // Device copies of COO arrays
    thrust::device_vector<matgen_index_t> d_rows(coo->row_indices,
                                                 coo->row_indices + nnz);
    thrust::device_vector<matgen_index_t> d_cols(coo->col_indices,
                                                 coo->col_indices + nnz);
    thrust::device_vector<matgen_value_t> d_vals(coo->values,
                                                 coo->values + nnz);

    // If not sorted, sort by (row, col) on device (stable to preserve order for
    // equal keys)
    if (!coo->is_sorted) {
      auto keys_begin = thrust::make_zip_iterator(
          thrust::make_tuple(d_rows.begin(), d_cols.begin()));
      auto keys_end = keys_begin + nnz;
      thrust::stable_sort_by_key(thrust::device, keys_begin, keys_end,
                                 d_vals.begin());
    }

    // --- Phase 1: compute counts per row via reduce_by_key on sorted rows ---
    // reduce_by_key over d_rows with constant 1 produces unique_rows and counts
    thrust::device_vector<matgen_index_t> unique_rows(nnz);
    thrust::device_vector<matgen_size_t> counts(nnz);

    auto end_pair =
        thrust::reduce_by_key(thrust::device, d_rows.begin(), d_rows.end(),
                              thrust::make_constant_iterator((matgen_size_t)1),
                              unique_rows.begin(), counts.begin());

    size_t unique_count = end_pair.first - unique_rows.begin();

    // Create device row_counts array of length nrows, initialize to 0
    thrust::device_vector<matgen_size_t> d_row_counts((size_t)nrows,
                                                      (matgen_size_t)0);

    // Scatter counts into d_row_counts at indices unique_rows[0:unique_count]
    thrust::scatter(thrust::device, counts.begin(),
                    counts.begin() + unique_count, unique_rows.begin(),
                    d_row_counts.begin());

    // --- Phase 2: exclusive scan to build row_ptr (length nrows+1) ---
    thrust::device_vector<matgen_size_t> d_row_ptr((size_t)nrows + 1);
    thrust::exclusive_scan(thrust::device, d_row_counts.begin(),
                           d_row_counts.end(), d_row_ptr.begin());

    // The last element (total nnz) = row_ptr[nrows-1] + counts[nrows-1]
    // compute total nnz and set last entry
    matgen_size_t last_prefix = 0;
    matgen_size_t last_count = 0;
    if (nrows > 0) {
      CUDA_CHECK_RET_NULL(
          cudaMemcpy(&last_prefix,
                     thrust::raw_pointer_cast(d_row_ptr.data()) + (nrows - 1),
                     sizeof(matgen_size_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK_RET_NULL(cudaMemcpy(
          &last_count,
          thrust::raw_pointer_cast(d_row_counts.data()) + (nrows - 1),
          sizeof(matgen_size_t), cudaMemcpyDeviceToHost));
    }
    matgen_size_t total_nnz = last_prefix + last_count;
    CUDA_CHECK_RET_NULL(
        cudaMemcpy(thrust::raw_pointer_cast(d_row_ptr.data()) + nrows,
                   &total_nnz, sizeof(matgen_size_t), cudaMemcpyHostToDevice));

    // --- Phase 3: copy columns & values into CSR arrays (order-preserving) ---
    // Since COO is sorted by (row, col), we can directly copy to CSR
    // No atomic scatter needed - the sorted order is exactly what CSR needs
    thrust::device_vector<matgen_index_t> d_dst_cols = d_cols;
    thrust::device_vector<matgen_value_t> d_dst_vals = d_vals;

    // Copy row_ptr, col_indices, values back to host CSR structure
    // row_ptr length = nrows+1
    CUDA_CHECK_RET_NULL(cudaMemcpy(
        csr->row_ptr, thrust::raw_pointer_cast(d_row_ptr.data()),
        ((size_t)nrows + 1) * sizeof(matgen_size_t), cudaMemcpyDeviceToHost));

    // copy col_indices and values
    CUDA_CHECK_RET_NULL(cudaMemcpy(
        csr->col_indices, thrust::raw_pointer_cast(d_dst_cols.data()),
        nnz * sizeof(matgen_index_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RET_NULL(
        cudaMemcpy(csr->values, thrust::raw_pointer_cast(d_dst_vals.data()),
                   nnz * sizeof(matgen_value_t), cudaMemcpyDeviceToHost));

    csr->nnz = nnz;

    MATGEN_LOG_DEBUG("COO -> CSR conversion complete (CUDA)");
    return csr;

  } catch (const thrust::system_error& e) {
    MATGEN_LOG_ERROR("Thrust error during COO->CSR conversion: %s", e.what());
    matgen_csr_destroy(csr);
    return NULL;
  }
}

// -----------------------------------------------------------------------------
// CSR -> COO (CUDA)
// -----------------------------------------------------------------------------
matgen_coo_matrix_t* matgen_csr_to_coo_cuda(const matgen_csr_matrix_t* csr) {
  if (!csr) {
    MATGEN_LOG_ERROR("NULL CSR matrix pointer");
    return NULL;
  }

  if (!matgen_csr_validate(csr)) {
    MATGEN_LOG_ERROR("Invalid CSR matrix");
    return NULL;
  }

  MATGEN_LOG_DEBUG("Converting CSR (%llu x %llu, nnz=%zu) to COO (CUDA)",
                   (unsigned long long)csr->rows, (unsigned long long)csr->cols,
                   csr->nnz);

  matgen_coo_matrix_t* coo = matgen_coo_create(csr->rows, csr->cols, csr->nnz);
  if (!coo) return NULL;

  if (csr->nnz == 0) {
    MATGEN_LOG_DEBUG("Empty matrix, conversion trivial");
    return coo;
  }

  size_t nnz = csr->nnz;
  matgen_index_t nrows = csr->rows;

  try {
    // Copy csr row_ptr / col_indices / values to device
    thrust::device_vector<matgen_size_t> d_row_ptr((size_t)nrows + 1);
    thrust::device_vector<matgen_index_t> d_cols(nnz);
    thrust::device_vector<matgen_value_t> d_vals(nnz);

    CUDA_CHECK_RET_NULL(cudaMemcpy(
        thrust::raw_pointer_cast(d_row_ptr.data()), csr->row_ptr,
        ((size_t)nrows + 1) * sizeof(matgen_size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_RET_NULL(
        cudaMemcpy(thrust::raw_pointer_cast(d_cols.data()), csr->col_indices,
                   nnz * sizeof(matgen_index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_RET_NULL(cudaMemcpy(thrust::raw_pointer_cast(d_vals.data()),
                                   csr->values, nnz * sizeof(matgen_value_t),
                                   cudaMemcpyHostToDevice));

    // Allocate device array for coo row indices
    thrust::device_vector<matgen_index_t> d_coo_rows(nnz);

    // Launch kernel: one thread per row; each writes its row into assigned
    // range
    int block = 256;
    int grid = (int)((nrows + block - 1) / block);
    expand_csr_rows_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(d_row_ptr.data()),
        thrust::raw_pointer_cast(d_coo_rows.data()), nnz, nrows);
    CUDA_CHECK_RET_NULL(cudaGetLastError());
    CUDA_CHECK_RET_NULL(cudaDeviceSynchronize());

    // Copy back to host coo structure
    CUDA_CHECK_RET_NULL(cudaMemcpy(
        coo->row_indices, thrust::raw_pointer_cast(d_coo_rows.data()),
        nnz * sizeof(matgen_index_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RET_NULL(
        cudaMemcpy(coo->col_indices, thrust::raw_pointer_cast(d_cols.data()),
                   nnz * sizeof(matgen_index_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RET_NULL(
        cudaMemcpy(coo->values, thrust::raw_pointer_cast(d_vals.data()),
                   nnz * sizeof(matgen_value_t), cudaMemcpyDeviceToHost));

    coo->nnz = nnz;
    coo->is_sorted = true;  // CSR => row-sorted COO

    MATGEN_LOG_DEBUG("CSR -> COO conversion complete (CUDA)");
    return coo;

  } catch (const thrust::system_error& e) {
    MATGEN_LOG_ERROR("Thrust error during CSR->COO conversion: %s", e.what());
    matgen_coo_destroy(coo);
    return NULL;
  }
}
