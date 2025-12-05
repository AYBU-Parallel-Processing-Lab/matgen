// backends/cuda/internal/coo_cuda.cu
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "backends/cuda/internal/coo_cuda.h"
#include "matgen/core/matrix/coo.h"
#include "matgen/utils/log.h"

// =============================================================================
// Configuration / Macros
// =============================================================================

#define DEFAULT_INITIAL_CAPACITY 1024

#define CUDA_CHECK(call)                                              \
  do {                                                                \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
      MATGEN_LOG_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, \
                       cudaGetErrorString(err));                      \
      return MATGEN_ERROR_CUDA;                                       \
    }                                                                 \
  } while (0)

// =============================================================================
// Thrust helpers: key type, equality comparator, and reducers
// =============================================================================

using key_t = thrust::tuple<matgen_index_t, matgen_index_t>;

struct key_equal {
  __host__ __device__ bool operator()(const key_t& a, const key_t& b) const {
    return thrust::get<0>(a) == thrust::get<0>(b) &&
           thrust::get<1>(a) == thrust::get<1>(b);
  }
};

struct reducer_sum {
  __host__ __device__ matgen_value_t operator()(matgen_value_t a,
                                                matgen_value_t b) const {
    return a + b;
  }
};

struct reducer_max {
  __host__ __device__ matgen_value_t operator()(matgen_value_t a,
                                                matgen_value_t b) const {
    return a > b ? a : b;
  }
};

struct reducer_min {
  __host__ __device__ matgen_value_t operator()(matgen_value_t a,
                                                matgen_value_t b) const {
    return a < b ? a : b;
  }
};

struct reducer_last {
  __host__ __device__ matgen_value_t operator()(matgen_value_t a,
                                                matgen_value_t b) const {
    (void)a;
    return b;
  }
};

// =============================================================================
// Utility: safe host reallocation (returns MATGEN_SUCCESS or
// MATGEN_ERROR_OUT_OF_MEMORY)
// =============================================================================
static matgen_error_t safe_realloc_host_arrays(matgen_coo_matrix_t* matrix,
                                               matgen_size_t new_capacity) {
  if (!matrix) return MATGEN_ERROR_INVALID_ARGUMENT;
  if (new_capacity <= matrix->capacity) return MATGEN_SUCCESS;

  // Allocate new buffers first (avoid losing old pointers on partial failure)
  matgen_index_t* new_rows =
      (matgen_index_t*)malloc(new_capacity * sizeof(matgen_index_t));
  matgen_index_t* new_cols =
      (matgen_index_t*)malloc(new_capacity * sizeof(matgen_index_t));
  matgen_value_t* new_vals =
      (matgen_value_t*)malloc(new_capacity * sizeof(matgen_value_t));

  if (!new_rows || !new_cols || !new_vals) {
    MATGEN_LOG_ERROR(
        "Failed to allocate host buffers during reallocation (requested %zu)",
        new_capacity);
    free(new_rows);
    free(new_cols);
    free(new_vals);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Copy existing data
  if (matrix->nnz > 0) {
    memcpy(new_rows, matrix->row_indices, matrix->nnz * sizeof(matgen_index_t));
    memcpy(new_cols, matrix->col_indices, matrix->nnz * sizeof(matgen_index_t));
    memcpy(new_vals, matrix->values, matrix->nnz * sizeof(matgen_value_t));
  }

  // Free old and assign
  free(matrix->row_indices);
  free(matrix->col_indices);
  free(matrix->values);

  matrix->row_indices = new_rows;
  matrix->col_indices = new_cols;
  matrix->values = new_vals;
  matrix->capacity = new_capacity;

  return MATGEN_SUCCESS;
}

// =============================================================================
// Public API: create / destroy
// =============================================================================

matgen_coo_matrix_t* matgen_coo_create_cuda(matgen_index_t rows,
                                            matgen_index_t cols,
                                            matgen_size_t nnz_hint) {
  if (rows == 0 || cols == 0) {
    MATGEN_LOG_ERROR("Invalid matrix dimensions: %llu x %llu",
                     (unsigned long long)rows, (unsigned long long)cols);
    return NULL;
  }

  matgen_coo_matrix_t* matrix =
      (matgen_coo_matrix_t*)malloc(sizeof(matgen_coo_matrix_t));
  if (!matrix) {
    MATGEN_LOG_ERROR("Failed to allocate COO matrix structure");
    return NULL;
  }

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = 0;
  matrix->capacity = (nnz_hint > 0) ? nnz_hint : DEFAULT_INITIAL_CAPACITY;
  matrix->is_sorted = true;  // empty is trivially sorted

  matrix->row_indices =
      (matgen_index_t*)malloc(matrix->capacity * sizeof(matgen_index_t));
  matrix->col_indices =
      (matgen_index_t*)malloc(matrix->capacity * sizeof(matgen_index_t));
  matrix->values =
      (matgen_value_t*)malloc(matrix->capacity * sizeof(matgen_value_t));

  if (!matrix->row_indices || !matrix->col_indices || !matrix->values) {
    MATGEN_LOG_ERROR("Failed to allocate COO matrix arrays");
    matgen_coo_destroy(matrix);
    return NULL;
  }

  MATGEN_LOG_DEBUG("Created COO matrix (CUDA) %llu x %llu with capacity %zu",
                   (unsigned long long)rows, (unsigned long long)cols,
                   matrix->capacity);

  return matrix;
}

// =============================================================================
// Sorting: stable sort by (row, col) using thrust
// =============================================================================

matgen_error_t matgen_coo_sort_cuda(matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (matrix->nnz <= 1) {
    matrix->is_sorted = true;
    return MATGEN_SUCCESS;
  }

  MATGEN_LOG_DEBUG("Sorting COO matrix (CUDA) with %zu entries", matrix->nnz);

  size_t nnz = matrix->nnz;

  try {
    // Device vectors copy from host
    thrust::device_vector<matgen_index_t> d_rows(matrix->row_indices,
                                                 matrix->row_indices + nnz);
    thrust::device_vector<matgen_index_t> d_cols(matrix->col_indices,
                                                 matrix->col_indices + nnz);
    thrust::device_vector<matgen_value_t> d_vals(matrix->values,
                                                 matrix->values + nnz);

    auto keys_begin = thrust::make_zip_iterator(
        thrust::make_tuple(d_rows.begin(), d_cols.begin()));
    auto keys_end = keys_begin + nnz;

    // stable sort by (row, col) and permute values accordingly
    thrust::stable_sort_by_key(thrust::device, keys_begin, keys_end,
                               d_vals.begin());

    // copy back to host
    thrust::copy(d_rows.begin(), d_rows.end(), matrix->row_indices);
    thrust::copy(d_cols.begin(), d_cols.end(), matrix->col_indices);
    thrust::copy(d_vals.begin(), d_vals.end(), matrix->values);

    matrix->is_sorted = true;
    MATGEN_LOG_DEBUG("Matrix sorted successfully (CUDA)");
    return MATGEN_SUCCESS;
  } catch (const thrust::system_error& e) {
    MATGEN_LOG_ERROR("Thrust error during sort: %s", e.what());
    return MATGEN_ERROR_CUDA;
  }
}

// =============================================================================
// Deduplication (SUM) using reduce_by_key
// =============================================================================

matgen_error_t matgen_coo_sum_duplicates_cuda(matgen_coo_matrix_t* matrix) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (matrix->nnz <= 1) return MATGEN_SUCCESS;
  if (!matrix->is_sorted) {
    MATGEN_LOG_ERROR("Matrix must be sorted before sum_duplicates");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  MATGEN_LOG_DEBUG("Summing duplicates in COO matrix (CUDA) with %zu entries",
                   matrix->nnz);

  size_t nnz = matrix->nnz;

  try {
    thrust::device_vector<matgen_index_t> d_rows(matrix->row_indices,
                                                 matrix->row_indices + nnz);
    thrust::device_vector<matgen_index_t> d_cols(matrix->col_indices,
                                                 matrix->col_indices + nnz);
    thrust::device_vector<matgen_value_t> d_vals(matrix->values,
                                                 matrix->values + nnz);

    auto keys_begin = thrust::make_zip_iterator(
        thrust::make_tuple(d_rows.begin(), d_cols.begin()));
    auto keys_end = keys_begin + nnz;

    // Output buffers (max size = nnz)
    thrust::device_vector<matgen_index_t> out_rows(nnz);
    thrust::device_vector<matgen_index_t> out_cols(nnz);
    thrust::device_vector<matgen_value_t> out_vals(nnz);

    auto out_keys_begin = thrust::make_zip_iterator(
        thrust::make_tuple(out_rows.begin(), out_cols.begin()));

    auto new_end = thrust::reduce_by_key(
        thrust::device, keys_begin, keys_end, d_vals.begin(), out_keys_begin,
        out_vals.begin(), key_equal(), reducer_sum());

    // new_nnz = number of reduced keys
    size_t new_nnz = new_end.second - out_vals.begin();

    // Ensure host capacity
    if (new_nnz > matrix->capacity) {
      matgen_size_t new_cap = new_nnz;
      // round up a bit for amortized growth
      if (new_cap < matrix->capacity * 2) new_cap = matrix->capacity * 2;
      matgen_error_t r = safe_realloc_host_arrays(matrix, new_cap);
      if (r != MATGEN_SUCCESS) return r;
    }

    // Copy results to host
    thrust::copy(out_rows.begin(), out_rows.begin() + new_nnz,
                 matrix->row_indices);
    thrust::copy(out_cols.begin(), out_cols.begin() + new_nnz,
                 matrix->col_indices);
    thrust::copy(out_vals.begin(), out_vals.begin() + new_nnz, matrix->values);

    matgen_size_t old_nnz = matrix->nnz;
    matrix->nnz = new_nnz;
    matrix->is_sorted = true;

    MATGEN_LOG_DEBUG("Reduced nnz from %zu to %zu (removed %zu duplicates)",
                     old_nnz, matrix->nnz, (size_t)(old_nnz - matrix->nnz));
    return MATGEN_SUCCESS;

  } catch (const thrust::system_error& e) {
    MATGEN_LOG_ERROR("Thrust error during duplicate sum: %s", e.what());
    return MATGEN_ERROR_CUDA;
  }
}

// =============================================================================
// Merge duplicates with policy: SUM | AVG | MAX | MIN | LAST
// Implemented using reduce_by_key (AVG done via counts pass)
// =============================================================================

matgen_error_t matgen_coo_merge_duplicates_cuda(
    matgen_coo_matrix_t* matrix, matgen_collision_policy_t policy) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (matrix->nnz <= 1) return MATGEN_SUCCESS;
  if (!matrix->is_sorted) {
    MATGEN_LOG_ERROR("Matrix must be sorted before merge_duplicates");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  MATGEN_LOG_DEBUG(
      "Merging duplicates in COO matrix (CUDA) with %zu entries (policy=%d)",
      matrix->nnz, (int)policy);

  size_t nnz = matrix->nnz;

  try {
    thrust::device_vector<matgen_index_t> d_rows(matrix->row_indices,
                                                 matrix->row_indices + nnz);
    thrust::device_vector<matgen_index_t> d_cols(matrix->col_indices,
                                                 matrix->col_indices + nnz);
    thrust::device_vector<matgen_value_t> d_vals(matrix->values,
                                                 matrix->values + nnz);

    auto keys_begin = thrust::make_zip_iterator(
        thrust::make_tuple(d_rows.begin(), d_cols.begin()));
    auto keys_end = keys_begin + nnz;

    // Output buffers
    thrust::device_vector<matgen_index_t> out_rows(nnz);
    thrust::device_vector<matgen_index_t> out_cols(nnz);
    thrust::device_vector<matgen_value_t> out_vals(nnz);

    auto out_keys_begin = thrust::make_zip_iterator(
        thrust::make_tuple(out_rows.begin(), out_cols.begin()));

    // Choose reducer
    // For AVG, we'll reduce sums first and also compute counts in a second
    // reduce_by_key.
    if (policy == MATGEN_COLLISION_SUM) {
      auto new_end = thrust::reduce_by_key(
          thrust::device, keys_begin, keys_end, d_vals.begin(), out_keys_begin,
          out_vals.begin(), key_equal(), reducer_sum());
      size_t new_nnz = new_end.second - out_vals.begin();

      if (new_nnz > matrix->capacity) {
        matgen_size_t new_cap = new_nnz;
        if (new_cap < matrix->capacity * 2) new_cap = matrix->capacity * 2;
        matgen_error_t r = safe_realloc_host_arrays(matrix, new_cap);
        if (r != MATGEN_SUCCESS) return r;
      }

      thrust::copy(out_rows.begin(), out_rows.begin() + new_nnz,
                   matrix->row_indices);
      thrust::copy(out_cols.begin(), out_cols.begin() + new_nnz,
                   matrix->col_indices);
      thrust::copy(out_vals.begin(), out_vals.begin() + new_nnz,
                   matrix->values);

      matrix->nnz = new_nnz;
      matrix->is_sorted = true;
      return MATGEN_SUCCESS;
    }

    if (policy == MATGEN_COLLISION_MAX) {
      auto new_end = thrust::reduce_by_key(
          thrust::device, keys_begin, keys_end, d_vals.begin(), out_keys_begin,
          out_vals.begin(), key_equal(), reducer_max());
      size_t new_nnz = new_end.second - out_vals.begin();

      if (new_nnz > matrix->capacity) {
        matgen_size_t new_cap = new_nnz;
        if (new_cap < matrix->capacity * 2) new_cap = matrix->capacity * 2;
        matgen_error_t r = safe_realloc_host_arrays(matrix, new_cap);
        if (r != MATGEN_SUCCESS) return r;
      }

      thrust::copy(out_rows.begin(), out_rows.begin() + new_nnz,
                   matrix->row_indices);
      thrust::copy(out_cols.begin(), out_cols.begin() + new_nnz,
                   matrix->col_indices);
      thrust::copy(out_vals.begin(), out_vals.begin() + new_nnz,
                   matrix->values);

      matrix->nnz = new_nnz;
      matrix->is_sorted = true;
      return MATGEN_SUCCESS;
    }

    if (policy == MATGEN_COLLISION_MIN) {
      auto new_end = thrust::reduce_by_key(
          thrust::device, keys_begin, keys_end, d_vals.begin(), out_keys_begin,
          out_vals.begin(), key_equal(), reducer_min());
      size_t new_nnz = new_end.second - out_vals.begin();

      if (new_nnz > matrix->capacity) {
        matgen_size_t new_cap = new_nnz;
        if (new_cap < matrix->capacity * 2) new_cap = matrix->capacity * 2;
        matgen_error_t r = safe_realloc_host_arrays(matrix, new_cap);
        if (r != MATGEN_SUCCESS) return r;
      }

      thrust::copy(out_rows.begin(), out_rows.begin() + new_nnz,
                   matrix->row_indices);
      thrust::copy(out_cols.begin(), out_cols.begin() + new_nnz,
                   matrix->col_indices);
      thrust::copy(out_vals.begin(), out_vals.begin() + new_nnz,
                   matrix->values);

      matrix->nnz = new_nnz;
      matrix->is_sorted = true;
      return MATGEN_SUCCESS;
    }

    if (policy == MATGEN_COLLISION_LAST) {
      // reduce_by_key with reducer_last yields last value of each group
      auto new_end = thrust::reduce_by_key(
          thrust::device, keys_begin, keys_end, d_vals.begin(), out_keys_begin,
          out_vals.begin(), key_equal(), reducer_last());
      size_t new_nnz = new_end.second - out_vals.begin();

      if (new_nnz > matrix->capacity) {
        matgen_size_t new_cap = new_nnz;
        if (new_cap < matrix->capacity * 2) new_cap = matrix->capacity * 2;
        matgen_error_t r = safe_realloc_host_arrays(matrix, new_cap);
        if (r != MATGEN_SUCCESS) return r;
      }

      thrust::copy(out_rows.begin(), out_rows.begin() + new_nnz,
                   matrix->row_indices);
      thrust::copy(out_cols.begin(), out_cols.begin() + new_nnz,
                   matrix->col_indices);
      thrust::copy(out_vals.begin(), out_vals.begin() + new_nnz,
                   matrix->values);

      matrix->nnz = new_nnz;
      matrix->is_sorted = true;
      return MATGEN_SUCCESS;
    }

    if (policy == MATGEN_COLLISION_AVG) {
      // First pass: sums
      thrust::device_vector<matgen_value_t> summed_vals(nnz);
      thrust::device_vector<matgen_index_t> summed_rows(nnz);
      thrust::device_vector<matgen_index_t> summed_cols(nnz);
      auto summed_keys_begin = thrust::make_zip_iterator(
          thrust::make_tuple(summed_rows.begin(), summed_cols.begin()));

      auto sums_end = thrust::reduce_by_key(
          thrust::device, keys_begin, keys_end, d_vals.begin(),
          summed_keys_begin, summed_vals.begin(), key_equal(), reducer_sum());

      size_t sums_n = sums_end.second - summed_vals.begin();

      // Second pass: counts (use constant iterator of 1)
      thrust::device_vector<matgen_value_t> ones(
          nnz, static_cast<matgen_value_t>(1));
      thrust::device_vector<matgen_value_t> counts(nnz);
      thrust::device_vector<matgen_index_t> counts_rows(nnz);
      thrust::device_vector<matgen_index_t> counts_cols(nnz);
      auto counts_keys_begin = thrust::make_zip_iterator(
          thrust::make_tuple(counts_rows.begin(), counts_cols.begin()));

      auto counts_end = thrust::reduce_by_key(
          thrust::device, keys_begin, keys_end, ones.begin(), counts_keys_begin,
          counts.begin(), key_equal(), reducer_sum());

      size_t counts_n = counts_end.second - counts.begin();

      // sums_n and counts_n must be equal and correspond to unique keys
      if (sums_n != counts_n) {
        MATGEN_LOG_ERROR("Internal error: sums_n (%zu) != counts_n (%zu)",
                         sums_n, counts_n);
        return MATGEN_ERROR_CUDA;
      }

      // Compute averages in-place (on device)
      for (size_t i = 0; i < sums_n; ++i) {
        summed_vals[i] = summed_vals[i] / counts[i];
      }

      // Ensure host capacity
      if (sums_n > matrix->capacity) {
        matgen_size_t new_cap = sums_n;
        if (new_cap < matrix->capacity * 2) new_cap = matrix->capacity * 2;
        matgen_error_t r = safe_realloc_host_arrays(matrix, new_cap);
        if (r != MATGEN_SUCCESS) return r;
      }

      // Copy keys and averaged vals to host
      thrust::copy(summed_rows.begin(), summed_rows.begin() + sums_n,
                   matrix->row_indices);
      thrust::copy(summed_cols.begin(), summed_cols.begin() + sums_n,
                   matrix->col_indices);
      thrust::copy(summed_vals.begin(), summed_vals.begin() + sums_n,
                   matrix->values);

      matrix->nnz = sums_n;
      matrix->is_sorted = true;
      return MATGEN_SUCCESS;
    }

    // Unknown policy: fallback to SUM
    MATGEN_LOG_ERROR("Unknown collision policy %d (falling back to SUM)",
                     (int)policy);
    return matgen_coo_sum_duplicates_cuda(matrix);
  } catch (const thrust::system_error& e) {
    MATGEN_LOG_ERROR("Thrust error during merge duplicates: %s", e.what());
    return MATGEN_ERROR_CUDA;
  }
}
