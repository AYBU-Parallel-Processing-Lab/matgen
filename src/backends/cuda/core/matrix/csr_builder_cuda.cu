#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "backends/cuda/internal/csr_builder_cuda.cuh"
#include "core/matrix/csr_builder_internal.h"
#include "matgen/core/matrix/csr.h"
#include "matgen/core/matrix/csr_builder.h"
#include "matgen/utils/log.h"

// -----------------------------------------------------------------------------
// Configuration / helpers
// -----------------------------------------------------------------------------
#define INITIAL_CAPACITY 1024
#define CUDA_CHECK_RET_NULL(call)                                     \
  do {                                                                \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
      MATGEN_LOG_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, \
                       cudaGetErrorString(err));                      \
      return NULL;                                                    \
    }                                                                 \
  } while (0)

#define CUDA_CHECK_RET(call, retval)                                  \
  do {                                                                \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
      MATGEN_LOG_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, \
                       cudaGetErrorString(err));                      \
      return retval;                                                  \
    }                                                                 \
  } while (0)

// -----------------------------------------------------------------------------
// Thrust helpers
// -----------------------------------------------------------------------------
using key_pair_t = thrust::tuple<matgen_index_t, matgen_index_t>;

struct key_equal {
  __host__ __device__ bool operator()(const key_pair_t& a,
                                      const key_pair_t& b) const {
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

// -----------------------------------------------------------------------------
// Functors that must live at namespace/global scope
// -----------------------------------------------------------------------------
struct decrement_functor {
  __host__ __device__ matgen_size_t operator()(matgen_size_t x) const {
    return x - 1;
  }
};

struct compute_pos_functor {
  const matgen_size_t* row_ptr;
  const matgen_index_t* rows;
  const matgen_size_t* offsets;

  __host__ __device__ compute_pos_functor(const matgen_size_t* rp,
                                          const matgen_index_t* r,
                                          const matgen_size_t* o)
      : row_ptr(rp), rows(r), offsets(o) {}

  __host__ __device__ matgen_size_t operator()(size_t idx) const {
    return row_ptr[rows[idx]] + offsets[idx];
  }
};

// For safety, we'll fetch pointers via offset-like access. But most projects
// will already include a backend.cuda member. If your struct differs, adapt.

static matgen_error_t ensure_capacity_cuda(matgen_csr_builder_t* builder,
                                           matgen_size_t min_capacity) {
  if (!builder) return MATGEN_ERROR_INVALID_ARGUMENT;

  // Access backend.cuda fields (assumed)
  matgen_size_t cap = builder->backend.cuda.capacity;
  if (cap >= min_capacity) return MATGEN_SUCCESS;

  // new capacity: double until >= min_capacity
  matgen_size_t new_cap = (cap == 0) ? INITIAL_CAPACITY : cap;
  while (new_cap < min_capacity) new_cap *= 2;

  // Allocate pinned host memory (cudaHostAlloc) for faster transfer
  matgen_index_t* new_rows = nullptr;
  matgen_index_t* new_cols = nullptr;
  matgen_value_t* new_vals = nullptr;

  CUDA_CHECK_RET(
      cudaHostAlloc((void**)&new_rows, new_cap * sizeof(matgen_index_t),
                    cudaHostAllocPortable),
      MATGEN_ERROR_OUT_OF_MEMORY);
  CUDA_CHECK_RET(
      cudaHostAlloc((void**)&new_cols, new_cap * sizeof(matgen_index_t),
                    cudaHostAllocPortable),
      MATGEN_ERROR_OUT_OF_MEMORY);
  CUDA_CHECK_RET(
      cudaHostAlloc((void**)&new_vals, new_cap * sizeof(matgen_value_t),
                    cudaHostAllocPortable),
      MATGEN_ERROR_OUT_OF_MEMORY);

  // copy existing data if any
  if (cap > 0 && builder->backend.cuda.host_rows) {
    memcpy(new_rows, builder->backend.cuda.host_rows,
           builder->backend.cuda.nnz_count * sizeof(matgen_index_t));
    memcpy(new_cols, builder->backend.cuda.host_cols,
           builder->backend.cuda.nnz_count * sizeof(matgen_index_t));
    memcpy(new_vals, builder->backend.cuda.host_vals,
           builder->backend.cuda.nnz_count * sizeof(matgen_value_t));

    // free old pinned memory
    cudaFreeHost(builder->backend.cuda.host_rows);
    cudaFreeHost(builder->backend.cuda.host_cols);
    cudaFreeHost(builder->backend.cuda.host_vals);
  }

  builder->backend.cuda.host_rows = new_rows;
  builder->backend.cuda.host_cols = new_cols;
  builder->backend.cuda.host_vals = new_vals;
  builder->backend.cuda.capacity = new_cap;
  return MATGEN_SUCCESS;
}

// -----------------------------------------------------------------------------
// Public API: create / destroy
// -----------------------------------------------------------------------------

matgen_csr_builder_t* matgen_csr_builder_create_cuda(matgen_index_t rows,
                                                     matgen_index_t cols,
                                                     matgen_size_t est_nnz) {
  matgen_csr_builder_t* builder =
      (matgen_csr_builder_t*)malloc(sizeof(matgen_csr_builder_t));
  if (!builder) return NULL;

  builder->rows = rows;
  builder->cols = cols;
  builder->est_nnz = est_nnz;
  builder->policy = MATGEN_EXEC_AUTO;  // or MATGEN_EXEC_CUDA if defined
  builder->collision_policy = MATGEN_COLLISION_SUM;
  builder->finalized = false;

  // initialize backend.cuda
  builder->backend.cuda.host_rows = NULL;
  builder->backend.cuda.host_cols = NULL;
  builder->backend.cuda.host_vals = NULL;
  builder->backend.cuda.capacity = 0;
  builder->backend.cuda.nnz_count = 0;

  // preallocate estimate or initial capacity
  matgen_size_t initial = est_nnz > 0 ? est_nnz : INITIAL_CAPACITY;
  if (ensure_capacity_cuda(builder, initial) != MATGEN_SUCCESS) {
    free(builder);
    return NULL;
  }

  MATGEN_LOG_DEBUG("Created CSR builder (CUDA) %llu x %llu, est_nnz=%zu",
                   (unsigned long long)rows, (unsigned long long)cols, est_nnz);

  return builder;
}

void matgen_csr_builder_destroy_cuda(matgen_csr_builder_t* builder) {
  if (!builder) return;

  if (builder->backend.cuda.host_rows)
    cudaFreeHost(builder->backend.cuda.host_rows);
  if (builder->backend.cuda.host_cols)
    cudaFreeHost(builder->backend.cuda.host_cols);
  if (builder->backend.cuda.host_vals)
    cudaFreeHost(builder->backend.cuda.host_vals);

  free(builder);
}

// -----------------------------------------------------------------------------
// Add entries (single and batch)
// -----------------------------------------------------------------------------

matgen_error_t matgen_csr_builder_add_cuda(matgen_csr_builder_t* builder,
                                           matgen_index_t row,
                                           matgen_index_t col,
                                           matgen_value_t value) {
  if (!builder || builder->finalized) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }
  if (row >= builder->rows || col >= builder->cols) {
    MATGEN_LOG_ERROR("Index out of bounds (%llu,%llu)", (unsigned long long)row,
                     (unsigned long long)col);
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // append
  matgen_size_t idx = builder->backend.cuda.nnz_count;
  if (ensure_capacity_cuda(builder, idx + 1) != MATGEN_SUCCESS) {
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  builder->backend.cuda.host_rows[idx] = row;
  builder->backend.cuda.host_cols[idx] = col;
  builder->backend.cuda.host_vals[idx] = value;
  builder->backend.cuda.nnz_count++;
  return MATGEN_SUCCESS;
}

matgen_error_t matgen_csr_builder_add_with_policy_cuda(
    matgen_csr_builder_t* builder, matgen_index_t row, matgen_index_t col,
    matgen_value_t value, matgen_collision_policy_t policy) {
  // we just record policy for finalize
  if (!builder) return MATGEN_ERROR_INVALID_ARGUMENT;
  builder->collision_policy = policy;
  return matgen_csr_builder_add_cuda(builder, row, col, value);
}

matgen_error_t matgen_csr_builder_add_batch_cuda(matgen_csr_builder_t* builder,
                                                 matgen_size_t count,
                                                 const matgen_index_t* rows,
                                                 const matgen_index_t* cols,
                                                 const matgen_value_t* vals) {
  if (!builder || builder->finalized) return MATGEN_ERROR_INVALID_ARGUMENT;
  if (count == 0) return MATGEN_SUCCESS;

  matgen_size_t start = builder->backend.cuda.nnz_count;
  if (ensure_capacity_cuda(builder, start + count) != MATGEN_SUCCESS)
    return MATGEN_ERROR_OUT_OF_MEMORY;

  // Bulk copy into pinned host arrays
  memcpy(builder->backend.cuda.host_rows + start, rows,
         count * sizeof(matgen_index_t));
  memcpy(builder->backend.cuda.host_cols + start, cols,
         count * sizeof(matgen_index_t));
  memcpy(builder->backend.cuda.host_vals + start, vals,
         count * sizeof(matgen_value_t));
  builder->backend.cuda.nnz_count += count;
  return MATGEN_SUCCESS;
}

matgen_size_t matgen_csr_builder_get_nnz_cuda(
    const matgen_csr_builder_t* builder) {
  if (!builder) return 0;
  return builder->backend.cuda.nnz_count;
}

// -----------------------------------------------------------------------------
// Finalize: convert appended COO -> CSR using GPU
// -----------------------------------------------------------------------------

matgen_csr_matrix_t* matgen_csr_builder_finalize_cuda(
    matgen_csr_builder_t* builder) {
  if (!builder || builder->finalized) return NULL;
  builder->finalized = true;

  MATGEN_LOG_DEBUG("Finalizing CSR builder (CUDA) entries=%zu",
                   (size_t)builder->backend.cuda.nnz_count);

  size_t nnz = builder->backend.cuda.nnz_count;
  matgen_index_t nrows = builder->rows;
  matgen_index_t ncols = builder->cols;

  // Create CSR matrix (host)
  matgen_csr_matrix_t* csr = matgen_csr_create(nrows, ncols, nnz);
  if (!csr) return NULL;

  if (nnz == 0) {
    MATGEN_LOG_DEBUG("No entries, returning empty CSR");
    return csr;
  }

  try {
    // Copy host pinned buffers to device (wrap in device_vectors)
    thrust::device_vector<matgen_index_t> d_rows(
        builder->backend.cuda.host_rows, builder->backend.cuda.host_rows + nnz);
    thrust::device_vector<matgen_index_t> d_cols(
        builder->backend.cuda.host_cols, builder->backend.cuda.host_cols + nnz);
    thrust::device_vector<matgen_value_t> d_vals(
        builder->backend.cuda.host_vals, builder->backend.cuda.host_vals + nnz);

    // Sort by (row, col) to prepare for reduce_by_key
    auto keys_begin = thrust::make_zip_iterator(
        thrust::make_tuple(d_rows.begin(), d_cols.begin()));
    auto keys_end = keys_begin + nnz;
    thrust::stable_sort_by_key(thrust::device, keys_begin, keys_end,
                               d_vals.begin());

    // Reduce duplicates according to policy: produce uniq_rows, uniq_cols,
    // uniq_vals
    thrust::device_vector<matgen_index_t> uniq_rows(nnz);
    thrust::device_vector<matgen_index_t> uniq_cols(nnz);
    thrust::device_vector<matgen_value_t> uniq_vals(nnz);
    auto uniq_keys_begin = thrust::make_zip_iterator(
        thrust::make_tuple(uniq_rows.begin(), uniq_cols.begin()));

    matgen_collision_policy_t policy = builder->collision_policy;

    // Helper lambda-like: select reducer & run reduce_by_key on pair keys
    if (policy == MATGEN_COLLISION_SUM || policy == MATGEN_COLLISION_AVG) {
      auto end_pair = thrust::reduce_by_key(
          thrust::device, keys_begin, keys_end, d_vals.begin(), uniq_keys_begin,
          uniq_vals.begin(), key_equal(), reducer_sum());
      size_t uniq_n = end_pair.second - uniq_vals.begin();

      // If AVG: need counts per unique key
      if (policy == MATGEN_COLLISION_AVG) {
        // compute counts per unique key (reduce_by_key on keys but with
        // constant 1)
        thrust::device_vector<matgen_value_t> ones(nnz, (matgen_value_t)1);
        thrust::device_vector<matgen_value_t> uniq_counts(nnz);
        auto tmp_keys_begin = thrust::make_zip_iterator(
            thrust::make_tuple(d_rows.begin(), d_cols.begin()));
        auto counts_end = thrust::reduce_by_key(
            thrust::device, tmp_keys_begin, tmp_keys_begin + nnz, ones.begin(),
            thrust::make_zip_iterator(
                thrust::make_tuple(uniq_rows.begin(), uniq_cols.begin())),
            uniq_counts.begin(), key_equal(), reducer_sum());
        size_t counts_n = counts_end.second - uniq_counts.begin();
        if (counts_n != uniq_n) {
          MATGEN_LOG_ERROR("Internal error: counts mismatch in AVG");
          matgen_csr_destroy(csr);
          return NULL;
        }
        // divide sums by counts
        thrust::transform(uniq_vals.begin(), uniq_vals.begin() + uniq_n,
                          uniq_counts.begin(), uniq_vals.begin(),
                          thrust::divides<matgen_value_t>());
      }

      // Now uniq_rows/cols/vals contain unique entries
      // Proceed to build row_ptr and CSR layout below using only first uniq_n
      // entries We'll shorten vectors by resetting end iterators
      uniq_rows.resize(uniq_n);
      uniq_cols.resize(uniq_n);
      uniq_vals.resize(uniq_n);
    } else if (policy == MATGEN_COLLISION_MAX) {
      auto end_pair = thrust::reduce_by_key(
          thrust::device, keys_begin, keys_end, d_vals.begin(), uniq_keys_begin,
          uniq_vals.begin(), key_equal(), reducer_max());
      size_t uniq_n = end_pair.second - uniq_vals.begin();
      uniq_rows.resize(uniq_n);
      uniq_cols.resize(uniq_n);
      uniq_vals.resize(uniq_n);
    } else if (policy == MATGEN_COLLISION_MIN) {
      auto end_pair = thrust::reduce_by_key(
          thrust::device, keys_begin, keys_end, d_vals.begin(), uniq_keys_begin,
          uniq_vals.begin(), key_equal(), reducer_min());
      size_t uniq_n = end_pair.second - uniq_vals.begin();
      uniq_rows.resize(uniq_n);
      uniq_cols.resize(uniq_n);
      uniq_vals.resize(uniq_n);
    } else if (policy == MATGEN_COLLISION_LAST) {
      auto end_pair = thrust::reduce_by_key(
          thrust::device, keys_begin, keys_end, d_vals.begin(), uniq_keys_begin,
          uniq_vals.begin(), key_equal(), reducer_last());
      size_t uniq_n = end_pair.second - uniq_vals.begin();
      uniq_rows.resize(uniq_n);
      uniq_cols.resize(uniq_n);
      uniq_vals.resize(uniq_n);
    } else {
      // default: SUM
      auto end_pair = thrust::reduce_by_key(
          thrust::device, keys_begin, keys_end, d_vals.begin(), uniq_keys_begin,
          uniq_vals.begin(), key_equal(), reducer_sum());
      size_t uniq_n = end_pair.second - uniq_vals.begin();
      uniq_rows.resize(uniq_n);
      uniq_cols.resize(uniq_n);
      uniq_vals.resize(uniq_n);
    }

    size_t uniq_n = uniq_vals.size();

    // Edge: if no unique entries (shouldn't happen), return empty CSR
    if (uniq_n == 0) {
      csr->nnz = 0;
      return csr;
    }

    // Build row_counts: reduce_by_key on uniq_rows (single-key) to get counts
    // per row
    thrust::device_vector<matgen_size_t> d_row_counts((size_t)nrows,
                                                      (matgen_size_t)0);

    // reduce_by_key over uniq_rows with constant 1
    thrust::device_vector<matgen_index_t> uniq_row_ids(uniq_n);
    thrust::device_vector<matgen_size_t> uniq_row_counts(uniq_n);
    auto rows_begin = uniq_rows.begin();
    auto counts_end_pair =
        thrust::reduce_by_key(thrust::device, rows_begin, rows_begin + uniq_n,
                              thrust::make_constant_iterator((matgen_size_t)1),
                              uniq_row_ids.begin(), uniq_row_counts.begin());
    size_t uniq_row_count = counts_end_pair.second - uniq_row_counts.begin();

    // Scatter uniq_row_counts into d_row_counts at indices uniq_row_ids
    thrust::scatter(thrust::device, uniq_row_counts.begin(),
                    uniq_row_counts.begin() + uniq_row_count,
                    uniq_row_ids.begin(), d_row_counts.begin());

    // Exclusive scan to build row_ptr (length nrows+1)
    thrust::device_vector<matgen_size_t> d_row_ptr((size_t)nrows + 1);
    thrust::exclusive_scan(thrust::device, d_row_counts.begin(),
                           d_row_counts.end(), d_row_ptr.begin());
    // Compute total nnz
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

    // Compute intra-row positions using inclusive_scan_by_key on uniq_rows:
    // offsets = inclusive_scan_by_key(uniq_rows, 1, init=0) - 1
    thrust::device_vector<matgen_size_t> offsets(uniq_n);
    // inclusive_scan_by_key with constant 1 to compute 1..k per segment
    thrust::inclusive_scan_by_key(
        thrust::device, uniq_rows.begin(), uniq_rows.begin() + uniq_n,
        thrust::make_constant_iterator((matgen_size_t)1), offsets.begin());
    // convert to 0-based offsets (subtract 1)
    thrust::transform(thrust::device, offsets.begin(), offsets.end(),
                      offsets.begin(), decrement_functor{});

    // Now compute final positions: pos[i] = d_row_ptr[uniq_rows[i]] +
    // offsets[i]
    thrust::device_vector<matgen_size_t> positions(uniq_n);

    // Create counting iterator and transform
    thrust::counting_iterator<size_t> cnt_first(0);
    thrust::transform(
        thrust::device, cnt_first, cnt_first + uniq_n, positions.begin(),
        compute_pos_functor(thrust::raw_pointer_cast(d_row_ptr.data()),
                            thrust::raw_pointer_cast(uniq_rows.data()),
                            thrust::raw_pointer_cast(offsets.data())));

    // Prepare destination arrays (device)
    thrust::device_vector<matgen_index_t> d_dst_cols((size_t)total_nnz);
    thrust::device_vector<matgen_value_t> d_dst_vals((size_t)total_nnz);

    // scatter uniq_cols/uniq_vals into final positions
    thrust::scatter(thrust::device, uniq_cols.begin(),
                    uniq_cols.begin() + uniq_n, positions.begin(),
                    d_dst_cols.begin());
    thrust::scatter(thrust::device, uniq_vals.begin(),
                    uniq_vals.begin() + uniq_n, positions.begin(),
                    d_dst_vals.begin());

    // Copy row_ptr, col_indices and values back to host CSR structure
    CUDA_CHECK_RET_NULL(cudaMemcpy(
        csr->row_ptr, thrust::raw_pointer_cast(d_row_ptr.data()),
        ((size_t)nrows + 1) * sizeof(matgen_size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RET_NULL(cudaMemcpy(
        csr->col_indices, thrust::raw_pointer_cast(d_dst_cols.data()),
        total_nnz * sizeof(matgen_index_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RET_NULL(
        cudaMemcpy(csr->values, thrust::raw_pointer_cast(d_dst_vals.data()),
                   total_nnz * sizeof(matgen_value_t), cudaMemcpyDeviceToHost));

    csr->nnz = total_nnz;

    MATGEN_LOG_DEBUG("CSR builder finalized (CUDA): %llu x %llu, nnz=%zu",
                     (unsigned long long)csr->rows,
                     (unsigned long long)csr->cols, (size_t)csr->nnz);

    return csr;

  } catch (const thrust::system_error& e) {
    MATGEN_LOG_ERROR("Thrust error during finalize: %s", e.what());
    matgen_csr_destroy(csr);
    return NULL;
  }
}
