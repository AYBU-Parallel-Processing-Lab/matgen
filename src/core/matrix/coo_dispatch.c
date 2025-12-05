#include "matgen/core/execution/dispatch.h"
#include "matgen/core/execution/policy.h"
#include "matgen/core/matrix/coo.h"
#include "matgen/utils/log.h"

// Backend-specific headers
#include "backends/seq/internal/coo_seq.h"

#ifdef MATGEN_HAS_OPENMP
#include "backends/omp/internal/coo_omp.h"
#endif

#ifdef MATGEN_HAS_CUDA
#include "backends/cuda/internal/coo_cuda.h"
#endif

// =============================================================================
// Creation and Destruction
// =============================================================================

matgen_coo_matrix_t* matgen_coo_create(matgen_index_t rows, matgen_index_t cols,
                                       matgen_size_t nnz_hint) {
  // For creation, use AUTO policy (default to best available backend)
  matgen_exec_policy_t policy = matgen_exec_resolve(MATGEN_EXEC_AUTO);
  matgen_dispatch_context_t ctx = matgen_dispatch_create(policy);

  MATGEN_DISPATCH_BEGIN(ctx, "matgen_coo_create") {
  MATGEN_DISPATCH_CASE_SEQ:
    return matgen_coo_create_seq(rows, cols, nnz_hint);

#ifdef MATGEN_HAS_OPENMP
  MATGEN_DISPATCH_CASE_PAR:
    return matgen_coo_create_omp(rows, cols, nnz_hint);
#endif

#ifdef MATGEN_HAS_CUDA
  MATGEN_DISPATCH_CASE_PAR_UNSEQ:
    return matgen_coo_create_cuda(rows, cols, nnz_hint);
#endif

  MATGEN_DISPATCH_DEFAULT:
    // Fallback to sequential
    return matgen_coo_create_seq(rows, cols, nnz_hint);
  }
  MATGEN_DISPATCH_END();

  // Should never reach here
  return NULL;
}

// =============================================================================
// COO Operations with Policy
// =============================================================================

matgen_error_t matgen_coo_sort_with_policy(matgen_coo_matrix_t* matrix,
                                           matgen_exec_policy_t policy) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (matrix->is_sorted) {
    MATGEN_LOG_DEBUG("Matrix already sorted, skipping sort");
    return MATGEN_SUCCESS;
  }

  matgen_exec_policy_t resolved = matgen_exec_resolve(policy);
  matgen_dispatch_context_t ctx = matgen_dispatch_create(resolved);

  MATGEN_DISPATCH_BEGIN(ctx, "matgen_coo_sort") {
  MATGEN_DISPATCH_CASE_SEQ:
    return matgen_coo_sort_seq(matrix);

#ifdef MATGEN_HAS_OPENMP
  MATGEN_DISPATCH_CASE_PAR:
    return matgen_coo_sort_omp(matrix);
#endif

#ifdef MATGEN_HAS_CUDA
  MATGEN_DISPATCH_CASE_PAR_UNSEQ:
    return matgen_coo_sort_cuda(matrix);
#endif

  MATGEN_DISPATCH_DEFAULT:
    return matgen_coo_sort_seq(matrix);
  }
  MATGEN_DISPATCH_END();

  return MATGEN_ERROR_UNKNOWN;
}

matgen_error_t matgen_coo_sum_duplicates_with_policy(
    matgen_coo_matrix_t* matrix, matgen_exec_policy_t policy) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (!matrix->is_sorted) {
    MATGEN_LOG_ERROR("Matrix must be sorted before sum_duplicates");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  matgen_exec_policy_t resolved = matgen_exec_resolve(policy);
  matgen_dispatch_context_t ctx = matgen_dispatch_create(resolved);

  MATGEN_DISPATCH_BEGIN(ctx, "matgen_coo_sum_duplicates") {
  MATGEN_DISPATCH_CASE_SEQ:
    return matgen_coo_sum_duplicates_seq(matrix);

#ifdef MATGEN_HAS_OPENMP
  MATGEN_DISPATCH_CASE_PAR:
    return matgen_coo_sum_duplicates_omp(matrix);
#endif

#ifdef MATGEN_HAS_CUDA
  MATGEN_DISPATCH_CASE_PAR_UNSEQ:
    return matgen_coo_sum_duplicates_cuda(matrix);
#endif

  MATGEN_DISPATCH_DEFAULT:
    return matgen_coo_sum_duplicates_seq(matrix);
  }
  MATGEN_DISPATCH_END();

  return MATGEN_ERROR_UNKNOWN;
}

matgen_error_t matgen_coo_merge_duplicates_with_policy(
    matgen_coo_matrix_t* matrix, matgen_collision_policy_t collision_policy,
    matgen_exec_policy_t exec_policy) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (!matrix->is_sorted) {
    MATGEN_LOG_ERROR("Matrix must be sorted before merge_duplicates");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  matgen_exec_policy_t resolved = matgen_exec_resolve(exec_policy);
  matgen_dispatch_context_t ctx = matgen_dispatch_create(resolved);

  MATGEN_DISPATCH_BEGIN(ctx, "matgen_coo_merge_duplicates") {
  MATGEN_DISPATCH_CASE_SEQ:
    return matgen_coo_merge_duplicates_seq(matrix, collision_policy);

#ifdef MATGEN_HAS_OPENMP
  MATGEN_DISPATCH_CASE_PAR:
    return matgen_coo_merge_duplicates_omp(matrix, collision_policy);
#endif

#ifdef MATGEN_HAS_CUDA
  MATGEN_DISPATCH_CASE_PAR_UNSEQ:
    return matgen_coo_merge_duplicates_cuda(matrix, collision_policy);
#endif

  MATGEN_DISPATCH_DEFAULT:
    return matgen_coo_merge_duplicates_seq(matrix, collision_policy);
  }
  MATGEN_DISPATCH_END();

  return MATGEN_ERROR_UNKNOWN;
}

// =============================================================================
// Public API Wrappers (use sequential by default for compatibility)
// =============================================================================

matgen_error_t matgen_coo_sort(matgen_coo_matrix_t* matrix) {
  return matgen_coo_sort_with_policy(matrix, MATGEN_EXEC_AUTO);
}

matgen_error_t matgen_coo_sum_duplicates(matgen_coo_matrix_t* matrix) {
  return matgen_coo_sum_duplicates_with_policy(matrix, MATGEN_EXEC_AUTO);
}

matgen_error_t matgen_coo_merge_duplicates(matgen_coo_matrix_t* matrix,
                                           matgen_collision_policy_t policy) {
  return matgen_coo_merge_duplicates_with_policy(matrix, policy,
                                                 MATGEN_EXEC_AUTO);
}
