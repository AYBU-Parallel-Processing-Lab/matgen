#include "matgen/algorithms/scaling.h"
#include "matgen/core/execution/dispatch.h"
#include "matgen/core/execution/policy.h"
#include "matgen/utils/log.h"

// Include backend-specific headers
#include "backends/seq/internal/nearest_neighbor_seq.h"

#ifdef MATGEN_HAS_OPENMP
#include "backends/omp/internal/nearest_neighbor_omp.h"
#endif

#ifdef MATGEN_HAS_CUDA
#include "backends/cuda/internal/nearest_neighbor_cuda.h"
#endif

matgen_error_t matgen_scale_nearest_neighbor_with_policy(
    matgen_exec_policy_t policy, const matgen_csr_matrix_t* source,
    matgen_index_t new_rows, matgen_index_t new_cols,
    matgen_csr_matrix_t** result) {
  // Use default collision policy (SUM)
  return matgen_scale_nearest_neighbor_with_policy_detailed(
      policy, source, new_rows, new_cols, MATGEN_COLLISION_SUM, result);
}

matgen_error_t matgen_scale_nearest_neighbor_with_policy_detailed(
    matgen_exec_policy_t policy, const matgen_csr_matrix_t* source,
    matgen_index_t new_rows, matgen_index_t new_cols,
    matgen_collision_policy_t collision_policy, matgen_csr_matrix_t** result) {
  // Validate inputs
  if (source == NULL || result == NULL) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Handle auto policy: select based on problem size
  if (policy == MATGEN_EXEC_AUTO) {
    policy = matgen_exec_select_auto(source->nnz, source->rows, source->cols);
    MATGEN_LOG_DEBUG("Auto policy selected: %s",
                     matgen_exec_policy_name(policy));
  }

  // Resolve policy and create dispatch context
  matgen_exec_policy_t resolved = matgen_exec_resolve(policy);
  matgen_dispatch_context_t ctx = matgen_dispatch_create(resolved);

  // Dispatch to appropriate backend
  MATGEN_DISPATCH_BEGIN(ctx, "nearest_neighbor_scale") {
  MATGEN_DISPATCH_CASE_SEQ:
    return matgen_scale_nearest_neighbor_seq(source, new_rows, new_cols,
                                             collision_policy, result);

#ifdef MATGEN_HAS_OPENMP
  MATGEN_DISPATCH_CASE_PAR:
    return matgen_scale_nearest_neighbor_omp(source, new_rows, new_cols,
                                             collision_policy, result);
#endif

#ifdef MATGEN_HAS_CUDA
  MATGEN_DISPATCH_CASE_PAR_UNSEQ:
    return matgen_scale_nearest_neighbor_cuda(source, new_rows, new_cols,
                                              collision_policy, result);
#endif

  MATGEN_DISPATCH_DEFAULT:
    // Fallback to sequential
    return matgen_scale_nearest_neighbor_seq(source, new_rows, new_cols,
                                             collision_policy, result);
  }
  MATGEN_DISPATCH_END();

  return MATGEN_ERROR_UNKNOWN;
}
