#include "matgen/core/execution/dispatch.h"
#include "matgen/core/execution/policy.h"
#include "matgen/core/matrix/csr.h"

// Backend-specific headers
#include "backends/seq/internal/csr_seq.h"

#ifdef MATGEN_HAS_OPENMP
#include "backends/omp/internal/csr_omp.h"
#endif

#ifdef MATGEN_HAS_CUDA
#include "backends/cuda/internal/csr_cuda.h"
#endif

// =============================================================================
// Creation and Destruction (Dispatched)
// =============================================================================

matgen_csr_matrix_t* matgen_csr_create_with_policy(
    matgen_index_t rows, matgen_index_t cols, matgen_size_t nnz,
    matgen_exec_policy_t policy) {
  matgen_exec_policy_t resolved = matgen_exec_resolve(policy);
  matgen_dispatch_context_t ctx = matgen_dispatch_create(resolved);

  MATGEN_DISPATCH_BEGIN(ctx, "matgen_csr_create") {
  MATGEN_DISPATCH_CASE_SEQ:
    return matgen_csr_create_seq(rows, cols, nnz);

#ifdef MATGEN_HAS_OPENMP
  MATGEN_DISPATCH_CASE_PAR:
    return matgen_csr_create_omp(rows, cols, nnz);
#endif

#ifdef MATGEN_HAS_CUDA
  MATGEN_DISPATCH_CASE_PAR_UNSEQ:
    return matgen_csr_create_cuda(rows, cols, nnz);
#endif

  MATGEN_DISPATCH_DEFAULT:
    // Fallback to sequential
    return matgen_csr_create_seq(rows, cols, nnz);
  }
  MATGEN_DISPATCH_END();

  // Should never reach here
  return NULL;
}

// Public API uses AUTO policy by default
matgen_csr_matrix_t* matgen_csr_create(matgen_index_t rows, matgen_index_t cols,
                                       matgen_size_t nnz) {
  return matgen_csr_create_with_policy(rows, cols, nnz, MATGEN_EXEC_AUTO);
}
