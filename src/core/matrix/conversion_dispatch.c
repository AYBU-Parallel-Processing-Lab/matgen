#include "matgen/core/execution/dispatch.h"
#include "matgen/core/execution/policy.h"
#include "matgen/core/matrix/conversion.h"
#include "matgen/utils/log.h"

// Backend-specific headers
#include "backends/seq/internal/conversion_seq.h"

#ifdef MATGEN_HAS_OPENMP
#include "backends/omp/internal/conversion_omp.h"
#endif

#ifdef MATGEN_HAS_CUDA
#include "backends/cuda/internal/conversion_cuda.cuh"
#endif

#ifdef MATGEN_HAS_MPI
#include "backends/mpi/internal/conversion_mpi.h"
#endif

// =============================================================================
// COO to CSR Conversion
// =============================================================================

matgen_csr_matrix_t* matgen_coo_to_csr_with_policy(
    const matgen_coo_matrix_t* coo, matgen_exec_policy_t policy) {
  if (!coo) {
    MATGEN_LOG_ERROR("NULL COO matrix pointer");
    return NULL;
  }

  if (!matgen_coo_validate(coo)) {
    MATGEN_LOG_ERROR("Invalid COO matrix");
    return NULL;
  }

  matgen_exec_policy_t resolved = matgen_exec_resolve(policy);
  matgen_dispatch_context_t ctx = matgen_dispatch_create(resolved);

  MATGEN_DISPATCH_BEGIN(ctx, "matgen_coo_to_csr") {
  MATGEN_DISPATCH_CASE_SEQ:
    return matgen_coo_to_csr_seq(coo);

#ifdef MATGEN_HAS_OPENMP
  MATGEN_DISPATCH_CASE_PAR:
    return matgen_coo_to_csr_omp(coo);
#endif

#ifdef MATGEN_HAS_MPI
  MATGEN_DISPATCH_CASE_MPI:
    return matgen_coo_to_csr_mpi(coo);
#endif

#ifdef MATGEN_HAS_CUDA
  MATGEN_DISPATCH_CASE_PAR_UNSEQ:
    return matgen_coo_to_csr_cuda(coo);
#endif

  MATGEN_DISPATCH_DEFAULT:
    return matgen_coo_to_csr_seq(coo);
  }
  MATGEN_DISPATCH_END();

  return NULL;
}

matgen_csr_matrix_t* matgen_coo_to_csr(const matgen_coo_matrix_t* coo) {
  return matgen_coo_to_csr_with_policy(coo, MATGEN_EXEC_AUTO);
}

// =============================================================================
// CSR to COO Conversion
// =============================================================================

matgen_coo_matrix_t* matgen_csr_to_coo_with_policy(
    const matgen_csr_matrix_t* csr, matgen_exec_policy_t policy) {
  if (!csr) {
    MATGEN_LOG_ERROR("NULL CSR matrix pointer");
    return NULL;
  }

  if (!matgen_csr_validate(csr)) {
    MATGEN_LOG_ERROR("Invalid CSR matrix");
    return NULL;
  }

  matgen_exec_policy_t resolved = matgen_exec_resolve(policy);
  matgen_dispatch_context_t ctx = matgen_dispatch_create(resolved);

  MATGEN_DISPATCH_BEGIN(ctx, "matgen_csr_to_coo") {
  MATGEN_DISPATCH_CASE_SEQ:
    return matgen_csr_to_coo_seq(csr);

#ifdef MATGEN_HAS_OPENMP
  MATGEN_DISPATCH_CASE_PAR:
    return matgen_csr_to_coo_omp(csr);
#endif

#ifdef MATGEN_HAS_MPI
  MATGEN_DISPATCH_CASE_MPI:
    return matgen_csr_to_coo_mpi(csr);
#endif

#ifdef MATGEN_HAS_CUDA
  MATGEN_DISPATCH_CASE_PAR_UNSEQ:
    return matgen_csr_to_coo_cuda(csr);
#endif

  MATGEN_DISPATCH_DEFAULT:
    return matgen_csr_to_coo_seq(csr);
  }
  MATGEN_DISPATCH_END();

  return NULL;
}

matgen_coo_matrix_t* matgen_csr_to_coo(const matgen_csr_matrix_t* csr) {
  return matgen_csr_to_coo_with_policy(csr, MATGEN_EXEC_AUTO);
}
