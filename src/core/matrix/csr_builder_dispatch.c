#include "backends/seq/internal/csr_builder_seq.h"
#include "core/matrix/csr_builder_internal.h"  // IWYU pragma: keep
#include "matgen/core/execution/policy.h"
#include "matgen/core/matrix/csr.h"
#include "matgen/core/matrix/csr_builder.h"

#ifdef MATGEN_HAS_OPENMP
#include "backends/omp/internal/csr_builder_omp.h"
#endif

#ifdef MATGEN_HAS_CUDA
#include "backends/cuda/internal/csr_builder_cuda.cuh"
#endif

#ifdef MATGEN_HAS_MPI
#include "backends/mpi/internal/csr_builder_mpi.h"
#endif

// =============================================================================
// CSR Builder Creation
// =============================================================================

matgen_csr_builder_t* matgen_csr_builder_create(matgen_index_t rows,
                                                matgen_index_t cols,
                                                matgen_size_t est_nnz) {
  // Default to best available backend
  matgen_exec_policy_t policy = matgen_exec_resolve(MATGEN_EXEC_AUTO);
  return matgen_csr_builder_create_with_policy(rows, cols, est_nnz, policy);
}

matgen_csr_builder_t* matgen_csr_builder_create_with_policy(
    matgen_index_t rows, matgen_index_t cols, matgen_size_t est_nnz,
    matgen_exec_policy_t policy) {
  matgen_exec_policy_t resolved = matgen_exec_resolve(policy);

  switch (resolved) {
#ifdef MATGEN_HAS_OPENMP
    case MATGEN_EXEC_PAR:
      return matgen_csr_builder_create_omp(rows, cols, est_nnz);
#endif

#ifdef MATGEN_HAS_MPI
    case MATGEN_EXEC_MPI:
      return matgen_csr_builder_create_mpi(rows, cols, est_nnz);
#endif

#ifdef MATGEN_HAS_CUDA
    case MATGEN_EXEC_PAR_UNSEQ:
      return matgen_csr_builder_create_cuda(rows, cols, est_nnz);
#endif

    case MATGEN_EXEC_SEQ:
    default:
      return matgen_csr_builder_create_seq(rows, cols, est_nnz);
  }
}

void matgen_csr_builder_destroy(matgen_csr_builder_t* builder) {
  if (!builder) {
    return;
  }

  // Dispatch to backend-specific destroy
  switch (builder->policy) {
#ifdef MATGEN_HAS_OPENMP
    case MATGEN_EXEC_PAR:
      matgen_csr_builder_destroy_omp(builder);
      break;
#endif

#ifdef MATGEN_HAS_MPI
    case MATGEN_EXEC_MPI:
      matgen_csr_builder_destroy_mpi(builder);
      break;
#endif

#ifdef MATGEN_HAS_CUDA
    case MATGEN_EXEC_PAR_UNSEQ:
      matgen_csr_builder_destroy_cuda(builder);
      break;
#endif

    case MATGEN_EXEC_SEQ:
    default:
      matgen_csr_builder_destroy_seq(builder);
      break;
  }
}

// =============================================================================
// Entry Addition
// =============================================================================

matgen_error_t matgen_csr_builder_add(matgen_csr_builder_t* builder,
                                      matgen_index_t row, matgen_index_t col,
                                      matgen_value_t value) {
  if (!builder) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Dispatch based on builder's policy
  switch (builder->policy) {
#ifdef MATGEN_HAS_OPENMP
    case MATGEN_EXEC_PAR:
      return matgen_csr_builder_add_omp(builder, row, col, value);
#endif

#ifdef MATGEN_HAS_MPI
    case MATGEN_EXEC_MPI:
      return matgen_csr_builder_add_mpi(builder, row, col, value);
#endif

#ifdef MATGEN_HAS_CUDA
    case MATGEN_EXEC_PAR_UNSEQ:
      return matgen_csr_builder_add_cuda(builder, row, col, value);
#endif

    case MATGEN_EXEC_SEQ:
    default:
      return matgen_csr_builder_add_seq(builder, row, col, value);
  }
}

matgen_error_t matgen_csr_builder_add_with_policy(
    matgen_csr_builder_t* builder, matgen_index_t row, matgen_index_t col,
    matgen_value_t value, matgen_collision_policy_t policy) {
  // For now, only SUM is supported
  // TODO: Implement other collision policies
  (void)policy;
  return matgen_csr_builder_add(builder, row, col, value);
}

matgen_error_t matgen_csr_builder_add_batch(matgen_csr_builder_t* builder,
                                            matgen_size_t count,
                                            const matgen_index_t* rows,
                                            const matgen_index_t* cols,
                                            const matgen_value_t* values) {
  if (!builder || !rows || !cols || !values) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Simple implementation: call add for each entry
  // TODO: Optimize with batch insertion
  for (matgen_size_t i = 0; i < count; i++) {
    matgen_error_t err =
        matgen_csr_builder_add(builder, rows[i], cols[i], values[i]);
    if (err != MATGEN_SUCCESS) {
      return err;
    }
  }

  return MATGEN_SUCCESS;
}

// =============================================================================
// Finalization
// =============================================================================

matgen_csr_matrix_t* matgen_csr_builder_finalize(
    matgen_csr_builder_t* builder) {
  if (!builder) {
    return NULL;
  }

  switch (builder->policy) {
#ifdef MATGEN_HAS_OPENMP
    case MATGEN_EXEC_PAR:
      return matgen_csr_builder_finalize_omp(builder);
#endif

#ifdef MATGEN_HAS_MPI
    case MATGEN_EXEC_MPI:
      return matgen_csr_builder_finalize_mpi(builder);
#endif

#ifdef MATGEN_HAS_CUDA
    case MATGEN_EXEC_PAR_UNSEQ:
      return matgen_csr_builder_finalize_cuda(builder);
#endif

    case MATGEN_EXEC_SEQ:
    default:
      return matgen_csr_builder_finalize_seq(builder);
  }
}

// =============================================================================
// Query Functions
// =============================================================================

matgen_size_t matgen_csr_builder_get_nnz(const matgen_csr_builder_t* builder) {
  if (!builder) {
    return 0;
  }

  switch (builder->policy) {
#ifdef MATGEN_HAS_OPENMP
    case MATGEN_EXEC_PAR:
      return matgen_csr_builder_get_nnz_omp(builder);
#endif

#ifdef MATGEN_HAS_MPI
    case MATGEN_EXEC_MPI:
      return matgen_csr_builder_get_nnz_mpi(builder);
#endif

#ifdef MATGEN_HAS_CUDA
    case MATGEN_EXEC_PAR_UNSEQ:
      return matgen_csr_builder_get_nnz_cuda(builder);
#endif

    case MATGEN_EXEC_SEQ:
    default:
      return matgen_csr_builder_get_nnz_seq(builder);
  }
}

void matgen_csr_builder_get_dims(const matgen_csr_builder_t* builder,
                                 matgen_index_t* rows, matgen_index_t* cols) {
  if (!builder) {
    if (rows) {
      *rows = 0;
    }

    if (cols) {
      *cols = 0;
    }

    return;
  }

  // Both backends have rows/cols in the same position
  if (rows) {
    *rows = builder->rows;
  }

  if (cols) {
    *cols = *cols = builder->cols;
  }
}
