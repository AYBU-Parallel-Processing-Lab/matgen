#include "core/matrix/csr_builder_internal.h"  // IWYU pragma: keep
#include "matgen/core/execution/policy.h"
#include "matgen/core/matrix/csr.h"
#include "matgen/core/matrix/csr_builder.h"

// Forward declarations for CSR matrix backend-specific functions
#ifdef MATGEN_HAS_OPENMP
extern matgen_csr_matrix_t* matgen_csr_create_omp(matgen_index_t rows,
                                                  matgen_index_t cols,
                                                  matgen_size_t nnz);
#endif

extern matgen_csr_matrix_t* matgen_csr_create_seq(matgen_index_t rows,
                                                  matgen_index_t cols,
                                                  matgen_size_t nnz);

// Forward declarations for CSR builder backend-specific functions
#ifdef MATGEN_HAS_OPENMP
extern matgen_csr_builder_t* matgen_csr_builder_create_omp(
    matgen_index_t rows, matgen_index_t cols, matgen_size_t est_nnz);
extern void matgen_csr_builder_destroy_omp(matgen_csr_builder_t* builder);
extern matgen_error_t matgen_csr_builder_add_omp(matgen_csr_builder_t* builder,
                                                 matgen_index_t row,
                                                 matgen_index_t col,
                                                 matgen_value_t value);
extern matgen_csr_matrix_t* matgen_csr_builder_finalize_omp(
    matgen_csr_builder_t* builder);
extern matgen_size_t matgen_csr_builder_get_nnz_omp(
    const matgen_csr_builder_t* builder);
#endif

extern matgen_csr_builder_t* matgen_csr_builder_create_seq(
    matgen_index_t rows, matgen_index_t cols, matgen_size_t est_nnz);
extern void matgen_csr_builder_destroy_seq(matgen_csr_builder_t* builder);
extern matgen_error_t matgen_csr_builder_add_seq(matgen_csr_builder_t* builder,
                                                 matgen_index_t row,
                                                 matgen_index_t col,
                                                 matgen_value_t value);
extern matgen_csr_matrix_t* matgen_csr_builder_finalize_seq(
    matgen_csr_builder_t* builder);
extern matgen_size_t matgen_csr_builder_get_nnz_seq(
    const matgen_csr_builder_t* builder);

// =============================================================================
// CSR Matrix Creation
// =============================================================================

matgen_csr_matrix_t* matgen_csr_create(matgen_index_t rows, matgen_index_t cols,
                                       matgen_size_t nnz) {
  // Default to best available backend
  matgen_exec_policy_t policy = matgen_exec_resolve(MATGEN_EXEC_AUTO);
  return matgen_csr_create_with_policy(rows, cols, nnz, policy);
}

matgen_csr_matrix_t* matgen_csr_create_with_policy(
    matgen_index_t rows, matgen_index_t cols, matgen_size_t nnz,
    matgen_exec_policy_t policy) {
  matgen_exec_policy_t resolved = matgen_exec_resolve(policy);

  switch (resolved) {
#ifdef MATGEN_HAS_OPENMP
    case MATGEN_EXEC_PAR:
      return matgen_csr_create_omp(rows, cols, nnz);
#endif

    case MATGEN_EXEC_SEQ:
    default:
      return matgen_csr_create_seq(rows, cols, nnz);
  }
}

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
