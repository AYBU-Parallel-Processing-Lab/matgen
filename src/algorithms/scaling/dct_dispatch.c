#include "matgen/algorithms/scaling.h"
#include "matgen/core/execution/dispatch.h"
#include "matgen/core/execution/policy.h"
#include "matgen/utils/log.h"

#include "backends/seq/internal/dct_seq.h"

#ifdef MATGEN_HAS_OPENMP
#include "backends/omp/internal/dct_omp.h"
#endif

#ifdef MATGEN_HAS_CUDA
#include "backends/cuda/internal/dct_cuda.cuh"
#endif

matgen_error_t matgen_scale_dct_with_policy(
    matgen_exec_policy_t policy,
    const matgen_csr_matrix_t* source,
    matgen_index_t new_rows,
    matgen_index_t new_cols,
    matgen_csr_matrix_t** result) {
    if (!source || !result || new_rows == 0 || new_cols == 0) return MATGEN_ERROR_INVALID_ARGUMENT;

    if (policy == MATGEN_EXEC_AUTO) {
        policy = matgen_exec_select_auto(source->nnz, source->rows, source->cols);
    }

    matgen_exec_policy_t resolved = matgen_exec_resolve(policy);
    matgen_dispatch_context_t ctx = matgen_dispatch_create(resolved);

    MATGEN_DISPATCH_BEGIN(ctx, "dct_scale") {
    MATGEN_DISPATCH_CASE_SEQ:
        return matgen_scale_dct_seq(source, new_rows, new_cols, result);

#ifdef MATGEN_HAS_OPENMP
    MATGEN_DISPATCH_CASE_PAR:
        return matgen_scale_dct_omp(source, new_rows, new_cols, result);
#endif

#ifdef MATGEN_HAS_CUDA
    MATGEN_DISPATCH_CASE_PAR_UNSEQ:
        return matgen_scale_dct_cuda(source, new_rows, new_cols, result);
#endif

    MATGEN_DISPATCH_DEFAULT:
        return matgen_scale_dct_seq(source, new_rows, new_cols, result);
    }
    MATGEN_DISPATCH_END();

    return MATGEN_ERROR_UNKNOWN;
}
