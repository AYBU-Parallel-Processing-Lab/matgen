#ifndef MATGEN_CORE_EXECUTION_DISPATCH_H
#define MATGEN_CORE_EXECUTION_DISPATCH_H

/**
 * @file dispatch.h
 * @brief Algorithm dispatch based on execution policy
 *
 * Provides macros and utilities to dispatch algorithms to the appropriate
 * backend based on execution policy, ensuring type safety and compile-time
 * optimization where possible.
 */

#include "matgen/core/execution/policy.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Dispatch Context
// =============================================================================

/**
 * @brief Dispatch context for runtime backend selection
 *
 * Stores resolved execution policy and runtime parameters for algorithm
 * execution. Created from an execution policy and used by dispatched functions.
 */
typedef struct {
  matgen_exec_policy_t
      resolved_policy; /**< Resolved backend (after fallback) */

  // OpenMP parameters
  int num_threads; /**< Number of threads for OpenMP (0 = default) */

  // CUDA parameters
  int cuda_device_id;  /**< CUDA device ID (-1 = default) */
  int cuda_block_size; /**< CUDA block size (0 = default) */

  // MPI parameters
  void* mpi_comm; /**< MPI communicator (NULL = MPI_COMM_WORLD) */

} matgen_dispatch_context_t;

/**
 * @brief Create a dispatch context from an execution policy
 *
 * Resolves the policy (handling fallbacks) and extracts runtime parameters.
 *
 * @param policy Execution policy (can be any of the policy types)
 * @return Dispatch context ready for use
 */
matgen_dispatch_context_t matgen_dispatch_create(matgen_exec_policy_t policy);

/**
 * @brief Create a dispatch context from a policy union
 *
 * Extracts parameters from the union and creates a dispatch context.
 *
 * @param policy_union Union containing execution policy and parameters
 * @return Dispatch context ready for use
 */
matgen_dispatch_context_t matgen_dispatch_create_from_union(
    const matgen_exec_policy_union_t* policy_union);

/**
 * @brief Log dispatch information for debugging
 *
 * @param ctx Dispatch context
 * @param algorithm_name Name of the algorithm being dispatched
 */
void matgen_dispatch_log(const matgen_dispatch_context_t* ctx,
                         const char* algorithm_name);

// =============================================================================
// Dispatch Macros
// =============================================================================

/**
 * @brief Dispatch algorithm based on execution policy
 *
 * This macro provides a clean switch-based dispatch to different backend
 * implementations. It automatically handles policy resolution and logs
 * the dispatch decision.
 *
 * Example usage:
 * ```c
 * matgen_error_t my_algorithm(const matgen_exec_policy_union_t* policy, ...) {
 *   matgen_dispatch_context_t ctx = matgen_dispatch_create_from_union(policy);
 *
 *   MATGEN_DISPATCH_BEGIN(ctx, "my_algorithm")
 *     MATGEN_DISPATCH_CASE_SEQ:
 *       return my_algorithm_seq(...);
 *     MATGEN_DISPATCH_CASE_PAR:
 *       return my_algorithm_omp(..., ctx.num_threads);
 *     MATGEN_DISPATCH_CASE_PAR_UNSEQ:
 *       return my_algorithm_cuda(..., ctx.cuda_device_id);
 *     MATGEN_DISPATCH_CASE_MPI:
 *       return my_algorithm_mpi(..., ctx.mpi_comm);
 *   MATGEN_DISPATCH_END()
 * }
 * ```
 */
#define MATGEN_DISPATCH_BEGIN(ctx, algo_name) \
  do {                                        \
    matgen_dispatch_log(&(ctx), algo_name);   \
    switch ((ctx).resolved_policy) {
#define MATGEN_DISPATCH_CASE_SEQ case MATGEN_EXEC_SEQ

#define MATGEN_DISPATCH_CASE_PAR case MATGEN_EXEC_PAR

#define MATGEN_DISPATCH_CASE_PAR_UNSEQ case MATGEN_EXEC_PAR_UNSEQ

#define MATGEN_DISPATCH_CASE_MPI case MATGEN_EXEC_MPI

#define MATGEN_DISPATCH_DEFAULT default

#define MATGEN_DISPATCH_END() \
  }                           \
  }                           \
  while (0)

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_CORE_EXECUTION_DISPATCH_H
