#include "matgen/core/execution/dispatch.h"

#include "matgen/utils/log.h"

matgen_dispatch_context_t matgen_dispatch_create(matgen_exec_policy_t policy) {
  matgen_dispatch_context_t ctx;

  // Resolve the policy (handle fallbacks)
  ctx.resolved_policy = matgen_exec_resolve(policy);

  // Set default parameters
  ctx.num_threads = 0;      // Use OpenMP default
  ctx.cuda_device_id = -1;  // Use CUDA default
  ctx.cuda_block_size = 0;  // Use default block size (256)
  ctx.mpi_comm = NULL;      // Use MPI_COMM_WORLD

  return ctx;
}

matgen_dispatch_context_t matgen_dispatch_create_from_union(
    const matgen_exec_policy_union_t* policy_union) {
  matgen_dispatch_context_t ctx;

  if (policy_union == NULL) {
    // Default to sequential
    return matgen_dispatch_create(MATGEN_EXEC_SEQ);
  }

  // Resolve the base policy
  ctx.resolved_policy = matgen_exec_resolve(policy_union->base);

  // Extract parameters based on policy type
  switch (policy_union->base) {
    case MATGEN_EXEC_SEQ:
      ctx.num_threads = 0;
      ctx.cuda_device_id = -1;
      ctx.cuda_block_size = 0;
      ctx.mpi_comm = NULL;
      break;

    case MATGEN_EXEC_PAR:
      ctx.num_threads = policy_union->par.num_threads;
      ctx.cuda_device_id = -1;
      ctx.cuda_block_size = 0;
      ctx.mpi_comm = NULL;
      break;

    case MATGEN_EXEC_PAR_UNSEQ:
      ctx.num_threads = 0;
      ctx.cuda_device_id = policy_union->par_unseq.device_id;
      ctx.cuda_block_size = policy_union->par_unseq.block_size;
      ctx.mpi_comm = NULL;
      break;

    case MATGEN_EXEC_MPI:
      ctx.num_threads = 0;
      ctx.cuda_device_id = -1;
      ctx.cuda_block_size = 0;
      ctx.mpi_comm = policy_union->mpi.mpi_comm;
      break;

    case MATGEN_EXEC_AUTO:
    default:
      // Unknown policy, use defaults
      ctx.num_threads = 0;
      ctx.cuda_device_id = -1;
      ctx.cuda_block_size = 0;
      ctx.mpi_comm = NULL;
      break;
  }

  return ctx;
}

void matgen_dispatch_log(const matgen_dispatch_context_t* ctx,
                         const char* algorithm_name) {
  if (ctx == NULL || algorithm_name == NULL) {
    return;
  }

  const char* policy_name = matgen_exec_policy_name(ctx->resolved_policy);

  switch (ctx->resolved_policy) {
    case MATGEN_EXEC_SEQ:
      MATGEN_LOG_DEBUG("Dispatching %s to Sequential backend", algorithm_name);
      break;

    case MATGEN_EXEC_PAR:
      if (ctx->num_threads > 0) {
        MATGEN_LOG_DEBUG("Dispatching %s to OpenMP backend (%d threads)",
                         algorithm_name, ctx->num_threads);
      } else {
        MATGEN_LOG_DEBUG("Dispatching %s to OpenMP backend (default threads)",
                         algorithm_name);
      }
      break;

    case MATGEN_EXEC_PAR_UNSEQ:
      if (ctx->cuda_device_id >= 0) {
        MATGEN_LOG_DEBUG(
            "Dispatching %s to CUDA backend (device %d, block size %d)",
            algorithm_name, ctx->cuda_device_id,
            ctx->cuda_block_size > 0 ? ctx->cuda_block_size : 256);
      } else {
        MATGEN_LOG_DEBUG("Dispatching %s to CUDA backend (default device)",
                         algorithm_name);
      }
      break;

    case MATGEN_EXEC_MPI:
      MATGEN_LOG_DEBUG("Dispatching %s to MPI backend", algorithm_name);
      break;

    default:
      MATGEN_LOG_DEBUG("Dispatching %s to %s backend", algorithm_name,
                       policy_name);
      break;
  }
}
