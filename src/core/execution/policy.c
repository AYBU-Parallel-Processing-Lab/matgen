#include "matgen/core/execution/policy.h"

#include "matgen/utils/log.h"

#ifdef MATGEN_HAS_OPENMP
#include <omp.h>
#endif

#ifdef MATGEN_HAS_CUDA
#include <cuda_runtime.h>
#endif

#ifdef MATGEN_HAS_MPI
#include <mpi.h>
#endif

// =============================================================================
// Predefined Execution Policy Instances
// =============================================================================

const matgen_exec_seq_t matgen_exec_seq = {.policy = MATGEN_EXEC_SEQ};

const matgen_exec_par_t matgen_exec_par = {.policy = MATGEN_EXEC_PAR,
                                           .num_threads = 0};

const matgen_exec_par_unseq_t matgen_exec_par_unseq = {
    .policy = MATGEN_EXEC_PAR_UNSEQ, .device_id = -1, .block_size = 0};

const matgen_exec_mpi_t matgen_exec_mpi = {.policy = MATGEN_EXEC_MPI,
                                           .mpi_comm = NULL};

// =============================================================================
// Execution Policy Utilities
// =============================================================================

bool matgen_exec_is_available(matgen_exec_policy_t policy) {
  switch (policy) {
    case MATGEN_EXEC_SEQ:  // NOLINT
      return true;         // Always available

    case MATGEN_EXEC_PAR:  // NOLINT
#ifdef MATGEN_HAS_OPENMP
      return true;
#else
      return false;
#endif

    case MATGEN_EXEC_PAR_UNSEQ:  // NOLINT
#ifdef MATGEN_HAS_CUDA
    {
      int device_count = 0;
      cudaError_t err = cudaGetDeviceCount(&device_count);
      return (err == cudaSuccess && device_count > 0);
    }
#else
      return false;
#endif

    case MATGEN_EXEC_MPI:  // NOLINT
#ifdef MATGEN_HAS_MPI
    {
      int initialized = 0;
      MPI_Initialized(&initialized);
      return initialized != 0;
    }
#else
      return false;
#endif

    case MATGEN_EXEC_AUTO:  // NOLINT
      return true;  // Auto always "available" (will resolve to something)

    default:
      return false;
  }
}

const char* matgen_exec_policy_name(matgen_exec_policy_t policy) {
  switch (policy) {
    case MATGEN_EXEC_SEQ:
      return "Sequential";
    case MATGEN_EXEC_PAR:
      return "OpenMP";
    case MATGEN_EXEC_PAR_UNSEQ:
      return "CUDA";
    case MATGEN_EXEC_MPI:
      return "MPI";
    case MATGEN_EXEC_AUTO:
      return "Auto";
    default:
      return "Unknown";
  }
}

matgen_exec_policy_t matgen_exec_select_auto(matgen_size_t nnz,
                                             matgen_index_t rows,
                                             matgen_index_t cols) {
  // Suppress unused parameter warnings
  MATGEN_UNUSED(rows);
  MATGEN_UNUSED(cols);

  // Thresholds for automatic selection
  const matgen_size_t SMALL_PROBLEM_THRESHOLD = 1000000;    // 1M elements
  const matgen_size_t LARGE_PROBLEM_THRESHOLD = 100000000;  // 100M elements

  // Small problems: sequential is often faster due to overhead
  if (nnz < SMALL_PROBLEM_THRESHOLD) {
    return MATGEN_EXEC_SEQ;
  }

  // Very large problems: prefer GPU if available
  if (nnz >= LARGE_PROBLEM_THRESHOLD) {
    if (matgen_exec_is_available(MATGEN_EXEC_PAR_UNSEQ)) {
      return MATGEN_EXEC_PAR_UNSEQ;
    }
  }

  // Medium to large problems: prefer OpenMP
  if (matgen_exec_is_available(MATGEN_EXEC_PAR)) {
    return MATGEN_EXEC_PAR;
  }

  // Fallback to sequential
  return MATGEN_EXEC_SEQ;
}

matgen_exec_policy_t matgen_exec_resolve(matgen_exec_policy_t policy) {
  // Auto policy: select based on heuristics (needs problem size)
  if (policy == MATGEN_EXEC_AUTO) {
    // Without problem size info, prefer the fastest available backend
    if (matgen_exec_is_available(MATGEN_EXEC_PAR_UNSEQ)) {
      return MATGEN_EXEC_PAR_UNSEQ;
    }

    if (matgen_exec_is_available(MATGEN_EXEC_PAR)) {
      return MATGEN_EXEC_PAR;
    }

    return MATGEN_EXEC_SEQ;
  }

  // If requested policy is available, use it
  if (matgen_exec_is_available(policy)) {
    return policy;
  }

  // Fallback logic
  switch (policy) {
    case MATGEN_EXEC_PAR_UNSEQ:
      // CUDA not available, try OpenMP
      if (matgen_exec_is_available(MATGEN_EXEC_PAR)) {
        MATGEN_LOG_WARN("CUDA not available, falling back to OpenMP execution");
        return MATGEN_EXEC_PAR;
      }
      // OpenMP not available either, fall through to sequential
      MATGEN_LOG_WARN(
          "CUDA and OpenMP not available, falling back to "
          "sequential execution");
      return MATGEN_EXEC_SEQ;

    case MATGEN_EXEC_PAR:
      // OpenMP not available, fall back to sequential
      MATGEN_LOG_WARN(
          "OpenMP not available, falling back to sequential "
          "execution");
      return MATGEN_EXEC_SEQ;

    case MATGEN_EXEC_MPI:
      // MPI not available, fall back to sequential
      // (could fall back to OpenMP, but MPI is typically a deliberate choice)
      MATGEN_LOG_WARN(
          "MPI not available or not initialized, falling back to "
          "sequential execution");
      return MATGEN_EXEC_SEQ;

    default:
      return MATGEN_EXEC_SEQ;
  }
}

matgen_exec_par_t matgen_exec_par_with_threads(int num_threads) {
  matgen_exec_par_t policy = {.policy = MATGEN_EXEC_PAR,
                              .num_threads = num_threads};
  return policy;
}

matgen_exec_par_unseq_t matgen_exec_par_unseq_with_params(int device_id,
                                                          int block_size) {
  matgen_exec_par_unseq_t policy = {.policy = MATGEN_EXEC_PAR_UNSEQ,
                                    .device_id = device_id,
                                    .block_size = block_size};
  return policy;
}

matgen_exec_mpi_t matgen_exec_mpi_with_comm(void* mpi_comm) {
  matgen_exec_mpi_t policy = {.policy = MATGEN_EXEC_MPI, .mpi_comm = mpi_comm};
  return policy;
}

// =============================================================================
// Backend Capability Queries
// =============================================================================

int matgen_exec_get_num_threads(void) {
#ifdef MATGEN_HAS_OPENMP
  return omp_get_max_threads();
#else
  return 0;
#endif
}

int matgen_exec_get_num_cuda_devices(void) {
#ifdef MATGEN_HAS_CUDA
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    return 0;
  }
  return device_count;
#else
  return 0;
#endif
}

int matgen_exec_get_mpi_size(void) {
#ifdef MATGEN_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    return 1;
  }

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
#else
  return 1;
#endif
}

int matgen_exec_get_mpi_rank(void) {
#ifdef MATGEN_HAS_MPI
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    return 0;
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
#else
  return 0;
#endif
}

bool matgen_exec_is_mpi_root(void) { return matgen_exec_get_mpi_rank() == 0; }
