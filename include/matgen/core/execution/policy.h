#ifndef MATGEN_CORE_EXECUTION_POLICY_H
#define MATGEN_CORE_EXECUTION_POLICY_H

/**
 * @file execution_policy.h
 * @brief Execution policy for backend dispatch (Sequential, OpenMP, CUDA, MPI)
 *
 * Provides a type-safe way to select execution backends at compile-time or
 * runtime, ensuring parallel pipelines follow parallel paths and sequential
 * pipelines follow sequential paths.
 */

#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Execution Policy Types
// =============================================================================

/**
 * @brief Execution policy enumeration
 *
 * Defines the available execution backends. Use these to dispatch algorithms
 * to the appropriate implementation.
 */
typedef enum {
  /**
   * @brief Sequential execution policy
   *
   * Single-threaded CPU execution. Guaranteed to be available on all platforms.
   * Use for small datasets or when deterministic ordering is required.
   */
  MATGEN_EXEC_SEQ = 0,

  /**
   * @brief Parallel execution policy (OpenMP)
   *
   * Multi-threaded CPU execution using OpenMP. Available when compiled with
   * MATGEN_HAS_OPENMP. Falls back to sequential if OpenMP is not available.
   * Use for medium to large datasets on shared-memory systems.
   */
  MATGEN_EXEC_PAR = 1,

  /**
   * @brief Parallel unsequenced execution policy (CUDA)
   *
   * GPU execution using CUDA. Available when compiled with MATGEN_HAS_CUDA.
   * Falls back to parallel (OpenMP) or sequential if CUDA is not available.
   * Use for very large datasets with suitable GPU kernels.
   */
  MATGEN_EXEC_PAR_UNSEQ = 2,

  /**
   * @brief Distributed execution policy (MPI)
   *
   * Distributed-memory parallel execution using MPI. Available when compiled
   * with MATGEN_HAS_MPI. Falls back to sequential if MPI is not available.
   * Use for extremely large datasets across multiple nodes.
   */
  MATGEN_EXEC_MPI = 3,

  /**
   * @brief Automatic execution policy selection
   *
   * Automatically selects the best available backend based on:
   * - Problem size
   * - Available hardware (GPU, number of CPU cores)
   * - Compiled backends
   *
   * The selection logic prefers: CUDA > OpenMP > Sequential for local
   * execution, and MPI for distributed execution.
   */
  MATGEN_EXEC_AUTO = 4

} matgen_exec_policy_t;

// =============================================================================
// Execution Policy Tags (Type-Safe Compile-Time Dispatch)
// =============================================================================

/**
 * @brief Type tag for sequential execution
 */
typedef struct {
  matgen_exec_policy_t policy;
} matgen_exec_seq_t;

/**
 * @brief Type tag for parallel execution (OpenMP)
 */
typedef struct {
  matgen_exec_policy_t policy;
  int num_threads; /**< Number of threads (0 = use default) */
} matgen_exec_par_t;

/**
 * @brief Type tag for parallel unsequenced execution (CUDA)
 */
typedef struct {
  matgen_exec_policy_t policy;
  int device_id;  /**< CUDA device ID (-1 = use default) */
  int block_size; /**< Thread block size (0 = use default) */
} matgen_exec_par_unseq_t;

/**
 * @brief Type tag for MPI execution
 */
typedef struct {
  matgen_exec_policy_t policy;
  void* mpi_comm; /**< MPI communicator (MPI_Comm*), NULL = MPI_COMM_WORLD */
} matgen_exec_mpi_t;

/**
 * @brief Union of all execution policy types
 */
typedef union {
  matgen_exec_policy_t base;
  matgen_exec_seq_t seq;
  matgen_exec_par_t par;
  matgen_exec_par_unseq_t par_unseq;
  matgen_exec_mpi_t mpi;
} matgen_exec_policy_union_t;

// =============================================================================
// Predefined Execution Policy Instances
// =============================================================================

/**
 * @brief Sequential execution policy instance
 */
extern const matgen_exec_seq_t matgen_exec_seq;

/**
 * @brief Parallel execution policy instance (default settings)
 */
extern const matgen_exec_par_t matgen_exec_par;

/**
 * @brief Parallel unsequenced execution policy instance (default settings)
 */
extern const matgen_exec_par_unseq_t matgen_exec_par_unseq;

/**
 * @brief MPI execution policy instance (default settings)
 */
extern const matgen_exec_mpi_t matgen_exec_mpi;

// =============================================================================
// Execution Policy Utilities
// =============================================================================

/**
 * @brief Check if an execution policy is available on this platform
 *
 * @param policy Execution policy to check
 * @return true if the policy is available, false otherwise
 */
bool matgen_exec_is_available(matgen_exec_policy_t policy);

/**
 * @brief Get the name of an execution policy as a string
 *
 * @param policy Execution policy
 * @return Human-readable name (e.g., "Sequential", "OpenMP", "CUDA", "MPI")
 */
const char* matgen_exec_policy_name(matgen_exec_policy_t policy);

/**
 * @brief Select the best available execution policy for a given problem size
 *
 * Heuristic-based selection:
 * - Small problems (< 1M elements): Sequential
 * - Medium problems (1M - 100M elements): OpenMP
 * - Large problems (> 100M elements): CUDA (if available) or OpenMP
 *
 * @param nnz Number of non-zero elements in the problem
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Recommended execution policy
 */
matgen_exec_policy_t matgen_exec_select_auto(matgen_size_t nnz,
                                             matgen_index_t rows,
                                             matgen_index_t cols);

/**
 * @brief Resolve an execution policy to the actual backend that will be used
 *
 * Handles fallback logic if a requested policy is not available.
 * For example, MATGEN_EXEC_PAR_UNSEQ falls back to MATGEN_EXEC_PAR if CUDA
 * is not available, and MATGEN_EXEC_PAR falls back to MATGEN_EXEC_SEQ if
 * OpenMP is not available.
 *
 * @param policy Requested execution policy
 * @return Actual execution policy that will be used
 */
matgen_exec_policy_t matgen_exec_resolve(matgen_exec_policy_t policy);

/**
 * @brief Create a custom parallel execution policy with specific thread count
 *
 * @param num_threads Number of OpenMP threads (0 = use default)
 * @return Parallel execution policy
 */
matgen_exec_par_t matgen_exec_par_with_threads(int num_threads);

/**
 * @brief Create a custom CUDA execution policy with specific device and block
 * size
 *
 * @param device_id CUDA device ID (-1 = use default)
 * @param block_size Thread block size (0 = use default, typically 256)
 * @return Parallel unsequenced execution policy
 */
matgen_exec_par_unseq_t matgen_exec_par_unseq_with_params(int device_id,
                                                          int block_size);

/**
 * @brief Create a custom MPI execution policy with specific communicator
 *
 * @param mpi_comm MPI communicator (NULL = MPI_COMM_WORLD)
 * @return MPI execution policy
 */
matgen_exec_mpi_t matgen_exec_mpi_with_comm(void* mpi_comm);

// =============================================================================
// Backend Capability Queries
// =============================================================================

/**
 * @brief Get the number of available OpenMP threads
 *
 * @return Number of threads (0 if OpenMP is not available)
 */
int matgen_exec_get_num_threads(void);

/**
 * @brief Get the number of available CUDA devices
 *
 * @return Number of CUDA devices (0 if CUDA is not available)
 */
int matgen_exec_get_num_cuda_devices(void);

/**
 * @brief Get the MPI world size
 *
 * @return Number of MPI processes (1 if MPI is not available)
 */
int matgen_exec_get_mpi_size(void);

/**
 * @brief Get the MPI rank of this process
 *
 * @return MPI rank (0 if MPI is not available)
 */
int matgen_exec_get_mpi_rank(void);

/**
 * @brief Check if the current process is the MPI root
 *
 * @return true if this is the root process (always true if MPI is not
 * available)
 */
bool matgen_exec_is_mpi_root(void);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_CORE_EXECUTION_POLICY_H
