#ifndef MATGEN_ALGORITHMS_SCALING_BILINEAR_CUDA_H
#define MATGEN_ALGORITHMS_SCALING_BILINEAR_CUDA_H

/**
 * @file bilinear_cuda.h
 * @brief Bilinear interpolation for sparse matrix scaling (CUDA GPU)
 */

#ifdef MATGEN_HAS_CUDA

#include "matgen/core/csr_matrix.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Scale sparse matrix using bilinear interpolation (CUDA GPU)
 *
 * Distributes each source entry's value to neighboring target cells
 * using bilinear weights based on distance. Values are accumulated (summed)
 * at each target cell.
 *
 * This is a CUDA GPU implementation. The source matrix is copied to device
 * memory, processed on the GPU using thread blocks, and the result is copied
 * back to host memory.
 *
 * @param source Source matrix (CSR format, on host)
 * @param new_rows Target number of rows
 * @param new_cols Target number of columns
 * @param result Output: scaled matrix (CSR format, on host)
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @note CUDA GPU implementation
 * @note Requires CUDA-capable GPU
 * @note For small matrices, may be slower than CPU versions due to transfer
 * overhead
 * @note Uses bilinear interpolation with distance-based weights
 */
matgen_error_t matgen_scale_bilinear_cuda(const matgen_csr_matrix_t* source,
                                          matgen_index_t new_rows,
                                          matgen_index_t new_cols,
                                          matgen_csr_matrix_t** result);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_HAS_CUDA

#endif  // MATGEN_ALGORITHMS_SCALING_BILINEAR_CUDA_H
