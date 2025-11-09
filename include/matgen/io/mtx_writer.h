#ifndef MATGEN_IO_MTX_WRITER_H
#define MATGEN_IO_MTX_WRITER_H

/**
 * @file mtx_writer.h
 * @brief Matrix Market format writer for sparse matrices
 *
 * Writes Matrix Market (.mtx) coordinate format files.
 */

#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Write a COO matrix to Matrix Market file
 *
 * Writes a COO matrix in Matrix Market coordinate format.
 * Always writes as general (non-symmetric) real matrix.
 *
 * @param filename Path to output .mtx file
 * @param matrix COO matrix to write
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_mtx_write_coo(const char* filename,
                                    const matgen_coo_matrix_t* matrix);

/**
 * @brief Write a CSR matrix to Matrix Market file
 *
 * Writes a CSR matrix in Matrix Market coordinate format.
 * Always writes as general (non-symmetric) real matrix.
 *
 * @param filename Path to output .mtx file
 * @param matrix CSR matrix to write
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_mtx_write_csr(const char* filename,
                                    const matgen_csr_matrix_t* matrix);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_IO_MTX_WRITER_H
