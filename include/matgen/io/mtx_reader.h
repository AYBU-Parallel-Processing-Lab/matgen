#ifndef MATGEN_IO_MTX_READER_H
#define MATGEN_IO_MTX_READER_H

/**
 * @file mtx_reader.h
 * @brief Matrix Market format reader for sparse matrices
 *
 * Reads Matrix Market (.mtx) coordinate format files.
 */

#include "matgen/core/matrix/coo.h"
#include "matgen/core/types.h"
#include "matgen/io/mtx_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Read a Matrix Market file into a COO matrix
 *
 * Reads a Matrix Market coordinate format file and returns a COO matrix.
 * For symmetric matrices, expands to full storage (adds transpose entries).
 * For pattern matrices, sets all values to 1.0.
 *
 * @param filename Path to .mtx file
 * @param info Optional output for matrix info (can be NULL)
 * @return New COO matrix, or NULL on error
 */
matgen_coo_matrix_t* matgen_mtx_read(const char* filename,
                                     matgen_mm_info_t* info);

/**
 * @brief Read only the header of a Matrix Market file
 *
 * Useful for inspecting matrix dimensions without loading the full matrix.
 *
 * @param filename Path to .mtx file
 * @param info Output matrix info (required)
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_mtx_read_header(const char* filename,
                                      matgen_mm_info_t* info);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_IO_MTX_READER_H
