#ifndef MATGEN_CORE_MATRIX_IO_H
#define MATGEN_CORE_MATRIX_IO_H

/**
 * @file matrix_io.h
 * @brief Matrix Market format I/O for sparse matrices
 *
 * Implements reading and writing of Matrix Market (.mtx) coordinate format
 * files. Supports:
 *   - Real and pattern matrices
 *   - General, symmetric matrices
 *   - Comments in files
 *
 * @see https://math.nist.gov/MatrixMarket/formats.html
 */

#include <stdio.h>

#include "coo_matrix.h"
#include "csr_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Matrix Market format types
 */
typedef enum {
  MATGEN_MM_REAL,     // Real valued matrix
  MATGEN_MM_INTEGER,  // Integer valued matrix
  MATGEN_MM_PATTERN,  // Pattern matrix (no values, all ones)
  MATGEN_MM_COMPLEX   // Complex valued (not yet supported)
} matgen_mm_value_type_t;

/**
 * @brief Matrix Market symmetry types
 */
typedef enum {
  MATGEN_MM_GENERAL,         // No symmetry
  MATGEN_MM_SYMMETRIC,       // Symmetric matrix
  MATGEN_MM_SKEW_SYMMETRIC,  // Skew-symmetric matrix
  MATGEN_MM_HERMITIAN        // Hermitian matrix (not yet supported)
} matgen_mm_symmetry_t;

/**
 * @brief Matrix Market file information
 */
typedef struct {
  matgen_mm_value_type_t value_type;  // Value type
  matgen_mm_symmetry_t symmetry;      // Symmetry type

  size_t rows;  // Number of rows
  size_t cols;  // Number of columns
  size_t nnz;   // Number of non-zeros (as stored in file)
} matgen_mm_info_t;

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
matgen_coo_matrix_t* matgen_mm_read(const char* filename,
                                    matgen_mm_info_t* info);

/**
 * @brief Write a COO matrix to Matrix Market file
 *
 * Writes a COO matrix in Matrix Market coordinate format.
 * Always writes as general (non-symmetric) real matrix.
 *
 * @param filename Path to output .mtx file
 * @param matrix COO matrix to write
 * @return 0 on success, -1 on error
 */
int matgen_mm_write_coo(const char* filename,
                        const matgen_coo_matrix_t* matrix);

/**
 * @brief Write a CSR matrix to Matrix Market file
 *
 * Writes a CSR matrix in Matrix Market coordinate format.
 * Always writes as general (non-symmetric) real matrix.
 *
 * @param filename Path to output .mtx file
 * @param matrix CSR matrix to write
 * @return 0 on success, -1 on error
 */
int matgen_mm_write_csr(const char* filename,
                        const matgen_csr_matrix_t* matrix);

/**
 * @brief Read only the header of a Matrix Market file
 *
 * Useful for inspecting matrix dimensions without loading the full matrix.
 *
 * @param filename Path to .mtx file
 * @param info Output matrix info
 * @return 0 on success, -1 on error
 */
int matgen_mm_read_info(const char* filename, matgen_mm_info_t* info);

#ifdef __cplusplus
}
#endif

#endif
