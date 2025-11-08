#include "matgen/io/mtx_writer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MATGEN_HAS_OPENMP
#include <omp.h>
#endif

#include "matgen/utils/log.h"

// =============================================================================
// Configuration
// =============================================================================
#define FILE_BUFFER_SIZE (64 * 1024 * 1024)   // 64MB file buffer
#define CHUNK_BUFFER_SIZE (32 * 1024 * 1024)  // 32MB per thread chunk
#define MIN_ENTRIES_PER_THREAD 10000  // Minimum entries for parallelization

// =============================================================================
// Fast Integer to String Conversion with Lookup Table
// =============================================================================

// Lookup table for 2-digit pairs (00-99)
static const char digit_pairs[200] = {
    '0', '0', '0', '1', '0', '2', '0', '3', '0', '4', '0', '5', '0', '6', '0',
    '7', '0', '8', '0', '9', '1', '0', '1', '1', '1', '2', '1', '3', '1', '4',
    '1', '5', '1', '6', '1', '7', '1', '8', '1', '9', '2', '0', '2', '1', '2',
    '2', '2', '3', '2', '4', '2', '5', '2', '6', '2', '7', '2', '8', '2', '9',
    '3', '0', '3', '1', '3', '2', '3', '3', '3', '4', '3', '5', '3', '6', '3',
    '7', '3', '8', '3', '9', '4', '0', '4', '1', '4', '2', '4', '3', '4', '4',
    '4', '5', '4', '6', '4', '7', '4', '8', '4', '9', '5', '0', '5', '1', '5',
    '2', '5', '3', '5', '4', '5', '5', '5', '6', '5', '7', '5', '8', '5', '9',
    '6', '0', '6', '1', '6', '2', '6', '3', '6', '4', '6', '5', '6', '6', '6',
    '7', '6', '8', '6', '9', '7', '0', '7', '1', '7', '2', '7', '3', '7', '4',
    '7', '5', '7', '6', '7', '7', '7', '8', '7', '9', '8', '0', '8', '1', '8',
    '2', '8', '3', '8', '4', '8', '5', '8', '6', '8', '7', '8', '8', '8', '9',
    '9', '0', '9', '1', '9', '2', '9', '3', '9', '4', '9', '5', '9', '6', '9',
    '7', '9', '8', '9', '9'};

// Ultra-fast unsigned integer to string (optimized with lookup table)
static inline int fast_uint64_to_str(unsigned long long val, char* buf) {
  if (val == 0) {
    *buf = '0';
    return 1;
  }

  // Handle common small cases with lookup
  if (val < 100) {
    if (val < 10) {
      *buf = (char)('0' + val);
      return 1;
    }
    const char* p = &digit_pairs[val * 2];
    buf[0] = p[0];
    buf[1] = p[1];
    return 2;
  }

  // Count digits first
  unsigned long long temp = val;
  int digits = 0;
  while (temp > 0) {
    temp /= 10;
    digits++;
  }

  int len = digits;
  char* p = buf + digits;

  // Write pairs of digits when possible
  while (val >= 100) {
    unsigned long long q = val / 100;
    unsigned long long r = val % 100;
    val = q;
    p -= 2;
    memcpy(p, &digit_pairs[r * 2], 2);
  }

  // Write remaining 1-2 digits
  if (val >= 10) {
    p -= 2;
    memcpy(p, &digit_pairs[val * 2], 2);
  } else {
    *--p = (char)('0' + val);
  }

  return len;
}

// =============================================================================
// Fast Double to String Conversion (Simplified Ryu-style)
// =============================================================================

// Fast double to string for scientific notation
// Optimized for the %.16g format used in MTX files
static inline int fast_double_to_str(matgen_value_t val, char* buf) {
  // Handle special cases
  if (val == 0.0) {
    *buf = '0';
    return 1;
  }

  char* start = buf;

  // Handle negative
  if (val < 0.0) {
    *buf++ = '-';
    val = -val;
  }

  // For very small or very large numbers, use scientific notation
  // For normal range, use fixed-point
  if (val < 1e-4 || val >= 1e10) {
    // Use scientific notation
    int exp = 0;
    if (val >= 1.0) {
      while (val >= 10.0) {
        val /= 10.0;
        exp++;
      }
    } else {
      while (val < 1.0) {
        val *= 10.0;
        exp--;
      }
    }

    // Write mantissa (1 digit before decimal)
    int digit = (int)val;
    *buf++ = (char)('0' + digit);
    val -= digit;

    // Write up to 15 decimal places
    if (val > 1e-15) {
      *buf++ = '.';
      for (int i = 0; i < 15 && val > 1e-15; i++) {
        val *= 10.0;
        digit = (int)val;
        *buf++ = (char)('0' + digit);
        val -= digit;
      }
    }

    // Write exponent
    *buf++ = 'e';
    if (exp < 0) {
      *buf++ = '-';
      exp = -exp;
    } else {
      *buf++ = '+';
    }
    buf += fast_uint64_to_str(exp, buf);
  } else {
    // Use fixed-point notation
    unsigned long long int_part = (unsigned long long)val;
    buf += fast_uint64_to_str(int_part, buf);

    matgen_value_t frac = val - (matgen_value_t)int_part;
    if (frac > 1e-15) {
      *buf++ = '.';

      // Write up to 15 decimal places
      for (int i = 0; i < 15 && frac > 1e-15; i++) {
        frac *= 10.0;
        int digit = (int)frac;
        *buf++ = (char)('0' + digit);
        frac -= digit;
      }
    }
  }

  return (int)(buf - start);
}

// =============================================================================
// Optimized Entry Writing
// =============================================================================

// Write a single entry (row col value) to buffer
static inline int write_entry_fast(char* buf, unsigned long long row,
                                   unsigned long long col,
                                   matgen_value_t value) {
  char* start = buf;

  // Write row
  buf += fast_uint64_to_str(row, buf);
  *buf++ = ' ';

  // Write col
  buf += fast_uint64_to_str(col, buf);
  *buf++ = ' ';

  // Write value
  buf += fast_double_to_str(value, buf);
  *buf++ = '\n';

  return (int)(buf - start);
}

// =============================================================================
// Parallel Formatting
// =============================================================================

typedef struct {
  char* buffer;     // Formatted data buffer
  size_t size;      // Size of formatted data
  size_t capacity;  // Buffer capacity
} chunk_buffer_t;

// Format a chunk of COO entries into a buffer
static matgen_error_t format_coo_chunk(const matgen_index_t* row_indices,
                                       const matgen_index_t* col_indices,
                                       const matgen_value_t* values,
                                       size_t start_idx, size_t end_idx,
                                       chunk_buffer_t* chunk) {
  size_t num_entries = end_idx - start_idx;
  size_t estimated_size = num_entries * 70;  // ~70 bytes per entry worst case

  chunk->buffer = (char*)malloc(estimated_size);
  if (!chunk->buffer) {
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  chunk->capacity = estimated_size;
  chunk->size = 0;

  char* ptr = chunk->buffer;

  for (size_t i = start_idx; i < end_idx; i++) {
    ptr +=
        write_entry_fast(ptr, (unsigned long long)(row_indices[i] + 1),
                         (unsigned long long)(col_indices[i] + 1), values[i]);
  }

  chunk->size = (size_t)(ptr - chunk->buffer);
  return MATGEN_SUCCESS;
}

// Format a chunk of CSR entries into a buffer
static matgen_error_t format_csr_chunk(const matgen_csr_matrix_t* matrix,
                                       matgen_index_t start_row,
                                       matgen_index_t end_row,
                                       chunk_buffer_t* chunk) {
  // Count entries in this row range
  size_t num_entries = matrix->row_ptr[end_row] - matrix->row_ptr[start_row];
  size_t estimated_size = num_entries * 70;

  chunk->buffer = (char*)malloc(estimated_size);
  if (!chunk->buffer) {
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  chunk->capacity = estimated_size;
  chunk->size = 0;

  char* ptr = chunk->buffer;

  for (matgen_index_t row = start_row; row < end_row; row++) {
    size_t row_start = matrix->row_ptr[row];
    size_t row_end = matrix->row_ptr[row + 1];

    for (size_t j = row_start; j < row_end; j++) {
      ptr += write_entry_fast(ptr, (unsigned long long)(row + 1),
                              (unsigned long long)(matrix->col_indices[j] + 1),
                              matrix->values[j]);
    }
  }

  chunk->size = (size_t)(ptr - chunk->buffer);
  return MATGEN_SUCCESS;
}

// =============================================================================
// COO Writer (Parallel)
// =============================================================================
matgen_error_t matgen_mtx_write_coo(const char* filename,
                                    const matgen_coo_matrix_t* matrix) {
  if (!filename || !matrix) {
    MATGEN_LOG_ERROR("NULL pointer argument");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (!matgen_coo_validate(matrix)) {
    MATGEN_LOG_ERROR("Invalid COO matrix");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  MATGEN_LOG_DEBUG("Writing COO matrix to %s (%zu x %zu, nnz=%zu)", filename,
                   matrix->rows, matrix->cols, matrix->nnz);

  FILE* f = fopen(filename, "wb");  // Binary mode for better performance
  if (!f) {
    MATGEN_LOG_ERROR("Failed to open file for writing: %s", filename);
    return MATGEN_ERROR_IO;
  }

  // Large file buffer
  char* file_buffer = (char*)malloc((int)FILE_BUFFER_SIZE);
  if (file_buffer) {
    setvbuf(f, file_buffer, _IOFBF, (int)FILE_BUFFER_SIZE);
  }

  // Write header
  fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(f, "%% Generated by MatGen\n");
  fprintf(f, "%zu %zu %zu\n", matrix->rows, matrix->cols, matrix->nnz);

  matgen_error_t result = MATGEN_SUCCESS;

#ifdef MATGEN_HAS_OPENMP
  // Use parallel formatting for large matrices
  if (matrix->nnz >= (matgen_size_t)MIN_ENTRIES_PER_THREAD * 2) {
    int num_threads = omp_get_max_threads();
    if (num_threads > 1) {
      size_t chunk_size = (matrix->nnz + num_threads - 1) / num_threads;

      MATGEN_LOG_DEBUG("Parallel write with %d threads, chunk_size=%zu",
                       num_threads, chunk_size);

      chunk_buffer_t* chunks =
          (chunk_buffer_t*)calloc(num_threads, sizeof(chunk_buffer_t));
      if (!chunks) {
        result = MATGEN_ERROR_OUT_OF_MEMORY;
        goto cleanup;
      }

      // Parallel formatting
      int format_error = 0;
#pragma omp parallel num_threads(num_threads)
      {
        int tid = omp_get_thread_num();
        size_t start_idx = tid * chunk_size;
        size_t end_idx = start_idx + chunk_size;
        if (end_idx > matrix->nnz) {
          end_idx = matrix->nnz;
        }

        if (start_idx < matrix->nnz) {
          matgen_error_t err = format_coo_chunk(
              matrix->row_indices, matrix->col_indices, matrix->values,
              start_idx, end_idx, &chunks[tid]);

          if (err != MATGEN_SUCCESS) {
#pragma omp atomic write
            format_error = 1;
          }
        }
      }

      if (format_error) {
        result = MATGEN_ERROR_OUT_OF_MEMORY;
        for (int i = 0; i < num_threads; i++) {
          free(chunks[i].buffer);
        }
        free(chunks);
        goto cleanup;
      }

      // Sequential writing of formatted chunks
      for (int i = 0; i < num_threads; i++) {
        if (chunks[i].size > 0) {
          size_t written = fwrite(chunks[i].buffer, 1, chunks[i].size, f);
          if (written != chunks[i].size) {
            MATGEN_LOG_ERROR("Failed to write chunk %d", i);
            result = MATGEN_ERROR_IO;
          }
          free(chunks[i].buffer);
        }
      }
      free(chunks);
      goto cleanup;
    }
  }
#endif

  // Sequential fallback for small matrices or no OpenMP
  MATGEN_LOG_DEBUG("Sequential write");
  chunk_buffer_t chunk;
  result = format_coo_chunk(matrix->row_indices, matrix->col_indices,
                            matrix->values, 0, matrix->nnz, &chunk);

  if (result == MATGEN_SUCCESS) {
    size_t written = fwrite(chunk.buffer, 1, chunk.size, f);
    if (written != chunk.size) {
      MATGEN_LOG_ERROR("Failed to write data");
      result = MATGEN_ERROR_IO;
    }
    free(chunk.buffer);
  }

cleanup:
  fclose(f);
  free(file_buffer);

  if (result == MATGEN_SUCCESS) {
    MATGEN_LOG_DEBUG("Successfully wrote MTX file");
  }

  return result;
}

// =============================================================================
// CSR Writer (Parallel)
// =============================================================================
matgen_error_t matgen_mtx_write_csr(const char* filename,
                                    const matgen_csr_matrix_t* matrix) {
  if (!filename || !matrix) {
    MATGEN_LOG_ERROR("NULL pointer argument");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (!matgen_csr_validate(matrix)) {
    MATGEN_LOG_ERROR("Invalid CSR matrix");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  MATGEN_LOG_DEBUG("Writing CSR matrix to %s (%zu x %zu, nnz=%zu)", filename,
                   matrix->rows, matrix->cols, matrix->nnz);

  FILE* f = fopen(filename, "wb");
  if (!f) {
    MATGEN_LOG_ERROR("Failed to open file for writing: %s", filename);
    return MATGEN_ERROR_IO;
  }

  char* file_buffer = (char*)malloc((int)FILE_BUFFER_SIZE);
  if (file_buffer) {
    setvbuf(f, file_buffer, _IOFBF, (int)FILE_BUFFER_SIZE);
  }

  // Write header
  fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(f, "%% Generated by MatGen\n");
  fprintf(f, "%zu %zu %zu\n", matrix->rows, matrix->cols, matrix->nnz);

  matgen_error_t result = MATGEN_SUCCESS;

#ifdef MATGEN_HAS_OPENMP
  // Use parallel formatting for large matrices
  if (matrix->nnz >= (matgen_size_t)MIN_ENTRIES_PER_THREAD * 2) {
    int num_threads = omp_get_max_threads();
    if (num_threads > 1) {
      matgen_index_t rows_per_thread =
          (matrix->rows + num_threads - 1) / num_threads;

      MATGEN_LOG_DEBUG("Parallel write with %d threads, rows_per_thread=%zu",
                       num_threads, rows_per_thread);

      chunk_buffer_t* chunks =
          (chunk_buffer_t*)calloc(num_threads, sizeof(chunk_buffer_t));
      if (!chunks) {
        result = MATGEN_ERROR_OUT_OF_MEMORY;
        goto cleanup_csr;
      }

      // Parallel formatting
      int format_error = 0;
#pragma omp parallel num_threads(num_threads)
      {
        int tid = omp_get_thread_num();
        matgen_index_t start_row = tid * rows_per_thread;
        matgen_index_t end_row = start_row + rows_per_thread;
        if (end_row > matrix->rows) {
          end_row = matrix->rows;
        }

        if (start_row < matrix->rows) {
          matgen_error_t err =
              format_csr_chunk(matrix, start_row, end_row, &chunks[tid]);

          if (err != MATGEN_SUCCESS) {
#pragma omp atomic write
            format_error = 1;
          }
        }
      }

      if (format_error) {
        result = MATGEN_ERROR_OUT_OF_MEMORY;
        for (int i = 0; i < num_threads; i++) {
          free(chunks[i].buffer);
        }
        free(chunks);
        goto cleanup_csr;
      }

      // Sequential writing
      for (int i = 0; i < num_threads; i++) {
        if (chunks[i].size > 0) {
          size_t written = fwrite(chunks[i].buffer, 1, chunks[i].size, f);
          if (written != chunks[i].size) {
            MATGEN_LOG_ERROR("Failed to write chunk %d", i);
            result = MATGEN_ERROR_IO;
          }
          free(chunks[i].buffer);
        }
      }
      free(chunks);
      goto cleanup_csr;
    }
  }
#endif

  // Sequential fallback
  MATGEN_LOG_DEBUG("Sequential write");
  chunk_buffer_t chunk;
  result = format_csr_chunk(matrix, 0, matrix->rows, &chunk);

  if (result == MATGEN_SUCCESS) {
    size_t written = fwrite(chunk.buffer, 1, chunk.size, f);
    if (written != chunk.size) {
      MATGEN_LOG_ERROR("Failed to write data");
      result = MATGEN_ERROR_IO;
    }
    free(chunk.buffer);
  }

cleanup_csr:
  fclose(f);
  free(file_buffer);

  if (result == MATGEN_SUCCESS) {
    MATGEN_LOG_DEBUG("Successfully wrote MTX file");
  }

  return result;
}
