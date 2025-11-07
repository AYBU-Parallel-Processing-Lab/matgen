#include "matgen/io/mtx_writer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/utils/log.h"

// =============================================================================
// Configuration
// =============================================================================
#define FILE_BUFFER_SIZE (8 * 1024 * 1024)    // 8MB file buffer (increased)
#define BATCH_BUFFER_SIZE (16 * 1024 * 1024)  // 16MB batch buffer (increased)
#define BATCH_SAFETY_MARGIN 512               // Safety margin for last line

// =============================================================================
// Fast Number to String Conversion
// =============================================================================

// Fast unsigned integer to string conversion (returns length written)
static inline int fast_uint64_to_str(unsigned long long val, char* buf) {
  if (val == 0) {
    *buf = '0';
    return 1;
  }

  // Count digits
  unsigned long long temp = val;
  int digits = 0;
  while (temp > 0) {
    temp /= 10;
    digits++;
  }

  // Write digits in reverse
  int len = digits;
  while (val > 0) {
    buf[--digits] = (char)('0' + (val % 10));
    val /= 10;
  }

  return len;
}

// Fast double to string conversion for scientific notation
// This is a simplified version optimized for %.16g format
static inline int fast_double_to_str(double val, char* buf) {
  // Handle special cases
  if (val == 0.0) {
    *buf++ = '0';
    return 1;
  }

  int len = 0;

  // Handle negative
  if (val < 0) {
    *buf++ = '-';
    val = -val;
    len++;
  }

  // Use snprintf for doubles (still reasonably fast for the value part)
  // A fully custom implementation would be complex for proper rounding
  int written = snprintf(buf, 32, "%.16g", val);
  return len + written;
}

// Optimized version that writes directly to buffer without bounds checking
// Assumes buffer has enough space (caller must ensure this)
static inline int write_entry_unchecked(char* buf, unsigned long long row,
                                        unsigned long long col, double value) {
  char* start = buf;

  // Write row
  int row_len = fast_uint64_to_str(row, buf);
  buf += row_len;
  *buf++ = ' ';

  // Write col
  int col_len = fast_uint64_to_str(col, buf);
  buf += col_len;
  *buf++ = ' ';

  // Write value
  int val_len = fast_double_to_str(value, buf);
  buf += val_len;
  *buf++ = '\n';

  return (int)(buf - start);
}

// =============================================================================
// Helper Function: Write Buffered (Optimized)
// =============================================================================
static matgen_error_t write_matrix_entries_buffered(
    FILE* f, const matgen_index_t* row_indices,
    const matgen_index_t* col_indices, const double* values,
    matgen_size_t nnz) {
  // Allocate batch buffer
  char* batch_buffer = (char*)malloc((int)BATCH_BUFFER_SIZE);
  if (!batch_buffer) {
    MATGEN_LOG_ERROR("Failed to allocate batch buffer");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  size_t buffer_pos = 0;

  // Pre-calculate worst-case line length:
  // 20 digits + space + 20 digits + space + 24 chars (double) + newline ≈ 66
  // bytes
  const size_t max_line_length = 66;
  const size_t flush_threshold = (int)BATCH_BUFFER_SIZE - max_line_length;

  for (matgen_size_t i = 0; i < nnz; i++) {
    // Check if we need to flush before writing
    if (buffer_pos >= flush_threshold) {
      size_t written_bytes = fwrite(batch_buffer, 1, buffer_pos, f);
      if (written_bytes != buffer_pos) {
        MATGEN_LOG_ERROR("Failed to write batch buffer to file");
        free(batch_buffer);
        return MATGEN_ERROR_IO;
      }
      buffer_pos = 0;
    }

    // Write entry directly (unchecked - we know we have space)
    int written = write_entry_unchecked(
        batch_buffer + buffer_pos, (unsigned long long)(row_indices[i] + 1),
        (unsigned long long)(col_indices[i] + 1), values[i]);

    buffer_pos += written;
  }

  // Flush remaining buffer content
  if (buffer_pos > 0) {
    size_t written_bytes = fwrite(batch_buffer, 1, buffer_pos, f);
    if (written_bytes != buffer_pos) {
      MATGEN_LOG_ERROR("Failed to write final batch buffer to file");
      free(batch_buffer);
      return MATGEN_ERROR_IO;
    }
  }

  free(batch_buffer);
  return MATGEN_SUCCESS;
}

// =============================================================================
// COO Writer
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

  MATGEN_LOG_DEBUG("Writing COO matrix to %s (%llu x %llu, nnz=%zu)", filename,
                   (unsigned long long)matrix->rows,
                   (unsigned long long)matrix->cols, matrix->nnz);

  FILE* f = fopen(filename, "w");
  if (!f) {
    MATGEN_LOG_ERROR("Failed to open file for writing: %s", filename);
    return MATGEN_ERROR_IO;
  }

  // Allocate file buffer for better I/O performance
  char* file_buffer = (char*)malloc((int)FILE_BUFFER_SIZE);
  if (file_buffer) {
    setvbuf(f, file_buffer, _IOFBF, (int)FILE_BUFFER_SIZE);
  } else {
    MATGEN_LOG_WARN("Failed to allocate file buffer, using default buffering");
  }

  // Write header
  fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(f, "%% Generated by MatGen\n");

  // Write dimensions
  fprintf(f, "%llu %llu %zu\n", (unsigned long long)matrix->rows,
          (unsigned long long)matrix->cols, matrix->nnz);

  // Write entries using optimized buffered batch writing
  matgen_error_t result = write_matrix_entries_buffered(
      f, matrix->row_indices, matrix->col_indices, matrix->values, matrix->nnz);

  fclose(f);

  if (file_buffer) {
    free(file_buffer);
  }

  if (result != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to write matrix entries");
    return result;
  }

  MATGEN_LOG_DEBUG("Successfully wrote MTX file");
  return MATGEN_SUCCESS;
}

// =============================================================================
// CSR Writer (Optimized)
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

  MATGEN_LOG_DEBUG("Writing CSR matrix to %s (%llu x %llu, nnz=%zu)", filename,
                   (unsigned long long)matrix->rows,
                   (unsigned long long)matrix->cols, matrix->nnz);

  FILE* f = fopen(filename, "w");
  if (!f) {
    MATGEN_LOG_ERROR("Failed to open file for writing: %s", filename);
    return MATGEN_ERROR_IO;
  }

  // Allocate file buffer for better I/O performance
  char* file_buffer = (char*)malloc((int)FILE_BUFFER_SIZE);
  if (file_buffer) {
    setvbuf(f, file_buffer, _IOFBF, (int)FILE_BUFFER_SIZE);
  } else {
    MATGEN_LOG_WARN("Failed to allocate file buffer, using default buffering");
  }

  // Write header
  fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(f, "%% Generated by MatGen\n");

  // Write dimensions
  fprintf(f, "%llu %llu %zu\n", (unsigned long long)matrix->rows,
          (unsigned long long)matrix->cols, matrix->nnz);

  // Allocate batch buffer
  char* batch_buffer = (char*)malloc((int)BATCH_BUFFER_SIZE);
  if (!batch_buffer) {
    MATGEN_LOG_ERROR("Failed to allocate batch buffer");
    fclose(f);
    if (file_buffer) {
      free(file_buffer);
    }
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  size_t buffer_pos = 0;
  matgen_error_t result = MATGEN_SUCCESS;

  const size_t max_line_length = 66;
  const size_t flush_threshold = (int)BATCH_BUFFER_SIZE - max_line_length;

  // Write entries by iterating through CSR structure
  for (matgen_index_t row = 0; row < matrix->rows; row++) {
    matgen_size_t row_start = matrix->row_ptr[row];
    matgen_size_t row_end = matrix->row_ptr[row + 1];

    for (matgen_size_t j = row_start; j < row_end; j++) {
      // Check if we need to flush
      if (buffer_pos >= flush_threshold) {
        size_t written_bytes = fwrite(batch_buffer, 1, buffer_pos, f);
        if (written_bytes != buffer_pos) {
          MATGEN_LOG_ERROR("Failed to write batch buffer to file");
          result = MATGEN_ERROR_IO;
          goto cleanup;
        }
        buffer_pos = 0;
      }

      // Write entry directly (unchecked - we know we have space)
      int written = write_entry_unchecked(
          batch_buffer + buffer_pos, (unsigned long long)(row + 1),
          (unsigned long long)(matrix->col_indices[j] + 1), matrix->values[j]);

      buffer_pos += written;
    }
  }

  // Flush remaining buffer content
  if (buffer_pos > 0) {
    size_t written_bytes = fwrite(batch_buffer, 1, buffer_pos, f);
    if (written_bytes != buffer_pos) {
      MATGEN_LOG_ERROR("Failed to write final batch buffer to file");
      result = MATGEN_ERROR_IO;
    }
  }

cleanup:
  free(batch_buffer);
  fclose(f);
  if (file_buffer) {
    free(file_buffer);
  }

  if (result == MATGEN_SUCCESS) {
    MATGEN_LOG_DEBUG("Successfully wrote MTX file");
  }

  return result;
}
