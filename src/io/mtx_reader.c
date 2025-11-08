#include "matgen/io/mtx_reader.h"

#include <stdio.h>
#include <string.h>

#include "matgen/utils/log.h"

// =============================================================================
// Helper Functions
// =============================================================================

// Skip whitespace and comments
static int skip_comments(FILE* f) {
  int c;
  while ((c = fgetc(f)) == '%') {
    // Skip to end of line
    while ((c = fgetc(f)) != '\n' && c != EOF) {
      // Skip
    }
  }
  ungetc(c, f);
  return c != EOF ? 0 : -1;
}

// Parse header line
static matgen_error_t parse_header(const char* header, matgen_mm_info_t* info) {
  char obj_str[32];
  char fmt_str[32];
  char val_str[32];
  char sym_str[32];

  // Format: %%MatrixMarket matrix coordinate real general
  int n = sscanf(header, "%%%%MatrixMarket %31s %31s %31s %31s", obj_str,
                 fmt_str, val_str, sym_str);

  if (n != 4) {
    MATGEN_LOG_ERROR("Invalid Matrix Market header");
    return MATGEN_ERROR_INVALID_FORMAT;
  }

  // Parse object type
  if (strcmp(obj_str, "matrix") == 0) {
    info->object = MATGEN_MM_MATRIX;
  } else if (strcmp(obj_str, "vector") == 0) {
    info->object = MATGEN_MM_VECTOR;
  } else {
    MATGEN_LOG_ERROR("Unsupported object type: %s", obj_str);
    return MATGEN_ERROR_UNSUPPORTED;
  }

  // Parse format
  if (strcmp(fmt_str, "coordinate") == 0) {
    info->format = MATGEN_MM_COORDINATE;
  } else if (strcmp(fmt_str, "array") == 0) {
    info->format = MATGEN_MM_ARRAY;
  } else {
    MATGEN_LOG_ERROR("Unsupported format: %s", fmt_str);
    return MATGEN_ERROR_UNSUPPORTED;
  }

  // Parse value type
  if (strcmp(val_str, "real") == 0) {
    info->value_type = MATGEN_MM_REAL;
  } else if (strcmp(val_str, "integer") == 0) {
    info->value_type = MATGEN_MM_INTEGER;
  } else if (strcmp(val_str, "pattern") == 0) {
    info->value_type = MATGEN_MM_PATTERN;
  } else if (strcmp(val_str, "complex") == 0) {
    info->value_type = MATGEN_MM_COMPLEX;
    MATGEN_LOG_ERROR("Complex matrices not yet supported");
    return MATGEN_ERROR_UNSUPPORTED;
  } else {
    MATGEN_LOG_ERROR("Unsupported value type: %s", val_str);
    return MATGEN_ERROR_UNSUPPORTED;
  }

  // Parse symmetry
  if (strcmp(sym_str, "general") == 0) {
    info->symmetry = MATGEN_MM_GENERAL;
  } else if (strcmp(sym_str, "symmetric") == 0) {
    info->symmetry = MATGEN_MM_SYMMETRIC;
  } else if (strcmp(sym_str, "skew-symmetric") == 0) {
    info->symmetry = MATGEN_MM_SKEW_SYMMETRIC;
  } else if (strcmp(sym_str, "hermitian") == 0) {
    info->symmetry = MATGEN_MM_HERMITIAN;
    MATGEN_LOG_ERROR("Hermitian matrices not yet supported");
    return MATGEN_ERROR_UNSUPPORTED;
  } else {
    MATGEN_LOG_ERROR("Unsupported symmetry: %s", sym_str);
    return MATGEN_ERROR_UNSUPPORTED;
  }

  return MATGEN_SUCCESS;
}

// =============================================================================
// Public API
// =============================================================================

matgen_error_t matgen_mtx_read_header(const char* filename,
                                      matgen_mm_info_t* info) {
  if (!filename || !info) {
    MATGEN_LOG_ERROR("NULL pointer argument");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  FILE* f = fopen(filename, "r");
  if (!f) {
    MATGEN_LOG_ERROR("Failed to open file: %s", filename);
    return MATGEN_ERROR_IO;
  }

  // Read header line
  char header[256];
  if (!fgets(header, sizeof(header), f)) {
    MATGEN_LOG_ERROR("Failed to read header");
    fclose(f);
    return MATGEN_ERROR_IO;
  }

  // Parse header
  matgen_error_t err = parse_header(header, info);
  if (err != MATGEN_SUCCESS) {
    fclose(f);
    return err;
  }

  // Skip comments
  if (skip_comments(f) != 0) {
    MATGEN_LOG_ERROR("Unexpected EOF after header");
    fclose(f);
    return MATGEN_ERROR_IO;
  }

  // Read dimensions
  unsigned long long rows;
  unsigned long long cols;
  unsigned long long nnz;
  if (fscanf(f, "%llu %llu %llu", &rows, &cols, &nnz) != 3) {
    MATGEN_LOG_ERROR("Failed to read matrix dimensions");
    fclose(f);
    return MATGEN_ERROR_INVALID_FORMAT;
  }

  info->rows = (matgen_index_t)rows;
  info->cols = (matgen_index_t)cols;
  info->nnz = (matgen_size_t)nnz;

  fclose(f);

  MATGEN_LOG_DEBUG("Read MTX header: %llu x %llu, nnz=%zu",
                   (unsigned long long)info->rows,
                   (unsigned long long)info->cols, info->nnz);

  return MATGEN_SUCCESS;
}

matgen_coo_matrix_t* matgen_mtx_read(const char* filename,
                                     matgen_mm_info_t* info_out) {
  if (!filename) {
    MATGEN_LOG_ERROR("NULL filename");
    return NULL;
  }

  // Read header
  matgen_mm_info_t info;
  matgen_error_t err = matgen_mtx_read_header(filename, &info);
  if (err != MATGEN_SUCCESS) {
    return NULL;
  }

  // Check we support this format
  if (info.format != MATGEN_MM_COORDINATE) {
    MATGEN_LOG_ERROR("Only coordinate format supported");
    return NULL;
  }

  MATGEN_LOG_DEBUG("Reading MTX file: %s (%llu x %llu, nnz=%zu)", filename,
                   (unsigned long long)info.rows, (unsigned long long)info.cols,
                   info.nnz);

  // Determine actual nnz (symmetric matrices will be expanded)
  matgen_size_t actual_nnz = info.nnz;
  if (info.symmetry == MATGEN_MM_SYMMETRIC) {
    // Estimate expanded size (will have at most 2*nnz - diag_count)
    actual_nnz = info.nnz * 2;
    MATGEN_LOG_DEBUG("Symmetric matrix will be expanded");
  }

  // Create matrix
  matgen_coo_matrix_t* matrix =
      matgen_coo_create(info.rows, info.cols, actual_nnz);
  if (!matrix) {
    return NULL;
  }

  // Reopen file to read data
  FILE* f = fopen(filename, "r");
  if (!f) {
    MATGEN_LOG_ERROR("Failed to reopen file");
    matgen_coo_destroy(matrix);
    return NULL;
  }

  // Skip header and comments again
  char line[256];
  fgets(line, sizeof(line), f);  // Skip header
  skip_comments(f);

  // Skip dimension line
  unsigned long long dummy;
  fscanf(f, "%llu %llu %llu", &dummy, &dummy, &dummy);

  // Read entries
  for (matgen_size_t i = 0; i < info.nnz; i++) {
    unsigned long long row_1based;
    unsigned long long col_1based;
    f64 value = 1.0;  // Default for pattern matrices

    if (info.value_type == MATGEN_MM_PATTERN) {
      if (fscanf(f, "%llu %llu", &row_1based, &col_1based) != 2) {
        MATGEN_LOG_ERROR("Failed to read entry %zu", i);
        matgen_coo_destroy(matrix);
        fclose(f);
        return NULL;
      }
    } else {
      if (fscanf(f, "%llu %llu %lf", &row_1based, &col_1based, &value) != 3) {
        MATGEN_LOG_ERROR("Failed to read entry %zu", i);
        matgen_coo_destroy(matrix);
        fclose(f);
        return NULL;
      }
    }

    // Convert to 0-based indices
    matgen_index_t row = (matgen_index_t)(row_1based - 1);
    matgen_index_t col = (matgen_index_t)(col_1based - 1);

    // Add entry
    if (matgen_coo_add_entry(matrix, row, col, (matgen_value_t)value) !=
        MATGEN_SUCCESS) {
      MATGEN_LOG_ERROR("Failed to add entry (%llu, %llu)",
                       (unsigned long long)row, (unsigned long long)col);
      matgen_coo_destroy(matrix);
      fclose(f);
      return NULL;
    }

    // For symmetric matrices, add transpose (if not diagonal)
    if (info.symmetry == MATGEN_MM_SYMMETRIC && row != col) {
      f64 transpose_value = value;
      if (info.symmetry == MATGEN_MM_SKEW_SYMMETRIC) {
        transpose_value = -value;
      }

      // Its correct to add the transpose entry here
      // NOLINTNEXTLINE(readability-suspicious-call-argument)
      if (matgen_coo_add_entry(matrix, col, row,
                               (matgen_value_t)transpose_value) !=
          MATGEN_SUCCESS) {
        MATGEN_LOG_ERROR("Failed to add transpose entry");
        matgen_coo_destroy(matrix);
        fclose(f);
        return NULL;
      }
    }
  }

  fclose(f);

  // Sort the matrix
  if (matgen_coo_sort(matrix) != MATGEN_SUCCESS) {
    MATGEN_LOG_ERROR("Failed to sort matrix");
    matgen_coo_destroy(matrix);
    return NULL;
  }

  MATGEN_LOG_DEBUG("Successfully read MTX file, actual nnz=%zu", matrix->nnz);

  // Return info if requested
  if (info_out) {
    *info_out = info;
  }

  return matrix;
}
