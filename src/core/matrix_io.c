#include "matgen/core/matrix_io.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/core/coo_matrix.h"
#include "matgen/core/csr_matrix.h"

/* Helper: Skip whitespace and comments */
static void skip_comments(FILE* f) {
  int c;
  while ((c = fgetc(f)) != EOF) {
    if (c == '%') {
      /* Skip comment line */
      while ((c = fgetc(f)) != EOF && c != '\n');
    } else if (!isspace(c)) {
      ungetc(c, f);
      break;
    }
  }
}

/* Helper: Parse Matrix Market header */
static int parse_header(FILE* f, matgen_mm_info_t* info) {
  char banner[64];
  char mtx[64];
  char format[64];
  char field[64];
  char symmetry[64];

  /* Read banner line: %%MatrixMarket matrix coordinate real general */
  if (fscanf(f, "%s %s %s %s %s", banner, mtx, format, field, symmetry) != 5) {
    return -1;
  }

  /* Validate banner */
  if (strcmp(banner, "%%MatrixMarket") != 0) {
    return -1;
  }
  if (strcmp(mtx, "matrix") != 0) {
    return -1;
  }
  if (strcmp(format, "coordinate") != 0) {
    return -1; /* We only support coordinate format */
  }

  /* Parse field (value type) */
  if (strcmp(field, "real") == 0) {
    info->value_type = MATGEN_MM_REAL;
  } else if (strcmp(field, "integer") == 0) {
    info->value_type = MATGEN_MM_INTEGER;
  } else if (strcmp(field, "pattern") == 0) {
    info->value_type = MATGEN_MM_PATTERN;
  } else if (strcmp(field, "complex") == 0) {
    info->value_type = MATGEN_MM_COMPLEX;
    return -1; /* Not supported yet */
  } else {
    return -1;
  }

  /* Parse symmetry */
  if (strcmp(symmetry, "general") == 0) {
    info->symmetry = MATGEN_MM_GENERAL;
  } else if (strcmp(symmetry, "symmetric") == 0) {
    info->symmetry = MATGEN_MM_SYMMETRIC;
  } else if (strcmp(symmetry, "skew-symmetric") == 0) {
    info->symmetry = MATGEN_MM_SKEW_SYMMETRIC;
    return -1; /* Not supported yet */
  } else if (strcmp(symmetry, "hermitian") == 0) {
    info->symmetry = MATGEN_MM_HERMITIAN;
    return -1; /* Not supported yet */
  } else {
    return -1;
  }

  return 0;
}

int matgen_mm_read_info(const char* filename, matgen_mm_info_t* info) {
  FILE* f = fopen(filename, "r");
  if (!f) {
    return -1;
  }

  /* Parse header */
  if (parse_header(f, info) != 0) {
    fclose(f);
    return -1;
  }

  /* Skip comments */
  skip_comments(f);

  /* Read dimensions: rows cols nnz */
  if (fscanf(f, "%zu %zu %zu", &info->rows, &info->cols, &info->nnz) != 3) {
    fclose(f);
    return -1;
  }

  fclose(f);
  return 0;
}

matgen_coo_matrix_t* matgen_mm_read(const char* filename,
                                    matgen_mm_info_t* info) {
  matgen_mm_info_t local_info;
  matgen_mm_info_t* info_ptr = info ? info : &local_info;

  /* Read header and dimensions */
  if (matgen_mm_read_info(filename, info_ptr) != 0) {
    return NULL;
  }

  /* Open file again for reading data */
  FILE* f = fopen(filename, "r");
  if (!f) {
    return NULL;
  }

  /* Skip header and comments again */
  char line[1024];
  while (fgets(line, sizeof(line), f)) {
    if (line[0] != '%') {
      break; /* Found dimension line */
    }
  }

  /* Skip dimension line (already read) */
  size_t rows;
  size_t cols;
  size_t nnz;
  if (sscanf(line, "%zu %zu %zu", &rows, &cols, &nnz) != 3) {
    fclose(f);
    return NULL;
  }

  /* Initialize matrix */
  int expand_symmetric = (info_ptr->symmetry == MATGEN_MM_SYMMETRIC);
  size_t capacity = expand_symmetric ? nnz * 2 : nnz;

  matgen_coo_matrix_t* matrix = matgen_coo_create(rows, cols, capacity);
  if (!matrix) {
    fclose(f);
    return NULL;
  }

  /* Read entries */
  size_t row;
  size_t col;
  double value;
  size_t count = 0;

  while (count < nnz) {
    if (info_ptr->value_type == MATGEN_MM_PATTERN) {
      /* Pattern matrix: row col (no value) */
      if (fscanf(f, "%zu %zu", &row, &col) != 2) {
        matgen_coo_destroy(matrix);
        fclose(f);
        return NULL;
      }
      value = 1.0; /* Default for pattern matrices */
    } else {
      /* Real or integer matrix: row col value */
      if (fscanf(f, "%zu %zu %lf", &row, &col, &value) != 3) {
        matgen_coo_destroy(matrix);
        fclose(f);
        return NULL;
      }
    }

    /* Convert to 0-based indexing (Matrix Market is 1-based) */
    row--;
    col--;

    /* Insert entry */
    if (matgen_coo_add_entry(matrix, row, col, value) != 0) {
      matgen_coo_destroy(matrix);
      fclose(f);
      return NULL;
    }

    /* For symmetric matrices, add transpose entry (unless on diagonal) */
    if (expand_symmetric && row != col) {
      // NOLINTNEXTLINE(readability-suspicious-call-argument)
      if (matgen_coo_add_entry(matrix, col, row, value) != 0) {
        matgen_coo_destroy(matrix);
        fclose(f);
        return NULL;
      }
    }

    count++;
  }

  fclose(f);
  return matrix;
}

int matgen_mm_write_coo(const char* filename,
                        const matgen_coo_matrix_t* matrix) {
  if (!matrix || !filename) {
    return -1;
  }

  FILE* f = fopen(filename, "w");
  if (!f) {
    return -1;
  }

  /* Write header */
  fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(f, "%% Generated by MatGen\n");

  /* Write dimensions */
  fprintf(f, "%zu %zu %zu\n", matrix->rows, matrix->cols, matrix->nnz);

  /* Write entries (convert to 1-based indexing) */
  for (size_t i = 0; i < matrix->nnz; i++) {
    fprintf(f, "%zu %zu %.16g\n", matrix->row_indices[i] + 1,
            matrix->col_indices[i] + 1, matrix->values[i]);
  }

  fclose(f);
  return 0;
}

int matgen_mm_write_csr(const char* filename,
                        const matgen_csr_matrix_t* matrix) {
  if (!matrix || !filename) {
    return -1;
  }

  FILE* f = fopen(filename, "w");
  if (!f) {
    return -1;
  }

  /* Write header */
  fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(f, "%% Generated by MatGen\n");

  /* Write dimensions */
  fprintf(f, "%zu %zu %zu\n", matrix->rows, matrix->cols, matrix->nnz);

  /* Write entries by iterating through CSR structure (convert to 1-based) */
  for (size_t i = 0; i < matrix->rows; i++) {
    size_t row_start = matrix->row_ptr[i];
    size_t row_end = matrix->row_ptr[i + 1];

    for (size_t j = row_start; j < row_end; j++) {
      fprintf(f, "%zu %zu %.16g\n", i + 1, matrix->col_indices[j] + 1,
              matrix->values[j]);
    }
  }

  fclose(f);
  return 0;
}
