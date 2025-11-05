#include "matgen/core/csr_matrix.h"

#include <stdlib.h>
#include <string.h>

#include "matgen/util/log.h"

// =============================================================================
// Creation and Destruction
// =============================================================================

matgen_csr_matrix_t* matgen_csr_create(size_t rows, size_t cols, size_t nnz) {
  if (rows == 0 || cols == 0) {
    MATGEN_LOG_ERROR("Invalid matrix dimensions: %zu x %zu", rows, cols);
    return NULL;
  }

  matgen_csr_matrix_t* matrix =
      (matgen_csr_matrix_t*)malloc(sizeof(matgen_csr_matrix_t));
  if (!matrix) {
    MATGEN_LOG_ERROR("Failed to allocate CSR matrix structure");
    return NULL;
  }

  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = nnz;

  // Allocate arrays
  matrix->row_ptr = (size_t*)calloc(rows + 1, sizeof(size_t));
  matrix->col_indices = (size_t*)malloc(nnz * sizeof(size_t));
  matrix->values = (double*)malloc(nnz * sizeof(double));

  if (!matrix->row_ptr || !matrix->col_indices || !matrix->values) {
    MATGEN_LOG_ERROR("Failed to allocate CSR matrix arrays");
    matgen_csr_destroy(matrix);
    return NULL;
  }

  // Initialize row_ptr to 0 (empty rows)
  // Already done by calloc

  MATGEN_LOG_DEBUG("Created CSR matrix %zu x %zu with nnz %zu", rows, cols,
                   nnz);

  return matrix;
}

void matgen_csr_destroy(matgen_csr_matrix_t* matrix) {
  if (!matrix) {
    return;
  }

  MATGEN_LOG_DEBUG("Destroying CSR matrix %zu x %zu (nnz: %zu)", matrix->rows,
                   matrix->cols, matrix->nnz);

  free(matrix->row_ptr);
  free(matrix->col_indices);
  free(matrix->values);
  free(matrix);
}

// =============================================================================
// Matrix Access
// =============================================================================

double matgen_csr_get(const matgen_csr_matrix_t* matrix, size_t row,
                      size_t col) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return 0.0;
  }

  if (row >= matrix->rows || col >= matrix->cols) {
    MATGEN_LOG_ERROR("Index out of bounds: (%zu, %zu) for %zu x %zu matrix",
                     row, col, matrix->rows, matrix->cols);
    return 0.0;
  }

  // Binary search in row
  size_t start = matrix->row_ptr[row];
  size_t end = matrix->row_ptr[row + 1];

  while (start < end) {
    size_t mid = start + ((end - start) / 2);
    size_t mid_col = matrix->col_indices[mid];

    if (mid_col == col) {
      return matrix->values[mid];
    }

    if (mid_col < col) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }

  return 0.0;  // Not found
}

size_t matgen_csr_row_nnz(const matgen_csr_matrix_t* matrix, size_t row) {
  if (!matrix || row >= matrix->rows) {
    return 0;
  }

  return matrix->row_ptr[row + 1] - matrix->row_ptr[row];
}

// =============================================================================
// Utility Functions
// =============================================================================

void matgen_csr_print_info(const matgen_csr_matrix_t* matrix, FILE* stream) {
  if (!matrix || !stream) {
    return;
  }

  double sparsity = (matrix->rows * matrix->cols > 0)
                        ? (100.0 * (double)matrix->nnz) /
                              (double)(matrix->rows * matrix->cols)
                        : 0.0;

  fprintf(stream, "CSR Matrix Information:\n");
  fprintf(stream, "  Dimensions: %zu x %zu\n", matrix->rows, matrix->cols);
  fprintf(stream, "  Non-zeros:  %zu\n", matrix->nnz);
  fprintf(stream, "  Sparsity:   %.4f%%\n", sparsity);

  // Row statistics
  size_t min_nnz = matrix->nnz;
  size_t max_nnz = 0;
  size_t empty_rows = 0;

  for (size_t i = 0; i < matrix->rows; i++) {
    size_t row_nnz = matgen_csr_row_nnz(matrix, i);
    if (row_nnz == 0) {
      empty_rows++;
    }
    if (row_nnz < min_nnz) {
      min_nnz = row_nnz;
    }

    if (row_nnz > max_nnz) {
      max_nnz = row_nnz;
    }
  }

  fprintf(stream, "  Empty rows: %zu\n", empty_rows);
  fprintf(stream, "  Min/Max nnz per row: %zu / %zu\n", min_nnz, max_nnz);
}

size_t matgen_csr_memory_usage(const matgen_csr_matrix_t* matrix) {
  if (!matrix) {
    return 0;
  }

  size_t memory = sizeof(matgen_csr_matrix_t);
  memory += (matrix->rows + 1) * sizeof(size_t);  // row_ptr
  memory += matrix->nnz * sizeof(size_t);         // col_indices
  memory += matrix->nnz * sizeof(double);         // values

  return memory;
}

bool matgen_csr_validate(const matgen_csr_matrix_t* matrix) {
  if (!matrix) {
    MATGEN_LOG_ERROR("NULL matrix pointer");
    return false;
  }

  // Check row_ptr is monotonically increasing
  for (size_t i = 0; i < matrix->rows; i++) {
    if (matrix->row_ptr[i] > matrix->row_ptr[i + 1]) {
      MATGEN_LOG_ERROR("row_ptr not monotonic at row %zu", i);
      return false;
    }
  }

  // Check last row_ptr equals nnz
  if (matrix->row_ptr[matrix->rows] != matrix->nnz) {
    MATGEN_LOG_ERROR("row_ptr[%zu] = %zu, expected %zu", matrix->rows,
                     matrix->row_ptr[matrix->rows], matrix->nnz);
    return false;
  }

  // Check column indices and sorting
  for (size_t i = 0; i < matrix->rows; i++) {
    size_t row_start = matrix->row_ptr[i];
    size_t row_end = matrix->row_ptr[i + 1];

    for (size_t j = row_start; j < row_end; j++) {
      // Check column in range
      if (matrix->col_indices[j] >= matrix->cols) {
        MATGEN_LOG_ERROR("Column index %zu out of range at position %zu",
                         matrix->col_indices[j], j);
        return false;
      }

      // Check sorted within row
      if (j > row_start &&
          matrix->col_indices[j] <= matrix->col_indices[j - 1]) {
        MATGEN_LOG_ERROR("Column indices not sorted in row %zu", i);
        return false;
      }
    }
  }

  MATGEN_LOG_DEBUG("CSR matrix validation passed");
  return true;
}
