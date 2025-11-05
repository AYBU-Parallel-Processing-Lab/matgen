#include "matgen/core/csr_matrix.h"

#include <gtest/gtest.h>

#include <cmath>

#include "matgen/util/log.h"

class CSRMatrixTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Suppress log output during tests
    matgen_log_set_level(MATGEN_LOG_LEVEL_ERROR);
  }

  void TearDown() override {
    if (matrix != nullptr) {
      matgen_csr_destroy(matrix);
      matrix = nullptr;
    }
  }

  // Helper to create a simple CSR matrix manually
  // Matrix: [1 0 2]
  //         [0 3 0]
  //         [4 0 5]
  matgen_csr_matrix_t* CreateTestMatrix() {
    ((void)this);

    matgen_csr_matrix_t* m = matgen_csr_create(3, 3, 5);
    if (m == nullptr) {
      return nullptr;
    }

    // Row 0: [1, 2] at cols [0, 2]
    m->row_ptr[0] = 0;
    m->row_ptr[1] = 2;
    m->col_indices[0] = 0;
    m->values[0] = 1.0;
    m->col_indices[1] = 2;
    m->values[1] = 2.0;

    // Row 1: [3] at col [1]
    m->row_ptr[2] = 3;
    m->col_indices[2] = 1;
    m->values[2] = 3.0;

    // Row 2: [4, 5] at cols [0, 2]
    m->row_ptr[3] = 5;
    m->col_indices[3] = 0;
    m->values[3] = 4.0;
    m->col_indices[4] = 2;
    m->values[4] = 5.0;

    return m;
  }

  matgen_csr_matrix_t* matrix{nullptr};  // NOLINT
};

// =============================================================================
// Creation and Destruction Tests
// =============================================================================

TEST_F(CSRMatrixTest, CreateAndDestroy) {
  matrix = matgen_csr_create(10, 10, 5);
  ASSERT_NE(matrix, nullptr);
  EXPECT_EQ(matrix->rows, 10);
  EXPECT_EQ(matrix->cols, 10);
  EXPECT_EQ(matrix->nnz, 5);
  EXPECT_NE(matrix->row_ptr, nullptr);
  EXPECT_NE(matrix->col_indices, nullptr);
  EXPECT_NE(matrix->values, nullptr);
}

TEST_F(CSRMatrixTest, InvalidDimensions) {
  matrix = matgen_csr_create(0, 10, 5);
  EXPECT_EQ(matrix, nullptr);

  matrix = matgen_csr_create(10, 0, 5);
  EXPECT_EQ(matrix, nullptr);
}

TEST_F(CSRMatrixTest, ZeroNonzeros) {
  matrix = matgen_csr_create(5, 5, 0);
  ASSERT_NE(matrix, nullptr);
  EXPECT_EQ(matrix->nnz, 0);

  // All rows should be empty
  for (size_t i = 0; i <= matrix->rows; i++) {
    EXPECT_EQ(matrix->row_ptr[i], 0);
  }
}

TEST_F(CSRMatrixTest, InitialState) {
  matrix = matgen_csr_create(5, 5, 10);
  ASSERT_NE(matrix, nullptr);

  // row_ptr should be all zeros (calloc)
  for (size_t i = 0; i <= matrix->rows; i++) {
    EXPECT_EQ(matrix->row_ptr[i], 0);
  }
}

// =============================================================================
// Matrix Access Tests
// =============================================================================

TEST_F(CSRMatrixTest, GetExistingValues) {
  matrix = CreateTestMatrix();
  ASSERT_NE(matrix, nullptr);

  EXPECT_DOUBLE_EQ(matgen_csr_get(matrix, 0, 0), 1.0);
  EXPECT_DOUBLE_EQ(matgen_csr_get(matrix, 0, 2), 2.0);
  EXPECT_DOUBLE_EQ(matgen_csr_get(matrix, 1, 1), 3.0);
  EXPECT_DOUBLE_EQ(matgen_csr_get(matrix, 2, 0), 4.0);
  EXPECT_DOUBLE_EQ(matgen_csr_get(matrix, 2, 2), 5.0);
}

TEST_F(CSRMatrixTest, GetZeroValues) {
  matrix = CreateTestMatrix();
  ASSERT_NE(matrix, nullptr);

  // These positions should be zero (not stored)
  EXPECT_DOUBLE_EQ(matgen_csr_get(matrix, 0, 1), 0.0);
  EXPECT_DOUBLE_EQ(matgen_csr_get(matrix, 1, 0), 0.0);
  EXPECT_DOUBLE_EQ(matgen_csr_get(matrix, 1, 2), 0.0);
  EXPECT_DOUBLE_EQ(matgen_csr_get(matrix, 2, 1), 0.0);
}

TEST_F(CSRMatrixTest, GetOutOfBounds) {
  matrix = CreateTestMatrix();
  ASSERT_NE(matrix, nullptr);

  EXPECT_DOUBLE_EQ(matgen_csr_get(matrix, 5, 0), 0.0);
  EXPECT_DOUBLE_EQ(matgen_csr_get(matrix, 0, 5), 0.0);
}

TEST_F(CSRMatrixTest, GetFromNullMatrix) {
  EXPECT_DOUBLE_EQ(matgen_csr_get(nullptr, 0, 0), 0.0);
}

// =============================================================================
// Row Operations Tests
// =============================================================================

TEST_F(CSRMatrixTest, RowNonzeroCount) {
  matrix = CreateTestMatrix();
  ASSERT_NE(matrix, nullptr);

  EXPECT_EQ(matgen_csr_row_nnz(matrix, 0), 2);
  EXPECT_EQ(matgen_csr_row_nnz(matrix, 1), 1);
  EXPECT_EQ(matgen_csr_row_nnz(matrix, 2), 2);
}

TEST_F(CSRMatrixTest, EmptyRow) {
  // Create matrix with empty row
  matrix = matgen_csr_create(3, 3, 2);
  ASSERT_NE(matrix, nullptr);

  // Row 0: [1] at col [0]
  matrix->row_ptr[0] = 0;
  matrix->row_ptr[1] = 1;
  matrix->col_indices[0] = 0;
  matrix->values[0] = 1.0;

  // Row 1: empty
  matrix->row_ptr[2] = 1;

  // Row 2: [2] at col [1]
  matrix->row_ptr[3] = 2;
  matrix->col_indices[1] = 1;
  matrix->values[1] = 2.0;

  EXPECT_EQ(matgen_csr_row_nnz(matrix, 0), 1);
  EXPECT_EQ(matgen_csr_row_nnz(matrix, 1), 0);  // Empty row
  EXPECT_EQ(matgen_csr_row_nnz(matrix, 2), 1);
}

// =============================================================================
// Validation Tests
// =============================================================================

TEST_F(CSRMatrixTest, ValidateCorrectMatrix) {
  matrix = CreateTestMatrix();
  ASSERT_NE(matrix, nullptr);

  EXPECT_TRUE(matgen_csr_validate(matrix));
}

TEST_F(CSRMatrixTest, ValidateNullMatrix) {
  EXPECT_FALSE(matgen_csr_validate(nullptr));
}

TEST_F(CSRMatrixTest, ValidateInvalidRowPtr) {
  matrix = CreateTestMatrix();
  ASSERT_NE(matrix, nullptr);

  // Make row_ptr non-monotonic
  size_t temp = matrix->row_ptr[1];
  matrix->row_ptr[1] = matrix->row_ptr[2];
  matrix->row_ptr[2] = temp;

  EXPECT_FALSE(matgen_csr_validate(matrix));
}

TEST_F(CSRMatrixTest, ValidateWrongFinalRowPtr) {
  matrix = CreateTestMatrix();
  ASSERT_NE(matrix, nullptr);

  // Make last row_ptr incorrect
  matrix->row_ptr[matrix->rows] = matrix->nnz + 1;

  EXPECT_FALSE(matgen_csr_validate(matrix));
}

TEST_F(CSRMatrixTest, ValidateColumnOutOfRange) {
  matrix = CreateTestMatrix();
  ASSERT_NE(matrix, nullptr);

  // Set column index out of range
  matrix->col_indices[0] = matrix->cols + 1;

  EXPECT_FALSE(matgen_csr_validate(matrix));
}

TEST_F(CSRMatrixTest, ValidateUnsortedColumns) {
  matrix = CreateTestMatrix();
  ASSERT_NE(matrix, nullptr);

  // Make columns in row 0 unsorted (swap 0 and 2)
  size_t temp = matrix->col_indices[0];
  matrix->col_indices[0] = matrix->col_indices[1];
  matrix->col_indices[1] = temp;

  EXPECT_FALSE(matgen_csr_validate(matrix));
}

// =============================================================================
// Utility Tests
// =============================================================================

TEST_F(CSRMatrixTest, PrintInfo) {
  matrix = CreateTestMatrix();
  ASSERT_NE(matrix, nullptr);

  // Should not crash
  matgen_csr_print_info(matrix, stdout);
}

TEST_F(CSRMatrixTest, MemoryUsage) {
  matrix = matgen_csr_create(10, 10, 20);
  ASSERT_NE(matrix, nullptr);

  size_t memory = matgen_csr_memory_usage(matrix);
  EXPECT_GT(memory, 0);

  // Should at least include the arrays
  size_t expected = ((matrix->rows + 1) * sizeof(size_t)) +  // row_ptr
                    (matrix->nnz * sizeof(size_t)) +         // col_indices
                    (matrix->nnz * sizeof(double));          // values
  EXPECT_GE(memory, expected);
}

TEST_F(CSRMatrixTest, MemoryUsageNull) {
  EXPECT_EQ(matgen_csr_memory_usage(nullptr), 0);
}

// =============================================================================
// Binary Search Tests (implicit in matgen_csr_get)
// =============================================================================

TEST_F(CSRMatrixTest, BinarySearchFirstElement) {
  matrix = CreateTestMatrix();
  ASSERT_NE(matrix, nullptr);

  // First element in row
  EXPECT_DOUBLE_EQ(matgen_csr_get(matrix, 0, 0), 1.0);
}

TEST_F(CSRMatrixTest, BinarySearchLastElement) {
  matrix = CreateTestMatrix();
  ASSERT_NE(matrix, nullptr);

  // Last element in row
  EXPECT_DOUBLE_EQ(matgen_csr_get(matrix, 0, 2), 2.0);
}

TEST_F(CSRMatrixTest, BinarySearchMiddleElement) {
  // Create a row with multiple elements
  matrix = matgen_csr_create(1, 10, 5);
  ASSERT_NE(matrix, nullptr);

  matrix->row_ptr[0] = 0;
  matrix->row_ptr[1] = 5;

  for (size_t i = 0; i < 5; i++) {
    matrix->col_indices[i] = i * 2;  // 0, 2, 4, 6, 8
    matrix->values[i] = (double)(i + 1);
  }

  EXPECT_DOUBLE_EQ(matgen_csr_get(matrix, 0, 4), 3.0);  // Middle element
}

TEST_F(CSRMatrixTest, BinarySearchNotFound) {
  matrix = CreateTestMatrix();
  ASSERT_NE(matrix, nullptr);

  // Search for non-existent element between stored ones
  EXPECT_DOUBLE_EQ(matgen_csr_get(matrix, 0, 1), 0.0);
}
