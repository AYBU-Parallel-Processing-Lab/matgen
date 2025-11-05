#include "matgen/core/coo_matrix.h"

#include <gtest/gtest.h>

#include "matgen/util/log.h"

class COOMatrixBasicTest : public ::testing::Test {
 protected:
  void SetUp() override { matgen_log_set_level(MATGEN_LOG_LEVEL_ERROR); }

  void TearDown() override {
    if (matrix != nullptr) {
      matgen_coo_destroy(matrix);
      matrix = nullptr;
    }
  }

  matgen_coo_matrix_t* matrix{nullptr};  // NOLINT
};

// =============================================================================
// Creation and Destruction Tests
// =============================================================================
TEST_F(COOMatrixBasicTest, CreateAndDestroy) {
  matrix = matgen_coo_create(10, 10, 5);
  ASSERT_NE(matrix, nullptr);
  EXPECT_EQ(matrix->rows, 10);
  EXPECT_EQ(matrix->cols, 10);
  EXPECT_EQ(matrix->nnz, 0);
  EXPECT_EQ(matrix->capacity, 5);
  EXPECT_TRUE(matrix->is_sorted);
}

TEST_F(COOMatrixBasicTest, InvalidDimensions) {
  matrix = matgen_coo_create(0, 10, 5);
  EXPECT_EQ(matrix, nullptr);

  matrix = matgen_coo_create(10, 0, 5);
  EXPECT_EQ(matrix, nullptr);
}

TEST_F(COOMatrixBasicTest, DefaultCapacity) {
  matrix = matgen_coo_create(5, 5, 0);
  ASSERT_NE(matrix, nullptr);
  EXPECT_GT(matrix->capacity, 0);
}

// =============================================================================
// Adding Entries Tests
// =============================================================================

TEST_F(COOMatrixBasicTest, AddSingleEntry) {
  matrix = matgen_coo_create(5, 5, 2);
  ASSERT_NE(matrix, nullptr);

  int result = matgen_coo_add_entry(matrix, 0, 0, 1.0);
  EXPECT_EQ(result, 0);
  EXPECT_EQ(matrix->nnz, 1);
  EXPECT_EQ(matrix->row_indices[0], 0);
  EXPECT_EQ(matrix->col_indices[0], 0);
  EXPECT_DOUBLE_EQ(matrix->values[0], 1.0);
  EXPECT_FALSE(matrix->is_sorted);  // Adding marks as unsorted
}

TEST_F(COOMatrixBasicTest, AddMultipleEntries) {
  matrix = matgen_coo_create(5, 5, 3);
  ASSERT_NE(matrix, nullptr);

  EXPECT_EQ(matgen_coo_add_entry(matrix, 0, 0, 1.0), 0);
  EXPECT_EQ(matgen_coo_add_entry(matrix, 2, 3, 2.5), 0);
  EXPECT_EQ(matgen_coo_add_entry(matrix, 4, 1, -3.7), 0);

  EXPECT_EQ(matrix->nnz, 3);
}

TEST_F(COOMatrixBasicTest, AutoGrow) {
  matrix = matgen_coo_create(10, 10, 2);
  ASSERT_NE(matrix, nullptr);

  // Add more than initial capacity
  for (size_t i = 0; i < 10; i++) {
    EXPECT_EQ(matgen_coo_add_entry(matrix, i, i, (double)i), 0);
  }

  EXPECT_EQ(matrix->nnz, 10);
  EXPECT_GE(matrix->capacity, 10);
}

TEST_F(COOMatrixBasicTest, OutOfBounds) {
  matrix = matgen_coo_create(5, 5, 5);
  ASSERT_NE(matrix, nullptr);

  EXPECT_NE(matgen_coo_add_entry(matrix, 5, 0, 1.0), 0);
  EXPECT_NE(matgen_coo_add_entry(matrix, 0, 5, 1.0), 0);
  EXPECT_NE(matgen_coo_add_entry(matrix, 10, 10, 1.0), 0);
}

// =============================================================================
// Utility Tests
// =============================================================================

TEST_F(COOMatrixBasicTest, PrintInfo) {
  matrix = matgen_coo_create(100, 50, 10);
  ASSERT_NE(matrix, nullptr);

  matgen_coo_add_entry(matrix, 0, 0, 1.0);
  matgen_coo_add_entry(matrix, 10, 20, 2.5);

  // Should not crash
  matgen_coo_print_info(matrix, stdout);
}

TEST_F(COOMatrixBasicTest, MemoryUsage) {
  matrix = matgen_coo_create(10, 10, 100);
  ASSERT_NE(matrix, nullptr);

  size_t memory = matgen_coo_memory_usage(matrix);
  EXPECT_GT(memory, 0);

  // Should at least include the arrays
  size_t expected = 100 * (sizeof(size_t) + sizeof(size_t) + sizeof(double));
  EXPECT_GE(memory, expected);
}
