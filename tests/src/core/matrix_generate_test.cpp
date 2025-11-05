#include "matgen/core/matrix_generate.h"

#include <gtest/gtest.h>

#include <cmath>
#include <set>

#include "matgen/core/coo_matrix.h"

// Test: Config initialization
TEST(MatrixGenerate, ConfigInit) {
  matgen_random_config_t config;
  matgen_random_config_init(&config, 100, 50, 1000);

  EXPECT_EQ(config.rows, 100);
  EXPECT_EQ(config.cols, 50);
  EXPECT_EQ(config.nnz, 1000);
  EXPECT_EQ(config.distribution, MATGEN_DIST_UNIFORM);
  EXPECT_DOUBLE_EQ(config.min_value, 0.0);
  EXPECT_DOUBLE_EQ(config.max_value, 1.0);
  EXPECT_EQ(config.allow_duplicates, false);
  EXPECT_EQ(config.sorted, true);
}

// Test: Basic random matrix generation
TEST(MatrixGenerate, BasicRandom) {
  matgen_random_config_t config;
  matgen_random_config_init(&config, 10, 10, 20);
  config.seed = 12345;

  matgen_coo_matrix_t* matrix = matgen_random_coo_create(&config);
  ASSERT_NE(matrix, nullptr);

  EXPECT_EQ(matrix->rows, 10);
  EXPECT_EQ(matrix->cols, 10);
  EXPECT_EQ(matrix->nnz, 20);

  // Check values are in range [0, 1]
  for (size_t i = 0; i < matrix->nnz; i++) {
    EXPECT_GE(matrix->values[i], 0.0);
    EXPECT_LE(matrix->values[i], 1.0);
  }

  matgen_coo_destroy(matrix);
}

// Test: Reproducibility with same seed
TEST(MatrixGenerate, Reproducibility) {
  matgen_random_config_t config;
  matgen_random_config_init(&config, 10, 10, 15);
  config.seed = 42;

  matgen_coo_matrix_t* matrix1 = matgen_random_coo_create(&config);
  matgen_coo_matrix_t* matrix2 = matgen_random_coo_create(&config);

  ASSERT_NE(matrix1, nullptr);
  ASSERT_NE(matrix2, nullptr);

  EXPECT_EQ(matrix1->nnz, matrix2->nnz);

  // Should generate identical matrices
  for (size_t i = 0; i < matrix1->nnz; i++) {
    EXPECT_EQ(matrix1->row_indices[i], matrix2->row_indices[i]);
    EXPECT_EQ(matrix1->col_indices[i], matrix2->col_indices[i]);
    EXPECT_DOUBLE_EQ(matrix1->values[i], matrix2->values[i]);
  }

  matgen_coo_destroy(matrix1);
  matgen_coo_destroy(matrix2);
}

// Test: Custom value range (uniform)
TEST(MatrixGenerate, CustomRange) {
  matgen_random_config_t config;
  matgen_random_config_init(&config, 20, 20, 50);
  config.min_value = -10.0;
  config.max_value = 10.0;
  config.seed = 999;

  matgen_coo_matrix_t* matrix = matgen_random_coo_create(&config);
  ASSERT_NE(matrix, nullptr);

  // Check all values are in range [-10, 10]
  for (size_t i = 0; i < matrix->nnz; i++) {
    EXPECT_GE(matrix->values[i], -10.0);
    EXPECT_LE(matrix->values[i], 10.0);
  }

  matgen_coo_destroy(matrix);
}

// Test: Normal distribution
TEST(MatrixGenerate, NormalDistribution) {
  matgen_random_config_t config;
  matgen_random_config_init(&config, 50, 50, 500);
  config.distribution = MATGEN_DIST_NORMAL;
  config.mean = 0.0;
  config.stddev = 1.0;
  config.seed = 777;

  matgen_coo_matrix_t* matrix = matgen_random_coo_create(&config);
  ASSERT_NE(matrix, nullptr);

  // Calculate sample mean and stddev
  double sum = 0.0;
  for (size_t i = 0; i < matrix->nnz; i++) {
    sum += matrix->values[i];
  }
  double mean = sum / static_cast<double>(matrix->nnz);

  double var_sum = 0.0;
  for (size_t i = 0; i < matrix->nnz; i++) {
    double diff = matrix->values[i] - mean;
    var_sum += diff * diff;
  }
  double stddev = sqrt(var_sum / static_cast<double>(matrix->nnz));

  // Should be close to N(0, 1) - allow some tolerance
  EXPECT_NEAR(mean, 0.0, 0.2);
  EXPECT_NEAR(stddev, 1.0, 0.2);

  matgen_coo_destroy(matrix);
}

// Test: Constant values
TEST(MatrixGenerate, ConstantDistribution) {
  matgen_random_config_t config;
  matgen_random_config_init(&config, 10, 10, 30);
  config.distribution = MATGEN_DIST_CONSTANT;
  config.constant_value = 42.0;
  config.seed = 111;

  matgen_coo_matrix_t* matrix = matgen_random_coo_create(&config);
  ASSERT_NE(matrix, nullptr);

  // All values should be exactly 42.0
  for (size_t i = 0; i < matrix->nnz; i++) {
    EXPECT_DOUBLE_EQ(matrix->values[i], 42.0);
  }

  matgen_coo_destroy(matrix);
}

// Test: No duplicates (default)
TEST(MatrixGenerate, NoDuplicates) {
  matgen_random_config_t config;
  matgen_random_config_init(&config, 10, 10, 30);
  config.seed = 555;

  matgen_coo_matrix_t* matrix = matgen_random_coo_create(&config);
  ASSERT_NE(matrix, nullptr);

  // Check for duplicate positions
  std::set<std::pair<size_t, size_t>> positions;
  for (size_t i = 0; i < matrix->nnz; i++) {
    auto pos = std::make_pair(matrix->row_indices[i], matrix->col_indices[i]);
    EXPECT_EQ(positions.count(pos), 0)
        << "Duplicate position found: (" << pos.first << ", " << pos.second
        << ")";
    positions.insert(pos);
  }

  EXPECT_EQ(positions.size(), matrix->nnz);

  matgen_coo_destroy(matrix);
}

// Test: Allow duplicates
TEST(MatrixGenerate, AllowDuplicates) {
  matgen_random_config_t config;
  matgen_random_config_init(&config, 5, 5, 30);
  config.allow_duplicates = true;
  config.seed = 333;

  matgen_coo_matrix_t* matrix = matgen_random_coo_create(&config);
  ASSERT_NE(matrix, nullptr);

  EXPECT_EQ(matrix->nnz, 30);

  matgen_coo_destroy(matrix);
}

// Test: Density parameter
TEST(MatrixGenerate, DensityParameter) {
  matgen_random_config_t config;
  matgen_random_config_init(&config, 100, 100, 0);  // nnz=0 initially
  config.density = 0.1;                             // 10% density
  config.seed = 888;

  matgen_coo_matrix_t* matrix = matgen_random_coo_create(&config);
  ASSERT_NE(matrix, nullptr);

  // Should have ~1000 entries (10% of 10000)
  EXPECT_EQ(matrix->nnz, 1000);

  matgen_coo_destroy(matrix);
}

// Test: Sorted output
TEST(MatrixGenerate, SortedOutput) {
  matgen_random_config_t config;
  matgen_random_config_init(&config, 20, 20, 50);
  config.sorted = true;
  config.seed = 222;

  matgen_coo_matrix_t* matrix = matgen_random_coo_create(&config);
  ASSERT_NE(matrix, nullptr);

  EXPECT_TRUE(matrix->is_sorted);

  // Verify sorted order
  for (size_t i = 1; i < matrix->nnz; i++) {
    size_t prev_row = matrix->row_indices[i - 1];
    size_t prev_col = matrix->col_indices[i - 1];
    size_t curr_row = matrix->row_indices[i];
    size_t curr_col = matrix->col_indices[i];

    bool is_sorted =
        (curr_row > prev_row) || (curr_row == prev_row && curr_col >= prev_col);
    EXPECT_TRUE(is_sorted);
  }

  matgen_coo_destroy(matrix);
}

// Test: Diagonal matrix
TEST(MatrixGenerate, DiagonalMatrix) {
  matgen_coo_matrix_t* matrix =
      matgen_random_diagonal(10, 10, MATGEN_DIST_UNIFORM, 1.0, 5.0, 12345);

  ASSERT_NE(matrix, nullptr);
  EXPECT_EQ(matrix->rows, 10);
  EXPECT_EQ(matrix->cols, 10);
  EXPECT_EQ(matrix->nnz, 10);

  // All entries should be on diagonal
  for (size_t i = 0; i < matrix->nnz; i++) {
    EXPECT_EQ(matrix->row_indices[i], matrix->col_indices[i]);
    EXPECT_GE(matrix->values[i], 1.0);
    EXPECT_LE(matrix->values[i], 5.0);
  }

  matgen_coo_destroy(matrix);
}

// Test: Rectangular diagonal matrix
TEST(MatrixGenerate, RectangularDiagonal) {
  matgen_coo_matrix_t* matrix =
      matgen_random_diagonal(15, 10, MATGEN_DIST_CONSTANT, 0.0, 1.0, 99);

  ASSERT_NE(matrix, nullptr);
  EXPECT_EQ(matrix->rows, 15);
  EXPECT_EQ(matrix->cols, 10);
  EXPECT_EQ(matrix->nnz, 10);  // min(15, 10) = 10

  matgen_coo_destroy(matrix);
}

// Test: Tridiagonal matrix
TEST(MatrixGenerate, TridiagonalMatrix) {
  matgen_coo_matrix_t* matrix =
      matgen_random_tridiagonal(5, MATGEN_DIST_UNIFORM, 0.0, 1.0, 42);

  ASSERT_NE(matrix, nullptr);
  EXPECT_EQ(matrix->rows, 5);
  EXPECT_EQ(matrix->cols, 5);
  EXPECT_EQ(matrix->nnz, 13);  // 5 + 4 + 4 = 13

  // Verify structure: all entries should be on 3 diagonals
  for (size_t i = 0; i < matrix->nnz; i++) {
    size_t row = matrix->row_indices[i];
    size_t col = matrix->col_indices[i];
    int diff = (int)col - (int)row;

    // Should be on main, upper, or lower diagonal
    EXPECT_TRUE(diff >= -1 && diff <= 1);
  }

  matgen_coo_destroy(matrix);
}

// Test: Error handling - impossible nnz without duplicates
TEST(MatrixGenerate, ImpossibleNNZ) {
  matgen_random_config_t config;
  matgen_random_config_init(&config, 5, 5,
                            30);  // Want 30 entries in 5x5 = 25 max
  config.allow_duplicates = false;
  config.seed = 1;

  matgen_coo_matrix_t* matrix = matgen_random_coo_create(&config);
  EXPECT_EQ(matrix, nullptr);  // Should fail
}

// Test: Empty matrix
TEST(MatrixGenerate, EmptyMatrix) {
  matgen_random_config_t config;
  matgen_random_config_init(&config, 10, 10, 0);
  config.seed = 1;

  matgen_coo_matrix_t* matrix = matgen_random_coo_create(&config);
  ASSERT_NE(matrix, nullptr);
  EXPECT_EQ(matrix->nnz, 0);

  matgen_coo_destroy(matrix);
}
