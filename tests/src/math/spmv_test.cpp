#include <gtest/gtest.h>
#include <matgen/core/conversion.h>
#include <matgen/core/coo_matrix.h>
#include <matgen/core/csr_matrix.h>
#include <matgen/math/spmv.h>

// =============================================================================
// CSR SpMV Tests
// =============================================================================

TEST(SpMVTest, CSRBasicMultiplication) {
  // Create a simple 3x3 matrix:
  // [1  0  2]
  // [0  3  0]
  // [4  0  5]
  matgen_csr_matrix_t* A = matgen_csr_create(3, 3, 5);
  ASSERT_NE(A, nullptr);

  // Fill CSR structure manually
  A->row_ptr[0] = 0;
  A->row_ptr[1] = 2;
  A->row_ptr[2] = 3;
  A->row_ptr[3] = 5;

  A->col_indices[0] = 0;
  A->values[0] = 1.0;
  A->col_indices[1] = 2;
  A->values[1] = 2.0;
  A->col_indices[2] = 1;
  A->values[2] = 3.0;
  A->col_indices[3] = 0;
  A->values[3] = 4.0;
  A->col_indices[4] = 2;
  A->values[4] = 5.0;

  matgen_value_t x[3] = {1.0, 2.0, 3.0};
  matgen_value_t y[3];

  // Compute y = A * x
  matgen_error_t err = matgen_csr_spmv(A, x, y);
  EXPECT_EQ(err, MATGEN_SUCCESS);

  // Expected: [1*1 + 2*3, 3*2, 4*1 + 5*3] = [7, 6, 19]
  EXPECT_DOUBLE_EQ(y[0], 7.0);
  EXPECT_DOUBLE_EQ(y[1], 6.0);
  EXPECT_DOUBLE_EQ(y[2], 19.0);

  matgen_csr_destroy(A);
}

TEST(SpMVTest, CSREmptyMatrix) {
  matgen_csr_matrix_t* A = matgen_csr_create(3, 3, 0);
  ASSERT_NE(A, nullptr);

  // All rows are empty
  for (size_t i = 0; i <= 3; i++) {
    A->row_ptr[i] = 0;
  }

  matgen_value_t x[3] = {1.0, 2.0, 3.0};
  matgen_value_t y[3];

  matgen_error_t err = matgen_csr_spmv(A, x, y);
  EXPECT_EQ(err, MATGEN_SUCCESS);

  // All output should be zero
  EXPECT_DOUBLE_EQ(y[0], 0.0);
  EXPECT_DOUBLE_EQ(y[1], 0.0);
  EXPECT_DOUBLE_EQ(y[2], 0.0);

  matgen_csr_destroy(A);
}

TEST(SpMVTest, CSRSingleRow) {
  // 1x3 matrix: [2.0  3.0  4.0]
  matgen_csr_matrix_t* A = matgen_csr_create(1, 3, 3);
  ASSERT_NE(A, nullptr);

  A->row_ptr[0] = 0;
  A->row_ptr[1] = 3;

  A->col_indices[0] = 0;
  A->values[0] = 2.0;
  A->col_indices[1] = 1;
  A->values[1] = 3.0;
  A->col_indices[2] = 2;
  A->values[2] = 4.0;

  matgen_value_t x[3] = {1.0, 2.0, 3.0};
  matgen_value_t y[1];

  matgen_error_t err = matgen_csr_spmv(A, x, y);
  EXPECT_EQ(err, MATGEN_SUCCESS);

  // Expected: 2*1 + 3*2 + 4*3 = 2 + 6 + 12 = 20
  EXPECT_DOUBLE_EQ(y[0], 20.0);

  matgen_csr_destroy(A);
}

TEST(SpMVTest, CSRNullPointers) {
  matgen_csr_matrix_t* A = matgen_csr_create(3, 3, 5);
  matgen_value_t x[3] = {1.0, 2.0, 3.0};
  matgen_value_t y[3];

  EXPECT_EQ(matgen_csr_spmv(nullptr, x, y), MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(matgen_csr_spmv(A, nullptr, y), MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(matgen_csr_spmv(A, x, nullptr), MATGEN_ERROR_INVALID_ARGUMENT);

  matgen_csr_destroy(A);
}

// =============================================================================
// CSR Transpose SpMV Tests
// =============================================================================

TEST(SpMVTest, CSRTransposeBasic) {
  // Create 3x2 matrix:
  // [1  2]
  // [0  3]
  // [4  0]
  matgen_csr_matrix_t* A = matgen_csr_create(3, 2, 4);
  ASSERT_NE(A, nullptr);

  A->row_ptr[0] = 0;
  A->row_ptr[1] = 2;
  A->row_ptr[2] = 3;
  A->row_ptr[3] = 4;

  A->col_indices[0] = 0;
  A->values[0] = 1.0;
  A->col_indices[1] = 1;
  A->values[1] = 2.0;
  A->col_indices[2] = 1;
  A->values[2] = 3.0;
  A->col_indices[3] = 0;
  A->values[3] = 4.0;

  matgen_value_t x[3] = {1.0, 2.0, 3.0};
  matgen_value_t y[2];

  // Compute y = A^T * x
  matgen_error_t err = matgen_csr_spmv_transpose(A, x, y);
  EXPECT_EQ(err, MATGEN_SUCCESS);

  // A^T is 2x3:
  // [1  0  4]
  // [2  3  0]
  // Expected: [1*1 + 4*3, 2*1 + 3*2] = [13, 8]
  EXPECT_DOUBLE_EQ(y[0], 13.0);
  EXPECT_DOUBLE_EQ(y[1], 8.0);

  matgen_csr_destroy(A);
}

TEST(SpMVTest, CSRTransposeEmpty) {
  matgen_csr_matrix_t* A = matgen_csr_create(3, 2, 0);
  ASSERT_NE(A, nullptr);

  for (size_t i = 0; i <= 3; i++) {
    A->row_ptr[i] = 0;
  }

  matgen_value_t x[3] = {1.0, 2.0, 3.0};
  matgen_value_t y[2];

  matgen_error_t err = matgen_csr_spmv_transpose(A, x, y);
  EXPECT_EQ(err, MATGEN_SUCCESS);

  EXPECT_DOUBLE_EQ(y[0], 0.0);
  EXPECT_DOUBLE_EQ(y[1], 0.0);

  matgen_csr_destroy(A);
}

TEST(SpMVTest, CSRTransposeNullPointers) {
  matgen_csr_matrix_t* A = matgen_csr_create(3, 2, 4);
  matgen_value_t x[3] = {1.0, 2.0, 3.0};
  matgen_value_t y[2];

  EXPECT_EQ(matgen_csr_spmv_transpose(nullptr, x, y),
            MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(matgen_csr_spmv_transpose(A, nullptr, y),
            MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(matgen_csr_spmv_transpose(A, x, nullptr),
            MATGEN_ERROR_INVALID_ARGUMENT);

  matgen_csr_destroy(A);
}

// =============================================================================
// COO SpMV Tests
// =============================================================================

TEST(SpMVTest, COOBasicMultiplication) {
  // Create same 3x3 matrix as CSR test
  matgen_coo_matrix_t* A = matgen_coo_create(3, 3, 5);
  ASSERT_NE(A, nullptr);

  matgen_coo_add_entry(A, 0, 0, 1.0);
  matgen_coo_add_entry(A, 0, 2, 2.0);
  matgen_coo_add_entry(A, 1, 1, 3.0);
  matgen_coo_add_entry(A, 2, 0, 4.0);
  matgen_coo_add_entry(A, 2, 2, 5.0);

  matgen_value_t x[3] = {1.0, 2.0, 3.0};
  matgen_value_t y[3] = {0.0, 0.0, 0.0};  // Must be zero-initialized

  matgen_error_t err = matgen_coo_spmv(A, x, y);
  EXPECT_EQ(err, MATGEN_SUCCESS);

  EXPECT_DOUBLE_EQ(y[0], 7.0);
  EXPECT_DOUBLE_EQ(y[1], 6.0);
  EXPECT_DOUBLE_EQ(y[2], 19.0);

  matgen_coo_destroy(A);
}

TEST(SpMVTest, COOWithDuplicates) {
  // Matrix with duplicate entries (should accumulate)
  matgen_coo_matrix_t* A = matgen_coo_create(2, 2, 4);
  ASSERT_NE(A, nullptr);

  // Add (0,0) twice
  matgen_coo_add_entry(A, 0, 0, 2.0);
  matgen_coo_add_entry(A, 0, 0, 3.0);  // Should accumulate to 5.0
  matgen_coo_add_entry(A, 1, 1, 4.0);

  matgen_value_t x[2] = {1.0, 2.0};
  matgen_value_t y[2] = {0.0, 0.0};

  matgen_error_t err = matgen_coo_spmv(A, x, y);
  EXPECT_EQ(err, MATGEN_SUCCESS);

  // Expected: [5*1, 4*2] = [5, 8]
  EXPECT_DOUBLE_EQ(y[0], 5.0);
  EXPECT_DOUBLE_EQ(y[1], 8.0);

  matgen_coo_destroy(A);
}

TEST(SpMVTest, COOEmpty) {
  matgen_coo_matrix_t* A = matgen_coo_create(3, 3, 0);
  ASSERT_NE(A, nullptr);

  matgen_value_t x[3] = {1.0, 2.0, 3.0};
  matgen_value_t y[3] = {0.0, 0.0, 0.0};

  matgen_error_t err = matgen_coo_spmv(A, x, y);
  EXPECT_EQ(err, MATGEN_SUCCESS);

  EXPECT_DOUBLE_EQ(y[0], 0.0);
  EXPECT_DOUBLE_EQ(y[1], 0.0);
  EXPECT_DOUBLE_EQ(y[2], 0.0);

  matgen_coo_destroy(A);
}

TEST(SpMVTest, COONullPointers) {
  matgen_coo_matrix_t* A = matgen_coo_create(3, 3, 5);
  matgen_value_t x[3] = {1.0, 2.0, 3.0};
  matgen_value_t y[3];

  EXPECT_EQ(matgen_coo_spmv(nullptr, x, y), MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(matgen_coo_spmv(A, nullptr, y), MATGEN_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(matgen_coo_spmv(A, x, nullptr), MATGEN_ERROR_INVALID_ARGUMENT);

  matgen_coo_destroy(A);
}

// =============================================================================
// Integration Test: COO -> CSR conversion and SpMV comparison
// =============================================================================

TEST(SpMVTest, COOvsCSRConsistency) {
  // Create a matrix in COO format
  matgen_coo_matrix_t* coo = matgen_coo_create(4, 4, 6);
  ASSERT_NE(coo, nullptr);

  matgen_coo_add_entry(coo, 0, 0, 1.5);
  matgen_coo_add_entry(coo, 0, 3, 2.5);
  matgen_coo_add_entry(coo, 1, 1, 3.5);
  matgen_coo_add_entry(coo, 2, 2, 4.5);
  matgen_coo_add_entry(coo, 3, 0, 5.5);
  matgen_coo_add_entry(coo, 3, 3, 6.5);

  // Convert to CSR
  matgen_csr_matrix_t* csr = matgen_coo_to_csr(coo);
  ASSERT_NE(csr, nullptr);

  matgen_value_t x[4] = {1.0, 2.0, 3.0, 4.0};
  matgen_value_t y_coo[4] = {0.0, 0.0, 0.0, 0.0};
  matgen_value_t y_csr[4];

  // Compute with both formats
  EXPECT_EQ(matgen_coo_spmv(coo, x, y_coo), MATGEN_SUCCESS);
  EXPECT_EQ(matgen_csr_spmv(csr, x, y_csr), MATGEN_SUCCESS);

  // Results should match
  for (int i = 0; i < 4; i++) {
    EXPECT_DOUBLE_EQ(y_coo[i], y_csr[i]) << "Mismatch at index " << i;
  }

  matgen_coo_destroy(coo);
  matgen_csr_destroy(csr);
}
