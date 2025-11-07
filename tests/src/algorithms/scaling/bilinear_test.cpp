#include <gtest/gtest.h>
#include <matgen/algorithms/scaling/bilinear.h>
#include <matgen/core/conversion.h>
#include <matgen/core/coo_matrix.h>
#include <matgen/core/csr_matrix.h>

TEST(BilinearTest, IdentityScaling) {
  matgen_coo_matrix_t* coo = matgen_coo_create(3, 3, 3);
  matgen_coo_add_entry(coo, 0, 0, 1.0);
  matgen_coo_add_entry(coo, 1, 1, 2.0);
  matgen_coo_add_entry(coo, 2, 2, 3.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_bilinear(source, 3, 3, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 3);
  EXPECT_EQ(result->cols, 3);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, ScaleUpWithFractionalCoordinates) {
  // Scale 2x2 -> 3x3 creates fractional coordinates (scale factor = 1.5)
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 2, 1);
  matgen_coo_add_entry(coo, 1, 1, 4.0);  // Entry at (1,1)

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_bilinear(source, 3, 3, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 3);
  EXPECT_EQ(result->cols, 3);

  // Entry at (1,1) maps to (1.5, 1.5) in 3x3
  // Should distribute to 4 neighbors with equal weights (0.5*0.5 = 0.25 each)
  EXPECT_EQ(result->nnz, 4);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, ScaleUpWithDistribution) {
  // Scale 3x3 -> 5x5 (scale factor = 5/3 = 1.666...)
  matgen_coo_matrix_t* coo = matgen_coo_create(3, 3, 2);
  matgen_coo_add_entry(coo, 1, 1, 9.0);  // Center
  matgen_coo_add_entry(coo, 2, 2, 6.0);  // Corner

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_bilinear(source, 5, 5, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 5);
  EXPECT_EQ(result->cols, 5);

  // Both entries should create fractional coordinates and distribute
  EXPECT_GT(result->nnz, 2);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, ScaleUpIntegerFactor) {
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 2, 1);
  matgen_coo_add_entry(coo, 0, 0, 4.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_bilinear(source, 4, 4, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 4);
  EXPECT_EQ(result->cols, 4);

  // Entry at (0,0) with scale factor 2.0 maps to exactly (0.0, 0.0)
  EXPECT_EQ(result->nnz, 1);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, NonSquareScaling) {
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 3, 2);
  matgen_coo_add_entry(coo, 0, 1, 1.0);  // Not at origin
  matgen_coo_add_entry(coo, 1, 2, 2.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_bilinear(source, 4, 6, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 4);
  EXPECT_EQ(result->cols, 6);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, DetailedDistributionCheck) {
  // 2x2 -> 3x3: entry at (1,1) should map to (1.5, 1.5)
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 2, 1);
  matgen_coo_add_entry(coo, 1, 1, 8.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_bilinear(source, 3, 3, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  EXPECT_EQ(result->nnz, 4);

  // Check that we have entries at (1,1), (1,2), (2,1), (2,2)
  bool found[4] = {false, false, false, false};
  for (matgen_index_t i = 0; i < result->rows; i++) {
    for (matgen_size_t j = result->row_ptr[i]; j < result->row_ptr[i + 1];
         j++) {
      matgen_index_t col = result->col_indices[j];
      matgen_value_t val = result->values[j];

      if (i == 1 && col == 1) {
        found[0] = true;
        EXPECT_DOUBLE_EQ(val, 2.0);
      }
      if (i == 1 && col == 2) {
        found[1] = true;
        EXPECT_DOUBLE_EQ(val, 2.0);
      }
      if (i == 2 && col == 1) {
        found[2] = true;
        EXPECT_DOUBLE_EQ(val, 2.0);
      }
      if (i == 2 && col == 2) {
        found[3] = true;
        EXPECT_DOUBLE_EQ(val, 2.0);
      }
    }
  }

  EXPECT_TRUE(found[0]) << "Missing entry at (1,1)";
  EXPECT_TRUE(found[1]) << "Missing entry at (1,2)";
  EXPECT_TRUE(found[2]) << "Missing entry at (2,1)";
  EXPECT_TRUE(found[3]) << "Missing entry at (2,2)";

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}
