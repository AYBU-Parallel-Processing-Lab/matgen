#include <gtest/gtest.h>
#include <matgen/algorithms/scaling/bilinear.h>
#include <matgen/core/conversion.h>
#include <matgen/core/coo_matrix.h>
#include <matgen/core/csr_matrix.h>

#include <algorithm>

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
  // Identity scaling: 1x1 blocks, NNZ stays the same
  EXPECT_EQ(result->nnz, 3);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, ScaleUp2x) {
  // Single entry at (0,0)
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 2, 1);
  matgen_coo_add_entry(coo, 0, 0, 4.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale 2x2 -> 4x4 (2x in each dimension)
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_bilinear(source, 4, 4, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 4);
  EXPECT_EQ(result->cols, 4);

  // Entry at (0,0) expands to 2x2 block = 4 entries
  EXPECT_EQ(result->nnz, 4);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, ScaleUp4x) {
  // Single entry test for 4x upscaling
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 2, 1);
  matgen_coo_add_entry(coo, 0, 0, 16.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale 2x2 -> 8x8 (4x in each dimension)
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_bilinear(source, 8, 8, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 8);
  EXPECT_EQ(result->cols, 8);

  // Entry at (0,0) expands to 4x4 block = 16 entries
  EXPECT_EQ(result->nnz, 16);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, MultipleEntriesScaleUp) {
  // Multiple entries to verify NNZ growth
  matgen_coo_matrix_t* coo = matgen_coo_create(3, 3, 3);
  matgen_coo_add_entry(coo, 0, 0, 1.0);
  matgen_coo_add_entry(coo, 1, 1, 2.0);
  matgen_coo_add_entry(coo, 2, 2, 3.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale 3x3 -> 6x6 (2x in each dimension)
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_bilinear(source, 6, 6, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 6);
  EXPECT_EQ(result->cols, 6);

  // 3 entries, each becoming 2x2 block = 12 entries total
  EXPECT_EQ(result->nnz, 12);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, NonSquareScaling) {
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 3, 2);
  matgen_coo_add_entry(coo, 0, 1, 1.0);
  matgen_coo_add_entry(coo, 1, 2, 2.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale 2x3 -> 4x6 (2x in both dimensions)
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_bilinear(source, 4, 6, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 4);
  EXPECT_EQ(result->cols, 6);

  // 2 entries, each expanding to 2x2 block = 8 total
  EXPECT_EQ(result->nnz, 8);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, ScaleDown) {
  // Test downscaling
  matgen_coo_matrix_t* coo = matgen_coo_create(4, 4, 4);
  matgen_coo_add_entry(coo, 0, 0, 1.0);
  matgen_coo_add_entry(coo, 1, 1, 2.0);
  matgen_coo_add_entry(coo, 2, 2, 3.0);
  matgen_coo_add_entry(coo, 3, 3, 4.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale 4x4 -> 2x2 (0.5x in each dimension)
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_bilinear(source, 2, 2, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 2);
  EXPECT_EQ(result->cols, 2);

  // Downscaling with block expansion semantics:
  // Each source entry creates a fractional block (< 1x1)
  // Source (0,0) and (1,1) both map to target (0,0) → collision
  // Source (2,2) and (3,3) both map to target (1,1) → collision
  // With SUM collision policy: 2 entries total
  EXPECT_EQ(result->nnz, 2);

  // Verify sum conservation
  double total_sum = 0.0;
  for (matgen_index_t i = 0; i < result->rows; i++) {
    for (matgen_size_t j = result->row_ptr[i]; j < result->row_ptr[i + 1];
         j++) {
      total_sum += result->values[j];
    }
  }
  EXPECT_NEAR(total_sum, 10.0, 1e-10);  // 1+2+3+4 = 10

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, FractionalScaling) {
  // Test non-integer scale factor
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 2, 1);
  matgen_coo_add_entry(coo, 0, 0, 6.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale 2x2 -> 3x3 (1.5x in each dimension)
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_bilinear(source, 3, 3, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 3);
  EXPECT_EQ(result->cols, 3);

  // Entry at (0,0) creates 1x1 block (fractional expansion)
  EXPECT_GE(result->nnz, 1);
  EXPECT_LE(result->nnz, 2);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, ValueConservation) {
  // Test that total value is conserved
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 2, 2);
  matgen_coo_add_entry(coo, 0, 0, 3.0);
  matgen_coo_add_entry(coo, 1, 1, 7.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Calculate sum of source
  double source_sum = 10.0;

  // Scale 2x2 -> 4x4
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_bilinear(source, 4, 4, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);

  // Calculate sum of result
  double result_sum = 0.0;
  for (matgen_index_t i = 0; i < result->rows; i++) {
    for (matgen_size_t j = result->row_ptr[i]; j < result->row_ptr[i + 1];
         j++) {
      result_sum += result->values[j];
    }
  }

  // Values should be conserved
  EXPECT_NEAR(source_sum, result_sum, 1e-9);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, DensityPreservation) {
  // Test that density is preserved during upscaling
  matgen_coo_matrix_t* coo = matgen_coo_create(4, 4, 4);
  matgen_coo_add_entry(coo, 0, 0, 1.0);
  matgen_coo_add_entry(coo, 1, 1, 2.0);
  matgen_coo_add_entry(coo, 2, 2, 3.0);
  matgen_coo_add_entry(coo, 3, 3, 4.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  double source_density =
      (double)source->nnz / (double)(source->rows * source->cols);

  // Test multiple scale factors
  matgen_index_t scales[] = {2, 3, 4};

  for (auto scale : scales) {
    matgen_csr_matrix_t* result = nullptr;
    matgen_error_t err =
        matgen_scale_bilinear(source, 4 * scale, 4 * scale, &result);

    ASSERT_EQ(err, MATGEN_SUCCESS);
    ASSERT_NE(result, nullptr);

    double result_density =
        (double)result->nnz / (double)(result->rows * result->cols);

    // Density should be preserved
    EXPECT_NEAR(result_density, source_density, 1e-10)
        << "Density not preserved for scale factor " << scale;

    matgen_csr_destroy(result);
  }

  matgen_csr_destroy(source);
}

TEST(BilinearTest, ComprehensiveValueConservation) {
  // Comprehensive value conservation test across various scales
  matgen_coo_matrix_t* coo = matgen_coo_create(3, 3, 5);
  matgen_coo_add_entry(coo, 0, 0, 1.5);
  matgen_coo_add_entry(coo, 0, 2, 2.5);
  matgen_coo_add_entry(coo, 1, 1, 3.5);
  matgen_coo_add_entry(coo, 2, 0, 4.5);
  matgen_coo_add_entry(coo, 2, 2, 5.5);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Calculate original sum
  double original_sum = 1.5 + 2.5 + 3.5 + 4.5 + 5.5;  // = 17.5

  // Scale by various factors
  struct {
    matgen_index_t new_rows;
    matgen_index_t new_cols;
  } test_cases[] = {
      {3, 3},    // 1x (identity)
      {6, 6},    // 2x
      {12, 12},  // 4x
      {9, 9},    // 3x
      {15, 15},  // 5x
  };

  for (const auto& test : test_cases) {
    matgen_csr_matrix_t* result = nullptr;
    matgen_error_t err =
        matgen_scale_bilinear(source, test.new_rows, test.new_cols, &result);

    ASSERT_EQ(err, MATGEN_SUCCESS);
    ASSERT_NE(result, nullptr);

    // Calculate result sum
    double result_sum = 0.0;
    for (matgen_index_t i = 0; i < result->rows; i++) {
      for (matgen_size_t j = result->row_ptr[i]; j < result->row_ptr[i + 1];
           j++) {
        result_sum += result->values[j];
      }
    }

    // Verify sum is conserved
    EXPECT_NEAR(result_sum, original_sum, 1e-9)
        << "Sum not conserved for " << test.new_rows << "x" << test.new_cols
        << " scaling";

    matgen_csr_destroy(result);
  }

  matgen_csr_destroy(source);
}

TEST(BilinearTest, WeightedDistribution) {
  // Verify that bilinear uses weighted distribution (not uniform)
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 2, 1);
  matgen_coo_add_entry(coo, 0, 0, 16.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale 2x2 -> 6x6 (3x in each dimension)
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_bilinear(source, 6, 6, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);

  // Entry at (0,0) creates 3x3 block
  EXPECT_EQ(result->nnz, 9);

  // Find min and max values
  double min_val = 1e10;
  double max_val = -1e10;

  for (matgen_index_t i = 0; i < result->rows; i++) {
    for (matgen_size_t j = result->row_ptr[i]; j < result->row_ptr[i + 1];
         j++) {
      min_val = std::min(result->values[j], min_val);
      max_val = std::max(result->values[j], max_val);
    }
  }

  // With bilinear weighting, values should NOT all be uniform
  // Center cells should have higher values than edge cells
  // For a 3x3 block, expect noticeable variation
  EXPECT_GT(max_val / min_val, 1.5);

  // Verify sum conservation
  double total_sum = 0.0;
  for (matgen_index_t i = 0; i < result->rows; i++) {
    for (matgen_size_t j = result->row_ptr[i]; j < result->row_ptr[i + 1];
         j++) {
      total_sum += result->values[j];
    }
  }
  EXPECT_NEAR(total_sum, 16.0, 1e-9);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}
