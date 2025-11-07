#include <gtest/gtest.h>
#include <matgen/algorithms/scaling/bilinear.h>
#include <matgen/core/conversion.h>
#include <matgen/core/coo_matrix.h>
#include <matgen/core/csr_matrix.h>

#include <cmath>

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
  // Identity scaling (1x): NNZ should stay the same
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

  // Entry at (0,0) should expand to a 2x2 block: (0,0), (0,1), (1,0), (1,1)
  // With bilinear weighting, expect 4 entries (could have some zeros filtered)
  EXPECT_GE(result->nnz, 1);
  EXPECT_LE(result->nnz, 4);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, ScaleUp4x) {
  // Single entry test for 4x upscaling
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 2, 1);
  matgen_coo_add_entry(coo, 1, 1, 8.0);

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

  // Entry at (1,1) should expand to a 4x4 block (16 entries)
  // With bilinear, might have fewer due to weighting
  EXPECT_GE(result->nnz, 1);
  EXPECT_LE(result->nnz, 16);

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

  // 3 entries, each becoming ~2x2 block = ~12 entries
  // Actual could vary due to bilinear weighting and filtering
  EXPECT_GT(result->nnz, 3);   // Must be more than original
  EXPECT_LE(result->nnz, 12);  // At most 4x growth per entry

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

  // 2 entries, each expanding to ~2x2 block
  EXPECT_GT(result->nnz, 2);
  EXPECT_LE(result->nnz, 8);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, ScaleDown) {
  // Test downscaling (should reduce NNZ)
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

  // Downscaling: multiple source entries may map to same target
  // Expect fewer entries than source
  EXPECT_LE(result->nnz, 4);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, FractionalScaling) {
  // Test non-integer scale factor
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 2, 1);
  matgen_coo_add_entry(coo, 0, 0, 5.0);

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

  // With fractional scaling, expect some distribution
  EXPECT_GE(result->nnz, 1);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(BilinearTest, ValueConservation) {
  // Test that total value is conserved (approximately)
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 2, 2);
  matgen_coo_add_entry(coo, 0, 0, 3.0);
  matgen_coo_add_entry(coo, 1, 1, 7.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Calculate sum of source
  double source_sum = 0.0;
  for (matgen_index_t i = 0; i < source->rows; i++) {
    for (matgen_size_t j = source->row_ptr[i]; j < source->row_ptr[i + 1];
         j++) {
      source_sum += source->values[j];
    }
  }

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

  // Values should be approximately conserved (within numerical precision)
  EXPECT_NEAR(source_sum, result_sum, 1e-6);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}
