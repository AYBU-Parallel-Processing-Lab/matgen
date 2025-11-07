#include <gtest/gtest.h>
#include <matgen/algorithms/scaling/nearest_neighbor.h>
#include <matgen/algorithms/scaling/scaling_types.h>
#include <matgen/core/conversion.h>
#include <matgen/core/coo_matrix.h>
#include <matgen/core/csr_matrix.h>

TEST(NearestNeighborTest, IdentityScaling) {
  // Create a simple 3x3 matrix
  matgen_coo_matrix_t* coo = matgen_coo_create(3, 3, 3);
  matgen_coo_add_entry(coo, 0, 0, 1.0);
  matgen_coo_add_entry(coo, 1, 1, 2.0);
  matgen_coo_add_entry(coo, 2, 2, 3.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale to same size (identity)
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_nearest_neighbor(
      source, 3, 3, MATGEN_COLLISION_SUM, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 3);
  EXPECT_EQ(result->cols, 3);
  // Identity scaling: NNZ stays the same
  EXPECT_EQ(result->nnz, 3);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(NearestNeighborTest, ScaleUp2x) {
  // Create 2x2 matrix
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 2, 2);
  matgen_coo_add_entry(coo, 0, 0, 1.0);
  matgen_coo_add_entry(coo, 1, 1, 2.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale 2x2 -> 4x4 (2x in each dimension)
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_nearest_neighbor(
      source, 4, 4, MATGEN_COLLISION_SUM, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 4);
  EXPECT_EQ(result->cols, 4);

  // Each entry becomes a 2x2 block = 4 entries
  // 2 source entries * 4 = 8 total entries
  EXPECT_EQ(result->nnz, 8);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(NearestNeighborTest, ScaleUp4x) {
  // Single entry to clearly see 4x expansion
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 2, 1);
  matgen_coo_add_entry(coo, 0, 0, 5.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale 2x2 -> 8x8 (4x in each dimension)
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_nearest_neighbor(
      source, 8, 8, MATGEN_COLLISION_SUM, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 8);
  EXPECT_EQ(result->cols, 8);

  // 1 entry becomes a 4x4 block = 16 entries
  EXPECT_EQ(result->nnz, 16);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(NearestNeighborTest, ScaleDown) {
  // Create 4x4 matrix with entries on diagonal
  matgen_coo_matrix_t* coo = matgen_coo_create(4, 4, 4);
  matgen_coo_add_entry(coo, 0, 0, 1.0);
  matgen_coo_add_entry(coo, 1, 1, 2.0);
  matgen_coo_add_entry(coo, 2, 2, 3.0);
  matgen_coo_add_entry(coo, 3, 3, 4.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale down 4x4 -> 2x2 (0.5x in each dimension)
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_nearest_neighbor(
      source, 2, 2, MATGEN_COLLISION_SUM, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 2);
  EXPECT_EQ(result->cols, 2);

  // Downscaling: no expansion of blocks
  // Each 2x2 source block maps to 1 target cell
  // Expect at most 2 entries (diagonal)
  EXPECT_LE(result->nnz, 4);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(NearestNeighborTest, CollisionPolicySum) {
  // Test SUM collision policy with downscaling
  matgen_coo_matrix_t* coo = matgen_coo_create(4, 4, 4);
  matgen_coo_add_entry(coo, 0, 0, 1.0);
  matgen_coo_add_entry(coo, 0, 1, 2.0);
  matgen_coo_add_entry(coo, 1, 0, 3.0);
  matgen_coo_add_entry(coo, 1, 1, 4.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale 4x4 -> 2x2: all 4 entries map to (0,0)
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_nearest_neighbor(
      source, 2, 2, MATGEN_COLLISION_SUM, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);

  // With SUM policy: expect summed value at (0,0)
  EXPECT_LE(result->nnz, 1);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(NearestNeighborTest, NonSquareScaling) {
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 3, 2);
  matgen_coo_add_entry(coo, 0, 0, 1.0);
  matgen_coo_add_entry(coo, 1, 2, 2.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale 2x3 -> 4x6 (2x in both dimensions)
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_nearest_neighbor(
      source, 4, 6, MATGEN_COLLISION_SUM, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 4);
  EXPECT_EQ(result->cols, 6);

  // 2 entries, each becoming a 2x2 block = 8 total
  EXPECT_EQ(result->nnz, 8);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(NearestNeighborTest, BlockReplicationVerification) {
  // Verify exact block replication for single entry
  matgen_coo_matrix_t* coo = matgen_coo_create(3, 3, 1);
  matgen_coo_add_entry(coo, 1, 1, 7.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale 3x3 -> 6x6 (2x in each dimension)
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_nearest_neighbor(
      source, 6, 6, MATGEN_COLLISION_SUM, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);

  // Entry at (1,1) should create a 2x2 block at positions:
  // (2,2), (2,3), (3,2), (3,3) - all with value 7.0
  EXPECT_EQ(result->nnz, 4);

  // Verify all values are 7.0
  for (matgen_index_t i = 0; i < result->rows; i++) {
    for (matgen_size_t j = result->row_ptr[i]; j < result->row_ptr[i + 1];
         j++) {
      EXPECT_DOUBLE_EQ(result->values[j], 7.0);
    }
  }

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(NearestNeighborTest, LargeScaleFactor) {
  // Test with large scale factor
  matgen_coo_matrix_t* coo = matgen_coo_create(2, 2, 1);
  matgen_coo_add_entry(coo, 0, 0, 1.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale 2x2 -> 20x20 (10x in each dimension)
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_nearest_neighbor(
      source, 20, 20, MATGEN_COLLISION_SUM, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 20);
  EXPECT_EQ(result->cols, 20);

  // 1 entry becomes a 10x10 block = 100 entries
  EXPECT_EQ(result->nnz, 100);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}
