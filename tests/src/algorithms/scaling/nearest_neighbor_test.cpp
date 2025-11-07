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

  // Scale to 4x4
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_nearest_neighbor(
      source, 4, 4, MATGEN_COLLISION_SUM, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 4);
  EXPECT_EQ(result->cols, 4);
  // Should have 2 entries (no collisions in this case)
  EXPECT_GE(result->nnz, 2);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}

TEST(NearestNeighborTest, ScaleDown) {
  // Create 4x4 matrix
  matgen_coo_matrix_t* coo = matgen_coo_create(4, 4, 4);
  matgen_coo_add_entry(coo, 0, 0, 1.0);
  matgen_coo_add_entry(coo, 1, 1, 2.0);
  matgen_coo_add_entry(coo, 2, 2, 3.0);
  matgen_coo_add_entry(coo, 3, 3, 4.0);

  matgen_csr_matrix_t* source = matgen_coo_to_csr(coo);
  ASSERT_NE(source, nullptr);
  matgen_coo_destroy(coo);

  // Scale down to 2x2
  matgen_csr_matrix_t* result = nullptr;
  matgen_error_t err = matgen_scale_nearest_neighbor(
      source, 2, 2, MATGEN_COLLISION_SUM, &result);

  ASSERT_EQ(err, MATGEN_SUCCESS);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->rows, 2);
  EXPECT_EQ(result->cols, 2);

  matgen_csr_destroy(source);
  matgen_csr_destroy(result);
}
