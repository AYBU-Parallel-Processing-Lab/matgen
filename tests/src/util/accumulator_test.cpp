#include "matgen/utils/accumulator.h"

#include <gtest/gtest.h>

// =============================================================================
// Basic Creation and Destruction Tests
// =============================================================================

TEST(AccumulatorTest, CreateAndDestroy) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(100, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);
  // Capacity is rounded up to next power of 2: 100 → 128
  EXPECT_EQ(matgen_accumulator_capacity(acc), 128);
  EXPECT_EQ(matgen_accumulator_size(acc), 0);
  EXPECT_EQ(acc->policy, MATGEN_COLLISION_SUM);

  matgen_accumulator_destroy(acc);
}

TEST(AccumulatorTest, CreateZeroCapacity) {
  // Zero capacity uses default (1024)
  matgen_accumulator_t* acc =
      matgen_accumulator_create(0, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);
  EXPECT_EQ(matgen_accumulator_capacity(acc), 1024);
  matgen_accumulator_destroy(acc);
}

TEST(AccumulatorTest, CreateSmallCapacity) {
  // Small capacities are rounded up to minimum (16)
  matgen_accumulator_t* acc =
      matgen_accumulator_create(5, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);
  EXPECT_GE(matgen_accumulator_capacity(acc), 16);
  matgen_accumulator_destroy(acc);
}

TEST(AccumulatorTest, DestroyNull) {
  // Should not crash
  matgen_accumulator_destroy(nullptr);
}

TEST(AccumulatorTest, AllPolicies) {
  matgen_accumulator_t* acc_sum =
      matgen_accumulator_create(100, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc_sum, nullptr);
  EXPECT_EQ(acc_sum->policy, MATGEN_COLLISION_SUM);
  matgen_accumulator_destroy(acc_sum);

  matgen_accumulator_t* acc_avg =
      matgen_accumulator_create(100, MATGEN_COLLISION_AVG);
  ASSERT_NE(acc_avg, nullptr);
  EXPECT_EQ(acc_avg->policy, MATGEN_COLLISION_AVG);
  matgen_accumulator_destroy(acc_avg);

  matgen_accumulator_t* acc_max =
      matgen_accumulator_create(100, MATGEN_COLLISION_MAX);
  ASSERT_NE(acc_max, nullptr);
  EXPECT_EQ(acc_max->policy, MATGEN_COLLISION_MAX);
  matgen_accumulator_destroy(acc_max);
}

// =============================================================================
// Basic Add Operations
// =============================================================================

TEST(AccumulatorTest, AddSingleEntry) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(100, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);

  matgen_error_t err = matgen_accumulator_add(acc, 5, 10, 3.14);
  EXPECT_EQ(err, MATGEN_SUCCESS);
  EXPECT_EQ(matgen_accumulator_size(acc), 1);

  // Verify using get function
  matgen_value_t value;
  err = matgen_accumulator_get(acc, 5, 10, &value);
  EXPECT_EQ(err, MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 3.14);

  matgen_accumulator_destroy(acc);
}

TEST(AccumulatorTest, AddMultipleUniqueEntries) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(100, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);

  matgen_accumulator_add(acc, 0, 0, 1.0);
  matgen_accumulator_add(acc, 1, 1, 2.0);
  matgen_accumulator_add(acc, 2, 2, 3.0);
  matgen_accumulator_add(acc, 3, 3, 4.0);

  EXPECT_EQ(matgen_accumulator_size(acc), 4);

  matgen_accumulator_destroy(acc);
}

TEST(AccumulatorTest, AddWithNullAccumulator) {
  matgen_error_t err = matgen_accumulator_add(nullptr, 0, 0, 1.0);
  EXPECT_EQ(err, MATGEN_ERROR_INVALID_ARGUMENT);
}

// =============================================================================
// Collision Policy: SUM
// =============================================================================

TEST(AccumulatorTest, CollisionPolicySumBasic) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(100, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);

  // Add same coordinate multiple times
  matgen_accumulator_add(acc, 5, 10, 1.0);
  matgen_accumulator_add(acc, 5, 10, 2.0);
  matgen_accumulator_add(acc, 5, 10, 3.0);

  EXPECT_EQ(matgen_accumulator_size(acc), 1);

  matgen_value_t value;
  matgen_error_t err = matgen_accumulator_get(acc, 5, 10, &value);
  EXPECT_EQ(err, MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 6.0);  // 1+2+3

  matgen_accumulator_destroy(acc);
}

TEST(AccumulatorTest, CollisionPolicySumNegativeValues) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(100, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);

  matgen_accumulator_add(acc, 0, 0, 10.0);
  matgen_accumulator_add(acc, 0, 0, -3.0);
  matgen_accumulator_add(acc, 0, 0, -2.0);

  matgen_value_t value;
  matgen_error_t err = matgen_accumulator_get(acc, 0, 0, &value);
  EXPECT_EQ(err, MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 5.0);  // 10-3-2

  matgen_accumulator_destroy(acc);
}

// =============================================================================
// Collision Policy: AVG
// =============================================================================

TEST(AccumulatorTest, CollisionPolicyAvgBasic) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(100, MATGEN_COLLISION_AVG);
  ASSERT_NE(acc, nullptr);

  // Add same coordinate multiple times
  matgen_accumulator_add(acc, 5, 10, 10.0);
  matgen_accumulator_add(acc, 5, 10, 20.0);
  matgen_accumulator_add(acc, 5, 10, 30.0);

  EXPECT_EQ(matgen_accumulator_size(acc), 1);

  // Get returns the average
  matgen_value_t value;
  matgen_error_t err = matgen_accumulator_get(acc, 5, 10, &value);
  EXPECT_EQ(err, MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 20.0);  // (10+20+30)/3

  matgen_accumulator_destroy(acc);
}

TEST(AccumulatorTest, CollisionPolicyAvgSingleValue) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(100, MATGEN_COLLISION_AVG);
  ASSERT_NE(acc, nullptr);

  matgen_accumulator_add(acc, 0, 0, 42.0);

  matgen_value_t value;
  matgen_error_t err = matgen_accumulator_get(acc, 0, 0, &value);
  EXPECT_EQ(err, MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 42.0);

  matgen_accumulator_destroy(acc);
}

// =============================================================================
// Collision Policy: MAX
// =============================================================================

TEST(AccumulatorTest, CollisionPolicyMaxBasic) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(100, MATGEN_COLLISION_MAX);
  ASSERT_NE(acc, nullptr);

  // Add values in mixed order
  matgen_accumulator_add(acc, 5, 10, 15.0);
  matgen_accumulator_add(acc, 5, 10, 42.0);  // Maximum
  matgen_accumulator_add(acc, 5, 10, 8.0);
  matgen_accumulator_add(acc, 5, 10, 23.0);

  EXPECT_EQ(matgen_accumulator_size(acc), 1);

  matgen_value_t value;
  matgen_error_t err = matgen_accumulator_get(acc, 5, 10, &value);
  EXPECT_EQ(err, MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 42.0);

  matgen_accumulator_destroy(acc);
}

TEST(AccumulatorTest, CollisionPolicyMaxNegativeValues) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(100, MATGEN_COLLISION_MAX);
  ASSERT_NE(acc, nullptr);

  matgen_accumulator_add(acc, 0, 0, -10.0);
  matgen_accumulator_add(acc, 0, 0, -5.0);  // Maximum
  matgen_accumulator_add(acc, 0, 0, -20.0);

  matgen_value_t value;
  matgen_error_t err = matgen_accumulator_get(acc, 0, 0, &value);
  EXPECT_EQ(err, MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, -5.0);

  matgen_accumulator_destroy(acc);
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST(AccumulatorTest, ManyUniqueEntries) {
  const size_t num_entries = 1000;
  matgen_accumulator_t* acc =
      matgen_accumulator_create(num_entries * 2, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);

  // Add many unique entries
  for (size_t i = 0; i < num_entries; i++) {
    matgen_error_t err = matgen_accumulator_add(acc, i, i, (matgen_value_t)i);
    EXPECT_EQ(err, MATGEN_SUCCESS);
  }

  EXPECT_EQ(matgen_accumulator_size(acc), num_entries);

  // Verify all entries exist
  for (size_t i = 0; i < num_entries; i++) {
    matgen_value_t value;
    matgen_error_t err = matgen_accumulator_get(acc, i, i, &value);
    EXPECT_EQ(err, MATGEN_SUCCESS);
    EXPECT_DOUBLE_EQ(value, (matgen_value_t)i);
  }

  matgen_accumulator_destroy(acc);
}

TEST(AccumulatorTest, ManyCollisions) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(100, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);

  const size_t num_adds = 1000;
  const matgen_index_t row = 42;
  const matgen_index_t col = 84;

  // Add to same coordinate many times
  for (size_t i = 0; i < num_adds; i++) {
    matgen_accumulator_add(acc, row, col, 1.0);
  }

  EXPECT_EQ(matgen_accumulator_size(acc), 1);

  matgen_value_t value;
  matgen_error_t err = matgen_accumulator_get(acc, row, col, &value);
  EXPECT_EQ(err, MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, (matgen_value_t)num_adds);

  matgen_accumulator_destroy(acc);
}

TEST(AccumulatorTest, AutoResize) {
  // Start with small capacity
  matgen_accumulator_t* acc =
      matgen_accumulator_create(16, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);

  size_t initial_capacity = matgen_accumulator_capacity(acc);

  // Add entries until resize triggers
  for (size_t i = 0; i < 100; i++) {
    matgen_accumulator_add(acc, i, i, (matgen_value_t)i);
  }

  // Capacity should have increased
  EXPECT_GT(matgen_accumulator_capacity(acc), initial_capacity);
  EXPECT_EQ(matgen_accumulator_size(acc), 100);

  matgen_accumulator_destroy(acc);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(AccumulatorTest, ZeroValues) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(100, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);

  matgen_accumulator_add(acc, 0, 0, 0.0);
  matgen_accumulator_add(acc, 0, 0, 0.0);
  matgen_accumulator_add(acc, 1, 1, 0.0);

  EXPECT_EQ(matgen_accumulator_size(acc), 2);

  matgen_accumulator_destroy(acc);
}

TEST(AccumulatorTest, VeryLargeIndices) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(100, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);

  matgen_index_t large_index = 1000000;
  matgen_accumulator_add(acc, large_index, large_index, 42.0);

  matgen_value_t value;
  matgen_error_t err =
      matgen_accumulator_get(acc, large_index, large_index, &value);
  EXPECT_EQ(err, MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 42.0);

  matgen_accumulator_destroy(acc);
}

TEST(AccumulatorTest, MixedRowCol) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(100, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);

  // Verify (5,10) and (10,5) are different
  matgen_accumulator_add(acc, 5, 10, 100.0);
  matgen_accumulator_add(acc, 10, 5, 200.0);

  EXPECT_EQ(matgen_accumulator_size(acc), 2);

  matgen_value_t value1;
  matgen_value_t value2;
  matgen_accumulator_get(acc, 5, 10, &value1);
  matgen_accumulator_get(acc, 10, 5, &value2);

  EXPECT_DOUBLE_EQ(value1, 100.0);
  EXPECT_DOUBLE_EQ(value2, 200.0);

  matgen_accumulator_destroy(acc);
}

TEST(AccumulatorTest, FloatingPointPrecision) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(100, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);

  // Add very small values
  matgen_accumulator_add(acc, 0, 0, 1e-10);
  matgen_accumulator_add(acc, 0, 0, 2e-10);
  matgen_accumulator_add(acc, 0, 0, 3e-10);

  matgen_value_t value;
  matgen_accumulator_get(acc, 0, 0, &value);
  EXPECT_NEAR(value, 6e-10, 1e-15);

  matgen_accumulator_destroy(acc);
}

TEST(AccumulatorTest, LoadFactor) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(128, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);

  EXPECT_DOUBLE_EQ(matgen_accumulator_load_factor(acc), 0.0);

  // Add 64 entries
  for (size_t i = 0; i < 64; i++) {
    matgen_accumulator_add(acc, i, i, 1.0);
  }

  // Load factor should be 64/128 = 0.5
  EXPECT_DOUBLE_EQ(matgen_accumulator_load_factor(acc), 0.5);

  matgen_accumulator_destroy(acc);
}

TEST(AccumulatorTest, GetNonExistent) {
  matgen_accumulator_t* acc =
      matgen_accumulator_create(100, MATGEN_COLLISION_SUM);
  ASSERT_NE(acc, nullptr);

  matgen_value_t value;
  matgen_error_t err = matgen_accumulator_get(acc, 99, 99, &value);
  EXPECT_NE(err, MATGEN_SUCCESS);

  matgen_accumulator_destroy(acc);
}
