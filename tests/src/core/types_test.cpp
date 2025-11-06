#include <gtest/gtest.h>
#include <matgen/core/types.h>

#include <limits>

// Test fixture for types
class TypesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup code if needed
  }
};

// =============================================================================
// Test Integer Type Sizes
// =============================================================================

TEST_F(TypesTest, IntegerTypeSizes) {
  // 8-bit types
  EXPECT_EQ(sizeof(i8), 1);
  EXPECT_EQ(sizeof(u8), 1);

  // 16-bit types
  EXPECT_EQ(sizeof(i16), 2);
  EXPECT_EQ(sizeof(u16), 2);

  // 32-bit types
  EXPECT_EQ(sizeof(i32), 4);
  EXPECT_EQ(sizeof(u32), 4);

  // 64-bit types
  EXPECT_EQ(sizeof(i64), 8);
  EXPECT_EQ(sizeof(u64), 8);
}

TEST_F(TypesTest, FloatTypeSizes) {
  EXPECT_EQ(sizeof(f32), 4);
  EXPECT_EQ(sizeof(f64), 8);
}

// =============================================================================
// Test Signed/Unsigned Ranges
// =============================================================================

TEST_F(TypesTest, SignedIntegerRanges) {
  // i8: -128 to 127
  EXPECT_EQ(std::numeric_limits<i8>::min(), -128);
  EXPECT_EQ(std::numeric_limits<i8>::max(), 127);

  // i16: -32768 to 32767
  EXPECT_EQ(std::numeric_limits<i16>::min(), -32768);
  EXPECT_EQ(std::numeric_limits<i16>::max(), 32767);

  // i32: -2^31 to 2^31-1
  EXPECT_EQ(std::numeric_limits<i32>::min(), -2147483647 - 1);
  EXPECT_EQ(std::numeric_limits<i32>::max(), 2147483647);

  // i64
  EXPECT_LT(std::numeric_limits<i64>::min(), 0);
  EXPECT_GT(std::numeric_limits<i64>::max(), 0);
}

TEST_F(TypesTest, UnsignedIntegerRanges) {
  // u8: 0 to 255
  EXPECT_EQ(std::numeric_limits<u8>::min(), 0);
  EXPECT_EQ(std::numeric_limits<u8>::max(), 255);

  // u16: 0 to 65535
  EXPECT_EQ(std::numeric_limits<u16>::min(), 0);
  EXPECT_EQ(std::numeric_limits<u16>::max(), 65535);

  // u32: 0 to 2^32-1
  EXPECT_EQ(std::numeric_limits<u32>::min(), 0U);
  EXPECT_EQ(std::numeric_limits<u32>::max(), 4294967295U);

  // u64
  EXPECT_EQ(std::numeric_limits<u64>::min(), 0U);
  EXPECT_GT(std::numeric_limits<u64>::max(), 0U);
}

// =============================================================================
// Test MatGen-Specific Types
// =============================================================================

TEST_F(TypesTest, MatGenIndexType) {
  // Should be 64-bit unsigned
  EXPECT_EQ(sizeof(matgen_index_t), 8);
  EXPECT_TRUE(std::is_unsigned<matgen_index_t>::value);

  // Should support very large matrices
  matgen_index_t large_index = 10000000000ULL;  // 10 billion
  EXPECT_GT(large_index, 0);
}

TEST_F(TypesTest, MatGenValueType) {
  // Should be double precision (f64)
  EXPECT_EQ(sizeof(matgen_value_t), 8);
  EXPECT_TRUE(std::is_floating_point<matgen_value_t>::value);
}

TEST_F(TypesTest, MatGenSizeType) {
  // Should be size_t
  EXPECT_EQ(sizeof(matgen_size_t), sizeof(size_t));
}

// =============================================================================
// Test Error Codes
// =============================================================================

TEST_F(TypesTest, ErrorCodes) {
  EXPECT_EQ(MATGEN_SUCCESS, 0);
  EXPECT_LT(MATGEN_ERROR_INVALID_ARGUMENT, 0);
  EXPECT_LT(MATGEN_ERROR_OUT_OF_MEMORY, 0);
  EXPECT_LT(MATGEN_ERROR_IO, 0);
  EXPECT_LT(MATGEN_ERROR_UNSUPPORTED, 0);
  EXPECT_LT(MATGEN_ERROR_INVALID_FORMAT, 0);
  EXPECT_LT(MATGEN_ERROR_MPI, 0);
  EXPECT_LT(MATGEN_ERROR_CUDA, 0);

  // All error codes should be unique
  EXPECT_NE(MATGEN_ERROR_INVALID_ARGUMENT, MATGEN_ERROR_OUT_OF_MEMORY);
  EXPECT_NE(MATGEN_ERROR_IO, MATGEN_ERROR_UNSUPPORTED);
}

// =============================================================================
// Test Macros
// =============================================================================

TEST_F(TypesTest, MinMaxMacros) {
  EXPECT_EQ(MATGEN_MIN(5, 10), 5);
  EXPECT_EQ(MATGEN_MIN(10, 5), 5);
  EXPECT_EQ(MATGEN_MIN(-5, 5), -5);

  EXPECT_EQ(MATGEN_MAX(5, 10), 10);
  EXPECT_EQ(MATGEN_MAX(10, 5), 10);
  EXPECT_EQ(MATGEN_MAX(-5, 5), 5);
}

TEST_F(TypesTest, ClampMacro) {
  EXPECT_EQ(MATGEN_CLAMP(5, 0, 10), 5);
  EXPECT_EQ(MATGEN_CLAMP(-5, 0, 10), 0);
  EXPECT_EQ(MATGEN_CLAMP(15, 0, 10), 10);
  EXPECT_EQ(MATGEN_CLAMP(0, 0, 10), 0);
  EXPECT_EQ(MATGEN_CLAMP(10, 0, 10), 10);  // NOLINT
}
