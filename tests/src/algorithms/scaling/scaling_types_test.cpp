#include <gtest/gtest.h>
#include <matgen/algorithms/scaling/scaling_types.h>
#include <matgen/core/types.h>

TEST(ScalingTypesTest, CoordinateMapperCreate) {
  matgen_coordinate_mapper_t mapper =
      matgen_coordinate_mapper_create(10, 10, 20, 20);

  EXPECT_EQ(mapper.src_rows, 10);
  EXPECT_EQ(mapper.src_cols, 10);
  EXPECT_EQ(mapper.dst_rows, 20);
  EXPECT_EQ(mapper.dst_cols, 20);
  EXPECT_DOUBLE_EQ(mapper.row_scale, 2.0);
  EXPECT_DOUBLE_EQ(mapper.col_scale, 2.0);
}

TEST(ScalingTypesTest, MapNearestScaleUp) {
  matgen_coordinate_mapper_t mapper =
      matgen_coordinate_mapper_create(10, 10, 20, 20);

  matgen_index_t dst_row;
  matgen_index_t dst_col;
  matgen_map_nearest(&mapper, 5, 5, &dst_row, &dst_col);

  // 5 * 2.0 = 10.0, round to 10
  EXPECT_EQ(dst_row, 10);
  EXPECT_EQ(dst_col, 10);
}

TEST(ScalingTypesTest, MapNearestScaleDown) {
  matgen_coordinate_mapper_t mapper =
      matgen_coordinate_mapper_create(20, 20, 10, 10);

  matgen_index_t dst_row;
  matgen_index_t dst_col;
  matgen_map_nearest(&mapper, 10, 10, &dst_row, &dst_col);

  // 10 * 0.5 = 5.0, round to 5
  EXPECT_EQ(dst_row, 5);
  EXPECT_EQ(dst_col, 5);
}

TEST(ScalingTypesTest, MapFractional) {
  matgen_coordinate_mapper_t mapper =
      matgen_coordinate_mapper_create(10, 10, 15, 15);

  matgen_fractional_coord_t coord = matgen_map_fractional(&mapper, 5, 5);

  // 5 * 1.5 = 7.5
  EXPECT_DOUBLE_EQ(coord.row, 7.5);
  EXPECT_DOUBLE_EQ(coord.col, 7.5);
  EXPECT_EQ(coord.row_floor, 7);
  EXPECT_EQ(coord.row_ceil, 8);
  EXPECT_EQ(coord.col_floor, 7);
  EXPECT_EQ(coord.col_ceil, 8);
}
