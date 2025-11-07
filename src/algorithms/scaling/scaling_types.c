#include "matgen/algorithms/scaling/scaling_types.h"

#include <math.h>

// =============================================================================
// Coordinate Mapper
// =============================================================================

matgen_coordinate_mapper_t matgen_coordinate_mapper_create(
    matgen_index_t src_rows, matgen_index_t src_cols, matgen_index_t dst_rows,
    matgen_index_t dst_cols) {
  matgen_coordinate_mapper_t mapper;

  mapper.src_rows = src_rows;
  mapper.src_cols = src_cols;
  mapper.dst_rows = dst_rows;
  mapper.dst_cols = dst_cols;

  // Compute scale factors: target / source
  mapper.row_scale = (matgen_value_t)dst_rows / (matgen_value_t)src_rows;
  mapper.col_scale = (matgen_value_t)dst_cols / (matgen_value_t)src_cols;

  return mapper;
}

void matgen_map_nearest(const matgen_coordinate_mapper_t* mapper,
                        matgen_index_t src_row, matgen_index_t src_col,
                        matgen_index_t* dst_row, matgen_index_t* dst_col) {
  // Map to fractional target coordinates
  matgen_value_t row_frac = (matgen_value_t)src_row * mapper->row_scale;
  matgen_value_t col_frac = (matgen_value_t)src_col * mapper->col_scale;

  // Round to nearest integer
  *dst_row = (matgen_index_t)round(row_frac);
  *dst_col = (matgen_index_t)round(col_frac);

  // Clamp to valid range (in case of rounding to boundary)
  if (*dst_row >= mapper->dst_rows) {
    *dst_row = mapper->dst_rows - 1;
  }
  if (*dst_col >= mapper->dst_cols) {
    *dst_col = mapper->dst_cols - 1;
  }
}

matgen_fractional_coord_t matgen_map_fractional(
    const matgen_coordinate_mapper_t* mapper, matgen_index_t src_row,
    matgen_index_t src_col) {
  matgen_fractional_coord_t coord;

  // Map to fractional target coordinates
  coord.row = (matgen_value_t)src_row * mapper->row_scale;
  coord.col = (matgen_value_t)src_col * mapper->col_scale;

  // Compute floor and ceil
  coord.row_floor = (matgen_index_t)floor(coord.row);
  coord.row_ceil = (matgen_index_t)ceil(coord.row);
  coord.col_floor = (matgen_index_t)floor(coord.col);
  coord.col_ceil = (matgen_index_t)ceil(coord.col);

  // Clamp to valid range
  if (coord.row_floor >= mapper->dst_rows) {
    coord.row_floor = mapper->dst_rows - 1;
  }
  if (coord.row_ceil >= mapper->dst_rows) {
    coord.row_ceil = mapper->dst_rows - 1;
  }
  if (coord.col_floor >= mapper->dst_cols) {
    coord.col_floor = mapper->dst_cols - 1;
  }
  if (coord.col_ceil >= mapper->dst_cols) {
    coord.col_ceil = mapper->dst_cols - 1;
  }

  return coord;
}
