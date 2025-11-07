#ifndef MATGEN_ALGORITHMS_SCALING_TYPES_H
#define MATGEN_ALGORITHMS_SCALING_TYPES_H

/**
 * @file scaling_types.h
 * @brief Common types for sparse matrix scaling algorithms
 */

#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief Interpolation methods
 */
typedef enum {
  MATGEN_INTERP_NEAREST = 0,  // Nearest neighbor
  MATGEN_INTERP_BILINEAR = 1  // Bilinear interpolation
} matgen_interpolation_method_t;

/**
 * @brief Collision handling policy for nearest neighbor
 *
 * When multiple source entries map to the same target cell
 */
typedef enum {
  MATGEN_COLLISION_SUM = 0,  // Sum all values
  MATGEN_COLLISION_AVG = 1,  // Average all values
  MATGEN_COLLISION_MAX = 2   // Take maximum value
} matgen_collision_policy_t;

// =============================================================================
// Structures
// =============================================================================

/**
 * @brief Coordinate mapper for scaling transformations
 */
typedef struct {
  matgen_value_t row_scale;  // target_rows / source_rows
  matgen_value_t col_scale;  // target_cols / source_cols
  matgen_index_t src_rows;   // Source matrix rows
  matgen_index_t src_cols;   // Source matrix columns
  matgen_index_t dst_rows;   // Target matrix rows
  matgen_index_t dst_cols;   // Target matrix columns
} matgen_coordinate_mapper_t;

/**
 * @brief Fractional coordinate in target space
 */
typedef struct {
  matgen_value_t row;        // Fractional row
  matgen_value_t col;        // Fractional column
  matgen_index_t row_floor;  // floor(row)
  matgen_index_t row_ceil;   // ceil(row)
  matgen_index_t col_floor;  // floor(col)
  matgen_index_t col_ceil;   // ceil(col)
} matgen_fractional_coord_t;

// =============================================================================
// Coordinate Mapping Functions
// =============================================================================

/**
 * @brief Create coordinate mapper
 *
 * @param src_rows Source matrix rows
 * @param src_cols Source matrix columns
 * @param dst_rows Target matrix rows
 * @param dst_cols Target matrix columns
 * @return Coordinate mapper structure
 */
matgen_coordinate_mapper_t matgen_coordinate_mapper_create(
    matgen_index_t src_rows, matgen_index_t src_cols, matgen_index_t dst_rows,
    matgen_index_t dst_cols);

/**
 * @brief Map source coordinates to target (nearest neighbor)
 *
 * @param mapper Coordinate mapper
 * @param src_row Source row index
 * @param src_col Source column index
 * @param dst_row Output: target row index
 * @param dst_col Output: target column index
 */
void matgen_map_nearest(const matgen_coordinate_mapper_t* mapper,
                        matgen_index_t src_row, matgen_index_t src_col,
                        matgen_index_t* dst_row, matgen_index_t* dst_col);

/**
 * @brief Map source coordinates to fractional target coordinates
 *
 * Used for bilinear interpolation.
 *
 * @param mapper Coordinate mapper
 * @param src_row Source row index
 * @param src_col Source column index
 * @return Fractional coordinates in target space
 */
matgen_fractional_coord_t matgen_map_fractional(
    const matgen_coordinate_mapper_t* mapper, matgen_index_t src_row,
    matgen_index_t src_col);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_ALGORITHMS_SCALING_TYPES_H
