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
 * @brief Collision policy for handling duplicate (row, col) pairs
 */
typedef enum {
  MATGEN_COLLISION_SUM, /**< Add values together (default) */
  MATGEN_COLLISION_AVG, /**< Average all values */
  MATGEN_COLLISION_MAX, /**< Keep maximum value */
  MATGEN_COLLISION_MIN, /**< Keep minimum value */
  MATGEN_COLLISION_LAST /**< Keep last inserted value */
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

// =============================================================================
// Coordinate Mapping Functions
// =============================================================================

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_ALGORITHMS_SCALING_TYPES_H
