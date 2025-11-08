#ifndef MATGEN_CORE_TYPES_H
#define MATGEN_CORE_TYPES_H

/**
 * @file types.h
 * @brief Common types and macros for MatGen
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Version Information
// =============================================================================

#define MATGEN_VERSION_MAJOR 0
#define MATGEN_VERSION_MINOR 1
#define MATGEN_VERSION_PATCH 0

// =============================================================================
// Integer Type Aliases
// =============================================================================

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef float f32;
typedef double f64;

// =============================================================================
// MatGen-Specific Types
// =============================================================================

/**
 * @brief Index type for matrix dimensions and indices
 *
 * Use 64-bit indices to support very large matrices.
 * Change to u32 if you only need matrices with < 4B elements.
 */
typedef u64 matgen_index_t;

/**
 * @brief Value type for matrix entries
 *
 * Currently using matgen_value_t precision (f64).
 * Change to f32 for single precision, or make it templated.
 */
typedef f32 matgen_value_t;

/**
 * @brief Size type for counts (nnz, capacity, etc.)
 */
typedef size_t matgen_size_t;

// =============================================================================
// Error Codes
// =============================================================================

/**
 * @brief MatGen error codes
 */
typedef enum {
  MATGEN_SUCCESS = 0,                  // No error
  MATGEN_ERROR_INVALID_ARGUMENT = -1,  // Invalid argument
  MATGEN_ERROR_OUT_OF_MEMORY = -2,     // Memory allocation failed
  MATGEN_ERROR_IO = -3,                // I/O error
  MATGEN_ERROR_UNSUPPORTED = -4,       // Unsupported operation
  MATGEN_ERROR_INVALID_FORMAT = -5,    // Invalid file format
  MATGEN_ERROR_MPI = -6,               // MPI-related error
  MATGEN_ERROR_CUDA = -7,              // CUDA-related error
  MATGEN_ERROR_UNKNOWN = -9999         // Unknown error
} matgen_error_t;

// =============================================================================
// Common Macros
// =============================================================================

// Minimum and maximum
#ifndef MATGEN_MIN
#define MATGEN_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MATGEN_MAX
#define MATGEN_MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// Clamp value between min and max
#ifndef MATGEN_CLAMP
#define MATGEN_CLAMP(x, min_val, max_val) \
  (MATGEN_MAX((min_val), MATGEN_MIN((x), (max_val))))
#endif

// Unused parameter (suppress warnings)
#ifndef MATGEN_UNUSED
#define MATGEN_UNUSED(x) ((void)(x))
#endif

// Likely/unlikely branch hints (for performance)
#ifdef __GNUC__
#define MATGEN_LIKELY(x) __builtin_expect(!!(x), 1)
#define MATGEN_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define MATGEN_LIKELY(x) (x)
#define MATGEN_UNLIKELY(x) (x)
#endif

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_CORE_TYPES_H
