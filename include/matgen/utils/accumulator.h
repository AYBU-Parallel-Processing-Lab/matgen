#ifndef MATGEN_UTILS_ACCUMULATOR_H
#define MATGEN_UTILS_ACCUMULATOR_H

#include <stdbool.h>

#include "matgen/algorithms/scaling/scaling_types.h"
#include "matgen/core/coo_matrix.h"
#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Types
// =============================================================================

/**
 * @brief Entry in the accumulator hash table
 */
typedef struct {
  matgen_index_t row;   /**< Row index (-1 indicates empty slot) */
  matgen_index_t col;   /**< Column index */
  matgen_value_t value; /**< Accumulated value */
  size_t count;         /**< Number of values accumulated */
} matgen_accum_entry_t;

/**
 * @brief Accumulator structure for building sparse matrices
 *
 * The accumulator uses a hash table to efficiently collect matrix entries
 * and handle duplicate coordinates according to a specified collision policy.
 */
typedef struct {
  matgen_accum_entry_t* entries;    /**< Hash table entries */
  size_t capacity;                  /**< Hash table capacity (power of 2) */
  size_t size;                      /**< Number of non-empty entries */
  matgen_collision_policy_t policy; /**< Collision handling policy */
} matgen_accumulator_t;

/**
 * @brief Callback function for iterating over accumulator entries
 *
 * @param row Row index
 * @param col Column index
 * @param value Final value (after policy applied)
 * @param count Number of times this coordinate was added
 * @param user_data User-provided data pointer
 * @return true to continue iteration, false to stop
 */
typedef bool (*matgen_accumulator_callback_t)(matgen_index_t row,
                                              matgen_index_t col,
                                              matgen_value_t value,
                                              size_t count, void* user_data);

// =============================================================================
// Core API
// =============================================================================

/**
 * @brief Create a new accumulator with specified capacity and collision policy
 *
 * @param capacity Initial capacity (0 uses default, will be rounded to power of
 * 2)
 * @param policy Collision policy for duplicate coordinates
 * @return Pointer to new accumulator, or NULL on failure
 *
 * @note The accumulator automatically resizes when load factor exceeds 0.7
 */
matgen_accumulator_t* matgen_accumulator_create(
    size_t capacity, matgen_collision_policy_t policy);

/**
 * @brief Destroy an accumulator and free its memory
 *
 * @param acc Accumulator to destroy (NULL safe)
 */
void matgen_accumulator_destroy(matgen_accumulator_t* acc);

/**
 * @brief Add a value to the accumulator at the given coordinate
 *
 * If an entry already exists at (row, col), the collision policy determines
 * how the values are combined.
 *
 * @param acc Accumulator to add to
 * @param row Row index
 * @param col Column index
 * @param value Value to add
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @note May trigger automatic resize if load factor exceeds threshold
 */
matgen_error_t matgen_accumulator_add(matgen_accumulator_t* acc,
                                      matgen_index_t row, matgen_index_t col,
                                      matgen_value_t value);

/**
 * @brief Get the accumulated value at a specific coordinate
 *
 * @param acc Accumulator to query
 * @param row Row index
 * @param col Column index
 * @param value Output: final value (after policy applied)
 * @return MATGEN_SUCCESS if found, MATGEN_ERROR_NOT_FOUND if not present
 */
matgen_error_t matgen_accumulator_get(const matgen_accumulator_t* acc,
                                      matgen_index_t row, matgen_index_t col,
                                      matgen_value_t* value);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Get the number of non-empty entries in the accumulator
 *
 * @param acc Accumulator to query
 * @return Number of entries (0 if acc is NULL)
 */
size_t matgen_accumulator_size(const matgen_accumulator_t* acc);

/**
 * @brief Get the current capacity of the accumulator
 *
 * @param acc Accumulator to query
 * @return Capacity (0 if acc is NULL)
 */
size_t matgen_accumulator_capacity(const matgen_accumulator_t* acc);

/**
 * @brief Get the current load factor (size / capacity)
 *
 * @param acc Accumulator to query
 * @return Load factor in range [0.0, 1.0]
 */
double matgen_accumulator_load_factor(const matgen_accumulator_t* acc);

/**
 * @brief Clear all entries from the accumulator
 *
 * Resets the accumulator to empty state without changing capacity.
 *
 * @param acc Accumulator to clear
 * @return MATGEN_SUCCESS on success, error code otherwise
 */
matgen_error_t matgen_accumulator_clear(matgen_accumulator_t* acc);

/**
 * @brief Reserve space for at least the specified number of entries
 *
 * Resizes the accumulator if new_capacity is larger than current capacity.
 * Does nothing if new_capacity is smaller.
 *
 * @param acc Accumulator to resize
 * @param new_capacity Desired capacity
 * @return MATGEN_SUCCESS on success, error code otherwise
 */
matgen_error_t matgen_accumulator_reserve(matgen_accumulator_t* acc,
                                          size_t new_capacity);

/**
 * @brief Iterate over all entries in the accumulator
 *
 * Calls the callback function for each non-empty entry. The callback receives
 * the final value after the collision policy has been applied.
 *
 * @param acc Accumulator to iterate over
 * @param callback Function to call for each entry
 * @param user_data User data to pass to callback
 * @return MATGEN_SUCCESS on success, error code otherwise
 *
 * @note Iteration order is not guaranteed and depends on hash table layout
 */
matgen_error_t matgen_accumulator_foreach(
    const matgen_accumulator_t* acc, matgen_accumulator_callback_t callback,
    void* user_data);

// =============================================================================
// Conversion Functions
// =============================================================================

/**
 * @brief Convert accumulator to COO matrix
 *
 * Creates a new COO matrix containing all entries from the accumulator.
 * Collision policies are applied when extracting values.
 *
 * @param acc Accumulator to convert
 * @param rows Number of rows in output matrix
 * @param cols Number of columns in output matrix
 * @return New COO matrix, or NULL on failure
 *
 * @note Caller is responsible for destroying the returned matrix
 */
matgen_coo_matrix_t* matgen_accumulator_to_coo(const matgen_accumulator_t* acc,
                                               matgen_index_t rows,
                                               matgen_index_t cols);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_UTILS_ACCUMULATOR_H
