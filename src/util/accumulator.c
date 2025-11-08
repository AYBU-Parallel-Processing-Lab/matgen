#include "matgen/utils/accumulator.h"

#include <stdlib.h>
#include <string.h>

// =============================================================================
// Configuration Constants
// =============================================================================

#define ACCUMULATOR_DEFAULT_CAPACITY 1024
#define ACCUMULATOR_LOAD_FACTOR_THRESHOLD 0.7
#define ACCUMULATOR_MIN_CAPACITY 16

// =============================================================================
// Internal Hash Function
// =============================================================================

/**
 * @brief Improved hash function using FNV-1a algorithm
 * @param row Row index
 * @param col Column index
 * @param capacity Hash table capacity (must be power of 2)
 * @return Hash value in range [0, capacity)
 */
static inline size_t hash_coord(matgen_index_t row, matgen_index_t col,
                                size_t capacity) {
  // FNV-1a hash provides better distribution than simple multiplication
  const size_t FNV_OFFSET = 2166136261U;
  const size_t FNV_PRIME = 16777619U;

  size_t hash = FNV_OFFSET;

  // Hash row
  hash ^= (size_t)row;
  hash *= FNV_PRIME;

  // Hash column
  hash ^= (size_t)col;
  hash *= FNV_PRIME;

  // Use bitwise AND for power-of-2 capacity (faster than modulo)
  return hash & (capacity - 1);
}

/**
 * @brief Calculate next power of 2 greater than or equal to n
 */
static inline size_t next_power_of_2(size_t n) {
  if (n == 0) {
    return 1;
  }

  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
#if SIZE_MAX > 0xFFFFFFFF
  n |= n >> 32;
#endif

  return n + 1;
}

// =============================================================================
// Internal Resize Function
// =============================================================================

/**
 * @brief Resize the accumulator hash table to a new capacity
 * @param acc Pointer to the accumulator
 * @param new_capacity New capacity (must be power of 2)
 * @return MATGEN_SUCCESS on success, error code otherwise
 */
static matgen_error_t accumulator_resize(matgen_accumulator_t* acc,
                                         size_t new_capacity) {
  if (!acc || new_capacity == 0) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Ensure new capacity is power of 2
  new_capacity = next_power_of_2(new_capacity);

  // Allocate new table
  matgen_accum_entry_t* new_entries =
      (matgen_accum_entry_t*)calloc(new_capacity, sizeof(matgen_accum_entry_t));
  if (!new_entries) {
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Initialize new entries
  for (size_t i = 0; i < new_capacity; i++) {
    new_entries[i].row = (matgen_index_t)-1;
  }

  // Rehash all existing entries
  size_t old_capacity = acc->capacity;
  matgen_accum_entry_t* old_entries = acc->entries;

  acc->entries = new_entries;
  acc->capacity = new_capacity;
  size_t old_size = acc->size;
  acc->size = 0;

  // Reinsert all non-empty entries
  for (size_t i = 0; i < old_capacity; i++) {
    if (old_entries[i].row != (matgen_index_t)-1) {
      size_t idx =
          hash_coord(old_entries[i].row, old_entries[i].col, new_capacity);

      // Find empty slot with linear probing
      while (new_entries[idx].row != (matgen_index_t)-1) {
        idx = (idx + 1) & (new_capacity - 1);  // Fast modulo for power of 2
      }

      // Copy entry
      new_entries[idx] = old_entries[i];
      acc->size++;
    }
  }

  // Verify all entries were reinserted
  if (acc->size != old_size) {
    // This should never happen - indicates a bug
    free(new_entries);
    acc->entries = old_entries;
    acc->capacity = old_capacity;
    acc->size = old_size;
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  free(old_entries);
  return MATGEN_SUCCESS;
}

// =============================================================================
// API Implementation
// =============================================================================

matgen_accumulator_t* matgen_accumulator_create(
    size_t capacity, matgen_collision_policy_t policy) {
  // Use default capacity if none specified
  if (capacity == 0) {
    capacity = ACCUMULATOR_DEFAULT_CAPACITY;
  }

  // Ensure minimum capacity
  if (capacity < ACCUMULATOR_MIN_CAPACITY) {
    capacity = ACCUMULATOR_MIN_CAPACITY;
  }

  // Round up to next power of 2 for efficient hashing
  capacity = next_power_of_2(capacity);

  matgen_accumulator_t* acc =
      (matgen_accumulator_t*)malloc(sizeof(matgen_accumulator_t));
  if (!acc) {
    return NULL;
  }

  acc->entries =
      (matgen_accum_entry_t*)calloc(capacity, sizeof(matgen_accum_entry_t));
  if (!acc->entries) {
    free(acc);
    return NULL;
  }

  acc->capacity = capacity;
  acc->size = 0;
  acc->policy = policy;

  // Initialize all entries as empty (row = -1 indicates empty)
  for (size_t i = 0; i < capacity; i++) {
    acc->entries[i].row = (matgen_index_t)-1;
    acc->entries[i].col = 0;
    acc->entries[i].value = 0.0;
    acc->entries[i].count = 0;
  }

  return acc;
}

void matgen_accumulator_destroy(matgen_accumulator_t* acc) {
  if (!acc) {
    return;
  }
  if (acc->entries) {
    free(acc->entries);
  }
  free(acc);
}

matgen_error_t matgen_accumulator_add(matgen_accumulator_t* acc,
                                      matgen_index_t row, matgen_index_t col,
                                      matgen_value_t value) {
  if (!acc) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Check if resize is needed (before insertion to avoid full table)
  double load_factor = (double)(acc->size + 1) / (double)acc->capacity;
  if (load_factor > ACCUMULATOR_LOAD_FACTOR_THRESHOLD) {
    matgen_error_t err = accumulator_resize(acc, acc->capacity * 2);
    if (err != MATGEN_SUCCESS) {
      return err;
    }
  }

  size_t idx = hash_coord(row, col, acc->capacity);
  size_t start_idx = idx;
  size_t probe_count = 0;

  // Linear probing to find empty slot or matching entry
  while (acc->entries[idx].row != (matgen_index_t)-1) {
    // Check if this is the same coordinate (collision case)
    if (acc->entries[idx].row == row && acc->entries[idx].col == col) {
      // Found existing entry - handle collision according to policy
      switch (acc->policy) {
        case MATGEN_COLLISION_SUM:
        case MATGEN_COLLISION_AVG:
          // Store sum for now, divide later when extracting
          acc->entries[idx].value += value;
          acc->entries[idx].count++;
          break;

        case MATGEN_COLLISION_MAX:
          if (value > acc->entries[idx].value) {
            acc->entries[idx].value = value;
          }
          acc->entries[idx].count++;
          break;

        case MATGEN_COLLISION_MIN:
          if (value < acc->entries[idx].value) {
            acc->entries[idx].value = value;
          }
          acc->entries[idx].count++;
          break;

        case MATGEN_COLLISION_LAST:
          acc->entries[idx].value = value;
          acc->entries[idx].count++;
          break;

        default:
          // Default to sum behavior
          acc->entries[idx].value += value;
          acc->entries[idx].count++;
          break;
      }
      return MATGEN_SUCCESS;
    }

    // Move to next slot (linear probing with fast modulo)
    idx = (idx + 1) & (acc->capacity - 1);
    probe_count++;

    // Safety check: detect infinite loop (should never happen with proper load
    // factor)
    if (probe_count >= acc->capacity || idx == start_idx) {
      // Table is full - this should have been prevented by resize
      return MATGEN_ERROR_OUT_OF_MEMORY;
    }
  }

  // Found empty slot - insert new entry
  acc->entries[idx].row = row;
  acc->entries[idx].col = col;
  acc->entries[idx].value = value;
  acc->entries[idx].count = 1;
  acc->size++;

  return MATGEN_SUCCESS;
}

matgen_error_t matgen_accumulator_get(const matgen_accumulator_t* acc,
                                      matgen_index_t row, matgen_index_t col,
                                      matgen_value_t* value) {
  if (!acc || !value) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  size_t idx = hash_coord(row, col, acc->capacity);
  size_t start_idx = idx;

  // Linear probing to find the entry
  while (acc->entries[idx].row != (matgen_index_t)-1) {
    if (acc->entries[idx].row == row && acc->entries[idx].col == col) {
      // Found the entry
      *value = acc->entries[idx].value;

      // Apply averaging if needed
      if (acc->policy == MATGEN_COLLISION_AVG && acc->entries[idx].count > 1) {
        *value /= (matgen_value_t)acc->entries[idx].count;
      }

      return MATGEN_SUCCESS;
    }

    idx = (idx + 1) & (acc->capacity - 1);

    // Prevent infinite loop
    if (idx == start_idx) {
      break;
    }
  }

  // Entry not found
  return MATGEN_ERROR_UNSUPPORTED;
}

size_t matgen_accumulator_size(const matgen_accumulator_t* acc) {
  return acc ? acc->size : 0;
}

size_t matgen_accumulator_capacity(const matgen_accumulator_t* acc) {
  return acc ? acc->capacity : 0;
}

double matgen_accumulator_load_factor(const matgen_accumulator_t* acc) {
  if (!acc || acc->capacity == 0) {
    return 0.0;
  }
  return (double)acc->size / (double)acc->capacity;
}

matgen_error_t matgen_accumulator_clear(matgen_accumulator_t* acc) {
  if (!acc) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Reset all entries to empty
  for (size_t i = 0; i < acc->capacity; i++) {
    acc->entries[i].row = (matgen_index_t)-1;
    acc->entries[i].col = 0;
    acc->entries[i].value = 0.0;
    acc->entries[i].count = 0;
  }

  acc->size = 0;
  return MATGEN_SUCCESS;
}

matgen_error_t matgen_accumulator_reserve(matgen_accumulator_t* acc,
                                          size_t new_capacity) {
  if (!acc) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  // Only resize if new capacity is larger
  if (new_capacity <= acc->capacity) {
    return MATGEN_SUCCESS;
  }

  return accumulator_resize(acc, new_capacity);
}

matgen_error_t matgen_accumulator_foreach(
    const matgen_accumulator_t* acc, matgen_accumulator_callback_t callback,
    void* user_data) {
  if (!acc || !callback) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  for (size_t i = 0; i < acc->capacity; i++) {
    if (acc->entries[i].row != (matgen_index_t)-1) {
      matgen_value_t value = acc->entries[i].value;

      // Apply averaging if needed
      if (acc->policy == MATGEN_COLLISION_AVG && acc->entries[i].count > 1) {
        value /= (matgen_value_t)acc->entries[i].count;
      }

      // Call user callback
      bool should_continue = callback(acc->entries[i].row, acc->entries[i].col,
                                      value, acc->entries[i].count, user_data);

      if (!should_continue) {
        break;
      }
    }
  }

  return MATGEN_SUCCESS;
}

matgen_coo_matrix_t* matgen_accumulator_to_coo(const matgen_accumulator_t* acc,
                                               matgen_index_t rows,
                                               matgen_index_t cols) {
  if (!acc) {
    return NULL;
  }

  // Create COO matrix with exact size needed
  matgen_coo_matrix_t* coo = matgen_coo_create(rows, cols, acc->size);
  if (!coo) {
    return NULL;
  }

  // Add all entries to COO matrix
  for (size_t i = 0; i < acc->capacity; i++) {
    if (acc->entries[i].row != (matgen_index_t)-1) {
      matgen_value_t value = acc->entries[i].value;

      // Apply averaging if needed
      if (acc->policy == MATGEN_COLLISION_AVG && acc->entries[i].count > 1) {
        value /= (matgen_value_t)acc->entries[i].count;
      }

      matgen_error_t err = matgen_coo_add_entry(coo, acc->entries[i].row,
                                                acc->entries[i].col, value);

      if (err != MATGEN_SUCCESS) {
        matgen_coo_destroy(coo);
        return NULL;
      }
    }
  }

  return coo;
}
