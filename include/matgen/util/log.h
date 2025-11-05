#ifndef MATGEN_UTIL_LOG_H
#define MATGEN_UTIL_LOG_H

/**
 * @file log.h
 * @brief Simple logging system for MatGen.
 *
 * Lightweight logging with configurable levels and output streams.
 * Can be completely disabled at compile time.
 */

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Log Levels
// =============================================================================

/**
 * @brief Log levels for MatGen logging system.
 */
typedef enum {
  MATGEN_LOG_LEVEL_TRACE = 0,  // Detailed trace information
  MATGEN_LOG_LEVEL_DEBUG = 1,  // Debug-level messages
  MATGEN_LOG_LEVEL_INFO = 2,   // Informational messages
  MATGEN_LOG_LEVEL_WARN = 3,   // Warning messages
  MATGEN_LOG_LEVEL_ERROR = 4,  // Error messages
  MATGEN_LOG_LEVEL_FATAL = 5,  // Fatal error messages
  MATGEN_LOG_LEVEL_OFF = 6     // Logging disabled
} matgen_log_level_t;

// =============================================================================
// Configuration
// =============================================================================

/**
 * @brief Set the global log level
 *
 * Only messages at or above this level will be logged.
 *
 * @param level Minimum log level to display
 */
void matgen_log_set_level(matgen_log_level_t level);

/**
 * @brief Get the current log level
 *
 * @return Current log level
 */
matgen_log_level_t matgen_log_get_level(void);

/**
 * @brief Set the output stream for logging
 *
 * @param stream Output stream (e.g., stdout, stderr, or a file)
 */
void matgen_log_set_stream(FILE* stream);

/**
 * @brief Enable or disable timestamps in log messages
 *
 * @param enabled true to show timestamps, false to hide
 */
void matgen_log_set_timestamps(bool enabled);

/**
 * @brief Enable or disable color output
 *
 * @param enabled true to use colored output, false for plain text
 */
void matgen_log_set_color(bool enabled);

// =============================================================================
// Logging Functions
// =============================================================================

/**
 * @brief Log a message at the specified level
 *
 * @param level Log level
 * @param file Source file name (__FILE__)
 * @param line Line number (__LINE__)
 * @param func Function name (__func__)
 * @param fmt Format string (printf-style)
 * @param ... Variable arguments
 */
void matgen_log(matgen_log_level_t level, const char* file, int line,
                const char* func, const char* fmt, ...);

// =============================================================================
// Convenience Macros
// =============================================================================

#ifndef MATGEN_LOG_DISABLE

#define MATGEN_LOG_TRACE(...) \
  matgen_log(MATGEN_LOG_LEVEL_TRACE, __FILE__, __LINE__, __func__, __VA_ARGS__)

#define MATGEN_LOG_DEBUG(...) \
  matgen_log(MATGEN_LOG_LEVEL_DEBUG, __FILE__, __LINE__, __func__, __VA_ARGS__)

#define MATGEN_LOG_INFO(...) \
  matgen_log(MATGEN_LOG_LEVEL_INFO, __FILE__, __LINE__, __func__, __VA_ARGS__)

#define MATGEN_LOG_WARN(...) \
  matgen_log(MATGEN_LOG_LEVEL_WARN, __FILE__, __LINE__, __func__, __VA_ARGS__)

#define MATGEN_LOG_ERROR(...) \
  matgen_log(MATGEN_LOG_LEVEL_ERROR, __FILE__, __LINE__, __func__, __VA_ARGS__)

#define MATGEN_LOG_FATAL(...) \
  matgen_log(MATGEN_LOG_LEVEL_FATAL, __FILE__, __LINE__, __func__, __VA_ARGS__)

#else

// Logging disabled at compile time - all macros become no-ops
#define MATGEN_LOG_TRACE(...) ((void)0)
#define MATGEN_LOG_DEBUG(...) ((void)0)
#define MATGEN_LOG_INFO(...) ((void)0)
#define MATGEN_LOG_WARN(...) ((void)0)
#define MATGEN_LOG_ERROR(...) ((void)0)
#define MATGEN_LOG_FATAL(...) ((void)0)

#endif  // MATGEN_LOG_DISABLE

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_UTIL_LOG_H
