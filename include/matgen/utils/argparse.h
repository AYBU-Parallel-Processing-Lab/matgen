#ifndef MATGEN_UTILS_ARGPARSE_H
#define MATGEN_UTILS_ARGPARSE_H

/**
 * @file argparse.h
 * @brief Simple command-line argument parser for MatGen
 *
 * Lightweight argument parsing with support for:
 * - Short options (-h, -v)
 * - Long options (--help, --version)
 * - Options with arguments (-o file, --output=file)
 * - Positional arguments
 * - Automatic help generation
 */

#include <stdio.h>

#include "matgen/core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Types
// =============================================================================

/**
 * @brief Argument types
 */
typedef enum {
  MATGEN_ARG_BOOL,       // Boolean flag (no argument)
  MATGEN_ARG_I32,        // 32-bit signed integer
  MATGEN_ARG_I64,        // 64-bit signed integer
  MATGEN_ARG_U32,        // 32-bit unsigned integer
  MATGEN_ARG_U64,        // 64-bit unsigned integer
  MATGEN_ARG_F64,        // Double precision float
  MATGEN_ARG_STRING,     // String argument
  MATGEN_ARG_POSITIONAL  // Positional argument
} matgen_arg_type_t;

/**
 * @brief Argument definition
 */
typedef struct {
  const char* short_opt;   // Short option (e.g., "h" for -h), can be NULL
  const char* long_opt;    // Long option (e.g., "help" for --help), can be NULL
  matgen_arg_type_t type;  // Argument type
  void* dest;              // Pointer to store the parsed value
  const char* help;        // Help text
  const char* metavar;     // Variable name for help (e.g., "FILE", "NUM")
  const char* default_str;  // Default value as string (for display)
  bool required;            // Whether the argument is required
} matgen_arg_t;

/**
 * @brief Argument parser context (opaque)
 */
typedef struct matgen_argparser matgen_argparser_t;

// =============================================================================
// Parser Creation and Destruction
// =============================================================================

/**
 * @brief Create a new argument parser
 *
 * @param program_name Name of the program
 * @param description Brief description of the program
 * @return New parser instance, or NULL on error
 */
matgen_argparser_t* matgen_argparser_create(const char* program_name,
                                            const char* description);

/**
 * @brief Destroy an argument parser
 *
 * @param parser Parser to destroy (can be NULL)
 */
void matgen_argparser_destroy(matgen_argparser_t* parser);

// =============================================================================
// Adding Arguments
// =============================================================================

/**
 * @brief Add an argument definition to the parser
 *
 * @param parser Parser instance
 * @param arg Argument definition
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_argparser_add(matgen_argparser_t* parser,
                                    const matgen_arg_t* arg);

/**
 * @brief Add a boolean flag (e.g., --verbose, -v)
 *
 * @param parser Parser instance
 * @param short_opt Short option (can be NULL)
 * @param long_opt Long option (can be NULL)
 * @param dest Pointer to bool to store result
 * @param help Help text
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_argparser_add_flag(matgen_argparser_t* parser,
                                         const char* short_opt,
                                         const char* long_opt, bool* dest,
                                         const char* help);

/**
 * @brief Add a 64-bit unsigned integer option (e.g., --count=10, -n 10)
 *
 * @param parser Parser instance
 * @param short_opt Short option (can be NULL)
 * @param long_opt Long option (can be NULL)
 * @param dest Pointer to u64 to store result
 * @param default_val Default value
 * @param help Help text
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_argparser_add_u64(matgen_argparser_t* parser,
                                        const char* short_opt,
                                        const char* long_opt, u64* dest,
                                        u64 default_val, const char* help);

/**
 * @brief Add a double precision float option (e.g., --threshold=0.5, -t 0.5)
 *
 * @param parser Parser instance
 * @param short_opt Short option (can be NULL)
 * @param long_opt Long option (can be NULL)
 * @param dest Pointer to f64 to store result
 * @param default_val Default value
 * @param help Help text
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_argparser_add_f64(matgen_argparser_t* parser,
                                        const char* short_opt,
                                        const char* long_opt, f64* dest,
                                        f64 default_val, const char* help);

/**
 * @brief Add a string option (e.g., --output=file.txt, -o file.txt)
 *
 * @param parser Parser instance
 * @param short_opt Short option (can be NULL)
 * @param long_opt Long option (can be NULL)
 * @param dest Pointer to const char* to store result
 * @param default_val Default value (can be NULL)
 * @param help Help text
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_argparser_add_string(
    matgen_argparser_t* parser, const char* short_opt, const char* long_opt,
    const char** dest, const char* default_val, const char* help);

// =============================================================================
// Parsing
// =============================================================================

/**
 * @brief Parse command-line arguments
 *
 * @param parser Parser instance
 * @param argc Argument count from main()
 * @param argv Argument vector from main()
 * @return MATGEN_SUCCESS on success, error code on failure
 */
matgen_error_t matgen_argparser_parse(matgen_argparser_t* parser, i32 argc,
                                      char** argv);

/**
 * @brief Print help message
 *
 * @param parser Parser instance
 * @param stream Output stream (e.g., stdout, stderr)
 */
void matgen_argparser_print_help(const matgen_argparser_t* parser,
                                 FILE* stream);

/**
 * @brief Print usage line
 *
 * @param parser Parser instance
 * @param stream Output stream (e.g., stdout, stderr)
 */
void matgen_argparser_print_usage(const matgen_argparser_t* parser,
                                  FILE* stream);

#ifdef __cplusplus
}
#endif

#endif  // MATGEN_UTILS_ARGPARSE_H
