#ifndef MATGEN_UTIL_ARGPARSE_H
#define MATGEN_UTIL_ARGPARSE_H

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

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

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
  MATGEN_ARG_INT,        // Integer argument
  MATGEN_ARG_DOUBLE,     // Double argument
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
  const char* default_value;  // Default value as string (for display purposes)
  bool required;              // Whether the argument is required
} matgen_arg_t;

/**
 * @brief Argument parser context
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
 * @param parser Parser to destroy
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
 * @return 0 on success, non-zero on error
 */
int matgen_argparser_add(matgen_argparser_t* parser, const matgen_arg_t* arg);

/**
 * @brief Add a boolean flag (e.g., --verbose, -v)
 *
 * @param parser Parser instance
 * @param short_opt Short option (can be NULL)
 * @param long_opt Long option (can be NULL)
 * @param dest Pointer to bool to store result
 * @param help Help text
 * @return 0 on success, non-zero on error
 */
int matgen_argparser_add_flag(matgen_argparser_t* parser, const char* short_opt,
                              const char* long_opt, bool* dest,
                              const char* help);

/**
 * @brief Add an integer option (e.g., --count=10, -n 10)
 *
 * @param parser Parser instance
 * @param short_opt Short option (can be NULL)
 * @param long_opt Long option (can be NULL)
 * @param dest Pointer to int to store result
 * @param default_val Default value
 * @param help Help text
 * @return 0 on success, non-zero on error
 */
int matgen_argparser_add_int(matgen_argparser_t* parser, const char* short_opt,
                             const char* long_opt, int* dest, int default_val,
                             const char* help);

/**
 * @brief Add a string option (e.g., --output=file.txt, -o file.txt)
 *
 * @param parser Parser instance
 * @param short_opt Short option (can be NULL)
 * @param long_opt Long option (can be NULL)
 * @param dest Pointer to const char* to store result
 * @param default_val Default value (can be NULL)
 * @param help Help text
 * @return 0 on success, non-zero on error
 */
int matgen_argparser_add_string(matgen_argparser_t* parser,
                                const char* short_opt, const char* long_opt,
                                const char** dest, const char* default_val,
                                const char* help);

// =============================================================================
// Parsing
// =============================================================================

/**
 * @brief Parse command-line arguments
 *
 * @param parser Parser instance
 * @param argc Argument count from main()
 * @param argv Argument vector from main()
 * @return 0 on success, non-zero on error
 */
int matgen_argparser_parse(matgen_argparser_t* parser, int argc, char** argv);

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

#endif  // MATGEN_UTIL_ARGPARSE_H
