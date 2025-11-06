#include "matgen/utils/argparse.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/utils/log.h"

// =============================================================================
// Internal Structures
// =============================================================================

#define MAX_ARGS 64

struct matgen_argparser {
  const char* program_name;
  const char* description;
  matgen_arg_t args[MAX_ARGS];
  char* allocated_strs[MAX_ARGS];
  u32 arg_count;
  u32 alloc_count;
};

// =============================================================================
// Parser Creation and Destruction
// =============================================================================

matgen_argparser_t* matgen_argparser_create(const char* program_name,
                                            const char* description) {
  if (!program_name) {
    MATGEN_LOG_ERROR("NULL program_name");
    return NULL;
  }

  matgen_argparser_t* parser =
      (matgen_argparser_t*)malloc(sizeof(matgen_argparser_t));
  if (!parser) {
    MATGEN_LOG_ERROR("Failed to allocate argparser");
    return NULL;
  }

  parser->program_name = program_name;
  parser->description = description;
  parser->arg_count = 0;
  parser->alloc_count = 0;

  return parser;
}

void matgen_argparser_destroy(matgen_argparser_t* parser) {
  if (!parser) {
    return;
  }

  // Free allocated default strings
  for (u32 i = 0; i < parser->alloc_count; i++) {
    free(parser->allocated_strs[i]);
  }

  free(parser);
}

// =============================================================================
// Adding Arguments
// =============================================================================

matgen_error_t matgen_argparser_add(matgen_argparser_t* parser,
                                    const matgen_arg_t* arg) {
  if (!parser || !arg) {
    MATGEN_LOG_ERROR("NULL pointer argument");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (parser->arg_count >= MAX_ARGS) {
    MATGEN_LOG_ERROR("Too many arguments (max %d)", MAX_ARGS);
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (!arg->short_opt && !arg->long_opt) {
    MATGEN_LOG_ERROR(
        "Argument must have at least one of short_opt or long_opt");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (!arg->dest) {
    MATGEN_LOG_ERROR("Argument dest cannot be NULL");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  parser->args[parser->arg_count] = *arg;
  parser->arg_count++;

  return MATGEN_SUCCESS;
}

matgen_error_t matgen_argparser_add_flag(matgen_argparser_t* parser,
                                         const char* short_opt,
                                         const char* long_opt, bool* dest,
                                         const char* help) {
  matgen_arg_t arg = {.short_opt = short_opt,
                      .long_opt = long_opt,
                      .type = MATGEN_ARG_BOOL,
                      .dest = dest,
                      .help = help,
                      .metavar = NULL,
                      .default_str = "false",
                      .required = false};

  // Set default value
  if (dest) {
    *dest = false;
  }

  return matgen_argparser_add(parser, &arg);
}

matgen_error_t matgen_argparser_add_u64(matgen_argparser_t* parser,
                                        const char* short_opt,
                                        const char* long_opt, u64* dest,
                                        u64 default_val, const char* help) {
  if (!parser || !dest) {
    MATGEN_LOG_ERROR("NULL pointer argument");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (parser->alloc_count >= MAX_ARGS) {
    MATGEN_LOG_ERROR("Too many allocated strings");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Allocate buffer for default string
  char* default_str = (char*)malloc(32);
  if (!default_str) {
    MATGEN_LOG_ERROR("Failed to allocate default string");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }
  snprintf(default_str, 32, "%llu", (unsigned long long)default_val);

  matgen_arg_t arg = {.short_opt = short_opt,
                      .long_opt = long_opt,
                      .type = MATGEN_ARG_U64,
                      .dest = dest,
                      .help = help,
                      .metavar = "NUM",
                      .default_str = default_str,
                      .required = false};

  // Set default value
  *dest = default_val;

  matgen_error_t err = matgen_argparser_add(parser, &arg);
  if (err != MATGEN_SUCCESS) {
    free(default_str);
    return err;
  }

  // Track allocation
  parser->allocated_strs[parser->alloc_count++] = default_str;

  return MATGEN_SUCCESS;
}

matgen_error_t matgen_argparser_add_f64(matgen_argparser_t* parser,
                                        const char* short_opt,
                                        const char* long_opt, f64* dest,
                                        f64 default_val, const char* help) {
  if (!parser || !dest) {
    MATGEN_LOG_ERROR("NULL pointer argument");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (parser->alloc_count >= MAX_ARGS) {
    MATGEN_LOG_ERROR("Too many allocated strings");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Allocate buffer for default string
  char* default_str = (char*)malloc(32);
  if (!default_str) {
    MATGEN_LOG_ERROR("Failed to allocate default string");
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }
  snprintf(default_str, 32, "%.6g", default_val);

  matgen_arg_t arg = {.short_opt = short_opt,
                      .long_opt = long_opt,
                      .type = MATGEN_ARG_F64,
                      .dest = dest,
                      .help = help,
                      .metavar = "NUM",
                      .default_str = default_str,
                      .required = false};

  // Set default value
  *dest = default_val;

  matgen_error_t err = matgen_argparser_add(parser, &arg);
  if (err != MATGEN_SUCCESS) {
    free(default_str);
    return err;
  }

  // Track allocation
  parser->allocated_strs[parser->alloc_count++] = default_str;

  return MATGEN_SUCCESS;
}

matgen_error_t matgen_argparser_add_string(
    matgen_argparser_t* parser, const char* short_opt, const char* long_opt,
    const char** dest, const char* default_val, const char* help) {
  matgen_arg_t arg = {.short_opt = short_opt,
                      .long_opt = long_opt,
                      .type = MATGEN_ARG_STRING,
                      .dest = (void*)dest,
                      .help = help,
                      .metavar = "STR",
                      .default_str = default_val ? default_val : "NULL",
                      .required = false};

  // Set default value
  if (dest) {
    *dest = default_val;
  }

  return matgen_argparser_add(parser, &arg);
}

// =============================================================================
// Parsing Helpers
// =============================================================================

static matgen_arg_t* find_short_opt(matgen_argparser_t* parser,
                                    const char* opt) {
  for (u32 i = 0; i < parser->arg_count; i++) {
    if (parser->args[i].short_opt &&
        strcmp(parser->args[i].short_opt, opt) == 0) {
      return &parser->args[i];
    }
  }
  return NULL;
}

static matgen_arg_t* find_long_opt(matgen_argparser_t* parser,
                                   const char* opt) {
  for (u32 i = 0; i < parser->arg_count; i++) {
    if (parser->args[i].long_opt &&
        strcmp(parser->args[i].long_opt, opt) == 0) {
      return &parser->args[i];
    }
  }
  return NULL;
}

static matgen_error_t parse_value(matgen_arg_t* arg, const char* value_str) {
  if (!arg || !value_str) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  switch (arg->type) {
    case MATGEN_ARG_BOOL:
      *(bool*)arg->dest = true;
      break;

    case MATGEN_ARG_U64: {
      char* endptr;
      unsigned long long val = strtoull(value_str, &endptr, 10);
      if (*endptr != '\0') {
        MATGEN_LOG_ERROR("Invalid u64 value: %s", value_str);
        return MATGEN_ERROR_INVALID_ARGUMENT;
      }
      *(u64*)arg->dest = (u64)val;
      break;
    }

    case MATGEN_ARG_I64: {
      char* endptr;
      long long val = strtoll(value_str, &endptr, 10);
      if (*endptr != '\0') {
        MATGEN_LOG_ERROR("Invalid i64 value: %s", value_str);
        return MATGEN_ERROR_INVALID_ARGUMENT;
      }
      *(i64*)arg->dest = (i64)val;
      break;
    }

    case MATGEN_ARG_F64: {
      char* endptr;
      double val = strtod(value_str, &endptr);
      if (*endptr != '\0') {
        MATGEN_LOG_ERROR("Invalid f64 value: %s", value_str);
        return MATGEN_ERROR_INVALID_ARGUMENT;
      }
      *(f64*)arg->dest = val;
      break;
    }

    case MATGEN_ARG_STRING:
      *(const char**)arg->dest = value_str;
      break;

    default:
      MATGEN_LOG_ERROR("Unknown argument type: %d", arg->type);
      return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  return MATGEN_SUCCESS;
}

// =============================================================================
// Parsing
// =============================================================================

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
matgen_error_t matgen_argparser_parse(matgen_argparser_t* parser, i32 argc,
                                      char** argv) {
  if (!parser || !argv) {
    MATGEN_LOG_ERROR("NULL pointer argument");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  for (i32 i = 1; i < argc; i++) {
    const char* arg_str = argv[i];

    // Check for long option (--option)
    if (strncmp(arg_str, "--", 2) == 0) {
      const char* opt_name = arg_str + 2;
      const char* eq_pos = strchr(opt_name, '=');
      char opt_buf[256];

      if (eq_pos) {
        // Format: --option=value
        size_t len = eq_pos - opt_name;
        if (len >= sizeof(opt_buf)) {
          len = sizeof(opt_buf) - 1;
        }
        strncpy(opt_buf, opt_name, len);
        opt_buf[len] = '\0';

        matgen_arg_t* arg = find_long_opt(parser, opt_buf);
        if (!arg) {
          MATGEN_LOG_ERROR("Unknown option: --%s", opt_buf);
          return MATGEN_ERROR_INVALID_ARGUMENT;
        }

        if (parse_value(arg, eq_pos + 1) != MATGEN_SUCCESS) {
          return MATGEN_ERROR_INVALID_ARGUMENT;
        }
      } else {
        // Format: --option [value]
        matgen_arg_t* arg = find_long_opt(parser, opt_name);
        if (!arg) {
          MATGEN_LOG_ERROR("Unknown option: --%s", opt_name);
          return MATGEN_ERROR_INVALID_ARGUMENT;
        }

        if (arg->type == MATGEN_ARG_BOOL) {
          parse_value(arg, "true");
        } else {
          // Need next argument
          if (i + 1 >= argc) {
            MATGEN_LOG_ERROR("Option --%s requires an argument", opt_name);
            return MATGEN_ERROR_INVALID_ARGUMENT;
          }
          i++;
          if (parse_value(arg, argv[i]) != MATGEN_SUCCESS) {
            return MATGEN_ERROR_INVALID_ARGUMENT;
          }
        }
      }
    }
    // Check for short option (-o)
    else if (arg_str[0] == '-' && arg_str[1] != '\0' && arg_str[1] != '-') {
      const char* opt_name = arg_str + 1;

      matgen_arg_t* arg = find_short_opt(parser, opt_name);
      if (!arg) {
        MATGEN_LOG_ERROR("Unknown option: -%s", opt_name);
        return MATGEN_ERROR_INVALID_ARGUMENT;
      }

      if (arg->type == MATGEN_ARG_BOOL) {
        parse_value(arg, "true");
      } else {
        // Need next argument
        if (i + 1 >= argc) {
          MATGEN_LOG_ERROR("Option -%s requires an argument", opt_name);
          return MATGEN_ERROR_INVALID_ARGUMENT;
        }
        i++;
        if (parse_value(arg, argv[i]) != MATGEN_SUCCESS) {
          return MATGEN_ERROR_INVALID_ARGUMENT;
        }
      }
    }
    // Positional argument
    else {
      MATGEN_LOG_WARN("Ignoring positional argument: %s", arg_str);
    }
  }

  return MATGEN_SUCCESS;
}

// =============================================================================
// Help
// =============================================================================

void matgen_argparser_print_usage(const matgen_argparser_t* parser,
                                  FILE* stream) {
  if (!parser || !stream) {
    return;
  }

  fprintf(stream, "Usage: %s [OPTIONS]\n", parser->program_name);
}

void matgen_argparser_print_help(const matgen_argparser_t* parser,
                                 FILE* stream) {
  if (!parser || !stream) {
    return;
  }

  // Print usage
  matgen_argparser_print_usage(parser, stream);

  if (parser->description) {
    fprintf(stream, "\n%s\n", parser->description);
  }

  fprintf(stream, "\nOptions:\n");

  // Print all arguments
  for (u32 i = 0; i < parser->arg_count; i++) {
    const matgen_arg_t* arg = &parser->args[i];

    fprintf(stream, "  ");

    // Print short option
    if (arg->short_opt) {
      fprintf(stream, "-%s", arg->short_opt);
      if (arg->long_opt) {
        fprintf(stream, ", ");
      }
    } else {
      fprintf(stream, "    ");
    }

    // Print long option
    if (arg->long_opt) {
      fprintf(stream, "--%s", arg->long_opt);
    }

    // Print metavar
    if (arg->type != MATGEN_ARG_BOOL && arg->metavar) {
      fprintf(stream, " <%s>", arg->metavar);
    }

    fprintf(stream, "\n");

    // Print help text
    if (arg->help) {
      fprintf(stream, "      %s", arg->help);
      if (arg->default_str && arg->type != MATGEN_ARG_BOOL) {
        fprintf(stream, " (default: %s)", arg->default_str);
      }
      fprintf(stream, "\n");
    }
  }
}
