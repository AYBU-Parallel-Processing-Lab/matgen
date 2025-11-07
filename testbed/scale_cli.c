/**
 * @file scale_matrix.c
 * @brief CLI tool for sparse matrix scaling with nearest neighbor and bilinear
 * interpolation
 *
 * Usage:
 *   scale_matrix -i <input.mtx> -o <output.mtx> -m <method> -r <rows> -c <cols>
 * [options]
 *
 * Methods:
 *   nearest  - Nearest neighbor interpolation
 *   bilinear - Bilinear interpolation
 *
 * Examples:
 *   # Scale to 100x100 using bilinear interpolation
 *   scale_matrix -i input.mtx -o output.mtx -m bilinear -r 100 -c 100
 *
 *   # Scale to 200x200 using nearest neighbor with averaging on collision
 *   scale_matrix -i input.mtx -o output.mtx -m nearest -r 200 -c 200
 * --collision avg
 *
 *   # Scale with verbose output and statistics
 *   scale_matrix -i input.mtx -o output.mtx -m bilinear -r 50 -c 50 -v -s
 */

#include <matgen/algorithms/scaling/bilinear.h>
#include <matgen/algorithms/scaling/nearest_neighbor.h>
#include <matgen/core/conversion.h>
#include <matgen/core/coo_matrix.h>
#include <matgen/core/csr_matrix.h>
#include <matgen/io/mtx_reader.h>
#include <matgen/io/mtx_writer.h>
#include <matgen/utils/log.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// =============================================================================
// Configuration Structure
// =============================================================================

typedef struct {
  const char* input_file;
  const char* output_file;
  const char* method;  // "nearest" or "bilinear"
  matgen_index_t new_rows;
  matgen_index_t new_cols;
  matgen_collision_policy_t collision_policy;
  bool verbose;
  bool show_stats;
  bool quiet;
} cli_config_t;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Print usage information
 */
static void print_usage(const char* prog_name) {
  printf("Usage: %s [options]\n", prog_name);
  printf("\n");
  printf("Required arguments:\n");
  printf("  -i, --input <file>     Input matrix file (.mtx format)\n");
  printf("  -o, --output <file>    Output matrix file (.mtx format)\n");
  printf("  -m, --method <method>  Scaling method: 'nearest' or 'bilinear'\n");
  printf("  -r, --rows <N>         Target number of rows\n");
  printf("  -c, --cols <N>         Target number of columns\n");
  printf("\n");
  printf("Optional arguments:\n");
  printf("  --collision <policy>   Collision policy for nearest neighbor:\n");
  printf("                         'sum' (default), 'avg', or 'max'\n");
  printf("  -v, --verbose          Enable verbose output\n");
  printf("  -s, --stats            Show detailed statistics\n");
  printf("  -q, --quiet            Suppress all non-error output\n");
  printf("  -h, --help             Show this help message\n");
  printf("\n");
  printf("Examples:\n");
  printf("  # Scale to 100x100 using bilinear interpolation\n");
  printf("  %s -i input.mtx -o output.mtx -m bilinear -r 100 -c 100\n",
         prog_name);
  printf("\n");
  printf(
      "  # Scale to 200x200 using nearest neighbor with max collision "
      "policy\n");
  printf(
      "  %s -i input.mtx -o output.mtx -m nearest -r 200 -c 200 --collision "
      "max\n",
      prog_name);
  printf("\n");
}

/**
 * @brief Parse command line arguments
 */
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static bool parse_args(int argc, char** argv, cli_config_t* config) {
  // Initialize defaults
  config->input_file = NULL;
  config->output_file = NULL;
  config->method = NULL;
  config->new_rows = 0;
  config->new_cols = 0;
  config->collision_policy = MATGEN_COLLISION_SUM;
  config->verbose = false;
  config->show_stats = false;
  config->quiet = false;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      return false;
    }

    if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: %s requires an argument\n", argv[i - 1]);
        return false;
      }
      config->input_file = argv[i];
    } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: %s requires an argument\n", argv[i - 1]);
        return false;
      }
      config->output_file = argv[i];
    } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--method") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: %s requires an argument\n", argv[i - 1]);
        return false;
      }
      config->method = argv[i];
    } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--rows") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: %s requires an argument\n", argv[i - 1]);
        return false;
      }
      config->new_rows = (matgen_index_t)atoll(argv[i]);
    } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--cols") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: %s requires an argument\n", argv[i - 1]);
        return false;
      }
      config->new_cols = (matgen_index_t)atoll(argv[i]);
    } else if (strcmp(argv[i], "--collision") == 0) {
      if (++i >= argc) {
        fprintf(stderr, "Error: --collision requires an argument\n");
        return false;
      }
      if (strcmp(argv[i], "sum") == 0) {
        config->collision_policy = MATGEN_COLLISION_SUM;
      } else if (strcmp(argv[i], "avg") == 0) {
        config->collision_policy = MATGEN_COLLISION_AVG;
      } else if (strcmp(argv[i], "max") == 0) {
        config->collision_policy = MATGEN_COLLISION_MAX;
      } else {
        fprintf(stderr, "Error: Unknown collision policy '%s'\n", argv[i]);
        return false;
      }
    } else if (strcmp(argv[i], "-v") == 0 ||
               strcmp(argv[i], "--verbose") == 0) {
      config->verbose = true;
    } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--stats") == 0) {
      config->show_stats = true;
    } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
      config->quiet = true;
    } else {
      fprintf(stderr, "Error: Unknown argument '%s'\n", argv[i]);
      return false;
    }
  }

  // Validate required arguments
  if (!config->input_file) {
    fprintf(stderr, "Error: Input file (-i) is required\n");
    return false;
  }
  if (!config->output_file) {
    fprintf(stderr, "Error: Output file (-o) is required\n");
    return false;
  }
  if (!config->method) {
    fprintf(stderr, "Error: Method (-m) is required\n");
    return false;
  }
  if (config->new_rows == 0) {
    fprintf(stderr, "Error: Target rows (-r) is required and must be > 0\n");
    return false;
  }
  if (config->new_cols == 0) {
    fprintf(stderr, "Error: Target cols (-c) is required and must be > 0\n");
    return false;
  }

  // Validate method
  if (strcmp(config->method, "nearest") != 0 &&
      strcmp(config->method, "bilinear") != 0) {
    fprintf(stderr, "Error: Method must be 'nearest' or 'bilinear', got '%s'\n",
            config->method);
    return false;
  }

  return true;
}

/**
 * @brief Compute and print matrix statistics
 */
static void print_matrix_stats(const char* label,
                               const matgen_csr_matrix_t* matrix) {
  if (!matrix) {
    printf("%s: NULL matrix\n", label);
    return;
  }

  // Compute statistics
  matgen_value_t sum = 0.0;
  matgen_value_t sum_sq = 0.0;
  matgen_value_t min_val = 0.0;
  matgen_value_t max_val = 0.0;
  bool first = true;

  for (matgen_index_t i = 0; i < matrix->rows; i++) {
    matgen_size_t row_start = matrix->row_ptr[i];
    matgen_size_t row_end = matrix->row_ptr[i + 1];

    for (matgen_size_t idx = row_start; idx < row_end; idx++) {
      matgen_value_t val = matrix->values[idx];
      sum += val;
      sum_sq += val * val;

      if (first) {
        min_val = val;
        max_val = val;
        first = false;
      } else {
        if (val < min_val) {
          min_val = val;
        }

        if (val > max_val) {
          max_val = val;
        }
      }
    }
  }

  matgen_value_t mean = matrix->nnz > 0 ? sum / (double)matrix->nnz : 0.0;
  matgen_value_t variance =
      matrix->nnz > 0 ? (sum_sq / (double)matrix->nnz) - (mean * mean) : 0.0;
  matgen_value_t std_dev = sqrt(variance);
  matgen_value_t frobenius = sqrt(sum_sq);

  double density = (double)matrix->nnz / (double)(matrix->rows * matrix->cols);
  double sparsity = 1.0 - density;

  printf("\n%s:\n", label);
  printf("  Dimensions:      %llu × %llu\n", (unsigned long long)matrix->rows,
         (unsigned long long)matrix->cols);
  printf("  Non-zeros (NNZ): %llu\n", (unsigned long long)matrix->nnz);
  printf("  Density:         %.6f (%.2f%%)\n", density, density * 100.0);
  printf("  Sparsity:        %.6f (%.2f%%)\n", sparsity, sparsity * 100.0);

  if (matrix->nnz > 0) {
    printf("  Value range:     [%.6e, %.6e]\n", min_val, max_val);
    printf("  Sum:             %.6e\n", sum);
    printf("  Mean:            %.6e\n", mean);
    printf("  Std Dev:         %.6e\n", std_dev);
    printf("  Frobenius norm:  %.6e\n", frobenius);
  }
}

/**
 * @brief Get collision policy name as string
 */
static const char* collision_policy_name(matgen_collision_policy_t policy) {
  switch (policy) {
    case MATGEN_COLLISION_SUM:
      return "sum";
    case MATGEN_COLLISION_AVG:
      return "average";
    case MATGEN_COLLISION_MAX:
      return "max";
    default:
      return "unknown";
  }
}

// =============================================================================
// Main Function
// =============================================================================

int main(int argc, char** argv) {
  cli_config_t config;

  // Parse arguments
  if (!parse_args(argc, argv, &config)) {
    print_usage(argv[0]);
    return 1;
  }

  // Configure logging
  if (config.quiet) {
    matgen_log_set_level(MATGEN_LOG_LEVEL_ERROR);
  } else if (config.verbose) {
    matgen_log_set_level(MATGEN_LOG_LEVEL_DEBUG);
  } else {
    matgen_log_set_level(MATGEN_LOG_LEVEL_INFO);
  }

  if (!config.quiet) {
    printf(
        "======================================================================"
        "=======\n");
    printf("MatGen Matrix Scaling Tool\n");
    printf(
        "======================================================================"
        "=======\n");
    printf("Input:           %s\n", config.input_file);
    printf("Output:          %s\n", config.output_file);
    printf("Method:          %s\n", config.method);
    printf("Target size:     %llu × %llu\n",
           (unsigned long long)config.new_rows,
           (unsigned long long)config.new_cols);
    if (strcmp(config.method, "nearest") == 0) {
      printf("Collision:       %s\n",
             collision_policy_name(config.collision_policy));
    }
    printf(
        "======================================================================"
        "=======\n");
  }

  // Step 1: Load input matrix
  if (!config.quiet) {
    printf("\n[1/4] Loading input matrix...\n");
  }

  matgen_coo_matrix_t* input_coo = matgen_mtx_read(config.input_file, NULL);
  if (!input_coo) {
    fprintf(stderr, "Error: Failed to load input matrix from '%s'\n",
            config.input_file);
    return 1;
  }

  if (config.verbose) {
    printf("  Loaded COO matrix: %llu × %llu, nnz = %llu\n",
           (unsigned long long)input_coo->rows,
           (unsigned long long)input_coo->cols,
           (unsigned long long)input_coo->nnz);
  }

  // Step 2: Convert to CSR
  if (!config.quiet) {
    printf("[2/4] Converting to CSR format...\n");
  }

  matgen_csr_matrix_t* input_csr = matgen_coo_to_csr(input_coo);
  matgen_coo_destroy(input_coo);  // Free COO, we only need CSR now

  if (!input_csr) {
    fprintf(stderr, "Error: Failed to convert matrix to CSR format\n");
    return 1;
  }

  if (config.show_stats) {
    print_matrix_stats("Input Matrix Statistics", input_csr);
  }

  // Step 3: Scale matrix
  if (!config.quiet) {
    printf("\n[3/4] Scaling matrix using %s interpolation...\n", config.method);
  }

  matgen_csr_matrix_t* output_csr = NULL;
  matgen_error_t err;

  clock_t start = clock();

  if (strcmp(config.method, "nearest") == 0) {
    err = matgen_scale_nearest_neighbor(input_csr, config.new_rows,
                                        config.new_cols,
                                        config.collision_policy, &output_csr);
  } else {  // bilinear
    err = matgen_scale_bilinear(input_csr, config.new_rows, config.new_cols,
                                &output_csr);
  }

  clock_t end = clock();
  double elapsed_sec = (double)(end - start) / CLOCKS_PER_SEC;

  if (err != MATGEN_SUCCESS) {
    fprintf(stderr, "Error: Matrix scaling failed with error code %d\n", err);
    matgen_csr_destroy(input_csr);
    return 1;
  }

  if (!config.quiet) {
    printf("  Scaling completed in %.3f ms (%.6f seconds)\n",
           elapsed_sec * 1000.0, elapsed_sec);
  }

  if (config.show_stats) {
    print_matrix_stats("Output Matrix Statistics", output_csr);
  }

  // Step 4: Write output matrix
  if (!config.quiet) {
    printf("\n[4/4] Writing output matrix...\n");
  }

  err = matgen_mtx_write_csr(config.output_file, output_csr);
  if (err != MATGEN_SUCCESS) {
    fprintf(stderr, "Error: Failed to write output matrix to '%s'\n",
            config.output_file);
    matgen_csr_destroy(input_csr);
    matgen_csr_destroy(output_csr);
    return 1;
  }

  if (!config.quiet) {
    printf("  Output written successfully to '%s'\n", config.output_file);
  }

  // Summary
  if (!config.quiet) {
    printf(
        "\n===================================================================="
        "=========\n");
    printf("Scaling Summary\n");
    printf(
        "======================================================================"
        "=======\n");
    printf("Input:           %llu × %llu, nnz = %llu\n",
           (unsigned long long)input_csr->rows,
           (unsigned long long)input_csr->cols,
           (unsigned long long)input_csr->nnz);
    printf("Output:          %llu × %llu, nnz = %llu\n",
           (unsigned long long)output_csr->rows,
           (unsigned long long)output_csr->cols,
           (unsigned long long)output_csr->nnz);

    double input_density =
        (double)input_csr->nnz / (double)(input_csr->rows * input_csr->cols);
    double output_density =
        (double)output_csr->nnz / (double)(output_csr->rows * output_csr->cols);

    printf("Input density:   %.6f (%.2f%%)\n", input_density,
           input_density * 100.0);
    printf("Output density:  %.6f (%.2f%%)\n", output_density,
           output_density * 100.0);
    printf("Time:            %.3f ms\n", elapsed_sec * 1000.0);
    printf(
        "======================================================================"
        "=======\n");
    printf("\nDone!\n");
  }

  // Cleanup
  matgen_csr_destroy(input_csr);
  matgen_csr_destroy(output_csr);

  return 0;
}
