#include "backends/omp/internal/lanczos_omp.h"

#ifdef MATGEN_HAS_OPENMP

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/core/matrix/csr.h"
#include "matgen/core/types.h"
#include "matgen/utils/log.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define LANCZOS_A 3  /* Kernel width parameter */

// =============================================================================
// Internal Structures
// =============================================================================

typedef struct {
  matgen_index_t row;
  matgen_index_t col;
  matgen_value_t val;
} entry_t;

// =============================================================================
// Helper Functions
// =============================================================================

static double lanczos_kernel(double x, int a) {
  if (x == 0.0) return 1.0;
  double ax = fabs(x);
  if (ax >= (double)a) return 0.0;

  double pix = M_PI * x;
  double pix_over_a = pix / (double)a;
  double num = (double)a * sin(pix) * sin(pix_over_a);
  double den = (M_PI * M_PI) * (x * x);
  return num / den;
}

static matgen_value_t fetch_csr_value(const matgen_csr_matrix_t* m,
                                      matgen_index_t row,
                                      matgen_index_t col) {
  if (row >= m->rows || col >= m->cols) return 0.0;

  matgen_size_t start = m->row_ptr[row];
  matgen_size_t end = m->row_ptr[row + 1];

  // Binary search in row
  while (start < end) {
    matgen_size_t mid = start + ((end - start) >> 1);
    matgen_index_t c = m->col_indices[mid];
    if (c == col) return m->values[mid];
    if (c < col) start = mid + 1;
    else end = mid;
  }
  return 0.0;
}

static int entry_compare(const void* a, const void* b) {
  const entry_t* ea = (const entry_t*)a;
  const entry_t* eb = (const entry_t*)b;
  if (ea->row != eb->row) return (ea->row < eb->row) ? -1 : 1;
  if (ea->col != eb->col) return (ea->col < eb->col) ? -1 : 1;
  return 0;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
matgen_error_t matgen_scale_lanczos_omp(const matgen_csr_matrix_t* source,
                                        matgen_index_t new_size,
                                        matgen_csr_matrix_t** result) {
  if (!source || !result) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (source->rows != source->cols) {
    MATGEN_LOG_ERROR("Lanczos scaling requires square matrices");
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  if (new_size == 0) {
    return MATGEN_ERROR_INVALID_ARGUMENT;
  }

  *result = NULL;

  matgen_index_t src_size = source->rows;
  double scale = (double)new_size / (double)src_size;
  double inv_scale = 1.0 / scale;

  int num_threads = omp_get_max_threads();
  MATGEN_LOG_DEBUG("Lanczos scaling (OMP): %llu -> %llu (scale: %.3f), threads=%d",
                   (unsigned long long)src_size,
                   (unsigned long long)new_size, scale, num_threads);

  // Calculate expansion radius
  int expansion_radius = 0;
  if (scale > 1.0) {
    expansion_radius = (int)ceil(sqrt(scale));
    if (expansion_radius < 2) expansion_radius = 2;
    if (expansion_radius > 6) expansion_radius = 6;
  }

  // Estimate max candidates
  size_t neighborhood_size =
      (size_t)(2 * expansion_radius + 1) * (2 * expansion_radius + 1);
  size_t max_candidates = (size_t)source->nnz * neighborhood_size;
  size_t size_limit = (size_t)new_size * (size_t)new_size;
  if (max_candidates > size_limit) max_candidates = size_limit;

  // Allocate candidate buffer
  entry_t* candidates = (entry_t*)malloc(max_candidates * sizeof(entry_t));
  if (!candidates) {
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  size_t num_candidates = 0;

  // Generate candidates
  for (matgen_index_t r = 0; r < src_size; ++r) {
    for (matgen_size_t idx = source->row_ptr[r]; idx < source->row_ptr[r + 1];
         ++idx) {
      matgen_index_t c = source->col_indices[idx];

      // Scale to new coordinates
      matgen_index_t base_row = (matgen_index_t)llround((double)r * scale);
      matgen_index_t base_col = (matgen_index_t)llround((double)c * scale);

      if (base_row >= new_size) base_row = new_size - 1;
      if (base_col >= new_size) base_col = new_size - 1;

      // Add center
      if (num_candidates < max_candidates) {
        candidates[num_candidates].row = base_row;
        candidates[num_candidates].col = base_col;
        candidates[num_candidates].val = 0.0;
        num_candidates++;
      }

      // Add neighbors
      for (int dr = -expansion_radius; dr <= expansion_radius; ++dr) {
        for (int dc = -expansion_radius; dc <= expansion_radius; ++dc) {
          if (dr == 0 && dc == 0) continue;
          if (num_candidates >= max_candidates) break;

          long long nr = (long long)base_row + dr;
          long long nc = (long long)base_col + dc;

          if (nr < 0 || nr >= (long long)new_size) continue;
          if (nc < 0 || nc >= (long long)new_size) continue;

          candidates[num_candidates].row = (matgen_index_t)nr;
          candidates[num_candidates].col = (matgen_index_t)nc;
          candidates[num_candidates].val = 0.0;
          num_candidates++;
        }
        if (num_candidates >= max_candidates) break;
      }
      if (num_candidates >= max_candidates) break;
    }
    if (num_candidates >= max_candidates) break;
  }

  MATGEN_LOG_DEBUG("Generated %zu candidates", num_candidates);

  // Sort and remove duplicates
  qsort(candidates, num_candidates, sizeof(entry_t), entry_compare);

  size_t num_unique = 0;
  for (size_t i = 0; i < num_candidates; ++i) {
    if (num_unique == 0 || candidates[i].row != candidates[num_unique - 1].row ||
        candidates[i].col != candidates[num_unique - 1].col) {
      candidates[num_unique] = candidates[i];
      num_unique++;
    }
  }

  MATGEN_LOG_DEBUG("Unique candidates: %zu", num_unique);

  // Compute interpolated values in parallel
  const int a = LANCZOS_A;

#pragma omp parallel for schedule(dynamic, 16)
  for (long long ii = 0; ii < (long long)num_unique; ++ii) {
    size_t i = (size_t)ii;
    matgen_index_t new_row = candidates[i].row;
    matgen_index_t new_col = candidates[i].col;

    double orig_row_f = (double)new_row * inv_scale;
    double orig_col_f = (double)new_col * inv_scale;

    int base_row = (int)floor(orig_row_f);
    int base_col = (int)floor(orig_col_f);

    double dx = orig_row_f - (double)base_row;
    double dy = orig_col_f - (double)base_col;

    double value_sum = 0.0;
    double weight_sum = 0.0;

    for (int di = -a + 1; di <= a; ++di) {
      int sample_row = base_row + di;
      if (sample_row < 0 || sample_row >= (int)src_size) continue;

      double wx = lanczos_kernel(dx - (double)di, a);
      if (wx == 0.0) continue;

      for (int dj = -a + 1; dj <= a; ++dj) {
        int sample_col = base_col + dj;
        if (sample_col < 0 || sample_col >= (int)src_size) continue;

        double wy = lanczos_kernel(dy - (double)dj, a);
        if (wy == 0.0) continue;

        double weight = wx * wy;
        matgen_value_t orig_val = fetch_csr_value(
            source, (matgen_index_t)sample_row, (matgen_index_t)sample_col);

        value_sum += (double)orig_val * weight;
        weight_sum += weight;
      }
    }

    candidates[i].val =
        (matgen_value_t)((weight_sum > 0.0) ? (value_sum / weight_sum) : 0.0);
  }

  // Filter zeros
  size_t num_entries = 0;
  const double zero_eps = 1e-10;
  for (size_t i = 0; i < num_unique; ++i) {
    if (fabs((double)candidates[i].val) > zero_eps) {
      candidates[num_entries++] = candidates[i];
    }
  }

  MATGEN_LOG_DEBUG("After filtering zeros: %zu entries", num_entries);

  if (num_entries == 0) {
    free(candidates);
    *result = matgen_csr_create(new_size, new_size, 0);
    return *result ? MATGEN_SUCCESS : MATGEN_ERROR_OUT_OF_MEMORY;
  }

  // Build CSR matrix (same as SEQ)
  *result = matgen_csr_create(new_size, new_size, num_entries);
  if (!*result) {
    free(candidates);
    return MATGEN_ERROR_OUT_OF_MEMORY;
  }

  matgen_csr_matrix_t* out = *result;

  // Fill row_ptr
  matgen_index_t current_row = 0;
  for (size_t i = 0; i < num_entries; ++i) {
    while (current_row < candidates[i].row) {
      out->row_ptr[current_row + 1] = (matgen_size_t)i;
      current_row++;
    }
    out->col_indices[i] = candidates[i].col;
    out->values[i] = candidates[i].val;
  }
  while (current_row < new_size) {
    out->row_ptr[current_row + 1] = (matgen_size_t)num_entries;
    current_row++;
  }

  free(candidates);

  MATGEN_LOG_DEBUG("Lanczos scaling (OMP) completed: output NNZ = %zu",
                   out->nnz);

  return MATGEN_SUCCESS;
}

#endif /* MATGEN_HAS_OPENMP */
