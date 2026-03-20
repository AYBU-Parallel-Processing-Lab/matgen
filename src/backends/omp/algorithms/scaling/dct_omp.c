#include "backends/omp/internal/dct_omp.h"
#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/conversion.h"
#include "matgen/utils/log.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DCT_BLOCK_SIZE 8
#define DCT_THRESHOLD 1e-5f
#define DCT_MAX_DIM 32

static void dct_1d(const matgen_value_t* in, matgen_value_t* out, int N) {
    for (int k = 0; k < N; k++) {
        double sum = 0.0;
        for (int n = 0; n < N; n++) sum += in[n] * cos(M_PI * (n + 0.5) * k / N);
        double alpha = (k == 0) ? 1.0 / sqrt(2.0) : 1.0;
        out[k] = sqrt(2.0 / N) * alpha * sum;
    }
}

static void idct_1d(const matgen_value_t* in, matgen_value_t* out, int N) {
    for (int n = 0; n < N; n++) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            double alpha = (k == 0) ? 1.0 / sqrt(2.0) : 1.0;
            sum += alpha * in[k] * cos(M_PI * (n + 0.5) * k / N);
        }
        out[n] = sqrt(2.0 / N) * sum;
    }
}

static void dct_2d(const matgen_value_t* in, matgen_value_t* out, int N) {
    matgen_value_t temp[DCT_MAX_DIM * DCT_MAX_DIM];
    for (int r = 0; r < N; r++) dct_1d(&in[r * N], &temp[r * N], N);
    for (int c = 0; c < N; c++) {
        matgen_value_t col_in[DCT_MAX_DIM], col_out[DCT_MAX_DIM];
        for (int r = 0; r < N; r++) col_in[r] = temp[r * N + c];
        dct_1d(col_in, col_out, N);
        for (int r = 0; r < N; r++) out[r * N + c] = col_out[r];
    }
}

static void idct_2d(const matgen_value_t* in, matgen_value_t* out, int N) {
    matgen_value_t temp[DCT_MAX_DIM * DCT_MAX_DIM];
    for (int c = 0; c < N; c++) {
        matgen_value_t col_in[DCT_MAX_DIM], col_out[DCT_MAX_DIM];
        for (int r = 0; r < N; r++) col_in[r] = in[r * N + c];
        idct_1d(col_in, col_out, N);
        for (int r = 0; r < N; r++) temp[r * N + c] = col_out[r];
    }
    for (int r = 0; r < N; r++) idct_1d(&temp[r * N], &out[r * N], N);
}

matgen_error_t matgen_scale_dct_omp(const matgen_csr_matrix_t* source, matgen_index_t new_rows, matgen_index_t new_cols, matgen_csr_matrix_t** result) {
    if (!source || !result || new_rows != new_cols) return MATGEN_ERROR_INVALID_ARGUMENT;
    matgen_index_t orig_max = (source->rows > source->cols) ? source->rows : source->cols;
    double scale = (double)new_rows / (double)orig_max;
    int new_block_size = (int)round(DCT_BLOCK_SIZE * scale);
    if (new_block_size < 1) new_block_size = 1;
    if (new_block_size > DCT_MAX_DIM) return MATGEN_ERROR_INVALID_ARGUMENT;

    matgen_index_t grid_width = (source->cols + DCT_BLOCK_SIZE - 1) / DCT_BLOCK_SIZE;
    matgen_index_t grid_height = (source->rows + DCT_BLOCK_SIZE - 1) / DCT_BLOCK_SIZE;
    matgen_index_t total_blocks = grid_width * grid_height;

    int max_threads = omp_get_max_threads();
    matgen_index_t** t_out_rows = (matgen_index_t**)malloc(max_threads * sizeof(matgen_index_t*));
    matgen_index_t** t_out_cols = (matgen_index_t**)malloc(max_threads * sizeof(matgen_index_t*));
    matgen_value_t** t_out_vals = (matgen_value_t**)malloc(max_threads * sizeof(matgen_value_t*));
    matgen_size_t* t_out_count = (matgen_size_t*)calloc(max_threads, sizeof(matgen_size_t));
    matgen_size_t* t_out_cap = (matgen_size_t*)malloc(max_threads * sizeof(matgen_size_t));

    for (int i = 0; i < max_threads; i++) {
        t_out_cap[i] = 1024;
        t_out_rows[i] = (matgen_index_t*)malloc(1024 * sizeof(matgen_index_t));
        t_out_cols[i] = (matgen_index_t*)malloc(1024 * sizeof(matgen_index_t));
        t_out_vals[i] = (matgen_value_t*)malloc(1024 * sizeof(matgen_value_t));
    }

    #pragma omp parallel for schedule(dynamic)
    for (matgen_index_t b = 0; b < total_blocks; b++) {
        int tid = omp_get_thread_num();
        matgen_index_t br = b / grid_width;
        matgen_index_t bc = b % grid_width;
        
        matgen_value_t block_dense[DCT_BLOCK_SIZE * DCT_BLOCK_SIZE] = {0};
        bool is_empty = true;
        for (matgen_index_t r = br * DCT_BLOCK_SIZE; r < (br + 1) * DCT_BLOCK_SIZE && r < source->rows; r++) {
            for (matgen_size_t j = source->row_ptr[r]; j < source->row_ptr[r + 1]; j++) {
                matgen_index_t c = source->col_indices[j];
                if (c >= bc * DCT_BLOCK_SIZE && c < (bc + 1) * DCT_BLOCK_SIZE) {
                    block_dense[(r - br * DCT_BLOCK_SIZE) * DCT_BLOCK_SIZE + (c - bc * DCT_BLOCK_SIZE)] = source->values[j];
                    is_empty = false;
                } else if (c >= (bc + 1) * DCT_BLOCK_SIZE) break;
            }
        }
        if (is_empty) continue;
        
        matgen_value_t block_f[DCT_BLOCK_SIZE * DCT_BLOCK_SIZE];
        dct_2d(block_dense, block_f, DCT_BLOCK_SIZE);
        
        matgen_value_t target_f[DCT_MAX_DIM * DCT_MAX_DIM] = {0};
        int copy_size = new_block_size < DCT_BLOCK_SIZE ? new_block_size : DCT_BLOCK_SIZE;
        for (int r = 0; r < copy_size; r++) {
            for (int c = 0; c < copy_size; c++) {
                target_f[r * new_block_size + c] = block_f[r * DCT_BLOCK_SIZE + c];
            }
        }
        
        matgen_value_t block_resized[DCT_MAX_DIM * DCT_MAX_DIM];
        idct_2d(target_f, block_resized, new_block_size);
        
        for (int r = 0; r < new_block_size; r++) {
            for (int c = 0; c < new_block_size; c++) {
                matgen_value_t val = block_resized[r * new_block_size + c];
                if (fabs(val) > DCT_THRESHOLD) {
                    matgen_index_t row_idx = (matgen_index_t)(br * DCT_BLOCK_SIZE * scale) + r;
                    matgen_index_t col_idx = (matgen_index_t)(bc * DCT_BLOCK_SIZE * scale) + c;
                    if (row_idx < new_rows && col_idx < new_cols) {
                        if (t_out_count[tid] >= t_out_cap[tid]) {
                            t_out_cap[tid] *= 2;
                            t_out_rows[tid] = (matgen_index_t*)realloc(t_out_rows[tid], t_out_cap[tid] * sizeof(matgen_index_t));
                            t_out_cols[tid] = (matgen_index_t*)realloc(t_out_cols[tid], t_out_cap[tid] * sizeof(matgen_index_t));
                            t_out_vals[tid] = (matgen_value_t*)realloc(t_out_vals[tid], t_out_cap[tid] * sizeof(matgen_value_t));
                        }
                        matgen_size_t idx = t_out_count[tid]++;
                        t_out_rows[tid][idx] = row_idx;
                        t_out_cols[tid][idx] = col_idx;
                        t_out_vals[tid][idx] = val;
                    }
                }
            }
        }
    }
    
    matgen_size_t total_out = 0;
    for (int i = 0; i < max_threads; i++) total_out += t_out_count[i];
    
    matgen_coo_matrix_t* coo = matgen_coo_create(new_rows, new_cols, total_out);
    matgen_size_t offset = 0;
    for (int i = 0; i < max_threads; i++) {
        memcpy(coo->row_indices + offset, t_out_rows[i], t_out_count[i] * sizeof(matgen_index_t));
        memcpy(coo->col_indices + offset, t_out_cols[i], t_out_count[i] * sizeof(matgen_index_t));
        memcpy(coo->values + offset, t_out_vals[i], t_out_count[i] * sizeof(matgen_value_t));
        offset += t_out_count[i];
        free(t_out_rows[i]);
        free(t_out_cols[i]);
        free(t_out_vals[i]);
    }
    free(t_out_rows);
    free(t_out_cols);
    free(t_out_vals);
    free(t_out_count);
    free(t_out_cap);
    
    coo->nnz = total_out;
    coo->is_sorted = false;
    
    matgen_coo_sort_with_policy(coo, MATGEN_EXEC_PAR);
    matgen_coo_sum_duplicates_with_policy(coo, MATGEN_EXEC_PAR);
    *result = matgen_coo_to_csr_with_policy(coo, MATGEN_EXEC_PAR);
    matgen_coo_destroy(coo);
    
    return *result ? MATGEN_SUCCESS : MATGEN_ERROR_OUT_OF_MEMORY;
}
