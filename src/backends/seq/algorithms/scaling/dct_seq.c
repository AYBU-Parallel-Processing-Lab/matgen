#include "backends/seq/internal/dct_seq.h"

#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/conversion.h"
#include "matgen/utils/log.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DCT_BLOCK_SIZE 8
#define DCT_THRESHOLD 1e-5f
#define DCT_MAX_DIM 32

static void dct_1d(const matgen_value_t* in, matgen_value_t* out, int N) {
    for (int k = 0; k < N; k++) {
        double sum = 0.0;
        for (int n = 0; n < N; n++) {
            sum += in[n] * cos(M_PI * (n + 0.5) * k / N);
        }
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

matgen_error_t matgen_scale_dct_seq(const matgen_csr_matrix_t* source, matgen_index_t new_rows, matgen_index_t new_cols, matgen_csr_matrix_t** result) {
    if (!source || !result || new_rows != new_cols) return MATGEN_ERROR_INVALID_ARGUMENT;
    matgen_index_t orig_max = (source->rows > source->cols) ? source->rows : source->cols;
    double scale = (double)new_rows / (double)orig_max;
    int new_block_size = (int)round(DCT_BLOCK_SIZE * scale);
    if (new_block_size < 1) new_block_size = 1;
    if (new_block_size > DCT_MAX_DIM) return MATGEN_ERROR_INVALID_ARGUMENT;

    matgen_index_t grid_width = (source->cols + DCT_BLOCK_SIZE - 1) / DCT_BLOCK_SIZE;
    matgen_index_t grid_height = (source->rows + DCT_BLOCK_SIZE - 1) / DCT_BLOCK_SIZE;

    matgen_size_t out_capacity = source->nnz * 2;
    if (out_capacity < 1024) out_capacity = 1024;
    matgen_size_t out_count = 0;

    matgen_index_t* out_rows_arr = (matgen_index_t*)malloc(out_capacity * sizeof(matgen_index_t));
    matgen_index_t* out_cols_arr = (matgen_index_t*)malloc(out_capacity * sizeof(matgen_index_t));
    matgen_value_t* out_vals_arr = (matgen_value_t*)malloc(out_capacity * sizeof(matgen_value_t));

    for (matgen_index_t br = 0; br < grid_height; br++) {
        for (matgen_index_t bc = 0; bc < grid_width; bc++) {
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
                            if (out_count >= out_capacity) {
                                out_capacity *= 2;
                                out_rows_arr = (matgen_index_t*)realloc(out_rows_arr, out_capacity * sizeof(matgen_index_t));
                                out_cols_arr = (matgen_index_t*)realloc(out_cols_arr, out_capacity * sizeof(matgen_index_t));
                                out_vals_arr = (matgen_value_t*)realloc(out_vals_arr, out_capacity * sizeof(matgen_value_t));
                            }
                            out_rows_arr[out_count] = row_idx;
                            out_cols_arr[out_count] = col_idx;
                            out_vals_arr[out_count] = val;
                            out_count++;
                        }
                    }
                }
            }
        }
    }

    matgen_coo_matrix_t* coo = matgen_coo_create(new_rows, new_cols, out_count);
    memcpy(coo->row_indices, out_rows_arr, out_count * sizeof(matgen_index_t));
    memcpy(coo->col_indices, out_cols_arr, out_count * sizeof(matgen_index_t));
    memcpy(coo->values, out_vals_arr, out_count * sizeof(matgen_value_t));
    coo->nnz = out_count;
    coo->is_sorted = false;

    free(out_rows_arr);
    free(out_cols_arr);
    free(out_vals_arr);

    matgen_coo_sort_with_policy(coo, MATGEN_EXEC_SEQ);
    matgen_coo_sum_duplicates_with_policy(coo, MATGEN_EXEC_SEQ);
    *result = matgen_coo_to_csr_with_policy(coo, MATGEN_EXEC_SEQ);
    matgen_coo_destroy(coo);

    return *result ? MATGEN_SUCCESS : MATGEN_ERROR_OUT_OF_MEMORY;
}
