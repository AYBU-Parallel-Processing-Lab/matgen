
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/remove.h>

#include "backends/cuda/internal/dct_cuda.cuh"
#include "matgen/core/matrix/coo.h"
#include "matgen/core/matrix/conversion.h"
#include "matgen/utils/log.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DCT_BLOCK_SIZE 8
#define DCT_THRESHOLD 1e-5f
#define DCT_MAX_DIM 32

__device__ void dct_1d_dev(const matgen_value_t* in, matgen_value_t* out, int N) {
    for (int k = 0; k < N; k++) {
        double sum = 0.0;
        for (int n = 0; n < N; n++) sum += in[n] * cos(M_PI * (n + 0.5) * k / N);
        double alpha = (k == 0) ? 1.0 / sqrt(2.0) : 1.0;
        out[k] = sqrt(2.0 / N) * alpha * sum;
    }
}

__device__ void idct_1d_dev(const matgen_value_t* in, matgen_value_t* out, int N) {
    for (int n = 0; n < N; n++) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            double alpha = (k == 0) ? 1.0 / sqrt(2.0) : 1.0;
            sum += alpha * in[k] * cos(M_PI * (n + 0.5) * k / N);
        }
        out[n] = sqrt(2.0 / N) * sum;
    }
}

__device__ void dct_2d_dev(const matgen_value_t* in, matgen_value_t* out, int N) {
    matgen_value_t temp[DCT_MAX_DIM * DCT_MAX_DIM];
    for (int r = 0; r < N; r++) dct_1d_dev(&in[r * N], &temp[r * N], N);
    for (int c = 0; c < N; c++) {
        matgen_value_t col_in[DCT_MAX_DIM], col_out[DCT_MAX_DIM];
        for (int r = 0; r < N; r++) col_in[r] = temp[r * N + c];
        dct_1d_dev(col_in, col_out, N);
        for (int r = 0; r < N; r++) out[r * N + c] = col_out[r];
    }
}

__device__ void idct_2d_dev(const matgen_value_t* in, matgen_value_t* out, int N) {
    matgen_value_t temp[DCT_MAX_DIM * DCT_MAX_DIM];
    for (int c = 0; c < N; c++) {
        matgen_value_t col_in[DCT_MAX_DIM], col_out[DCT_MAX_DIM];
        for (int r = 0; r < N; r++) col_in[r] = in[r * N + c];
        idct_1d_dev(col_in, col_out, N);
        for (int r = 0; r < N; r++) temp[r * N + c] = col_out[r];
    }
    for (int r = 0; r < N; r++) idct_1d_dev(&temp[r * N], &out[r * N], N);
}


__global__ void mark_active_blocks_cuda(
    matgen_index_t rows, const matgen_size_t* row_ptr, const matgen_index_t* col_indices,
    matgen_index_t grid_width, uint8_t* active_flags) {

    matgen_index_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        matgen_index_t br = r / DCT_BLOCK_SIZE;
        for (matgen_size_t j = row_ptr[r]; j < row_ptr[r + 1]; j++) {
            matgen_index_t c = col_indices[j];
            matgen_index_t bc = c / DCT_BLOCK_SIZE;
            matgen_index_t b = br * grid_width + bc;
            active_flags[b] = 1;
        }
    }
}

__global__ void process_active_blocks_cuda(
    matgen_index_t rows, const matgen_size_t* row_ptr, const matgen_index_t* col_indices, const matgen_value_t* values,
    matgen_index_t grid_width, const matgen_index_t* active_blocks, matgen_size_t num_active_blocks,
    int new_block_size, double scale, matgen_index_t new_rows, matgen_index_t new_cols,
    matgen_index_t* out_r, matgen_index_t* out_c, matgen_value_t* out_v, matgen_size_t max_out_per_block) {

    matgen_size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_active_blocks) {
        matgen_index_t b = active_blocks[idx];
        matgen_index_t br = b / grid_width;
        matgen_index_t bc = b % grid_width;

        matgen_value_t block_dense[DCT_BLOCK_SIZE * DCT_BLOCK_SIZE];
        for(int i=0; i<DCT_BLOCK_SIZE * DCT_BLOCK_SIZE; i++) block_dense[i] = 0;

        bool is_empty = true;

        for (matgen_index_t r = br * DCT_BLOCK_SIZE; r < (br + 1) * DCT_BLOCK_SIZE && r < rows; r++) {
            matgen_size_t r_low = row_ptr[r];
            matgen_size_t r_high = row_ptr[r + 1];
            matgen_size_t r_start_j = r_high;

            while (r_low < r_high) {
                matgen_size_t mid = r_low + (r_high - r_low) / 2;
                if (col_indices[mid] >= bc * DCT_BLOCK_SIZE) {
                    r_start_j = mid;
                    r_high = mid;
                } else {
                    r_low = mid + 1;
                }
            }

            for (matgen_size_t j = r_start_j; j < row_ptr[r + 1]; j++) {
                matgen_index_t c = col_indices[j];
                if (c >= (bc + 1) * DCT_BLOCK_SIZE) break;
                block_dense[(r - br * DCT_BLOCK_SIZE) * DCT_BLOCK_SIZE + (c - bc * DCT_BLOCK_SIZE)] = values[j];
                is_empty = false;
            }
        }

        matgen_size_t write_idx = idx * max_out_per_block;
        matgen_size_t count = 0;

        if (!is_empty) {
            matgen_value_t block_f[DCT_BLOCK_SIZE * DCT_BLOCK_SIZE];
            dct_2d_dev(block_dense, block_f, DCT_BLOCK_SIZE);

            matgen_value_t target_f[DCT_MAX_DIM * DCT_MAX_DIM];
            for(int i=0; i<DCT_MAX_DIM * DCT_MAX_DIM; i++) target_f[i] = 0;

            int copy_size = new_block_size < DCT_BLOCK_SIZE ? new_block_size : DCT_BLOCK_SIZE;
            for (int r = 0; r < copy_size; r++) {
                for (int c = 0; c < copy_size; c++) {
                    target_f[r * new_block_size + c] = block_f[r * DCT_BLOCK_SIZE + c];
                }
            }

            matgen_value_t block_resized[DCT_MAX_DIM * DCT_MAX_DIM];
            idct_2d_dev(target_f, block_resized, new_block_size);

            for (int r = 0; r < new_block_size; r++) {
                for (int c = 0; c < new_block_size; c++) {
                    matgen_value_t val = block_resized[r * new_block_size + c];
                    if (fabs(val) > DCT_THRESHOLD) {
                        matgen_index_t row_idx = (matgen_index_t)(br * DCT_BLOCK_SIZE * scale) + r;
                        matgen_index_t col_idx = (matgen_index_t)(bc * DCT_BLOCK_SIZE * scale) + c;

                        if (row_idx < new_rows && col_idx < new_cols && count < max_out_per_block) {
                            out_r[write_idx + count] = row_idx;
                            out_c[write_idx + count] = col_idx;
                            out_v[write_idx + count] = val;
                            count++;
                        }
                    }
                }
            }
        }

        // Fill the rest with invalid markers
        for (matgen_size_t i = count; i < max_out_per_block; i++) {
            out_r[write_idx + i] = (matgen_index_t)-1;
        }
    }
}


struct is_active {
    const uint8_t* active;
    is_active(const uint8_t* a) : active(a) {}
    __host__ __device__ bool operator()(const matgen_index_t& b) const { return active[b] > 0; }
};

struct is_valid {
    __host__ __device__ bool operator()(const thrust::tuple<matgen_index_t, matgen_index_t, matgen_value_t>& t) {
        return thrust::get<0>(t) != (matgen_index_t)-1;
    }
};


extern "C" matgen_error_t matgen_scale_dct_cuda(const matgen_csr_matrix_t* source, matgen_index_t new_rows, matgen_index_t new_cols, matgen_csr_matrix_t** result) {
    if (!source || !result || new_rows != new_cols) return MATGEN_ERROR_INVALID_ARGUMENT;
    matgen_index_t orig_max = (source->rows > source->cols) ? source->rows : source->cols;
    double scale = (double)new_rows / (double)orig_max;
    int new_block_size = (int)round(DCT_BLOCK_SIZE * scale);
    if (new_block_size < 1) new_block_size = 1;
    if (new_block_size > DCT_MAX_DIM) return MATGEN_ERROR_INVALID_ARGUMENT;

    matgen_index_t grid_width = (source->cols + DCT_BLOCK_SIZE - 1) / DCT_BLOCK_SIZE;
    matgen_index_t grid_height = (source->rows + DCT_BLOCK_SIZE - 1) / DCT_BLOCK_SIZE;
    matgen_index_t total_blocks = grid_width * grid_height;

    matgen_size_t* d_row_ptr;
    matgen_index_t* d_col_indices;
    matgen_value_t* d_values;
    cudaMalloc(&d_row_ptr, (source->rows + 1) * sizeof(matgen_size_t));
    cudaMalloc(&d_col_indices, source->nnz * sizeof(matgen_index_t));
    cudaMalloc(&d_values, source->nnz * sizeof(matgen_value_t));
    cudaMemcpy(d_row_ptr, source->row_ptr, (source->rows + 1) * sizeof(matgen_size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, source->col_indices, source->nnz * sizeof(matgen_index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, source->values, source->nnz * sizeof(matgen_value_t), cudaMemcpyHostToDevice);

    // Find active blocks
    thrust::device_vector<uint8_t> d_active_flags(total_blocks, 0);
    int threads_rows = 256;
    int blocks_rows = (source->rows + threads_rows - 1) / threads_rows;
    mark_active_blocks_cuda<<<blocks_rows, threads_rows>>>(
        source->rows, d_row_ptr, d_col_indices, grid_width, thrust::raw_pointer_cast(d_active_flags.data()));
    cudaDeviceSynchronize();

    thrust::device_vector<matgen_index_t> d_all_blocks(total_blocks);
    thrust::sequence(d_all_blocks.begin(), d_all_blocks.end());

    thrust::device_vector<matgen_index_t> d_active_blocks(total_blocks);



    auto end_active = thrust::copy_if(
        d_all_blocks.begin(), d_all_blocks.end(),
        d_active_blocks.begin(),
        is_active(thrust::raw_pointer_cast(d_active_flags.data())));

    matgen_size_t num_active_blocks = end_active - d_active_blocks.begin();

    matgen_size_t max_out_per_block = new_block_size * new_block_size;
    matgen_size_t max_total_out = num_active_blocks * max_out_per_block;

    thrust::device_vector<matgen_index_t> d_out_r(max_total_out);
    thrust::device_vector<matgen_index_t> d_out_c(max_total_out);
    thrust::device_vector<matgen_value_t> d_out_v(max_total_out);

    if (num_active_blocks > 0) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_active_blocks + threadsPerBlock - 1) / threadsPerBlock;
        process_active_blocks_cuda<<<blocksPerGrid, threadsPerBlock>>>(
            source->rows, d_row_ptr, d_col_indices, d_values,
            grid_width, thrust::raw_pointer_cast(d_active_blocks.data()), num_active_blocks,
            new_block_size, scale, new_rows, new_cols,
            thrust::raw_pointer_cast(d_out_r.data()),
            thrust::raw_pointer_cast(d_out_c.data()),
            thrust::raw_pointer_cast(d_out_v.data()),
            max_out_per_block
        );
        cudaDeviceSynchronize();
    }

    cudaFree(d_row_ptr);
    cudaFree(d_col_indices);
    cudaFree(d_values);

    auto zip_it = thrust::make_zip_iterator(thrust::make_tuple(d_out_r.begin(), d_out_c.begin(), d_out_v.begin()));
    auto new_end = thrust::copy_if(zip_it, zip_it + max_total_out, zip_it, is_valid());
    matgen_size_t total_out = new_end - zip_it;

    matgen_coo_matrix_t* coo = matgen_coo_create(new_rows, new_cols, total_out);
    thrust::copy(d_out_r.begin(), d_out_r.begin() + total_out, coo->row_indices);
    thrust::copy(d_out_c.begin(), d_out_c.begin() + total_out, coo->col_indices);
    thrust::copy(d_out_v.begin(), d_out_v.begin() + total_out, coo->values);
    coo->nnz = total_out;
    coo->is_sorted = false;

    matgen_coo_sort_with_policy(coo, MATGEN_EXEC_PAR_UNSEQ);
    matgen_coo_sum_duplicates_with_policy(coo, MATGEN_EXEC_PAR_UNSEQ);
    *result = matgen_coo_to_csr_with_policy(coo, MATGEN_EXEC_PAR_UNSEQ);
    matgen_coo_destroy(coo);

    return *result ? MATGEN_SUCCESS : MATGEN_ERROR_OUT_OF_MEMORY;
}
