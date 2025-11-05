#include "matgen/core/matrix_io.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <string>

#include "matgen/core/coo_matrix.h"
#include "matgen/core/csr_matrix.h"
#include "matgen/core/matrix_convert.h"

namespace {
// Helper: Create a temporary filename
std::string get_temp_filename() {
  static int counter = 0;
  char filename[256];
  snprintf(filename, sizeof(filename), "test_matrix_%d.mtx", counter++);
  return std::string(filename);
}

// Helper: Delete a file
void delete_file(const char* filename) { remove(filename); }
}  // namespace

TEST(MatrixIO, WriteReadCOO) {
  // Create a simple 3x3 COO matrix
  matgen_coo_matrix_t* original = matgen_coo_create(3, 3, 10);
  ASSERT_NE(original, nullptr);

  ASSERT_EQ(matgen_coo_add_entry(original, 0, 0, 1.0), 0);
  ASSERT_EQ(matgen_coo_add_entry(original, 0, 1, 2.0), 0);
  ASSERT_EQ(matgen_coo_add_entry(original, 1, 1, 3.0), 0);
  ASSERT_EQ(matgen_coo_add_entry(original, 2, 0, 4.0), 0);
  ASSERT_EQ(matgen_coo_add_entry(original, 2, 2, 5.0), 0);

  // Write to file
  std::string filename = get_temp_filename();
  ASSERT_EQ(matgen_mm_write_coo(filename.c_str(), original), 0);

  // Read back
  matgen_mm_info_t info;
  matgen_coo_matrix_t* loaded = matgen_mm_read(filename.c_str(), &info);
  ASSERT_NE(loaded, nullptr);

  // Verify info
  EXPECT_EQ(info.rows, 3);
  EXPECT_EQ(info.cols, 3);
  EXPECT_EQ(info.nnz, 5);
  EXPECT_EQ(info.value_type, MATGEN_MM_REAL);
  EXPECT_EQ(info.symmetry, MATGEN_MM_GENERAL);

  // Verify matrix dimensions
  EXPECT_EQ(loaded->rows, 3);
  EXPECT_EQ(loaded->cols, 3);
  EXPECT_EQ(loaded->nnz, 5);

  // Sort both for comparison
  matgen_coo_sort(original);
  matgen_coo_sort(loaded);

  // Verify entries
  for (size_t i = 0; i < loaded->nnz; i++) {
    EXPECT_EQ(loaded->row_indices[i], original->row_indices[i]);
    EXPECT_EQ(loaded->col_indices[i], original->col_indices[i]);
    EXPECT_DOUBLE_EQ(loaded->values[i], original->values[i]);
  }

  // Cleanup
  matgen_coo_destroy(original);
  matgen_coo_destroy(loaded);
  delete_file(filename.c_str());
}

TEST(MatrixIO, WriteReadCSR) {
  // Create a COO matrix
  matgen_coo_matrix_t* coo = matgen_coo_create(4, 4, 10);
  ASSERT_NE(coo, nullptr);

  ASSERT_EQ(matgen_coo_add_entry(coo, 0, 0, 10.0), 0);
  ASSERT_EQ(matgen_coo_add_entry(coo, 0, 2, 12.0), 0);
  ASSERT_EQ(matgen_coo_add_entry(coo, 1, 1, 21.0), 0);
  ASSERT_EQ(matgen_coo_add_entry(coo, 2, 2, 32.0), 0);
  ASSERT_EQ(matgen_coo_add_entry(coo, 3, 0, 40.0), 0);
  ASSERT_EQ(matgen_coo_add_entry(coo, 3, 3, 43.0), 0);

  // Convert to CSR
  matgen_csr_matrix_t* csr = matgen_coo_to_csr(coo);
  ASSERT_NE(csr, nullptr);

  // Write CSR to file
  std::string filename = get_temp_filename();
  ASSERT_EQ(matgen_mm_write_csr(filename.c_str(), csr), 0);

  // Read back as COO
  matgen_coo_matrix_t* loaded = matgen_mm_read(filename.c_str(), nullptr);
  ASSERT_NE(loaded, nullptr);

  // Verify dimensions
  EXPECT_EQ(loaded->rows, 4);
  EXPECT_EQ(loaded->cols, 4);
  EXPECT_EQ(loaded->nnz, 6);

  // Sort for comparison
  matgen_coo_sort(coo);
  matgen_coo_sort(loaded);

  // Verify entries match original COO
  for (size_t i = 0; i < loaded->nnz; i++) {
    EXPECT_EQ(loaded->row_indices[i], coo->row_indices[i]);
    EXPECT_EQ(loaded->col_indices[i], coo->col_indices[i]);
    EXPECT_DOUBLE_EQ(loaded->values[i], coo->values[i]);
  }

  // Cleanup
  matgen_coo_destroy(coo);
  matgen_csr_destroy(csr);
  matgen_coo_destroy(loaded);
  delete_file(filename.c_str());
}

TEST(MatrixIO, ReadInfoOnly) {
  // Create and write a matrix
  matgen_coo_matrix_t* mat = matgen_coo_create(100, 50, 10);
  ASSERT_NE(mat, nullptr);

  ASSERT_EQ(matgen_coo_add_entry(mat, 0, 0, 1.0), 0);
  ASSERT_EQ(matgen_coo_add_entry(mat, 50, 25, 2.0), 0);
  ASSERT_EQ(matgen_coo_add_entry(mat, 99, 49, 3.0), 0);

  std::string filename = get_temp_filename();
  ASSERT_EQ(matgen_mm_write_coo(filename.c_str(), mat), 0);

  // Read only info
  matgen_mm_info_t info;
  ASSERT_EQ(matgen_mm_read_info(filename.c_str(), &info), 0);

  // Verify info
  EXPECT_EQ(info.rows, 100);
  EXPECT_EQ(info.cols, 50);
  EXPECT_EQ(info.nnz, 3);
  EXPECT_EQ(info.value_type, MATGEN_MM_REAL);
  EXPECT_EQ(info.symmetry, MATGEN_MM_GENERAL);

  // Cleanup
  matgen_coo_destroy(mat);
  delete_file(filename.c_str());
}

TEST(MatrixIO, EmptyMatrix) {
  matgen_coo_matrix_t* mat = matgen_coo_create(5, 5, 0);
  ASSERT_NE(mat, nullptr);

  std::string filename = get_temp_filename();
  ASSERT_EQ(matgen_mm_write_coo(filename.c_str(), mat), 0);

  matgen_coo_matrix_t* loaded = matgen_mm_read(filename.c_str(), nullptr);
  ASSERT_NE(loaded, nullptr);

  EXPECT_EQ(loaded->rows, 5);
  EXPECT_EQ(loaded->cols, 5);
  EXPECT_EQ(loaded->nnz, 0);

  matgen_coo_destroy(mat);
  matgen_coo_destroy(loaded);
  delete_file(filename.c_str());
}

TEST(MatrixIO, ScientificNotation) {
  matgen_coo_matrix_t* mat = matgen_coo_create(2, 2, 10);
  ASSERT_NE(mat, nullptr);

  ASSERT_EQ(matgen_coo_add_entry(mat, 0, 0, 1.23456789e-10), 0);
  ASSERT_EQ(matgen_coo_add_entry(mat, 1, 1, 9.87654321e+15), 0);

  std::string filename = get_temp_filename();
  ASSERT_EQ(matgen_mm_write_coo(filename.c_str(), mat), 0);

  matgen_coo_matrix_t* loaded = matgen_mm_read(filename.c_str(), nullptr);
  ASSERT_NE(loaded, nullptr);

  matgen_coo_sort(mat);
  matgen_coo_sort(loaded);

  EXPECT_EQ(loaded->nnz, 2);
  EXPECT_NEAR(loaded->values[0], 1.23456789e-10, 1e-18);
  EXPECT_NEAR(loaded->values[1], 9.87654321e+15, 1e+7);

  matgen_coo_destroy(mat);
  matgen_coo_destroy(loaded);
  delete_file(filename.c_str());
}

TEST(MatrixIO, RectangularMatrix) {
  matgen_coo_matrix_t* mat = matgen_coo_create(10, 5, 10);
  ASSERT_NE(mat, nullptr);

  for (size_t i = 0; i < 5; i++) {
    ASSERT_EQ(matgen_coo_add_entry(mat, i, i, static_cast<double>(i + 1)), 0);
  }

  std::string filename = get_temp_filename();
  ASSERT_EQ(matgen_mm_write_coo(filename.c_str(), mat), 0);

  matgen_mm_info_t info;
  matgen_coo_matrix_t* loaded = matgen_mm_read(filename.c_str(), &info);
  ASSERT_NE(loaded, nullptr);

  EXPECT_EQ(info.rows, 10);
  EXPECT_EQ(info.cols, 5);
  EXPECT_EQ(loaded->rows, 10);
  EXPECT_EQ(loaded->cols, 5);
  EXPECT_EQ(loaded->nnz, 5);

  matgen_coo_destroy(mat);
  matgen_coo_destroy(loaded);
  delete_file(filename.c_str());
}

TEST(MatrixIO, InvalidFile) {
  matgen_coo_matrix_t* mat = matgen_mm_read("nonexistent_file.mtx", nullptr);
  EXPECT_EQ(mat, nullptr);

  matgen_mm_info_t info;
  EXPECT_EQ(matgen_mm_read_info("nonexistent_file.mtx", &info), -1);
}

TEST(MatrixIO, WriteToInvalidPath) {
  matgen_coo_matrix_t* mat = matgen_coo_create(2, 2, 1);
  ASSERT_NE(mat, nullptr);
  ASSERT_EQ(matgen_coo_add_entry(mat, 0, 0, 1.0), 0);

  // Try to write to an invalid path (directory that doesn't exist)
  EXPECT_EQ(matgen_mm_write_coo("nonexistent_dir/test.mtx", mat), -1);

  matgen_coo_destroy(mat);
}

TEST(MatrixIO, SymmetricMatrix) {
  // Create a symmetric matrix file manually
  std::string filename = get_temp_filename();
  FILE* f = fopen(filename.c_str(), "w");
  ASSERT_NE(f, nullptr);

  fprintf(f, "%%%%MatrixMarket matrix coordinate real symmetric\n");
  fprintf(f, "%% Test symmetric matrix\n");
  fprintf(f, "3 3 4\n");
  fprintf(f, "1 1 1.0\n");
  fprintf(f, "2 1 2.0\n");  // This will also create (1,2)
  fprintf(f, "2 2 3.0\n");
  fprintf(f, "3 2 4.0\n");  // This will also create (2,3)
  fclose(f);

  // Read the symmetric matrix
  matgen_mm_info_t info;
  matgen_coo_matrix_t* mat = matgen_mm_read(filename.c_str(), &info);
  ASSERT_NE(mat, nullptr);

  // Should expand symmetric entries
  EXPECT_EQ(info.symmetry, MATGEN_MM_SYMMETRIC);
  EXPECT_EQ(info.nnz, 4);  // As stored in file
  EXPECT_EQ(mat->nnz,
            6);  // After expansion: (1,1), (2,1), (1,2), (2,2), (3,2), (2,3)

  matgen_coo_sort(mat);

  // Check for symmetric entries
  // Should have: (0,0)=1.0, (1,0)=2.0, (0,1)=2.0, (1,1)=3.0, (2,1)=4.0,
  // (1,2)=4.0
  bool found_0_0 = false;
  bool found_1_0 = false;
  bool found_0_1 = false;
  bool found_1_1 = false;
  bool found_2_1 = false;
  bool found_1_2 = false;

  for (size_t i = 0; i < mat->nnz; i++) {
    if (mat->row_indices[i] == 0 && mat->col_indices[i] == 0) {
      found_0_0 = true;
      EXPECT_DOUBLE_EQ(mat->values[i], 1.0);
    }
    if (mat->row_indices[i] == 1 && mat->col_indices[i] == 0) {
      found_1_0 = true;
      EXPECT_DOUBLE_EQ(mat->values[i], 2.0);
    }
    if (mat->row_indices[i] == 0 && mat->col_indices[i] == 1) {
      found_0_1 = true;
      EXPECT_DOUBLE_EQ(mat->values[i], 2.0);
    }
    if (mat->row_indices[i] == 1 && mat->col_indices[i] == 1) {
      found_1_1 = true;
      EXPECT_DOUBLE_EQ(mat->values[i], 3.0);
    }
    if (mat->row_indices[i] == 2 && mat->col_indices[i] == 1) {
      found_2_1 = true;
      EXPECT_DOUBLE_EQ(mat->values[i], 4.0);
    }
    if (mat->row_indices[i] == 1 && mat->col_indices[i] == 2) {
      found_1_2 = true;
      EXPECT_DOUBLE_EQ(mat->values[i], 4.0);
    }
  }

  EXPECT_TRUE(found_0_0 && found_1_0 && found_0_1 && found_1_1 && found_2_1 &&
              found_1_2);

  matgen_coo_destroy(mat);
  delete_file(filename.c_str());
}

TEST(MatrixIO, PatternMatrix) {
  // Create a pattern matrix file
  std::string filename = get_temp_filename();
  FILE* f = fopen(filename.c_str(), "w");
  ASSERT_NE(f, nullptr);

  fprintf(f, "%%%%MatrixMarket matrix coordinate pattern general\n");
  fprintf(f, "4 4 5\n");
  fprintf(f, "1 1\n");
  fprintf(f, "2 2\n");
  fprintf(f, "3 3\n");
  fprintf(f, "4 4\n");
  fprintf(f, "1 4\n");
  fclose(f);

  // Read pattern matrix
  matgen_mm_info_t info;
  matgen_coo_matrix_t* mat = matgen_mm_read(filename.c_str(), &info);
  ASSERT_NE(mat, nullptr);

  EXPECT_EQ(info.value_type, MATGEN_MM_PATTERN);
  EXPECT_EQ(mat->nnz, 5);

  // All values should be 1.0
  for (size_t i = 0; i < mat->nnz; i++) {
    EXPECT_DOUBLE_EQ(mat->values[i], 1.0);
  }

  matgen_coo_destroy(mat);
  delete_file(filename.c_str());
}
