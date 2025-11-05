#include <gtest/gtest.h>
#include <cmath>

#include "matgen/core/coo_matrix.h"
#include "matgen/core/csr_matrix.h"
#include "matgen/core/matrix_convert.h"
#include "matgen/util/log.h"

class ConversionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Suppress log output during tests
        matgen_log_set_level(MATGEN_LOG_LEVEL_ERROR);
    }

    void TearDown() override {
        if (coo) {
            matgen_coo_destroy(coo);
            coo = nullptr;
        }
        if (csr) {
            matgen_csr_destroy(csr);
            csr = nullptr;
        }
    }

    // Helper to create test COO matrix
    // Matrix: [1 0 2]
    //         [0 3 0]
    //         [4 0 5]
    matgen_coo_matrix_t* CreateTestCOO() {
        matgen_coo_matrix_t* m = matgen_coo_create(3, 3, 5);
        if (!m) return nullptr;

        matgen_coo_add_entry(m, 0, 0, 1.0);
        matgen_coo_add_entry(m, 0, 2, 2.0);
        matgen_coo_add_entry(m, 1, 1, 3.0);
        matgen_coo_add_entry(m, 2, 0, 4.0);
        matgen_coo_add_entry(m, 2, 2, 5.0);

        return m;
    }

    matgen_coo_matrix_t* coo = nullptr;
    matgen_csr_matrix_t* csr = nullptr;
};

// =============================================================================
// COO to CSR Conversion Tests
// =============================================================================

TEST_F(ConversionTest, COOtoCSRBasic) {
    coo = CreateTestCOO();
    ASSERT_NE(coo, nullptr);

    csr = matgen_coo_to_csr(coo);
    ASSERT_NE(csr, nullptr);

    EXPECT_EQ(csr->rows, 3);
    EXPECT_EQ(csr->cols, 3);
    EXPECT_EQ(csr->nnz, 5);
}

TEST_F(ConversionTest, COOtoCSRNullInput) {
    csr = matgen_coo_to_csr(nullptr);
    EXPECT_EQ(csr, nullptr);
}

TEST_F(ConversionTest, COOtoCSRValues) {
    coo = CreateTestCOO();
    ASSERT_NE(coo, nullptr);

    csr = matgen_coo_to_csr(coo);
    ASSERT_NE(csr, nullptr);

    // Check that values are preserved
    EXPECT_DOUBLE_EQ(matgen_csr_get(csr, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matgen_csr_get(csr, 0, 2), 2.0);
    EXPECT_DOUBLE_EQ(matgen_csr_get(csr, 1, 1), 3.0);
    EXPECT_DOUBLE_EQ(matgen_csr_get(csr, 2, 0), 4.0);
    EXPECT_DOUBLE_EQ(matgen_csr_get(csr, 2, 2), 5.0);

    // Check zeros
    EXPECT_DOUBLE_EQ(matgen_csr_get(csr, 0, 1), 0.0);
    EXPECT_DOUBLE_EQ(matgen_csr_get(csr, 1, 0), 0.0);
}

TEST_F(ConversionTest, COOtoCSRRowPointers) {
    coo = CreateTestCOO();
    ASSERT_NE(coo, nullptr);

    csr = matgen_coo_to_csr(coo);
    ASSERT_NE(csr, nullptr);

    // Check row_ptr structure
    EXPECT_EQ(csr->row_ptr[0], 0);  // Row 0 starts at index 0
    EXPECT_EQ(csr->row_ptr[1], 2);  // Row 0 has 2 entries
    EXPECT_EQ(csr->row_ptr[2], 3);  // Row 1 has 1 entry
    EXPECT_EQ(csr->row_ptr[3], 5);  // Row 2 has 2 entries
}

TEST_F(ConversionTest, COOtoCSRValidation) {
    coo = CreateTestCOO();
    ASSERT_NE(coo, nullptr);

    csr = matgen_coo_to_csr(coo);
    ASSERT_NE(csr, nullptr);

    EXPECT_TRUE(matgen_csr_validate(csr));
}

TEST_F(ConversionTest, COOtoCSREmptyRows) {
    coo = matgen_coo_create(5, 5, 3);
    ASSERT_NE(coo, nullptr);

    // Add entries only in rows 0, 2, 4 (rows 1, 3 are empty)
    matgen_coo_add_entry(coo, 0, 0, 1.0);
    matgen_coo_add_entry(coo, 2, 1, 2.0);
    matgen_coo_add_entry(coo, 4, 2, 3.0);

    csr = matgen_coo_to_csr(coo);
    ASSERT_NE(csr, nullptr);

    EXPECT_EQ(matgen_csr_row_nnz(csr, 0), 1);
    EXPECT_EQ(matgen_csr_row_nnz(csr, 1), 0);  // Empty
    EXPECT_EQ(matgen_csr_row_nnz(csr, 2), 1);
    EXPECT_EQ(matgen_csr_row_nnz(csr, 3), 0);  // Empty
    EXPECT_EQ(matgen_csr_row_nnz(csr, 4), 1);
}

TEST_F(ConversionTest, COOtoCSRUnsortedInput) {
    // Create COO with unsorted entries
    coo = matgen_coo_create(3, 3, 3);
    ASSERT_NE(coo, nullptr);

    // Add entries out of order
    matgen_coo_add_entry(coo, 2, 1, 5.0);  // Last row first
    matgen_coo_add_entry(coo, 0, 2, 1.0);  // First row
    matgen_coo_add_entry(coo, 1, 0, 3.0);  // Middle row

    csr = matgen_coo_to_csr(coo);
    ASSERT_NE(csr, nullptr);

    // Should still work correctly
    EXPECT_DOUBLE_EQ(matgen_csr_get(csr, 0, 2), 1.0);
    EXPECT_DOUBLE_EQ(matgen_csr_get(csr, 1, 0), 3.0);
    EXPECT_DOUBLE_EQ(matgen_csr_get(csr, 2, 1), 5.0);

    EXPECT_TRUE(matgen_csr_validate(csr));
}

// =============================================================================
// CSR to COO Conversion Tests
// =============================================================================

TEST_F(ConversionTest, CSRtoCOOBasic) {
    // First create a CSR matrix via COO->CSR
    coo = CreateTestCOO();
    ASSERT_NE(coo, nullptr);

    csr = matgen_coo_to_csr(coo);
    ASSERT_NE(csr, nullptr);

    // Destroy original COO
    matgen_coo_destroy(coo);

    // Convert back to COO
    coo = matgen_csr_to_coo(csr);
    ASSERT_NE(coo, nullptr);

    EXPECT_EQ(coo->rows, 3);
    EXPECT_EQ(coo->cols, 3);
    EXPECT_EQ(coo->nnz, 5);
}

TEST_F(ConversionTest, CSRtoCOONullInput) {
    coo = matgen_csr_to_coo(nullptr);
    EXPECT_EQ(coo, nullptr);
}

TEST_F(ConversionTest, CSRtoCOOValues) {
    coo = CreateTestCOO();
    ASSERT_NE(coo, nullptr);

    csr = matgen_coo_to_csr(coo);
    ASSERT_NE(csr, nullptr);

    matgen_coo_destroy(coo);
    coo = matgen_csr_to_coo(csr);
    ASSERT_NE(coo, nullptr);

    // Values should be preserved
    EXPECT_DOUBLE_EQ(matgen_coo_get(coo, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matgen_coo_get(coo, 0, 2), 2.0);
    EXPECT_DOUBLE_EQ(matgen_coo_get(coo, 1, 1), 3.0);
    EXPECT_DOUBLE_EQ(matgen_coo_get(coo, 2, 0), 4.0);
    EXPECT_DOUBLE_EQ(matgen_coo_get(coo, 2, 2), 5.0);
}

TEST_F(ConversionTest, CSRtoCOOSorted) {
    coo = CreateTestCOO();
    ASSERT_NE(coo, nullptr);

    csr = matgen_coo_to_csr(coo);
    ASSERT_NE(csr, nullptr);

    matgen_coo_destroy(coo);
    coo = matgen_csr_to_coo(csr);
    ASSERT_NE(coo, nullptr);

    // Result should be sorted
    EXPECT_TRUE(coo->is_sorted);
}

// =============================================================================
// Round-Trip Conversion Tests
// =============================================================================

TEST_F(ConversionTest, RoundTripCOOtoCSRtoCOO) {
    matgen_coo_matrix_t* original_coo = CreateTestCOO();
    ASSERT_NE(original_coo, nullptr);

    // COO -> CSR
    csr = matgen_coo_to_csr(original_coo);
    ASSERT_NE(csr, nullptr);

    // CSR -> COO
    coo = matgen_csr_to_coo(csr);
    ASSERT_NE(coo, nullptr);

    // Check all values match
    EXPECT_EQ(coo->rows, original_coo->rows);
    EXPECT_EQ(coo->cols, original_coo->cols);
    EXPECT_EQ(coo->nnz, original_coo->nnz);

    for (size_t i = 0; i < coo->rows; i++) {
        for (size_t j = 0; j < coo->cols; j++) {
            EXPECT_DOUBLE_EQ(matgen_coo_get(coo, i, j),
                           matgen_coo_get(original_coo, i, j));
        }
    }

    matgen_coo_destroy(original_coo);
}

TEST_F(ConversionTest, RoundTripLargeMatrix) {
    // Create a larger matrix
    coo = matgen_coo_create(100, 100, 50);
    ASSERT_NE(coo, nullptr);

    // Add random entries
    for (size_t i = 0; i < 50; i++) {
        matgen_coo_add_entry(coo, i, i, (double)(i + 1));
    }

    // COO -> CSR -> COO
    csr = matgen_coo_to_csr(coo);
    ASSERT_NE(csr, nullptr);

    matgen_coo_matrix_t* coo2 = matgen_csr_to_coo(csr);
    ASSERT_NE(coo2, nullptr);

    EXPECT_EQ(coo2->nnz, 50);

    // Verify diagonal values
    for (size_t i = 0; i < 50; i++) {
        EXPECT_DOUBLE_EQ(matgen_coo_get(coo2, i, i), (double)(i + 1));
    }

    matgen_coo_destroy(coo2);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(ConversionTest, EmptyMatrix) {
    coo = matgen_coo_create(5, 5, 0);
    ASSERT_NE(coo, nullptr);

    csr = matgen_coo_to_csr(coo);
    ASSERT_NE(csr, nullptr);

    EXPECT_EQ(csr->nnz, 0);
    EXPECT_TRUE(matgen_csr_validate(csr));

    // All row_ptr should be 0
    for (size_t i = 0; i <= csr->rows; i++) {
        EXPECT_EQ(csr->row_ptr[i], 0);
    }
}

TEST_F(ConversionTest, SingleElement) {
    coo = matgen_coo_create(10, 10, 1);
    ASSERT_NE(coo, nullptr);

    matgen_coo_add_entry(coo, 5, 7, 42.0);

    csr = matgen_coo_to_csr(coo);
    ASSERT_NE(csr, nullptr);

    EXPECT_EQ(csr->nnz, 1);
    EXPECT_DOUBLE_EQ(matgen_csr_get(csr, 5, 7), 42.0);
    EXPECT_TRUE(matgen_csr_validate(csr));
}
