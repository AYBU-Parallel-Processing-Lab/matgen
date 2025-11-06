#include <gtest/gtest.h>
#include <matgen/core/types.h>
#include <matgen/utils/argparse.h>

// Test fixture
class ArgparseTest : public ::testing::Test {
 protected:
  matgen_argparser_t* parser;  // NOLINT

  void SetUp() override {
    parser =
        matgen_argparser_create("test_program", "Test program description");
    ASSERT_NE(parser, nullptr);
  }

  void TearDown() override {
    if (parser != nullptr) {
      matgen_argparser_destroy(parser);
      parser = nullptr;
    }
  }
};

// =============================================================================
// Creation/Destruction Tests
// =============================================================================

TEST_F(ArgparseTest, CreateDestroy) {
  // Already created in SetUp, just verify it's not NULL
  EXPECT_NE(parser, nullptr);
}

TEST_F(ArgparseTest, CreateNullName) {
  matgen_argparser_t* p = matgen_argparser_create(NULL, "desc");
  EXPECT_EQ(p, nullptr);
}

TEST_F(ArgparseTest, DestroyNull) {
  // Should not crash
  matgen_argparser_destroy(nullptr);
}

// =============================================================================
// Boolean Flag Tests
// =============================================================================

TEST_F(ArgparseTest, AddAndParseBoolFlag) {
  bool verbose = false;

  EXPECT_EQ(matgen_argparser_add_flag(parser, "v", "verbose", &verbose,
                                      "Enable verbose mode"),
            MATGEN_SUCCESS);

  // Initially false
  EXPECT_FALSE(verbose);

  // Parse with short option
  char* argv[] = {(char*)"prog", (char*)"-v"};
  EXPECT_EQ(matgen_argparser_parse(parser, 2, argv), MATGEN_SUCCESS);
  EXPECT_TRUE(verbose);
}

TEST_F(ArgparseTest, ParseBoolLongOption) {
  bool debug = false;

  EXPECT_EQ(
      matgen_argparser_add_flag(parser, "d", "debug", &debug, "Enable debug"),
      MATGEN_SUCCESS);

  char* argv[] = {(char*)"prog", (char*)"--debug"};
  EXPECT_EQ(matgen_argparser_parse(parser, 2, argv), MATGEN_SUCCESS);
  EXPECT_TRUE(debug);
}

// =============================================================================
// U64 Integer Tests
// =============================================================================

TEST_F(ArgparseTest, AddAndParseU64) {
  u64 count = 0;

  EXPECT_EQ(matgen_argparser_add_u64(parser, "n", "count", &count, 10,
                                     "Number of items"),
            MATGEN_SUCCESS);

  // Should have default value
  EXPECT_EQ(count, 10);

  // Parse with short option
  char* argv[] = {(char*)"prog", (char*)"-n", (char*)"42"};
  EXPECT_EQ(matgen_argparser_parse(parser, 3, argv), MATGEN_SUCCESS);
  EXPECT_EQ(count, 42);
}

TEST_F(ArgparseTest, ParseU64LongOption) {
  u64 size = 100;

  EXPECT_EQ(
      matgen_argparser_add_u64(parser, "s", "size", &size, 100, "Matrix size"),
      MATGEN_SUCCESS);

  char* argv[] = {(char*)"prog", (char*)"--size", (char*)"1000"};
  EXPECT_EQ(matgen_argparser_parse(parser, 3, argv), MATGEN_SUCCESS);
  EXPECT_EQ(size, 1000);
}

TEST_F(ArgparseTest, ParseU64WithEquals) {
  u64 value = 0;

  EXPECT_EQ(
      matgen_argparser_add_u64(parser, NULL, "value", &value, 5, "Some value"),
      MATGEN_SUCCESS);

  char* argv[] = {(char*)"prog", (char*)"--value=999"};
  EXPECT_EQ(matgen_argparser_parse(parser, 2, argv), MATGEN_SUCCESS);
  EXPECT_EQ(value, 999);
}

TEST_F(ArgparseTest, ParseU64Invalid) {
  u64 value = 0;

  EXPECT_EQ(matgen_argparser_add_u64(parser, NULL, "value", &value, 0, "Value"),
            MATGEN_SUCCESS);

  char* argv[] = {(char*)"prog", (char*)"--value", (char*)"not_a_number"};
  EXPECT_NE(matgen_argparser_parse(parser, 3, argv), MATGEN_SUCCESS);
}

// =============================================================================
// F64 Double Tests
// =============================================================================

TEST_F(ArgparseTest, AddAndParseF64) {
  f64 threshold = 0.0;

  EXPECT_EQ(matgen_argparser_add_f64(parser, "t", "threshold", &threshold, 0.5,
                                     "Threshold value"),
            MATGEN_SUCCESS);

  // Should have default value
  EXPECT_DOUBLE_EQ(threshold, 0.5);

  char* argv[] = {(char*)"prog", (char*)"-t", (char*)"0.75"};
  EXPECT_EQ(matgen_argparser_parse(parser, 3, argv), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(threshold, 0.75);
}

TEST_F(ArgparseTest, ParseF64Negative) {
  f64 value = 0.0;

  EXPECT_EQ(
      matgen_argparser_add_f64(parser, NULL, "value", &value, 1.0, "Value"),
      MATGEN_SUCCESS);

  char* argv[] = {(char*)"prog", (char*)"--value", (char*)"-3.14"};
  EXPECT_EQ(matgen_argparser_parse(parser, 3, argv), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, -3.14);
}

TEST_F(ArgparseTest, ParseF64Scientific) {
  f64 value = 0.0;

  EXPECT_EQ(
      matgen_argparser_add_f64(parser, NULL, "value", &value, 0.0, "Value"),
      MATGEN_SUCCESS);

  char* argv[] = {(char*)"prog", (char*)"--value", (char*)"1.5e-10"};
  EXPECT_EQ(matgen_argparser_parse(parser, 3, argv), MATGEN_SUCCESS);
  EXPECT_DOUBLE_EQ(value, 1.5e-10);
}

TEST_F(ArgparseTest, ParseF64Invalid) {
  f64 value = 0.0;

  EXPECT_EQ(
      matgen_argparser_add_f64(parser, NULL, "value", &value, 0.0, "Value"),
      MATGEN_SUCCESS);

  char* argv[] = {(char*)"prog", (char*)"--value", (char*)"not_a_number"};
  EXPECT_NE(matgen_argparser_parse(parser, 3, argv), MATGEN_SUCCESS);
}

// =============================================================================
// String Tests
// =============================================================================

TEST_F(ArgparseTest, AddAndParseString) {
  const char* output = nullptr;

  EXPECT_EQ(matgen_argparser_add_string(parser, "o", "output", &output,
                                        "default.txt", "Output file"),
            MATGEN_SUCCESS);

  // Should have default value
  EXPECT_STREQ(output, "default.txt");

  char* argv[] = {(char*)"prog", (char*)"-o", (char*)"result.txt"};
  EXPECT_EQ(matgen_argparser_parse(parser, 3, argv), MATGEN_SUCCESS);
  EXPECT_STREQ(output, "result.txt");
}

TEST_F(ArgparseTest, ParseStringWithEquals) {
  const char* input = nullptr;

  EXPECT_EQ(matgen_argparser_add_string(parser, NULL, "input", &input, NULL,
                                        "Input file"),
            MATGEN_SUCCESS);

  char* argv[] = {(char*)"prog", (char*)"--input=data.mtx"};
  EXPECT_EQ(matgen_argparser_parse(parser, 2, argv), MATGEN_SUCCESS);
  EXPECT_STREQ(input, "data.mtx");
}

// =============================================================================
// Multiple Arguments Tests
// =============================================================================

TEST_F(ArgparseTest, ParseMultipleArguments) {
  bool verbose = false;
  u64 count = 0;
  f64 threshold = 0.0;
  const char* output = nullptr;

  matgen_argparser_add_flag(parser, "v", "verbose", &verbose, "Verbose");
  matgen_argparser_add_u64(parser, "n", "count", &count, 10, "Count");
  matgen_argparser_add_f64(parser, "t", "threshold", &threshold, 0.5,
                           "Threshold");
  matgen_argparser_add_string(parser, "o", "output", &output, "out.txt",
                              "Output");

  char* argv[] = {(char*)"prog",
                  (char*)"-v",
                  (char*)"-n",
                  (char*)"100",
                  (char*)"--threshold=0.99",
                  (char*)"--output",
                  (char*)"result.txt"};

  EXPECT_EQ(matgen_argparser_parse(parser, 7, argv), MATGEN_SUCCESS);

  EXPECT_TRUE(verbose);
  EXPECT_EQ(count, 100);
  EXPECT_DOUBLE_EQ(threshold, 0.99);
  EXPECT_STREQ(output, "result.txt");
}

TEST_F(ArgparseTest, UnknownOption) {
  bool flag = false;
  matgen_argparser_add_flag(parser, "v", "verbose", &flag, "Verbose");

  char* argv[] = {(char*)"prog", (char*)"--unknown"};
  EXPECT_NE(matgen_argparser_parse(parser, 2, argv), MATGEN_SUCCESS);
}

TEST_F(ArgparseTest, MissingRequiredValue) {
  u64 count = 0;
  matgen_argparser_add_u64(parser, "n", "count", &count, 0, "Count");

  // -n without value
  char* argv[] = {(char*)"prog", (char*)"-n"};
  EXPECT_NE(matgen_argparser_parse(parser, 2, argv), MATGEN_SUCCESS);
}

// =============================================================================
// Help Tests
// =============================================================================

TEST_F(ArgparseTest, PrintHelp) {
  bool verbose = false;
  u64 count = 0;

  matgen_argparser_add_flag(parser, "v", "verbose", &verbose,
                            "Enable verbose output");
  matgen_argparser_add_u64(parser, "n", "count", &count, 10,
                           "Number of iterations");

  // Should not crash
  matgen_argparser_print_help(parser, stdout);
  matgen_argparser_print_usage(parser, stdout);
}

TEST_F(ArgparseTest, PrintHelpNull) {
  // Should not crash
  matgen_argparser_print_help(nullptr, stdout);
  matgen_argparser_print_usage(nullptr, stdout);

  matgen_argparser_print_help(parser, nullptr);
  matgen_argparser_print_usage(parser, nullptr);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(ArgparseTest, EmptyArguments) {
  bool flag = false;
  matgen_argparser_add_flag(parser, "v", "verbose", &flag, "Verbose");

  char* argv[] = {(char*)"prog"};
  EXPECT_EQ(matgen_argparser_parse(parser, 1, argv), MATGEN_SUCCESS);

  // Should keep default value
  EXPECT_FALSE(flag);
}

TEST_F(ArgparseTest, OnlyLongOption) {
  bool flag = false;

  EXPECT_EQ(
      matgen_argparser_add_flag(parser, NULL, "verbose", &flag, "Verbose"),
      MATGEN_SUCCESS);

  char* argv[] = {(char*)"prog", (char*)"--verbose"};
  EXPECT_EQ(matgen_argparser_parse(parser, 2, argv), MATGEN_SUCCESS);
  EXPECT_TRUE(flag);
}

TEST_F(ArgparseTest, OnlyShortOption) {
  bool flag = false;

  EXPECT_EQ(matgen_argparser_add_flag(parser, "v", NULL, &flag, "Verbose"),
            MATGEN_SUCCESS);

  char* argv[] = {(char*)"prog", (char*)"-v"};
  EXPECT_EQ(matgen_argparser_parse(parser, 2, argv), MATGEN_SUCCESS);
  EXPECT_TRUE(flag);
}
