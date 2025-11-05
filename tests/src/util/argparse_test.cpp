#include "matgen/util/argparse.h"

#include <gtest/gtest.h>

#include <cstring>

#include "matgen/util/log.h"

class ArgParseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    parser = nullptr;

    // Suppress log output during tests
    matgen_log_set_level(MATGEN_LOG_LEVEL_OFF);
  }

  void TearDown() override {
    if (parser != nullptr) {
      matgen_argparser_destroy(parser);
      parser = nullptr;
    }
  }

  // Helper to create argv from strings
  char** MakeArgv(const std::vector<const char*>& args) {
    ((void)this);

    char** argv = new char*[args.size()];
    for (size_t i = 0; i < args.size(); i++) {
      argv[i] = const_cast<char*>(args[i]);
    }
    return argv;
  }

  void FreeArgv(char** argv) {
    ((void)this);
    delete[] argv;
  }

  matgen_argparser_t* parser{nullptr};  // NOLINT
};

// =============================================================================
// Parser Creation Tests
// =============================================================================

TEST_F(ArgParseTest, CreateAndDestroy) {
  parser = matgen_argparser_create("test_prog", "Test program");
  ASSERT_NE(parser, nullptr);
}

TEST_F(ArgParseTest, CreateWithNullName) {
  parser = matgen_argparser_create(nullptr, "Description");
  EXPECT_EQ(parser, nullptr);
}

// =============================================================================
// Flag Tests
// =============================================================================

TEST_F(ArgParseTest, AddFlag) {
  parser = matgen_argparser_create("test", "Test");
  ASSERT_NE(parser, nullptr);

  bool verbose = false;
  int result = matgen_argparser_add_flag(parser, "v", "verbose", &verbose,
                                         "Enable verbose mode");
  EXPECT_EQ(result, 0);
  EXPECT_FALSE(verbose);  // Should be initialized to false
}

TEST_F(ArgParseTest, ParseShortFlag) {
  parser = matgen_argparser_create("test", "Test");
  bool verbose = false;
  matgen_argparser_add_flag(parser, "v", "verbose", &verbose, "Verbose");

  std::vector<const char*> args = {"program", "-v"};
  char** argv = MakeArgv(args);

  int result =
      matgen_argparser_parse(parser, static_cast<int>(args.size()), argv);

  EXPECT_EQ(result, 0);
  EXPECT_TRUE(verbose);

  FreeArgv(argv);
}

TEST_F(ArgParseTest, ParseLongFlag) {
  parser = matgen_argparser_create("test", "Test");
  bool verbose = false;
  matgen_argparser_add_flag(parser, "v", "verbose", &verbose, "Verbose");

  std::vector<const char*> args = {"program", "--verbose"};
  char** argv = MakeArgv(args);

  int result =
      matgen_argparser_parse(parser, static_cast<int>(args.size()), argv);

  EXPECT_EQ(result, 0);
  EXPECT_TRUE(verbose);

  FreeArgv(argv);
}

// =============================================================================
// Integer Argument Tests
// =============================================================================

TEST_F(ArgParseTest, AddIntWithDefault) {
  parser = matgen_argparser_create("test", "Test");
  int count = 0;

  int result =
      matgen_argparser_add_int(parser, "n", "count", &count, 42, "Count");

  EXPECT_EQ(result, 0);
  EXPECT_EQ(count, 42);  // Should have default value
}

TEST_F(ArgParseTest, ParseIntShort) {
  parser = matgen_argparser_create("test", "Test");
  int count = 0;
  matgen_argparser_add_int(parser, "n", "count", &count, 10, "Count");

  std::vector<const char*> args = {"program", "-n", "123"};
  char** argv = MakeArgv(args);

  int result =
      matgen_argparser_parse(parser, static_cast<int>(args.size()), argv);

  EXPECT_EQ(result, 0);
  EXPECT_EQ(count, 123);

  FreeArgv(argv);
}

TEST_F(ArgParseTest, ParseIntLong) {
  parser = matgen_argparser_create("test", "Test");
  int count = 0;
  matgen_argparser_add_int(parser, "n", "count", &count, 10, "Count");

  std::vector<const char*> args = {"program", "--count", "456"};
  char** argv = MakeArgv(args);

  int result =
      matgen_argparser_parse(parser, static_cast<int>(args.size()), argv);

  EXPECT_EQ(result, 0);
  EXPECT_EQ(count, 456);

  FreeArgv(argv);
}

TEST_F(ArgParseTest, ParseIntLongEquals) {
  parser = matgen_argparser_create("test", "Test");
  int count = 0;
  matgen_argparser_add_int(parser, "n", "count", &count, 10, "Count");

  std::vector<const char*> args = {"program", "--count=789"};
  char** argv = MakeArgv(args);

  int result =
      matgen_argparser_parse(parser, static_cast<int>(args.size()), argv);

  EXPECT_EQ(result, 0);
  EXPECT_EQ(count, 789);

  FreeArgv(argv);
}

TEST_F(ArgParseTest, ParseInvalidInt) {
  parser = matgen_argparser_create("test", "Test");
  int count = 0;
  matgen_argparser_add_int(parser, "n", "count", &count, 10, "Count");

  std::vector<const char*> args = {"program", "-n", "not_a_number"};
  char** argv = MakeArgv(args);

  int result =
      matgen_argparser_parse(parser, static_cast<int>(args.size()), argv);

  EXPECT_NE(result, 0);  // Should fail

  FreeArgv(argv);
}

// =============================================================================
// String Argument Tests
// =============================================================================

TEST_F(ArgParseTest, AddStringWithDefault) {
  parser = matgen_argparser_create("test", "Test");
  const char* output = nullptr;

  int result = matgen_argparser_add_string(parser, "o", "output", &output,
                                           "default.txt", "Output file");

  EXPECT_EQ(result, 0);
  EXPECT_STREQ(output, "default.txt");
}

TEST_F(ArgParseTest, ParseString) {
  parser = matgen_argparser_create("test", "Test");
  const char* output = nullptr;
  matgen_argparser_add_string(parser, "o", "output", &output, "default.txt",
                              "Output");

  std::vector<const char*> args = {"program", "-o", "myfile.txt"};
  char** argv = MakeArgv(args);

  int result =
      matgen_argparser_parse(parser, static_cast<int>(args.size()), argv);

  EXPECT_EQ(result, 0);
  EXPECT_STREQ(output, "myfile.txt");

  FreeArgv(argv);
}

TEST_F(ArgParseTest, ParseStringLongEquals) {
  parser = matgen_argparser_create("test", "Test");
  const char* output = nullptr;
  matgen_argparser_add_string(parser, "o", "output", &output, "default.txt",
                              "Output");

  std::vector<const char*> args = {"program", "--output=result.txt"};
  char** argv = MakeArgv(args);

  int result =
      matgen_argparser_parse(parser, static_cast<int>(args.size()), argv);

  EXPECT_EQ(result, 0);
  EXPECT_STREQ(output, "result.txt");

  FreeArgv(argv);
}

// =============================================================================
// Multiple Arguments Tests
// =============================================================================

TEST_F(ArgParseTest, ParseMultipleArguments) {
  parser = matgen_argparser_create("test", "Test");

  bool verbose = false;
  int count = 0;
  const char* output = nullptr;

  matgen_argparser_add_flag(parser, "v", "verbose", &verbose, "Verbose");
  matgen_argparser_add_int(parser, "n", "count", &count, 10, "Count");
  matgen_argparser_add_string(parser, "o", "output", &output, "out.txt",
                              "Output");

  std::vector<const char*> args = {"program", "-v", "-n", "50",
                                   "--output=file.txt"};
  char** argv = MakeArgv(args);

  int result =
      matgen_argparser_parse(parser, static_cast<int>(args.size()), argv);

  EXPECT_EQ(result, 0);
  EXPECT_TRUE(verbose);
  EXPECT_EQ(count, 50);
  EXPECT_STREQ(output, "file.txt");

  FreeArgv(argv);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(ArgParseTest, UnknownOption) {
  parser = matgen_argparser_create("test", "Test");

  std::vector<const char*> args = {"program", "--unknown"};
  char** argv = MakeArgv(args);

  int result =
      matgen_argparser_parse(parser, static_cast<int>(args.size()), argv);

  EXPECT_NE(result, 0);  // Should fail

  FreeArgv(argv);
}

TEST_F(ArgParseTest, MissingValue) {
  parser = matgen_argparser_create("test", "Test");
  int count = 0;
  matgen_argparser_add_int(parser, "n", "count", &count, 10, "Count");

  std::vector<const char*> args = {"program", "-n"};  // Missing value
  char** argv = MakeArgv(args);

  int result =
      matgen_argparser_parse(parser, static_cast<int>(args.size()), argv);

  EXPECT_NE(result, 0);  // Should fail

  FreeArgv(argv);
}

// =============================================================================
// Help Generation Tests
// =============================================================================

TEST_F(ArgParseTest, PrintHelp) {
  parser =
      matgen_argparser_create("test_program", "A test program for argparse");

  bool verbose = false;
  int count = 0;

  matgen_argparser_add_flag(parser, "v", "verbose", &verbose,
                            "Enable verbose output");
  matgen_argparser_add_int(parser, "n", "count", &count, 42,
                           "Number of iterations");

  // Redirect output to test (don't actually print during test)
  FILE* tmp = tmpfile();
  ASSERT_NE(tmp, nullptr);

  matgen_argparser_print_help(parser, tmp);

  // Check that something was written
  fflush(tmp);
  fseek(tmp, 0, SEEK_END);
  long size = ftell(tmp);
  EXPECT_GT(size, 0);

  fclose(tmp);
}
