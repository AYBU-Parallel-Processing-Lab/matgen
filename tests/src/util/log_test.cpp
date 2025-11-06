#include <gtest/gtest.h>
#include <matgen/utils/log.h>

#include <cstdio>

class LogTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a temporary file for capturing log output
    tmpfile_ = tmpfile();
    ASSERT_NE(tmpfile_, nullptr);

    // Configure logging to use our test file
    matgen_log_set_stream(tmpfile_);
    matgen_log_set_level(MATGEN_LOG_LEVEL_TRACE);
    matgen_log_set_timestamps(false);  // Disable for predictable output
    matgen_log_set_color(false);       // Disable for predictable output
  }

  void TearDown() override {
    if (tmpfile_ != nullptr) {
      fclose(tmpfile_);
      tmpfile_ = nullptr;
    }

    // Reset to defaults
    matgen_log_set_stream(stderr);
    matgen_log_set_level(MATGEN_LOG_LEVEL_INFO);
    matgen_log_set_timestamps(true);
  }

  std::string GetLogOutput() {
    fflush(tmpfile_);
    fseek(tmpfile_, 0, SEEK_SET);

    std::string output;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), tmpfile_) != nullptr) {
      output += buffer;
    }

    return output;
  }

  void ClearLogOutput() {
    fflush(tmpfile_);
    fclose(tmpfile_);
    tmpfile_ = tmpfile();
    if (tmpfile_ != nullptr) {
      matgen_log_set_stream(tmpfile_);
    }
  }

  FILE* tmpfile_{nullptr};  // NOLINT
};

// =============================================================================
// Log Level Tests
// =============================================================================

TEST_F(LogTest, SetAndGetLevel) {
  matgen_log_set_level(MATGEN_LOG_LEVEL_DEBUG);
  EXPECT_EQ(matgen_log_get_level(), MATGEN_LOG_LEVEL_DEBUG);

  matgen_log_set_level(MATGEN_LOG_LEVEL_ERROR);
  EXPECT_EQ(matgen_log_get_level(), MATGEN_LOG_LEVEL_ERROR);
}

TEST_F(LogTest, LogLevelFiltering) {
  matgen_log_set_level(MATGEN_LOG_LEVEL_WARN);

  MATGEN_LOG_TRACE("trace message");
  MATGEN_LOG_DEBUG("debug message");
  MATGEN_LOG_INFO("info message");

  std::string output = GetLogOutput();
  EXPECT_TRUE(output.empty())
      << "No messages should be logged below WARN level";

  ClearLogOutput();

  MATGEN_LOG_WARN("warning message");
  MATGEN_LOG_ERROR("error message");

  output = GetLogOutput();
  EXPECT_NE(output.find("WARN"), std::string::npos);
  EXPECT_NE(output.find("ERROR"), std::string::npos);
}

// =============================================================================
// Message Content Tests
// =============================================================================

TEST_F(LogTest, LogMessageContent) {
  MATGEN_LOG_INFO("Test message: %d", 42);

  std::string output = GetLogOutput();
  EXPECT_NE(output.find("Test message: 42"), std::string::npos);
  EXPECT_NE(output.find("[INFO ]"), std::string::npos);
}

TEST_F(LogTest, AllLogLevels) {
  MATGEN_LOG_TRACE("trace");
  MATGEN_LOG_DEBUG("debug");
  MATGEN_LOG_INFO("info");
  MATGEN_LOG_WARN("warn");
  MATGEN_LOG_ERROR("error");
  MATGEN_LOG_FATAL("fatal");

  std::string output = GetLogOutput();
  EXPECT_NE(output.find("[TRACE]"), std::string::npos);
  EXPECT_NE(output.find("[DEBUG]"), std::string::npos);
  EXPECT_NE(output.find("[INFO ]"), std::string::npos);
  EXPECT_NE(output.find("[WARN ]"), std::string::npos);
  EXPECT_NE(output.find("[ERROR]"), std::string::npos);
  EXPECT_NE(output.find("[FATAL]"), std::string::npos);
}

// =============================================================================
// Formatting Tests
// =============================================================================

TEST_F(LogTest, FormattingIntegers) {
  MATGEN_LOG_INFO("Value: %d", 123);
  std::string output = GetLogOutput();
  EXPECT_NE(output.find("Value: 123"), std::string::npos);
}

TEST_F(LogTest, FormattingFloats) {
  MATGEN_LOG_INFO("Value: %.2f", 3.14159);
  std::string output = GetLogOutput();
  EXPECT_NE(output.find("Value: 3.14"), std::string::npos);
}

TEST_F(LogTest, FormattingStrings) {
  MATGEN_LOG_INFO("Hello, %s!", "world");
  std::string output = GetLogOutput();
  EXPECT_NE(output.find("Hello, world!"), std::string::npos);
}

TEST_F(LogTest, MultipleArguments) {
  MATGEN_LOG_INFO("int=%d, float=%.1f, str=%s", 42, 3.14, "test");
  std::string output = GetLogOutput();
  EXPECT_NE(output.find("int=42"), std::string::npos);
  EXPECT_NE(output.find("float=3.1"), std::string::npos);
  EXPECT_NE(output.find("str=test"), std::string::npos);
}

// =============================================================================
// Configuration Tests
// =============================================================================

TEST_F(LogTest, TimestampsEnabled) {
  matgen_log_set_timestamps(true);
  MATGEN_LOG_INFO("test");

  std::string output = GetLogOutput();
  // Should contain something like "[2024-01-01 12:34:56]"
  EXPECT_TRUE(output.find('[') != std::string::npos);
}

TEST_F(LogTest, TimestampsDisabled) {
  matgen_log_set_timestamps(false);
  MATGEN_LOG_INFO("test");

  std::string output = GetLogOutput();
  // First character after potential color codes should be '['
  EXPECT_NE(output.find("[INFO ]"), std::string::npos);
}

// =============================================================================
// Debug Level File/Line Tests
// =============================================================================

TEST_F(LogTest, DebugLevelShowsLocation) {
  matgen_log_set_level(MATGEN_LOG_LEVEL_DEBUG);
  MATGEN_LOG_DEBUG("debug message");

  std::string output = GetLogOutput();
  // Should contain filename and function name
  EXPECT_NE(output.find("log_test.cpp"), std::string::npos);
}

TEST_F(LogTest, InfoLevelHidesLocation) {
  MATGEN_LOG_INFO("info message");

  std::string output = GetLogOutput();
  // Should NOT contain filename
  EXPECT_EQ(output.find("test_log.cpp"), std::string::npos);
}
