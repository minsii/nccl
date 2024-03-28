#include <stdlib.h>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <folly/ScopeGuard.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "Logger.h"
#include "comm.h"
#include "debug.h"
#include "gmock/gmock.h"
#include "nccl.h"
#include "nccl_cvars.h"

// Test in this file only works with "buck2 test"!!!
// Run it with "buck2 run" will result in unpredictable failure

class NcclLoggerChecker {
 public:
  static std::unique_ptr<NcclLogger>& getLogger() {
    return NcclLogger::singleton_;
  }
};

constexpr const char* kTestStr = "test1";
constexpr size_t kBufferSize = 4096;

static void setupCommonEnvVars() {
  NCCL_DEBUG_SUBSYS = "INIT";
  NCCL_DEBUG = "INFO";
}

static std::string getTestFilePath() {
  auto tmpPath = std::filesystem::temp_directory_path();
  tmpPath /= "loggertest" + std::to_string(getPidHash());
  return tmpPath.string();
}

TEST(LoggerTest, SyncLogStdout) {
  ncclCvarInit();
  setupCommonEnvVars();
  NCCL_LOGGER_MODE = NCCL_LOGGER_MODE::sync;
  NCCL_DEBUG_FILE = NCCL_DEBUG_FILE_DEFAULT;
  testing::internal::CaptureStdout();

  INFO(NCCL_INIT, kTestStr);

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST(LoggerTest, SyncLogSpecifyDebugFile) {
  ncclCvarInit();
  setupCommonEnvVars();

  auto testFile = getTestFilePath();
  NCCL_LOGGER_MODE = NCCL_LOGGER_MODE::sync;
  NCCL_DEBUG_FILE = testFile;

  INFO(NCCL_INIT, kTestStr);

  // Manually close the logging file and redirect ncclDebugFile to stdout so
  // that we can read and delete the logging file safely
  fclose(ncclDebugFile);
  ncclDebugFile = stdout;

  auto testFd = fopen(testFile.c_str(), "r");
  ASSERT_NE(testFd, nullptr);
  // Always close file and delete the file after the test
  auto fileGuard = folly::makeGuard([&]() {
    fclose(testFd);
    std::filesystem::remove(testFile);
  });

  char buffer[kBufferSize];
  auto len = fread(buffer, 1, kBufferSize, testFd);
  EXPECT_GT(len, 0);
  EXPECT_THAT(std::string(buffer, len), testing::HasSubstr(kTestStr));
}

TEST(LoggerTest, AsyncLogStdout) {
  ncclCvarInit();
  setupCommonEnvVars();
  NCCL_LOGGER_MODE = NCCL_LOGGER_MODE::async;
  NCCL_DEBUG_FILE = NCCL_DEBUG_FILE_DEFAULT;
  testing::internal::CaptureStdout();

  INFO(NCCL_INIT, kTestStr);

  // Check fater the GetCapturedStdout so the EXPECT_EQ will not be silently
  // captured by
  auto loggerPreviousPtr = NcclLoggerChecker::getLogger().get();
  // Free the logger so we will join the logging thread. Otherwise we cannot
  // guarantee the logging thread to finish before the test ends.
  NcclLoggerChecker::getLogger().reset();

  std::string output = testing::internal::GetCapturedStdout();

  // We should've already initialized the logger by logging
  EXPECT_NE(loggerPreviousPtr, nullptr);

  EXPECT_THAT(output, testing::HasSubstr(kTestStr));
}

TEST(LoggerTest, AsyncLogSpecifyDebugFile) {
  ncclCvarInit();
  setupCommonEnvVars();

  auto testFile = getTestFilePath();
  NCCL_LOGGER_MODE = NCCL_LOGGER_MODE::async;
  NCCL_DEBUG_FILE = testFile;

  INFO(NCCL_INIT, kTestStr);

  // We should've already initialized the logger by logging
  EXPECT_NE(NcclLoggerChecker::getLogger(), nullptr);
  // Free the logger so we will join the logging thread. Otherwise we cannot
  // guarantee the logging thread to finish before the test ends.
  NcclLoggerChecker::getLogger().reset();

  // Manually close the logging file and redirect ncclDebugFile to stdout so
  // that we can read and delete the logging file safely
  fclose(ncclDebugFile);
  ncclDebugFile = stdout;

  auto testFd = fopen(testFile.c_str(), "r");
  ASSERT_NE(testFd, nullptr);
  // Always close file and delete the file after the test
  auto fileGuard = folly::makeGuard([&]() {
    fclose(testFd);
    std::filesystem::remove(testFile);
  });

  char buffer[kBufferSize];
  auto len = fread(buffer, 1, kBufferSize, testFd);
  EXPECT_GT(len, 0);
  EXPECT_THAT(std::string(buffer, len), testing::HasSubstr(kTestStr));
}

TEST(LoggerTest, UseBeforeCvarInit) {
  unsetenv("NCCL_DEBUG_FILE");
  setenv("NCCL_DEBUG", "INFO", 1);
  setenv("NCCL_DEBUG_SUBSYS", "INIT", 1);

  // Before init, the logger should log to stdout by default
  EXPECT_EQ(ncclDebugFile, stdout);

  testing::internal::CaptureStdout();

  INFO(NCCL_INIT, kTestStr);

  std::string output = testing::internal::GetCapturedStdout();

  // Logging should initialize ncclDebugLevel. It would be NCCL_LOG_NONE because
  // we no longer check NCCL_DEBUG env var directly.
  EXPECT_EQ(ncclDebugLevel, NCCL_LOG_NONE);
  // Should not log anything because the ncclDebugLevel is NCCL_LOG_NONE.
  EXPECT_THAT(output, testing::Not(testing::HasSubstr(kTestStr)));

  // Start Cvar init
  ncclCvarInit();
  setupCommonEnvVars();

  auto testFile = getTestFilePath();
  NCCL_LOGGER_MODE = NCCL_LOGGER_MODE::async;
  NCCL_DEBUG_FILE = testFile;

  // Manually change debug.cc back to uninitialized state so the following log
  // will invoke logger init again. This should set logging file to the file
  // specified in NCCL_DEBUG_FILE
  ncclDebugLevel = -1;

  INFO(NCCL_INIT, kTestStr);
  // Free the logger so we will join the logging thread. Otherwise we cannot
  // guarantee the logging thread to finish before the test ends.
  NcclLoggerChecker::getLogger().reset();

  // Manually close the logging file and redirect ncclDebugFile to stdout so
  // that we can read and delete the logging file safely
  fclose(ncclDebugFile);
  ncclDebugFile = stdout;

  auto testFd = fopen(testFile.c_str(), "r");
  ASSERT_NE(testFd, nullptr);
  // Always close file and delete the file after the test
  auto fileGuard = folly::makeGuard([&]() {
    fclose(testFd);
    std::filesystem::remove(testFile);
  });

  char buffer[kBufferSize];
  auto len = fread(buffer, 1, kBufferSize, testFd);
  EXPECT_GT(len, 0);
  EXPECT_THAT(std::string(buffer, len), testing::HasSubstr(kTestStr));
}
