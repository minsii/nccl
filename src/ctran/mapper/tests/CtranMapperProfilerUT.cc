// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "CtranMapper.h"
#include "checks.h"
#include "comm.h"
#include "nccl_cvars.h"

class CtranMapperProfilerTest : public ::testing::Test {
 public:
  ncclComm* dummyComm;
  double expectedDurMS;
  CtranMapperProfilerTest() = default;

 protected:
  void SetUp() override {
    setenv("NCCL_CTRAN_BACKENDS", "", 1);
    setenv("NCCL_DEBUG", "INFO", 1);
    setenv("NCCL_DEBUG_SUBSYS", "INIT", 1);

    // A random duration between 0-5ms to test the timer
    srand(time(NULL));
    expectedDurMS = rand() % 5 + 1;
  }
  void TearDown() override {
    unsetenv("NCCL_CTRAN_BACKENDS");
  }
};

TEST_F(CtranMapperProfilerTest, Timer) {
  auto timer = std::unique_ptr<CtranMapperTimer>(new CtranMapperTimer());
  EXPECT_THAT(timer, testing::NotNull());

  usleep(expectedDurMS * 1000);

  double durMs = timer->durationMs();
  EXPECT_GE(durMs, expectedDurMS);
}

TEST_F(CtranMapperProfilerTest, TimestampPoint) {
  int peer = 0;
  auto tp = std::unique_ptr<CtranMapperTimestampPoint>(
      new CtranMapperTimestampPoint(peer));
  EXPECT_THAT(tp, testing::NotNull());
}

TEST_F(CtranMapperProfilerTest, Timestamp) {
  auto dummyAlgo = "Ring";
  auto ts = std::unique_ptr<CtranMapperTimestamp>(
      new CtranMapperTimestamp(dummyAlgo));
  EXPECT_THAT(ts, testing::NotNull());
}

TEST_F(CtranMapperProfilerTest, TimestampInsert) {
  auto dummyAlgo = "Ring";
  auto ts = std::unique_ptr<CtranMapperTimestamp>(
      new CtranMapperTimestamp(dummyAlgo));
  EXPECT_THAT(ts, testing::NotNull());

  auto begin = std::chrono::high_resolution_clock::now();

  usleep(expectedDurMS * 1000);
  ts->recvCtrl.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putIssued.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putComplete.push_back(CtranMapperTimestampPoint(0));

  double durMs = 0.0;
  durMs = std::chrono::duration_cast<std::chrono::milliseconds>(
              ts->recvCtrl[0].now.time_since_epoch() - begin.time_since_epoch())
              .count();
  // recvCtrl should take >= `expectedDurMS` ms
  EXPECT_GE(durMs, expectedDurMS);

  durMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          ts->putIssued[0].now.time_since_epoch() - begin.time_since_epoch())
          .count();
  // putIssued should take >= 2 * `expectedDurMS` ms
  EXPECT_GE(durMs, expectedDurMS * 2);

  durMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          ts->putComplete[0].now.time_since_epoch() - begin.time_since_epoch())
          .count();
  // putComplete should take >= 3 * `expectedDurMS` ms
  EXPECT_GE(durMs, expectedDurMS * 3);
}

TEST_F(CtranMapperProfilerTest, MapperFlushTimerStdout) {
  setenv("NCCL_CTRAN_PROFILING", "stdout", 1);
  ncclCvarInit();
  NCCLCHECKABORT(ncclCommInitAll(&dummyComm, 1, nullptr));

  auto dummyAlgo = "Ring";
  auto mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());

  auto ts = std::unique_ptr<CtranMapperTimestamp>(
      new CtranMapperTimestamp(dummyAlgo));
  EXPECT_THAT(ts, testing::NotNull());

  usleep(expectedDurMS * 1000);
  ts->recvCtrl.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putIssued.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putComplete.push_back(CtranMapperTimestampPoint(0));

  mapper->timestamps.push_back(std::move(ts));

  auto kExpectedOutput1 = "Communication Profiling";
  auto kExpectedOutput2 = dummyAlgo;
  testing::internal::CaptureStdout();

  mapper->reportProfiling(true);

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput1));
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput2));
  NCCLCHECKABORT(ncclCommDestroy(dummyComm));
}

TEST_F(CtranMapperProfilerTest, MapperFlushTimerInfo) {
  setenv("NCCL_CTRAN_PROFILING", "info", 1);
  ncclCvarInit();
  NCCLCHECKABORT(ncclCommInitAll(&dummyComm, 1, nullptr));

  auto dummyAlgo = "Ring";
  auto mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());

  auto ts = std::unique_ptr<CtranMapperTimestamp>(
      new CtranMapperTimestamp(dummyAlgo));
  EXPECT_THAT(ts, testing::NotNull());

  usleep(expectedDurMS * 1000);
  ts->recvCtrl.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putIssued.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putComplete.push_back(CtranMapperTimestampPoint(0));

  mapper->timestamps.push_back(std::move(ts));

  auto kExpectedOutput1 = "NCCL INFO";
  auto kExpectedOutput2 = "Communication Profiling";
  auto kExpectedOutput3 = dummyAlgo;
  testing::internal::CaptureStdout();

  mapper->reportProfiling(true);

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput1));
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput2));
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput3));
  NCCLCHECKABORT(ncclCommDestroy(dummyComm));
}

TEST_F(CtranMapperProfilerTest, MapperFlushTimerKineto) {
  auto outputDir = "/tmp";
  auto pid = getpid();
  // NOTE: this is default prefix of output name, need to be consistent with the
  // code in CtranMapper.cc
  auto prefix = "nccl_ctran_log." + std::to_string(pid);
  setenv("NCCL_CTRAN_PROFILING", "kineto", 1);
  setenv("NCCL_CTRAN_KINETO_PROFILE_DIR", outputDir, 1);
  ncclCvarInit();
  NCCLCHECKABORT(ncclCommInitAll(&dummyComm, 1, nullptr));

  auto dummyAlgo = "Ring";
  auto mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());

  auto ts = std::unique_ptr<CtranMapperTimestamp>(
      new CtranMapperTimestamp(dummyAlgo));
  EXPECT_THAT(ts, testing::NotNull());

  usleep(expectedDurMS * 1000);
  ts->recvCtrl.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putIssued.push_back(CtranMapperTimestampPoint(0));

  usleep(expectedDurMS * 1000);
  ts->putComplete.push_back(CtranMapperTimestampPoint(0));

  mapper->timestamps.push_back(std::move(ts));

  mapper->reportProfiling(true);

  bool foundFile = false;
  for (auto& entry : std::filesystem::directory_iterator(outputDir)) {
    if (entry.path().has_filename() &&
        entry.path().filename().string().rfind(prefix, 0) == 0) {
      foundFile = true;
      // NOTE: uncomment below to check the content of the file
      /*
      std::ifstream file(entry.path());
      std::string contents;
      if (file.is_open()) {
        file >> contents;
        std::cout << "Contents of " << entry.path() << ": " << contents
                  << std::endl;
        file.close();
      } else {
        std::cout << "Failed to open " << entry.path() << std::endl;
      }
      */
    }
  }
  EXPECT_TRUE(foundFile);
  NCCLCHECKABORT(ncclCommDestroy(dummyComm));
}

TEST_F(CtranMapperProfilerTest, regSnapshot) {
  setenv("NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT", "0", 1);
  ncclCvarInit();
  NCCLCHECKABORT(ncclCommInitAll(&dummyComm, 1, nullptr));

  auto mapper = std::unique_ptr<CtranMapper>(new CtranMapper(dummyComm));
  EXPECT_THAT(mapper, testing::NotNull());

  int nBufs = rand() % 10 + 1;
  size_t bufSize = 4096;
  // allocate & register nBufs buffers, which is a random number between [1, 10]
  for (int i = 0; i < nBufs; ++i) {
    void* buf = nullptr;
    CUDACHECKIGNORE(cudaMalloc(&buf, bufSize));
    void* hdl = nullptr;

    auto res = mapper->regMem(buf, bufSize, &hdl, true);

    EXPECT_EQ(res, ncclSuccess);
    EXPECT_THAT(hdl, testing::NotNull());

    res = mapper->deregMem(hdl);
    EXPECT_EQ(res, ncclSuccess);

    CUDACHECKIGNORE(cudaFree(buf));
  }

  testing::internal::CaptureStdout();

  mapper->reportRegSnapshot();

  // expected outputs
  // NOTE: this has to match the output defined in CtranMapper.cc
  auto kExpectedOutput1 = "NCCL INFO";
  auto kExpectedOutput2 = "total registered " + std::to_string(nBufs);

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput1));
  EXPECT_THAT(output, testing::HasSubstr(kExpectedOutput2));

  unsetenv("NCCL_CTRAN_REGISTER_REPORT_SNAPSHOT_COUNT");
}
