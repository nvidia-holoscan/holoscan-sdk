/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pure filesystem tests for publish_engine_file(). A TensorRT engine cache is trusted as soon as
// its path exists, so concurrent builders and loaders must never observe a partial engine.

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <utils.hpp>

#include "support/host_unique_test_name.hpp"

namespace {

namespace HoloInfer = holoscan::inference;

class EngineCachePublication : public ::testing::Test {
 protected:
  void SetUp() override {
    directory_ = std::filesystem::temp_directory_path() /
                 holoscan::test::make_host_unique_test_name("holoinfer_engine_publication");
    std::filesystem::create_directories(directory_);
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

  [[nodiscard]] std::vector<std::string> entries() const {
    std::vector<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(directory_)) {
      names.push_back(entry.path().filename().string());
    }
    return names;
  }

  [[nodiscard]] static std::vector<char> read_all(std::ifstream& stream) {
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
  }

  [[nodiscard]] static std::vector<char> pattern(const std::size_t size, const char seed) {
    std::vector<char> bytes(size);
    for (std::size_t index = 0U; index < size; ++index) {
      bytes[index] = static_cast<char>(seed + static_cast<char>(index % 31U));
    }
    return bytes;
  }

  std::filesystem::path directory_;
};

TEST_F(EngineCachePublication, PublishesCompleteFileWithoutTemporarySiblings) {
  const auto engine = (directory_ / "model.engine.fp32").string();
  const auto bytes = pattern(4096U, 'a');
  ASSERT_TRUE(HoloInfer::publish_engine_file(engine, bytes.data(), bytes.size()));

  std::ifstream stream(engine, std::ios::binary);
  ASSERT_TRUE(stream);
  EXPECT_EQ(read_all(stream), bytes);
  EXPECT_EQ(entries(), std::vector<std::string>{"model.engine.fp32"});
}

// Generated engine names keep the model stem and add GPU, TensorRT, configuration, and precision
// fields, so a valid destination can be close to the 255-byte component limit of common Linux
// filesystems. The temporary file's name must not extend the destination's name.
TEST_F(EngineCachePublication, PublishesDestinationNameAtComponentLengthLimit) {
  const std::string suffix = ".engine.fp32";
  const std::string name = std::string(250U - suffix.size(), 'm') + suffix;
  const auto engine = (directory_ / name).string();
  {
    const std::ofstream probe(engine, std::ios::binary);
    if (!probe) {
      GTEST_SKIP() << "the temporary filesystem rejects a 250-byte file name";
    }
  }
  std::filesystem::remove(engine);

  const auto bytes = pattern(1024U, 'g');
  ASSERT_TRUE(HoloInfer::publish_engine_file(engine, bytes.data(), bytes.size()));

  std::ifstream stream(engine, std::ios::binary);
  ASSERT_TRUE(stream);
  EXPECT_EQ(read_all(stream), bytes);
  EXPECT_EQ(entries(), std::vector<std::string>{name});
}

// A loader polling the cache path while a large engine is being written must only ever see the
// complete size. Writing the destination in place would expose every intermediate size.
TEST_F(EngineCachePublication, ConcurrentReaderNeverObservesPartialEngine) {
  const auto engine = (directory_ / "large.engine.fp32").string();
  const auto bytes = pattern(32U * 1024U * 1024U, 'k');
  std::atomic<bool> writer_done{false};
  std::atomic<std::uint64_t> partial_observations{0U};
  std::atomic<std::uint64_t> complete_observations{0U};
  std::thread reader([&] {
    while (true) {
      const bool finished = writer_done.load(std::memory_order_acquire);
      std::error_code error;
      const auto size = std::filesystem::file_size(engine, error);
      if (!error) {
        (size == bytes.size() ? complete_observations : partial_observations)
            .fetch_add(1U, std::memory_order_relaxed);
      }
      if (finished) {
        return;
      }
    }
  });
  const bool published = HoloInfer::publish_engine_file(engine, bytes.data(), bytes.size());
  writer_done.store(true, std::memory_order_release);
  reader.join();

  ASSERT_TRUE(published);
  EXPECT_EQ(partial_observations.load(), 0U);
  EXPECT_GE(complete_observations.load(), 1U);
}

// Replacing a cached engine must not change the bytes a loader already opened.
TEST_F(EngineCachePublication, ReplacementLeavesOpenReaderOnCompleteOldEngine) {
  const auto engine = (directory_ / "model.engine.fp16").string();
  const auto old_bytes = pattern(8192U, 'o');
  const auto new_bytes = pattern(1024U, 'n');
  ASSERT_TRUE(HoloInfer::publish_engine_file(engine, old_bytes.data(), old_bytes.size()));

  std::ifstream opened(engine, std::ios::binary);
  ASSERT_TRUE(opened);
  ASSERT_TRUE(HoloInfer::publish_engine_file(engine, new_bytes.data(), new_bytes.size()));
  EXPECT_EQ(read_all(opened), old_bytes);

  std::ifstream reopened(engine, std::ios::binary);
  ASSERT_TRUE(reopened);
  EXPECT_EQ(read_all(reopened), new_bytes);
  EXPECT_EQ(entries(), std::vector<std::string>{"model.engine.fp16"});
}

// A failed rename reports failure and removes the private temporary file it wrote.
TEST_F(EngineCachePublication, FailedPublicationRemovesTemporaryFile) {
  const auto blocked = directory_ / "occupied.engine.fp32";
  std::filesystem::create_directories(blocked / "child");
  const auto bytes = pattern(512U, 'f');
  EXPECT_FALSE(HoloInfer::publish_engine_file(blocked.string(), bytes.data(), bytes.size()));
  EXPECT_EQ(entries(), std::vector<std::string>{"occupied.engine.fp32"});

  const auto missing_directory = (directory_ / "missing" / "model.engine.fp32").string();
  EXPECT_FALSE(HoloInfer::publish_engine_file(missing_directory, bytes.data(), bytes.size()));
  EXPECT_EQ(entries(), std::vector<std::string>{"occupied.engine.fp32"});
}

}  // namespace
