/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <string>
#include <thread>

#if defined(__linux__)
#include <dirent.h>
#endif

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::benchmark_support {

struct ThreadCensus {
  std::size_t count{};
  int error{};
};

#if defined(__linux__)
[[nodiscard]] inline ThreadCensus read_process_thread_census() noexcept {
  DIR* const directory = opendir("/proc/self/task");
  if (directory == nullptr) {
    return {.error = errno};
  }
  std::size_t count{};
  int read_error{};
  for (;;) {
    errno = 0;
    const dirent* const entry = readdir(directory);
    if (entry == nullptr) {
      read_error = errno;
      break;
    }
    if (entry->d_name[0] == '.' &&
        (entry->d_name[1] == '\0' || (entry->d_name[1] == '.' && entry->d_name[2] == '\0'))) {
      continue;
    }
    ++count;
  }
  const int close_error = closedir(directory) == 0 ? 0 : errno;
  return {.count = count, .error = read_error != 0 ? read_error : close_error};
}
#endif

// A preceding benchmark's workers are joined, but Linux can still expose their task entries.
// Wait outside the measurement before starting a fresh managed domain. This does not replace the
// Runtime's admission checks, retry failed admission, or permit a live worker to outlast the wait.
template <typename CensusReader>
[[nodiscard]] expected<void, Error> wait_for_single_thread(
    // Invoke the same reader repeatedly as an lvalue while waiting for retirement.
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    CensusReader&& read_census, const std::chrono::nanoseconds timeout = std::chrono::seconds{1}) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    const ThreadCensus census = read_census();
    if (census.error != 0) {
      return make_unexpected(
          Error{ErrorCode::kFailure,
                "benchmark task census failed: errno=" + std::to_string(census.error)});
    }
    if (census.count == 1U) {
      return {};
    }
    if (census.count == 0U) {
      return make_unexpected(Error{ErrorCode::kFailure, "benchmark task census was empty"});
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return make_unexpected(
          Error{ErrorCode::kTimeout,
                "benchmark tasks did not retire before managed admission: observed_threads=" +
                    std::to_string(census.count)});
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
}

[[nodiscard]] inline expected<void, Error> wait_for_single_thread() {
#if defined(__linux__)
  return wait_for_single_thread(read_process_thread_census);
#else
  return make_unexpected(
      Error{ErrorCode::kNotSupported, "managed signal benchmarks require a Linux task census"});
#endif
}

}  // namespace holoscan::benchmark_support
