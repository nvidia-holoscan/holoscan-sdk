/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel_test_support.hpp"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <thread>

#include <sys/wait.h>
#include <unistd.h>

namespace holoscan::channel_test {

bool read_exact(const int fd, void* const buffer, const std::size_t length) noexcept {
  auto* output = static_cast<std::uint8_t*>(buffer);
  std::size_t completed = 0U;
  while (completed < length) {
    const ssize_t count = ::read(fd, output + completed, length - completed);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      return false;
    }
    completed += static_cast<std::size_t>(count);
  }
  return true;
}

bool write_exact(const int fd, const void* const buffer, const std::size_t length) noexcept {
  const auto* input = static_cast<const std::uint8_t*>(buffer);
  std::size_t completed = 0U;
  while (completed < length) {
    const ssize_t count = ::write(fd, input + completed, length - completed);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      return false;
    }
    completed += static_cast<std::size_t>(count);
  }
  return true;
}

ChildProcessCleanup::ChildProcessCleanup(const pid_t child, const int result_fd,
                                         const int start_fd) noexcept
    : child_(child), result_fd_(result_fd), start_fd_(start_fd) {}

ChildProcessCleanup::~ChildProcessCleanup() noexcept {
  close_start();
  close_result();
  terminate_and_wait();
}

void ChildProcessCleanup::close_fd(int& fd) noexcept {
  if (fd >= 0) {
    static_cast<void>(::close(fd));
    fd = -1;
  }
}

void ChildProcessCleanup::close_result() noexcept {
  close_fd(result_fd_);
}

void ChildProcessCleanup::close_start() noexcept {
  close_fd(start_fd_);
}

pid_t ChildProcessCleanup::wait(int* const status) noexcept {
  if (child_ <= 0) {
    return -1;
  }

  pid_t waited = -1;
  do {
    waited = ::waitpid(child_, status, 0);
  } while (waited < 0 && errno == EINTR);

  if (waited == child_ || (waited < 0 && errno == ECHILD)) {
    child_ = -1;
  }
  return waited;
}

void ChildProcessCleanup::terminate_and_wait() noexcept {
  if (child_ <= 0) {
    return;
  }

  int status = 0;
  pid_t waited = -1;
  do {
    waited = ::waitpid(child_, &status, WNOHANG);
  } while (waited < 0 && errno == EINTR);
  if (waited == child_ || (waited < 0 && errno == ECHILD)) {
    child_ = -1;
    return;
  }

  static_cast<void>(::kill(child_, SIGTERM));
  constexpr std::size_t kTerminationPollCount = 100U;
  for (std::size_t attempt = 0U; attempt < kTerminationPollCount; ++attempt) {
    do {
      waited = ::waitpid(child_, &status, WNOHANG);
    } while (waited < 0 && errno == EINTR);
    if (waited == child_ || (waited < 0 && errno == ECHILD)) {
      child_ = -1;
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  static_cast<void>(::kill(child_, SIGKILL));
  static_cast<void>(wait(&status));
}

}  // namespace holoscan::channel_test
