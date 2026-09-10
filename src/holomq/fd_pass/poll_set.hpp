/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <poll.h>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace holomq::fd_pass {

/// Thin wrapper around \c poll(2) for a fixed set of file descriptors, all registered with \c
/// POLLIN.
class PollSet {
 public:
  /// Outcome of \ref wait (maps \c poll(2) return value and \c errno).
  enum class PollStatus : std::uint8_t {
    /// \c poll returned \c >= 0 (one or more fds ready, or timed out with none).
    completed,
    /// \c poll returned \c -1 with \c errno == \c EINTR; the caller typically retries.
    interrupted,
    /// \c poll returned \c -1 with another \c errno.
    error,
  };

  /// Registers \p watch_fds for \c POLLIN. Each entry must be an open fd (\c fd >= 0); invalid fds
  /// get
  /// \c POLLNVAL from \c poll(2).
  explicit PollSet(std::initializer_list<int> watch_fds) : fds_() {
    fds_.reserve(watch_fds.size());
    for (const int fd : watch_fds) {
      fds_.push_back(::pollfd{fd, POLLIN, 0});
    }
  }

  /// Single \c poll(2) call with timeout in milliseconds (\c -1 blocks indefinitely).
  [[nodiscard]] PollStatus wait(int timeout_ms = 1) noexcept {
    const int r = ::poll(fds_.data(), static_cast<::nfds_t>(fds_.size()), timeout_ms);
    if (r < 0) {
      if (errno == EINTR) {
        return PollStatus::interrupted;
      }
      return PollStatus::error;
    }
    return PollStatus::completed;
  }

  [[nodiscard]] bool has_event(std::size_t index, short event) const noexcept {
    if (index >= fds_.size()) {
      return false;
    }
    return (fds_[index].revents & event) != 0;
  }

  [[nodiscard]] int fd(std::size_t index) const noexcept {
    if (index >= fds_.size()) {
      return -1;
    }
    return fds_[index].fd;
  }

  [[nodiscard]] std::size_t size() const noexcept { return fds_.size(); }

 private:
  std::vector<::pollfd> fds_;
};

}  // namespace holomq::fd_pass
