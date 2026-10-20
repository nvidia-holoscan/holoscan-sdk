/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cerrno>
#include <utility>

#include <unistd.h>

namespace holomq {

/// Owns a POSIX file descriptor until \ref release or destruction.
///
/// An empty guard holds \c fd == -1; \c explicit operator bool() is \c false. Closes with \c
/// ::close when
/// \c fd >= 0. On \c EINTR, \ref reset retries \c ::close (common Linux practice): if the
/// descriptor was already closed despite \c EINTR, the retry fails with \c EBADF and the loop
/// stops.
class ScopedFileDescriptor {
 public:
  ScopedFileDescriptor() noexcept = default;

  explicit ScopedFileDescriptor(int fd) noexcept : fd_(fd) {}

  ScopedFileDescriptor(const ScopedFileDescriptor&) = delete;
  ScopedFileDescriptor& operator=(const ScopedFileDescriptor&) = delete;

  ScopedFileDescriptor(ScopedFileDescriptor&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }

  ScopedFileDescriptor& operator=(ScopedFileDescriptor&& o) noexcept {
    if (this != &o) {
      reset();
      fd_ = o.fd_;
      o.fd_ = -1;
    }
    return *this;
  }

  ~ScopedFileDescriptor() { reset(); }

  [[nodiscard]] explicit operator bool() const noexcept { return fd_ >= 0; }

  [[nodiscard]] int get() const noexcept { return fd_; }

  /// Transfers ownership to the caller; this guard no longer closes the fd.
  [[nodiscard]] int release() noexcept {
    const int out = fd_;
    fd_ = -1;
    return out;
  }

  /// Exchanges owned fds with \p other.
  void swap(ScopedFileDescriptor& other) noexcept {
    const int tmp = fd_;
    fd_ = other.fd_;
    other.fd_ = tmp;
  }

  /// Closes the descriptor if owned; leaves this guard empty (\c fd == -1).
  ///
  /// Clears \c fd_ before calling \c ::close so this guard cannot double-close. Retries when \c
  /// errno is
  /// \c EINTR; any other error (including \c EBADF if the fd was already closed) ends the loop.
  void reset() noexcept {
    const int fd = fd_;
    fd_ = -1;
    if (fd < 0) {
      return;
    }
    // Linux: close may return EINTR while the fd is still open or already closed; retry until
    // !EINTR.
    while (::close(fd) != 0) {
      if (errno != EINTR) {
        break;
      }
    }
  }

 private:
  int fd_{-1};
};

}  // namespace holomq
