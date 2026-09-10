/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scoped_file_descriptor.hpp"

#include <fcntl.h>
#include <sys/socket.h>

namespace holomq::fd_pass {

/// \c accept(2) on \p listen_fd and set \c FD_CLOEXEC on the connected socket.
///
/// On a non-blocking \p listen_fd, an empty return sets \c errno to \c EAGAIN / \c EWOULDBLOCK when
/// the accept queue is empty.
[[nodiscard]] inline ScopedFileDescriptor accept_cloexec(const int listen_fd) noexcept {
  const int fd = ::accept(listen_fd, nullptr, nullptr);
  if (fd < 0) {
    return {};
  }
  ScopedFileDescriptor client(fd);
  if (::fcntl(client.get(), F_SETFD, FD_CLOEXEC) != 0) {
    return {};
  }
  return client;
}

}  // namespace holomq::fd_pass
