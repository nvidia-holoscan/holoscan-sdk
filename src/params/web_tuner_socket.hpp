/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <string_view>

#if defined(__linux__)

#include <poll.h>
#include <sys/socket.h>

namespace holoscan::experimental::params::detail {

[[nodiscard]] inline bool send_web_tuner_bytes(const std::atomic<bool>& stop_requested,
                                               int connection, std::string_view bytes) noexcept {
  std::size_t sent{};
  while (sent < bytes.size() && !stop_requested.load(std::memory_order_acquire)) {
    const auto count =
        ::send(connection, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (count > 0) {
      sent += static_cast<std::size_t>(count);
      continue;
    }
    if (count == 0) {
      return false;
    }
    if (errno == EINTR) {
      continue;
    }
    if (errno != EAGAIN && errno != EWOULDBLOCK) {
      return false;
    }
    pollfd descriptor{.fd = connection, .events = POLLOUT, .revents = 0};
    const int ready = ::poll(&descriptor, 1, 100);
    if (ready < 0 && errno == EINTR) {
      continue;
    }
    if (ready < 0 || (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      return false;
    }
  }
  return sent == bytes.size();
}

}  // namespace holoscan::experimental::params::detail

#endif
