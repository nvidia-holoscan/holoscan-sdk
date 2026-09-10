/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fd_pass/client.hpp"

#include <holomq/logger.hpp>
#include "fd_pass/sockaddr.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <string>
#include <utility>

#include <sys/socket.h>

namespace holomq::fd_pass {

Client::Client(holomq::ScopedFileDescriptor&& sock) noexcept : socket_(std::move(sock)) {}

Client::Client(Client&& o) noexcept : socket_(std::move(o.socket_)) {}

Client& Client::operator=(Client&& o) noexcept {
  if (this != &o) {
    socket_ = std::move(o.socket_);
  }
  return *this;
}

auto Client::create(const std::string& socket_path) -> std::optional<Client> {
  if (socket_path.empty()) {
    HOLOMQ_LOG_ERROR("fd_pass client create failed: empty path");
    return std::nullopt;
  }
  if (socket_path.size() > endpoint_max_length()) {
    HOLOMQ_LOG_ERROR("fd_pass client create failed: path too long");
    return std::nullopt;
  }
  holomq::ScopedFileDescriptor scoped_fd(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
  if (!scoped_fd) {
    HOLOMQ_LOG_ERROR(
        "fd_pass client create failed: socket failed (errno={}, {})", errno, std::strerror(errno));
    return std::nullopt;
  }

  ::sockaddr_un addr{};
  ::socklen_t len = 0;
  if (!fill_sockaddr_un(socket_path, &addr, &len)) {
    HOLOMQ_LOG_ERROR("fd_pass client create failed: invalid address");
    return std::nullopt;
  }
  if (::connect(scoped_fd.get(), reinterpret_cast<::sockaddr*>(&addr), len) != 0) {
    HOLOMQ_LOG_ERROR("fd_pass client create failed: connect failed");
    return std::nullopt;
  }
  return Client(std::move(scoped_fd));
}

auto Client::send_request(const Request& req) const -> std::optional<holomq::ScopedFileDescriptor> {
  if (::send(socket_.get(), &req, sizeof(req), MSG_NOSIGNAL) != static_cast<ssize_t>(sizeof(req))) {
    HOLOMQ_LOG_ERROR("fd_pass client send failed");
    return std::nullopt;
  }

  Response resp{};
  std::array<char, CMSG_SPACE(sizeof(int))> cmsg_buf{};
  ::iovec iov{};
  iov.iov_base = &resp;
  iov.iov_len = sizeof(resp);
  ::msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1U;
  msg.msg_control = cmsg_buf.data();
  msg.msg_controllen = cmsg_buf.size();

  const ssize_t nr = ::recvmsg(socket_.get(), &msg, MSG_WAITALL);
  if (nr != static_cast<ssize_t>(sizeof(Response))) {
    HOLOMQ_LOG_ERROR("fd_pass client receive failed");
    return std::nullopt;
  }
  if (resp.status != static_cast<std::uint32_t>(Status::ok)) {
    HOLOMQ_LOG_ERROR("fd_pass client received error response (status={})", resp.status);
    return std::nullopt;
  }

  int out_fd = -1;
  for (::cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
    if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
      // Pool fd pass: exactly one int per SCM_RIGHTS control message.
      const std::size_t data_len = cmsg->cmsg_len - CMSG_LEN(0);
      int* const fds = reinterpret_cast<int*>(CMSG_DATA(cmsg));
      // recvmsg installs one fd per sizeof(int) slot in this SCM_RIGHTS payload. Wrong size means
      // the protocol was violated: close every fd the kernel attached here so none leak; the
      // caller does not receive any of them.
      if (data_len != sizeof(int)) {
        const int n = static_cast<int>(data_len / sizeof(int));
        for (int i = 0; i < n; ++i) {
          const holomq::ScopedFileDescriptor discard(fds[i]);
        }
        HOLOMQ_LOG_ERROR("fd_pass client received invalid SCM_RIGHTS payload");
        return std::nullopt;
      }
      out_fd = fds[0];
      break;
    }
  }
  if (out_fd < 0) {
    HOLOMQ_LOG_ERROR("fd_pass client received no file descriptor");
    return std::nullopt;
  }
  return holomq::ScopedFileDescriptor(out_fd);
}

}  // namespace holomq::fd_pass
