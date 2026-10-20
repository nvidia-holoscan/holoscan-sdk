/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "fd_pass/protocol.hpp"
#include "scoped_file_descriptor.hpp"

#include <optional>
#include <string>

namespace holomq::fd_pass {

/// AF_UNIX stream client for the fd_pass protocol: obtain via \ref create, then \ref send_request.
class Client {
 public:
  [[nodiscard]] static auto create(const std::string& socket_path) -> std::optional<Client>;

  ~Client() = default;

  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;
  Client(Client&& o) noexcept;
  Client& operator=(Client&& o) noexcept;

  /// Sends \p req, receives \ref Response and at most one \c SCM_RIGHTS FD. The Unix socket stays
  /// open until
  /// \c Client is destroyed. On success returns a guard owning the passed pool fd.
  [[nodiscard]] auto send_request(const Request& req) const
      -> std::optional<holomq::ScopedFileDescriptor>;

 private:
  explicit Client(holomq::ScopedFileDescriptor&& sock) noexcept;

  holomq::ScopedFileDescriptor socket_;
};

}  // namespace holomq::fd_pass
