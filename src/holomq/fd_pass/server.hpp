/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "fd_pass/protocol.hpp"
#include "scoped_file_descriptor.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace holomq::fd_pass {

/// AF_UNIX SOCK_STREAM server: receives \ref Request, replies with \ref Response + optional
/// SCM_RIGHTS FDs.
class Server {
 public:
  /// Creates a listening AF_UNIX server for \p socket_path (see CMake \c HOLOMQ_FD_PASS_SOCKET);
  /// then call \ref start.
  /// \p listen_backlog is the second argument to \c listen(2): the maximum number of completed
  /// connections waiting for \c accept(2) before the kernel may refuse or drop new connects (the
  /// effective cap may be lower per OS limits).
  [[nodiscard]] static auto create(const std::string& socket_path, int listen_backlog = 4)
      -> std::optional<Server>;

  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;
  Server(Server&& o) noexcept;
  Server& operator=(Server&& o) noexcept;
  ~Server();

  /// Callable invoked per accepted request after magic/version checks. Receives \c pool_cookie;
  /// returns an FD to pass via \c SCM_RIGHTS on success, or an empty descriptor on failure (the
  /// server maps that to
  /// \ref Status::internal_error on the wire).
  using OnRequestHandler = std::function<holomq::ScopedFileDescriptor(std::uint64_t pool_cookie)>;

  /// Spawns a thread that \c poll(2)s on the wakeup \c eventfd and listen socket, then \c
  /// accept(2)s when the listen fd is readable until \ref stop(). Returns \c false if \ref start
  /// was already called without an intervening \ref stop, or if the worker thread could not be
  /// created. Accepted request reads are wake-aware and bounded so an incomplete peer cannot block
  /// shutdown or later connections indefinitely.
  [[nodiscard]] bool start(OnRequestHandler handler) noexcept;

  /// Sets the stop flag, clears the handler, pings the wakeup \c eventfd, joins the worker, and
  /// closes the listen socket. Interrupts an in-flight request read on an already-accepted
  /// connection. Idempotent: repeated calls after the first have no effect.
  void stop() noexcept;

  /// \c false after a successful \ref start until the worker thread has been joined by \ref stop or
  /// destruction.
  [[nodiscard]] bool stopped() const noexcept;

 private:
  struct Impl;

  explicit Server(std::unique_ptr<Impl>&& impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace holomq::fd_pass
