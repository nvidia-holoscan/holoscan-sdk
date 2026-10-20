/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fd_pass/server.hpp"

#include <holomq/logger.hpp>
#include "fd_pass/accept_cloexec.hpp"
#include "fd_pass/poll_set.hpp"
#include "fd_pass/sockaddr.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include <cstdint>

#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

namespace holomq::fd_pass {

namespace {

/// Maximum time one accepted peer may occupy the single worker without completing a request.
constexpr auto kRequestReadTimeout = std::chrono::seconds(1);

void send_response(int client_connection, const Response& response, ScopedFileDescriptor pass_fd) {
  Response resp_copy = response;
  ::iovec iov{};
  iov.iov_base = &resp_copy;
  iov.iov_len = sizeof(Response);

  std::array<char, CMSG_SPACE(sizeof(int))> cmsg_buf{};
  ::msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1U;

  if (pass_fd) {
    msg.msg_control = cmsg_buf.data();
    msg.msg_controllen = cmsg_buf.size();
    ::cmsghdr* const cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    const int fd_to_pass = pass_fd.get();
    std::memcpy(CMSG_DATA(cmsg), &fd_to_pass, sizeof(int));
  } else {
    msg.msg_control = nullptr;
    msg.msg_controllen = 0U;
  }

  if (::sendmsg(client_connection, &msg, MSG_NOSIGNAL) < 0) {
    HOLOMQ_LOG_ERROR("fd_pass server send response failed");
  }
}

}  // namespace

/// Wakeup primitive for the FD-pass worker thread: a Linux \c eventfd(2) registered with \c
/// poll(2). \ref notify increments the counter so the worker returns from \c poll; \ref consume_one
/// removes one increment (\c EFD_SEMAPHORE so multiple notifies are not merged into a single
/// level-triggered edge). Created with \c EFD_CLOEXEC and
/// \c EFD_NONBLOCK. If \c eventfd(2) fails, \ref operator bool is \c false and other members no-op.
class Event {
 public:
  /// Opens the \c eventfd; on failure the descriptor is invalid (\ref operator bool is \c false).
  Event() : event_fd_(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK | EFD_SEMAPHORE)) {}
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&&) noexcept = default;
  Event& operator=(Event&&) noexcept = default;
  ~Event() noexcept = default;

  /// \c true when the \c eventfd was created successfully.
  explicit operator bool() const noexcept { return static_cast<bool>(event_fd_); }

  /// Wakes waiters blocked on \c poll by posting one count to the \c eventfd (no-op if invalid).
  void notify() noexcept {
    if (!event_fd_) {
      return;
    }
    const std::uint64_t one = 1;
    const ssize_t w = ::write(event_fd_.get(), &one, sizeof(one));
    static_cast<void>(w);
  }

  /// Consumes one post from a prior \ref notify (retries only on \c EINTR). With \c EFD_SEMAPHORE,
  /// extra posts remain until a later call, so multiple notifies are not collapsed into one \c poll
  /// wakeup.
  void consume_one() noexcept {
    if (!event_fd_) {
      return;
    }
    for (;;) {
      std::uint64_t scratch{};
      const ssize_t n = ::read(event_fd_.get(), &scratch, sizeof(scratch));
      if (n < 0) {
        if (errno == EINTR) {
          continue;
        }
        return;
      }
      return;
    }
  }

  /// Raw fd for registering with \c poll(2) (e.g. alongside the listen socket).
  int get() const noexcept { return event_fd_.get(); }

 private:
  ScopedFileDescriptor event_fd_;
};

class Server::Impl {
 public:
  [[nodiscard]] static auto create(const std::string& socket_path, int listen_backlog)
      -> std::optional<std::unique_ptr<Impl>>;
  ~Impl() noexcept;

  [[nodiscard]] bool start(OnRequestHandler handler) noexcept;
  void stop() noexcept;

  [[nodiscard]] bool stopped() const noexcept;

 private:
  explicit Impl(ScopedFileDescriptor listen_socket, Event wakeup_event);

  void worker_loop();
  void handle_connection(int client_connection);
  [[nodiscard]] bool receive_request(int client_connection, Request& request) noexcept;

  std::mutex mutex_;
  OnRequestHandler handler_;
  ScopedFileDescriptor listen_socket_;
  Event wakeup_event_;
  std::atomic<bool> stop_requested_{false};
  std::thread worker_;
};

Server::Impl::Impl(ScopedFileDescriptor listen_socket, Event wakeup_event)
    : listen_socket_(std::move(listen_socket)), wakeup_event_(std::move(wakeup_event)) {}

Server::Impl::~Impl() noexcept {
  stop();
}

auto Server::Impl::create(const std::string& socket_path, int listen_backlog)
    -> std::optional<std::unique_ptr<Impl>> {
  if (socket_path.empty()) {
    HOLOMQ_LOG_ERROR("fd_pass server create failed: empty path");
    return std::nullopt;
  }
  if (socket_path.size() > endpoint_max_length()) {
    HOLOMQ_LOG_ERROR("fd_pass server create failed: path too long");
    return std::nullopt;
  }

  ScopedFileDescriptor listen_socket(
      ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
  if (!listen_socket) {
    HOLOMQ_LOG_ERROR("fd_pass server create failed: socket failed");
    return std::nullopt;
  }

  if (endpoint_uses_filesystem_path()) {
    ::unlink(socket_path.c_str());
  }

  ::sockaddr_un addr{};
  ::socklen_t len = 0;
  if (!fill_sockaddr_un(socket_path, &addr, &len)) {
    HOLOMQ_LOG_ERROR("fd_pass server create failed: invalid address");
    return std::nullopt;
  }
  if (::bind(listen_socket.get(), reinterpret_cast<::sockaddr*>(&addr), len) != 0) {
    HOLOMQ_LOG_ERROR("fd_pass server create failed: bind failed");
    return std::nullopt;
  }
  if (::listen(listen_socket.get(), listen_backlog) != 0) {
    HOLOMQ_LOG_ERROR("fd_pass server create failed: listen failed");
    return std::nullopt;
  }

  Event wakeup_event;
  if (!wakeup_event) {
    HOLOMQ_LOG_ERROR("fd_pass server create failed: eventfd failed");
    return std::nullopt;
  }

  return std::unique_ptr<Impl>(new Impl(std::move(listen_socket), std::move(wakeup_event)));
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool Server::Impl::receive_request(const int client_connection, Request& request) noexcept {
  constexpr std::size_t k_wakeup_poll_index = 0U;
  constexpr std::size_t k_client_poll_index = 1U;
  PollSet poll_set({wakeup_event_.get(), client_connection});

  auto* const request_bytes = reinterpret_cast<std::byte*>(&request);
  std::size_t bytes_received = 0U;
  const auto deadline = std::chrono::steady_clock::now() + kRequestReadTimeout;

  while (bytes_received < sizeof(request)) {
    if (stop_requested_.load(std::memory_order_relaxed)) {
      return false;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      return false;
    }
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
    const int timeout_ms = static_cast<int>(remaining > 0 ? remaining : 1);

    const auto wait_status = poll_set.wait(timeout_ms);
    if (wait_status == PollSet::PollStatus::interrupted) {
      continue;
    }
    if (wait_status == PollSet::PollStatus::error) {
      return false;
    }

    if (poll_set.has_event(k_wakeup_poll_index, POLLIN) ||
        poll_set.has_event(k_wakeup_poll_index, POLLERR)) {
      wakeup_event_.consume_one();
      if (stop_requested_.load(std::memory_order_relaxed)) {
        return false;
      }
    }

    if (poll_set.has_event(k_client_poll_index, POLLIN)) {
      const ssize_t received = ::recv(client_connection,
                                      request_bytes + bytes_received,
                                      sizeof(request) - bytes_received,
                                      MSG_DONTWAIT);
      if (received > 0) {
        bytes_received += static_cast<std::size_t>(received);
        continue;
      }
      if (received == 0) {
        return false;
      }
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      return false;
    }

    if (poll_set.has_event(k_client_poll_index, POLLERR) ||
        poll_set.has_event(k_client_poll_index, POLLHUP) ||
        poll_set.has_event(k_client_poll_index, POLLNVAL)) {
      return false;
    }
  }

  return true;
}

void Server::Impl::handle_connection(const int client_connection) {
  // One accept handles a single request/response on client_connection; the accepted fd is owned and
  // closed in worker_loop.
  Request req{};
  if (!receive_request(client_connection, req)) {
    return;
  }

  Response response{};
  holomq::ScopedFileDescriptor pass_fd_to_send{};
  if (req.magic != k_magic) {
    response.status = static_cast<std::uint32_t>(Status::bad_magic);
  } else if (req.version != k_version) {
    response.status = static_cast<std::uint32_t>(Status::bad_version);
  } else {
    OnRequestHandler handler;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      handler = handler_;
    }
    if (handler) {
      pass_fd_to_send = handler(req.pool_cookie);
      if (pass_fd_to_send) {
        response.status = static_cast<std::uint32_t>(Status::ok);
      } else {
        response.status = static_cast<std::uint32_t>(Status::internal_error);
      }
    } else {
      response.status = static_cast<std::uint32_t>(Status::internal_error);
    }
  }

  send_response(client_connection, response, std::move(pass_fd_to_send));
}

bool Server::Impl::start(OnRequestHandler handler) noexcept {
  if (!stopped()) {
    HOLOMQ_LOG_ERROR("fd_pass server start failed: thread already running");
    return false;
  }
  handler_ = std::move(handler);
  stop_requested_.store(false, std::memory_order_relaxed);
  try {
    worker_ = std::thread([this] { worker_loop(); });
  } catch (...) {
    HOLOMQ_LOG_ERROR("fd_pass server start failed: thread creation failed");
    return false;
  }
  return true;
}

void Server::Impl::stop() noexcept {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    stop_requested_.store(true, std::memory_order_relaxed);
    handler_ = {};
    wakeup_event_.notify();
  }
  if (worker_.joinable()) {
    worker_.join();
  }
  listen_socket_.reset();
}

bool Server::Impl::stopped() const noexcept {
  return !worker_.joinable();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void Server::Impl::worker_loop() {
  constexpr std::size_t k_wakeup_poll_index = 0;
  constexpr std::size_t k_listen_poll_index = 1;
  PollSet poll_set({wakeup_event_.get(), listen_socket_.get()});

  while (true) {
    // Block until \ref stop() posts to the wakeup \c eventfd or a client connects; \c -1 avoids
    // idle polling.
    auto wait_status = poll_set.wait(-1);
    if (wait_status == PollSet::PollStatus::interrupted) {
      continue;
    }
    if (wait_status == PollSet::PollStatus::error) {
      break;
    }

    // Check if the stop request has been set.
    if (stop_requested_.load(std::memory_order_relaxed)) {
      break;
    }

    if (poll_set.has_event(k_wakeup_poll_index, POLLIN) ||
        poll_set.has_event(k_wakeup_poll_index, POLLERR)) {
      wakeup_event_.consume_one();
    }

    if (poll_set.has_event(k_listen_poll_index, POLLNVAL)) {
      return;
    }
    if (poll_set.has_event(k_listen_poll_index, POLLIN) && listen_socket_) {
      const ScopedFileDescriptor client_connection = accept_cloexec(listen_socket_.get());
      // Failed \c accept: non-blocking listen returns \c EAGAIN / \c EWOULDBLOCK when the queue is
      // empty;
      // \c EINTR means interrupted by a signal — both are benign (retry on a later \c poll). Other
      // errors: treat \c EBADF / \c EINVAL as a dead listen fd and leave the worker thread.
      if (client_connection) {
        handle_connection(client_connection.get());
      } else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
        if (errno == EBADF || errno == EINVAL) {
          return;
        }
      }
    }

    if (stop_requested_.load(std::memory_order_relaxed)) {
      break;
    }
  }
}

Server::Server(std::unique_ptr<Impl>&& impl) noexcept : impl_(std::move(impl)) {}

auto Server::create(const std::string& socket_path, int listen_backlog) -> std::optional<Server> {
  auto impl_or = Impl::create(socket_path, listen_backlog);
  if (!impl_or.has_value()) {
    return std::nullopt;
  }
  return Server(std::move(impl_or).value());
}

Server::~Server() = default;

bool Server::start(OnRequestHandler handler) noexcept {
  return impl_->start(std::move(handler));
}

void Server::stop() noexcept {
  impl_->stop();
}

bool Server::stopped() const noexcept {
  return impl_->stopped();
}

Server::Server(Server&&) noexcept = default;
Server& Server::operator=(Server&&) noexcept = default;

}  // namespace holomq::fd_pass
