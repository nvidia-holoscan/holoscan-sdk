/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include "fd_pass/accept_cloexec.hpp"
#include "fd_pass/client.hpp"
#include "fd_pass/protocol.hpp"
#include "fd_pass/server.hpp"
#include "fd_pass/sockaddr.hpp"

#include <gtest/gtest.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <future>
#include <string>
#include <thread>
#include <utility>

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

constexpr std::uint64_t kPoolCookie = 0xDECAFBADULL;
constexpr unsigned char kPipeMarker = 0xA7U;
constexpr int kAcceptTimeoutMs = 1000;
constexpr auto kStopTimeout = std::chrono::milliseconds(500);
constexpr auto kStalledRequestTimeout = std::chrono::seconds(2);

[[nodiscard]] std::string make_socket_path(const char* suffix) {
  std::string path =
      "/tmp/holomq_fd_pass_test_" + std::to_string(static_cast<long>(::getpid())) + suffix;
  if (path.size() > holomq::fd_pass::endpoint_max_length()) {
    return {};
  }
  return path;
}

void cleanup_socket_path(const std::string& socket_path) {
  if (holomq::fd_pass::endpoint_uses_filesystem_path()) {
    static_cast<void>(::unlink(socket_path.c_str()));
  }
}

[[nodiscard]] holomq::ScopedFileDescriptor make_pipe_read_end_with_marker() {
  int fds[2]{};
  if (::pipe2(fds, O_CLOEXEC) != 0) {
    return {};
  }
  holomq::ScopedFileDescriptor read_end(fds[0]);
  holomq::ScopedFileDescriptor write_end(fds[1]);
  const unsigned char marker = kPipeMarker;
  if (::write(write_end.get(), &marker, 1) != 1) {
    return {};
  }
  return read_end;
}

[[nodiscard]] holomq::ScopedFileDescriptor connect_raw_client(const std::string& socket_path) {
  holomq::ScopedFileDescriptor client_fd(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
  if (!client_fd) {
    return {};
  }

  ::sockaddr_un addr{};
  ::socklen_t len = 0;
  if (!holomq::fd_pass::fill_sockaddr_un(socket_path, &addr, &len) ||
      ::connect(client_fd.get(), reinterpret_cast<::sockaddr*>(&addr), len) != 0) {
    return {};
  }
  return client_fd;
}

[[nodiscard]] bool wait_for_incoming_connection(const int listen_fd) noexcept {
  ::pollfd poll_fd{};
  poll_fd.fd = listen_fd;
  poll_fd.events = POLLIN;

  for (;;) {
    const int poll_result = ::poll(&poll_fd, 1, kAcceptTimeoutMs);
    if (poll_result > 0) {
      return (poll_fd.revents & (POLLIN | POLLERR | POLLHUP)) != 0;
    }
    if (poll_result == 0 || errno != EINTR) {
      return false;
    }
  }
}

void serve_ok_without_passed_fd(const std::string& socket_path, std::promise<bool> ready) {
  if (socket_path.empty()) {
    ready.set_value(false);
    return;
  }

  holomq::ScopedFileDescriptor listen_fd(
      ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
  if (!listen_fd) {
    ready.set_value(false);
    return;
  }

  if (holomq::fd_pass::endpoint_uses_filesystem_path()) {
    static_cast<void>(::unlink(socket_path.c_str()));
  }

  ::sockaddr_un addr{};
  ::socklen_t len = 0;
  if (!holomq::fd_pass::fill_sockaddr_un(socket_path, &addr, &len) ||
      ::bind(listen_fd.get(), reinterpret_cast<::sockaddr*>(&addr), len) != 0 ||
      ::listen(listen_fd.get(), 1) != 0) {
    ready.set_value(false);
    return;
  }

  ready.set_value(true);

  if (!wait_for_incoming_connection(listen_fd.get())) {
    return;
  }
  holomq::ScopedFileDescriptor client_fd = holomq::fd_pass::accept_cloexec(listen_fd.get());
  if (!client_fd) {
    return;
  }

  holomq::fd_pass::Request req{};
  if (::recv(client_fd.get(), &req, sizeof(req), MSG_WAITALL) !=
      static_cast<ssize_t>(sizeof(req))) {
    return;
  }

  holomq::fd_pass::Response response{};
  response.status = static_cast<std::uint32_t>(holomq::fd_pass::Status::ok);
  static_cast<void>(::send(client_fd.get(), &response, sizeof(response), MSG_NOSIGNAL));
}

}  // namespace

TEST(FdPassTest, PassesPipeReadEndOverUnixSocket) {
  using holomq::fd_pass::Client;
  using holomq::fd_pass::Request;
  using holomq::fd_pass::Server;

  const std::string socket_path = make_socket_path("_happy.sock");
  ASSERT_FALSE(socket_path.empty()) << "failed to build socket path";

  auto server_or = Server::create(socket_path);
  ASSERT_TRUE(server_or.has_value()) << "Server::create failed";
  Server server = std::move(server_or).value();

  const bool started = server.start([](std::uint64_t pool_cookie) {
    if (pool_cookie != kPoolCookie) {
      return holomq::ScopedFileDescriptor{};
    }
    return make_pipe_read_end_with_marker();
  });
  ASSERT_TRUE(started) << "Server::start failed";

  auto client_or = Client::create(socket_path);
  ASSERT_TRUE(client_or.has_value()) << "Client::create failed";
  Client client = std::move(client_or).value();

  const Request req{kPoolCookie};
  auto fd_or = client.send_request(req);
  server.stop();

  ASSERT_TRUE(fd_or.has_value()) << "Client::send_request failed";

  holomq::ScopedFileDescriptor passed = std::move(fd_or).value();
  ASSERT_TRUE(static_cast<bool>(passed)) << "empty passed fd";

  EXPECT_GE(::fcntl(passed.get(), F_GETFD), 0) << "passed fd is not valid";

  unsigned char byte{};
  const ssize_t nr = ::read(passed.get(), &byte, 1);
  ASSERT_EQ(nr, 1) << "expected 1 byte from pipe";
  EXPECT_EQ(byte, kPipeMarker) << "marker mismatch";

  cleanup_socket_path(socket_path);
}

TEST(FdPassTest, RejectsWrongPoolCookie) {
  using holomq::fd_pass::Client;
  using holomq::fd_pass::Request;
  using holomq::fd_pass::Server;

  const std::string socket_path = make_socket_path("_wrong_cookie.sock");
  ASSERT_FALSE(socket_path.empty());

  auto server_or = Server::create(socket_path);
  ASSERT_TRUE(server_or.has_value());
  Server server = std::move(server_or).value();
  ASSERT_TRUE(server.start([](std::uint64_t pool_cookie) {
    if (pool_cookie != kPoolCookie) {
      return holomq::ScopedFileDescriptor{};
    }
    return make_pipe_read_end_with_marker();
  }));

  auto client_or = Client::create(socket_path);
  ASSERT_TRUE(client_or.has_value());
  Client client = std::move(client_or).value();

  const Request req{kPoolCookie + 1U};
  EXPECT_FALSE(client.send_request(req).has_value());

  server.stop();
  cleanup_socket_path(socket_path);
}

TEST(FdPassTest, RejectsBadMagic) {
  using holomq::fd_pass::Client;
  using holomq::fd_pass::Request;
  using holomq::fd_pass::Server;

  const std::string socket_path = make_socket_path("_bad_magic.sock");
  ASSERT_FALSE(socket_path.empty());

  auto server_or = Server::create(socket_path);
  ASSERT_TRUE(server_or.has_value());
  Server server = std::move(server_or).value();
  ASSERT_TRUE(server.start([](std::uint64_t) { return make_pipe_read_end_with_marker(); }));

  auto client_or = Client::create(socket_path);
  ASSERT_TRUE(client_or.has_value());
  Client client = std::move(client_or).value();

  Request req{kPoolCookie};
  req.magic = 0U;
  EXPECT_FALSE(client.send_request(req).has_value());

  server.stop();
  cleanup_socket_path(socket_path);
}

TEST(FdPassTest, RejectsBadVersion) {
  using holomq::fd_pass::Client;
  using holomq::fd_pass::Request;
  using holomq::fd_pass::Server;

  const std::string socket_path = make_socket_path("_bad_version.sock");
  ASSERT_FALSE(socket_path.empty());

  auto server_or = Server::create(socket_path);
  ASSERT_TRUE(server_or.has_value());
  Server server = std::move(server_or).value();
  ASSERT_TRUE(server.start([](std::uint64_t) { return make_pipe_read_end_with_marker(); }));

  auto client_or = Client::create(socket_path);
  ASSERT_TRUE(client_or.has_value());
  Client client = std::move(client_or).value();

  Request req{kPoolCookie};
  req.version = 0U;
  EXPECT_FALSE(client.send_request(req).has_value());

  server.stop();
  cleanup_socket_path(socket_path);
}

// A client that sends only part of a request must not prevent the single worker from serving a
// complete request queued behind it.
TEST(FdPassTest, IncompleteRequestDoesNotBlockLaterClient) {
  using holomq::fd_pass::Client;
  using holomq::fd_pass::Request;
  using holomq::fd_pass::Server;

  const std::string socket_path = make_socket_path("_incomplete_request.sock");
  ASSERT_FALSE(socket_path.empty());

  auto server_or = Server::create(socket_path);
  ASSERT_TRUE(server_or.has_value());
  Server server = std::move(server_or).value();
  ASSERT_TRUE(server.start([](std::uint64_t pool_cookie) {
    return pool_cookie == kPoolCookie ? make_pipe_read_end_with_marker()
                                      : holomq::ScopedFileDescriptor{};
  }));

  holomq::ScopedFileDescriptor stalled_client = connect_raw_client(socket_path);
  ASSERT_TRUE(stalled_client);
  const Request partial_request{kPoolCookie};
  ASSERT_EQ(::send(stalled_client.get(), &partial_request, 1U, MSG_NOSIGNAL), 1);

  auto later_request = std::async(std::launch::async, [socket_path]() {
    auto client_or = Client::create(socket_path);
    if (!client_or.has_value()) {
      return false;
    }
    return std::move(client_or).value().send_request(Request{kPoolCookie}).has_value();
  });

  const auto initial_request_status = later_request.wait_for(kStalledRequestTimeout);
  auto request_status = initial_request_status;
  if (request_status != std::future_status::ready) {
    stalled_client.reset();
    request_status = later_request.wait_for(kStalledRequestTimeout);
  }
  EXPECT_EQ(initial_request_status, std::future_status::ready);
  ASSERT_EQ(request_status, std::future_status::ready);
  EXPECT_TRUE(later_request.get());

  server.stop();
  cleanup_socket_path(socket_path);
}

// Shutdown must wake a request read rather than waiting for a stalled peer or its request timeout.
TEST(FdPassTest, StopInterruptsIncompleteRequest) {
  using holomq::fd_pass::Request;
  using holomq::fd_pass::Server;

  const std::string socket_path = make_socket_path("_stop_incomplete.sock");
  ASSERT_FALSE(socket_path.empty());

  auto server_or = Server::create(socket_path);
  ASSERT_TRUE(server_or.has_value());
  Server server = std::move(server_or).value();
  ASSERT_TRUE(server.start([](std::uint64_t) { return make_pipe_read_end_with_marker(); }));

  holomq::ScopedFileDescriptor stalled_client = connect_raw_client(socket_path);
  ASSERT_TRUE(stalled_client);
  const Request partial_request{kPoolCookie};
  ASSERT_EQ(::send(stalled_client.get(), &partial_request, 1U, MSG_NOSIGNAL), 1);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  auto stopped = std::async(std::launch::async, [&server]() { server.stop(); });
  const auto initial_stop_status = stopped.wait_for(kStopTimeout);
  auto stop_status = initial_stop_status;
  if (stop_status != std::future_status::ready) {
    stalled_client.reset();
    stop_status = stopped.wait_for(kStopTimeout);
  }
  EXPECT_EQ(initial_stop_status, std::future_status::ready);
  ASSERT_EQ(stop_status, std::future_status::ready);
  stopped.get();

  cleanup_socket_path(socket_path);
}

TEST(FdPassTest, RejectsOkResponseWithoutPassedFd) {
  using holomq::fd_pass::Client;
  using holomq::fd_pass::Request;

  const std::string socket_path = make_socket_path("_ok_no_fd.sock");
  ASSERT_FALSE(socket_path.empty());

  std::promise<bool> ready;
  auto ready_future = ready.get_future();
  std::jthread server_thread(serve_ok_without_passed_fd, socket_path, std::move(ready));
  ASSERT_TRUE(ready_future.get());

  auto client_or = Client::create(socket_path);
  ASSERT_TRUE(client_or.has_value());
  Client client = std::move(client_or).value();

  const Request req{kPoolCookie};
  EXPECT_FALSE(client.send_request(req).has_value());

  server_thread.join();
  cleanup_socket_path(socket_path);
}

TEST(FdPassTest, AcceptCloexecSetsCloseOnExec) {
  const std::string socket_path = make_socket_path("_accept_cloexec.sock");
  ASSERT_FALSE(socket_path.empty());

  holomq::ScopedFileDescriptor listen_fd(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
  ASSERT_TRUE(listen_fd);

  if (holomq::fd_pass::endpoint_uses_filesystem_path()) {
    static_cast<void>(::unlink(socket_path.c_str()));
  }

  ::sockaddr_un addr{};
  ::socklen_t len = 0;
  ASSERT_TRUE(holomq::fd_pass::fill_sockaddr_un(socket_path, &addr, &len));
  ASSERT_EQ(::bind(listen_fd.get(), reinterpret_cast<::sockaddr*>(&addr), len), 0);
  ASSERT_EQ(::listen(listen_fd.get(), 1), 0);

  holomq::ScopedFileDescriptor client_fd(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
  ASSERT_TRUE(client_fd);
  ASSERT_EQ(::connect(client_fd.get(), reinterpret_cast<::sockaddr*>(&addr), len), 0);

  holomq::ScopedFileDescriptor accepted_fd = holomq::fd_pass::accept_cloexec(listen_fd.get());
  ASSERT_TRUE(accepted_fd);

  const int flags = ::fcntl(accepted_fd.get(), F_GETFD);
  ASSERT_GE(flags, 0);
  EXPECT_NE(static_cast<unsigned>(flags) & FD_CLOEXEC, 0U);

  cleanup_socket_path(socket_path);
}
