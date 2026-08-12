/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/context_builder.hpp>

#include <holomq/detail/transport/backend.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/logger.hpp>
#include "context_impl.hpp"
#include "fd_pass/server.hpp"
#include "fd_pass/sockaddr.hpp"

#include <chrono>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include <unistd.h>

namespace {

std::string default_fd_pass_socket_path() {
  constexpr std::size_t k_max_path = holomq::fd_pass::endpoint_max_length();
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();

  std::ostringstream path;
  path << "/tmp/holomq_" << ::getpid() << '_' << us << ".sock";
  std::string result = path.str();
  if (result.size() > k_max_path) {
    std::ostringstream fallback;
    fallback << "/tmp/holomq." << ::getpid() << ".sock";
    result = fallback.str();
    if (result.size() > k_max_path) {
      result.resize(k_max_path);
    }
  }
  return result;
}

}  // namespace

namespace holomq {

ContextBuilder& ContextBuilder::with_vmm_fd_export(bool enable) {
  vmm_fd_export_enabled_ = enable;
  return *this;
}

ContextBuilder& ContextBuilder::with_fd_pass_socket(std::string path) {
  fd_pass_socket_path_ = std::move(path);
  return *this;
}

ContextBuilder& ContextBuilder::with_transport_logging(bool enable) {
  transport_logging_enabled_ = enable;
  return *this;
}

auto ContextBuilder::create() const -> expected<Context, Error> {
  if (!vmm_fd_export_enabled_ && !fd_pass_socket_path_.empty()) {
    HOLOMQ_LOG_ERROR(
        "ContextBuilder create failed: fd_pass_socket requires with_vmm_fd_export(true)");
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  if (transport_logging_enabled_) {
    enable_transport_logging();
  }

  auto transport_or = detail::transport::create_runtime();
  if (!transport_or.has_value()) {
    HOLOMQ_LOG_ERROR("ContextBuilder create failed: {}", error_message(transport_or.error()));
    return holomq::forward_error(std::move(transport_or));
  }

  std::string fd_pass_socket_path = fd_pass_socket_path_;
  if (fd_pass_socket_path.empty()) {
    fd_pass_socket_path = default_fd_pass_socket_path();
  } else if (fd_pass_socket_path.size() > fd_pass::endpoint_max_length()) {
    HOLOMQ_LOG_ERROR("ContextBuilder create failed: fd_pass_socket path too long");
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  std::optional<fd_pass::Server> fd_pass_server;
  if (vmm_fd_export_enabled_) {
    auto server_or = fd_pass::Server::create(fd_pass_socket_path);
    if (!server_or.has_value()) {
      HOLOMQ_LOG_ERROR("ContextBuilder create failed: fd-pass server create failed");
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    fd_pass_server = std::move(server_or).value();
  }

  auto impl = std::make_shared<ContextImpl>(
      std::move(transport_or).value(), std::move(fd_pass_socket_path), std::move(fd_pass_server));

  if (impl->vmm_fd_export_enabled()) {
    auto& server = impl->fd_pass_server();
    if (!server.has_value()) {
      HOLOMQ_LOG_ERROR("ContextBuilder create failed: fd-pass server missing");
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    const std::weak_ptr<ContextImpl> weak_impl = impl;
    const auto started = server->start([weak_impl](std::uint64_t pool_cookie) {
      const auto locked_impl = weak_impl.lock();
      if (!locked_impl) {
        return ScopedFileDescriptor{};
      }
      return locked_impl->vmm_pool_export_registry().dup_exported_fd(pool_cookie);
    });
    if (!started) {
      HOLOMQ_LOG_ERROR("ContextBuilder create failed: fd-pass server start failed");
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    HOLOMQ_LOG_INFO("fd-pass server listening (AF_UNIX SCM_RIGHTS)");
  }

  return Context(std::move(impl));
}

}  // namespace holomq
