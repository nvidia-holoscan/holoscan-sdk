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

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

#include <unistd.h>

namespace {

struct DefaultFdPassEndpoint {
  std::string path;
  std::optional<std::filesystem::path> private_directory;
};

class GeneratedEndpointCleanup {
 public:
  GeneratedEndpointCleanup(const std::string& socket_path,
                           const std::optional<std::filesystem::path>& private_directory)
      : socket_path_(socket_path), private_directory_(private_directory) {}
  GeneratedEndpointCleanup(const GeneratedEndpointCleanup&) = delete;
  GeneratedEndpointCleanup& operator=(const GeneratedEndpointCleanup&) = delete;
  ~GeneratedEndpointCleanup() {
    if (!private_directory_) {
      return;
    }
    std::error_code ignored;
    std::filesystem::remove(socket_path_, ignored);
    ignored.clear();
    std::filesystem::remove(*private_directory_, ignored);
  }

  void release() noexcept { private_directory_.reset(); }

 private:
  std::string socket_path_;
  std::optional<std::filesystem::path> private_directory_;
};

[[nodiscard]] holomq::ErrorCode filesystem_error_code(const std::error_code& error) noexcept {
  if (error == std::errc::permission_denied || error == std::errc::operation_not_permitted) {
    return holomq::ErrorCode::kPermissionDenied;
  }
  if (error == std::errc::no_such_file_or_directory) {
    return holomq::ErrorCode::kNotFound;
  }
  if (error == std::errc::invalid_argument || error == std::errc::filename_too_long) {
    return holomq::ErrorCode::kInvalidArgument;
  }
  if (error == std::errc::no_space_on_device || error == std::errc::too_many_files_open ||
      error == std::errc::too_many_files_open_in_system) {
    return holomq::ErrorCode::kResourceExhausted;
  }
  if (error == std::errc::interrupted) {
    return holomq::ErrorCode::kInterrupted;
  }
  return holomq::ErrorCode::kFailure;
}

holomq::expected<DefaultFdPassEndpoint, holomq::Error> default_fd_pass_socket_endpoint(
    const bool create_filesystem_endpoint) {
  static std::atomic<std::uint64_t> ordinal{};
  constexpr std::size_t k_max_path = holomq::fd_pass::endpoint_max_length();
  const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
  const auto id = ordinal.fetch_add(1U, std::memory_order_relaxed);

  std::ostringstream name;
  name << "holomq_" << ::getpid() << '_' << us << '_' << id;
  if (!holomq::fd_pass::endpoint_uses_filesystem_path() || !create_filesystem_endpoint) {
    std::string result = name.str();
    if (result.size() > k_max_path) {
      result.resize(k_max_path);
    }
    return DefaultFdPassEndpoint{.path = std::move(result)};
  }

  std::error_code error;
  // The shared directory is only a parent; mkdtemp atomically creates the private 0700 child.
  const std::filesystem::path temporary_parent =
      std::filesystem::temp_directory_path(error);  // NOSONAR(cpp:S5443)
  if (error) {
    return holomq::make_unexpected(holomq::Error{
        filesystem_error_code(error), "temporary-directory lookup failed: " + error.message()});
  }
  std::string directory_template = (temporary_parent / "holomq-XXXXXX").string();
  directory_template.push_back('\0');
  char* const directory = ::mkdtemp(directory_template.data());
  if (directory == nullptr) {
    const std::error_code cause{errno, std::generic_category()};
    return holomq::make_unexpected(
        holomq::Error{filesystem_error_code(cause),
                      "private temporary-directory creation failed: " + cause.message()});
  }
  const std::filesystem::path private_directory{directory};
  const std::string socket_path = (private_directory / "fd-pass.sock").string();
  if (socket_path.size() > k_max_path) {
    std::filesystem::remove(private_directory, error);
    return holomq::make_unexpected(
        holomq::Error{holomq::ErrorCode::kInvalidArgument,
                      "generated fd-pass socket path exceeds the platform endpoint limit"});
  }
  return DefaultFdPassEndpoint{.path = socket_path, .private_directory = private_directory};
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
  std::optional<std::filesystem::path> generated_fd_pass_directory;
  if (fd_pass_socket_path.empty()) {
    auto generated = default_fd_pass_socket_endpoint(vmm_fd_export_enabled_);
    if (!generated) {
      HOLOMQ_LOG_ERROR("ContextBuilder create failed: {}", error_message(generated.error()));
      return forward_error(std::move(generated));
    }
    fd_pass_socket_path = std::move(generated->path);
    generated_fd_pass_directory = std::move(generated->private_directory);
  } else if (fd_pass_socket_path.size() > fd_pass::endpoint_max_length()) {
    HOLOMQ_LOG_ERROR("ContextBuilder create failed: fd_pass_socket path too long");
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  GeneratedEndpointCleanup generated_endpoint_cleanup(fd_pass_socket_path,
                                                      generated_fd_pass_directory);

  std::optional<fd_pass::Server> fd_pass_server;
  if (vmm_fd_export_enabled_) {
    auto server_or = fd_pass::Server::create(fd_pass_socket_path);
    if (!server_or.has_value()) {
      HOLOMQ_LOG_ERROR("ContextBuilder create failed: fd-pass server create failed");
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    fd_pass_server = std::move(server_or).value();
  }

  auto impl = std::make_shared<ContextImpl>(std::move(transport_or).value(),
                                            std::move(fd_pass_socket_path),
                                            std::move(fd_pass_server),
                                            generated_fd_pass_directory);
  generated_endpoint_cleanup.release();

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
