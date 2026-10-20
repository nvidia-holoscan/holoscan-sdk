/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "context_impl.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace holomq {

ContextImpl::ContextImpl(detail::transport::Runtime transport, std::string configured_fd_pass_path,
                         std::optional<fd_pass::Server> fd_pass_server,
                         std::optional<std::filesystem::path> generated_fd_pass_directory)
    : transport_(std::move(transport)),
      fd_pass_server_(std::move(fd_pass_server)),
      configured_fd_pass_path_(std::move(configured_fd_pass_path)),
      generated_fd_pass_directory_(std::move(generated_fd_pass_directory)) {
  if (configured_fd_pass_path_.empty()) {
    throw std::invalid_argument("configured_fd_pass_path must not be empty");
  }
}

ContextImpl::~ContextImpl() noexcept {
  // Close the listening socket before removing its private filesystem endpoint.
  fd_pass_server_.reset();
  if (generated_fd_pass_directory_) {
    std::error_code ignored;
    std::filesystem::remove(configured_fd_pass_path_, ignored);
    ignored.clear();
    std::filesystem::remove(*generated_fd_pass_directory_, ignored);
  }
}

bool ContextImpl::vmm_fd_export_enabled() const noexcept {
  return fd_pass_server_.has_value();
}

const std::string& ContextImpl::fd_pass_socket_path() const noexcept {
  return configured_fd_pass_path_;
}

}  // namespace holomq
