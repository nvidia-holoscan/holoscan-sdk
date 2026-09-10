/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/detail/transport/backend.hpp>
#include "buffer_pool/host/host_backing_segment_cache.hpp"
#include "buffer_pool/ref_count/ref_count_segment_cache.hpp"
#include "fd_pass/server.hpp"
#include "vmm_pool_export_registry.hpp"
#include "vmm_pool_import_cache.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace holomq {

class ContextBuilder;

/// Shared session state for \ref Context. \ref BufferPool implementations hold \c
/// std::shared_ptr<ContextImpl> so export registry and FD-pass state outlive the \ref Context shell
/// when pools remain.
class ContextImpl {
  friend class ContextBuilder;

 public:
  ContextImpl(detail::transport::Runtime transport, std::string configured_fd_pass_path,
              std::optional<fd_pass::Server> fd_pass_server,
              std::optional<std::filesystem::path> generated_fd_pass_directory = std::nullopt);
  ~ContextImpl() noexcept;

  [[nodiscard]] bool vmm_fd_export_enabled() const noexcept;
  [[nodiscard]] const std::string& fd_pass_socket_path() const noexcept;

  [[nodiscard]] VmmPoolExportRegistry& vmm_pool_export_registry() noexcept {
    return vmm_pool_export_registry_;
  }

  [[nodiscard]] const VmmPoolExportRegistry& vmm_pool_export_registry() const noexcept {
    return vmm_pool_export_registry_;
  }

  [[nodiscard]] detail::transport::Runtime& transport() noexcept { return transport_; }

  [[nodiscard]] const detail::transport::Runtime& transport() const noexcept { return transport_; }

  [[nodiscard]] RefCountSegmentCache& ref_count_segment_cache() noexcept {
    return ref_count_segment_cache_;
  }

  [[nodiscard]] HostBackingSegmentCache& host_backing_segment_cache() noexcept {
    return host_backing_segment_cache_;
  }

  [[nodiscard]] VmmPoolImportCache& vmm_pool_import_cache() noexcept {
    return vmm_pool_import_cache_;
  }

  [[nodiscard]] std::optional<fd_pass::Server>& fd_pass_server() noexcept {
    return fd_pass_server_;
  }

  [[nodiscard]] const std::optional<fd_pass::Server>& fd_pass_server() const noexcept {
    return fd_pass_server_;
  }

 private:
  detail::transport::Runtime transport_;
  /// Before \c fd_pass_server_ so destruction stops the worker before the registry is released.
  VmmPoolExportRegistry vmm_pool_export_registry_;
  std::optional<fd_pass::Server> fd_pass_server_;
  std::string configured_fd_pass_path_;
  std::optional<std::filesystem::path> generated_fd_pass_directory_;
  RefCountSegmentCache ref_count_segment_cache_;
  HostBackingSegmentCache host_backing_segment_cache_;
  VmmPoolImportCache vmm_pool_import_cache_;
};

}  // namespace holomq
