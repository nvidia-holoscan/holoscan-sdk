/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vmm_pool_export_registry.hpp"

#include <holomq/logger.hpp>

#include <utility>

namespace holomq {

void VmmPoolExportRegistry::register_vmm_fd_export(std::uint64_t pool_cookie,
                                                   std::function<ScopedFileDescriptor()> dup_fd) {
  if (!dup_fd) {
    HOLOMQ_LOG_WARN("vmm pool registry: ignoring empty fd export callback (pool_cookie={})",
                    pool_cookie);
    return;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  fd_exports_[pool_cookie] = std::move(dup_fd);
}

void VmmPoolExportRegistry::unregister_vmm_fd_export(std::uint64_t pool_cookie) {
  const std::lock_guard<std::mutex> lock(mutex_);
  fd_exports_.erase(pool_cookie);
}

ScopedFileDescriptor VmmPoolExportRegistry::dup_exported_fd(std::uint64_t pool_cookie) {
  std::function<ScopedFileDescriptor()> dup_fd;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = fd_exports_.find(pool_cookie);
    if (it == fd_exports_.end()) {
      return {};
    }
    dup_fd = it->second;
  }
  return dup_fd();
}

}  // namespace holomq
