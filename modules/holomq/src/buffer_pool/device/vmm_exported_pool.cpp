/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/device/vmm_exported_pool.hpp"

#include <holomq/logger.hpp>
#include "buffer_pool/splitmix64_id.hpp"
#include "context_impl.hpp"
#include "cuda/mem.hpp"

#include <cuda.h>

#include <memory>
#include <utility>

namespace holomq {

namespace {

[[nodiscard]] std::optional<ScopedFileDescriptor> export_vmm_allocation_fd(
    void* const backing_ptr) noexcept {
  if (backing_ptr == nullptr) {
    return std::nullopt;
  }

  const cuda::Mem retained_export_mem(backing_ptr);
  if (!retained_export_mem || !device_vmm_allocation_is_exportable(retained_export_mem.get())) {
    return std::nullopt;
  }

  int export_fd = -1;
  if (cuMemExportToShareableHandle(
          &export_fd, retained_export_mem.get(), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0ULL) !=
      CUDA_SUCCESS) {
    return std::nullopt;
  }

  return ScopedFileDescriptor(export_fd);
}

}  // namespace

VmmExportedPool::VmmExportedPool(VmmExportedPool&& other) noexcept
    : backing_(std::move(other.backing_)),
      exported_fd_(std::move(other.exported_fd_)),
      session_(std::move(other.session_)),
      pool_cookie_(other.pool_cookie_),
      registered_cookie_(other.registered_cookie_) {
  other.registered_cookie_ = 0U;
  other.pool_cookie_ = 0U;
}

VmmExportedPool& VmmExportedPool::operator=(VmmExportedPool&& other) noexcept {
  if (this != &other) {
    unregister_fd_export();
    backing_ = std::move(other.backing_);
    exported_fd_ = std::move(other.exported_fd_);
    session_ = std::move(other.session_);
    pool_cookie_ = other.pool_cookie_;
    registered_cookie_ = other.registered_cookie_;
    other.registered_cookie_ = 0U;
    other.pool_cookie_ = 0U;
  }
  return *this;
}

VmmExportedPool::~VmmExportedPool() {
  unregister_fd_export();
}

auto VmmExportedPool::create(const DeviceBufferAllocator& allocator,
                             const std::size_t region_bytes) noexcept
    -> std::optional<VmmExportedPool> {
  if (region_bytes == 0U) {
    HOLOMQ_LOG_ERROR("VmmExportedPool::create failed: zero region bytes");
    return std::nullopt;
  }
  if (!allocator) {
    HOLOMQ_LOG_ERROR("VmmExportedPool::create failed: null allocator");
    return std::nullopt;
  }
  if (!device_supports_vmm_posix_fd_export()) {
    HOLOMQ_LOG_ERROR("VmmExportedPool::create failed: device does not support VMM POSIX FD export");
    return std::nullopt;
  }

  std::unique_ptr<DeviceBufferPoolMemory> backing = allocator(region_bytes);
  if (!backing) {
    HOLOMQ_LOG_ERROR("VmmExportedPool::create failed: allocator returned null backing");
    return std::nullopt;
  }
  if (backing->size() < region_bytes) {
    HOLOMQ_LOG_ERROR("VmmExportedPool::create failed: backing size smaller than requested region");
    return std::nullopt;
  }

  auto exported_fd_or = export_vmm_allocation_fd(backing->ptr());
  if (!exported_fd_or.has_value()) {
    HOLOMQ_LOG_ERROR(
        "VmmExportedPool::create failed: VMM allocation is not exportable via POSIX FD");
    return std::nullopt;
  }

  VmmExportedPool pool;
  pool.backing_ = std::move(backing);
  pool.exported_fd_ = std::move(*exported_fd_or);
  pool.pool_cookie_ = detail::next_splitmix64_nonzero_id();
  return pool;
}

void* VmmExportedPool::ptr() const noexcept {
  return backing_ != nullptr ? backing_->ptr() : nullptr;
}

std::size_t VmmExportedPool::size() const noexcept {
  return backing_ != nullptr ? backing_->size() : 0U;
}

std::uint64_t VmmExportedPool::cookie() const noexcept {
  return pool_cookie_;
}

void VmmExportedPool::register_fd_export(const std::shared_ptr<ContextImpl>& session) {
  unregister_fd_export();
  if (!exported_fd_ || session == nullptr) {
    return;
  }

  const std::uint64_t pool_cookie = this->cookie();
  session->vmm_pool_export_registry().register_vmm_fd_export(
      pool_cookie, make_vmm_fd_dup_callback(exported_fd_.get()));
  session_ = session;
  registered_cookie_ = pool_cookie;
}

void VmmExportedPool::unregister_fd_export() noexcept {
  if (session_ != nullptr && registered_cookie_ != 0U) {
    session_->vmm_pool_export_registry().unregister_vmm_fd_export(registered_cookie_);
  }
  session_.reset();
  registered_cookie_ = 0U;
}

std::function<ScopedFileDescriptor()> make_vmm_fd_dup_callback(const int export_fd) noexcept {
  return [export_fd]() {
    if (export_fd < 0) {
      return ScopedFileDescriptor{};
    }
    const int dup_fd = ::dup(export_fd);
    return ScopedFileDescriptor(dup_fd >= 0 ? dup_fd : -1);
  };
}

}  // namespace holomq
