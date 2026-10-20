/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <holomq/device_buffer_allocator.hpp>
#include <holomq/device_buffer_pool_memory.hpp>
#include <memory>
#include <optional>
#include "scoped_file_descriptor.hpp"

namespace holomq {

class ContextImpl;

/// Owns a CUDA VMM pool mapping plus its exported POSIX FD and optional \ref ContextImpl session
/// registration.
class VmmExportedPool {
 public:
  VmmExportedPool() noexcept = default;
  VmmExportedPool(VmmExportedPool&& other) noexcept;
  VmmExportedPool& operator=(VmmExportedPool&& other) noexcept;
  VmmExportedPool(const VmmExportedPool&) = delete;
  VmmExportedPool& operator=(const VmmExportedPool&) = delete;
  ~VmmExportedPool();

  /// Allocates \p region_bytes via \p allocator, retains the handle, and exports a POSIX FD.
  [[nodiscard]] static auto create(const DeviceBufferAllocator& allocator,
                                   std::size_t region_bytes) noexcept
      -> std::optional<VmmExportedPool>;

  [[nodiscard]] explicit operator bool() const noexcept { return backing_ != nullptr; }

  [[nodiscard]] void* ptr() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::shared_ptr<void> backing_owner() const noexcept { return backing_; }
  /// Opaque per-pool id for fd-pass and subscriber import-cache keys
  /// (\c buffer_pool/splitmix64_id.hpp).
  [[nodiscard]] std::uint64_t cookie() const noexcept;

  /// Registers \ref cookie with \p session's export registry for FD-pass dup on import.
  void register_fd_export(const std::shared_ptr<ContextImpl>& session);

  /// Unregisters from the session export registry when previously registered.
  void unregister_fd_export() noexcept;

 private:
  std::shared_ptr<DeviceBufferPoolMemory> backing_;
  ScopedFileDescriptor exported_fd_;
  std::shared_ptr<ContextImpl> session_;
  std::uint64_t pool_cookie_{0U};
  std::uint64_t registered_cookie_{0U};
};

/// Returns a callback that dups \p export_fd for \ref VmmPoolExportRegistry registration.
[[nodiscard]] std::function<ScopedFileDescriptor()> make_vmm_fd_dup_callback(
    int export_fd) noexcept;

}  // namespace holomq
