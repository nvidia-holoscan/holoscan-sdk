/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/device_buffer_allocator.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include "buffer_pool/buffer_pool_impl.hpp"
#include "buffer_pool/device/vmm_exported_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace holomq {

class ContextImpl;

class DeviceBufferPoolImpl final : public BufferPool::Impl {
 public:
  [[nodiscard]] static auto try_create(std::size_t buffer_bytes, std::uint32_t buffer_count,
                                       std::string fd_pass_socket_path,
                                       const std::shared_ptr<ContextImpl>& session) noexcept
      -> expected<std::unique_ptr<DeviceBufferPoolImpl>, Error>;

  [[nodiscard]] static auto try_create(std::size_t buffer_bytes, std::uint32_t buffer_count,
                                       DeviceBufferAllocator allocator,
                                       std::string fd_pass_socket_path,
                                       const std::shared_ptr<ContextImpl>& session) noexcept
      -> expected<std::unique_ptr<DeviceBufferPoolImpl>, Error>;

  DeviceBufferPoolImpl(const DeviceBufferPoolImpl&) = delete;
  DeviceBufferPoolImpl& operator=(const DeviceBufferPoolImpl&) = delete;
  DeviceBufferPoolImpl(DeviceBufferPoolImpl&&) noexcept = delete;
  DeviceBufferPoolImpl& operator=(DeviceBufferPoolImpl&&) noexcept = delete;
  ~DeviceBufferPoolImpl() override = default;

  [[nodiscard]] BufferStreamRelease::Owners retirement_owners() const noexcept override {
    return {vmm_pool_.backing_owner(), {}};
  }
  [[nodiscard]] BufferMemoryType memory_type() const noexcept override;
  [[nodiscard]] std::optional<int32_t> cuda_device_ordinal() const noexcept override;
  [[nodiscard]] void* base_ptr() const noexcept override;
  [[nodiscard]] BufferHandle handle(std::uint32_t index) const noexcept override;
  [[nodiscard]] Buffer buffer_at_index(std::uint32_t index) noexcept override;

 private:
  struct PrivateTag {};

  DeviceBufferPoolImpl(PrivateTag, ShmRefCountMappingPtr ref_count_mapping,
                       VmmExportedPool vmm_pool, std::string fd_pass_socket_path,
                       std::size_t buffer_bytes, std::size_t slot_stride_bytes,
                       std::uint32_t buffer_count, int32_t cuda_device_ordinal) noexcept;

  [[nodiscard]] std::uint64_t pool_cookie() const noexcept;

  std::string fd_pass_socket_path_;
  VmmExportedPool vmm_pool_;
  int32_t cuda_device_ordinal_{-1};
};

}  // namespace holomq
