/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/buffer.hpp>
#include <holomq/detail/buffer_pool/device/handle_data.hpp>
#include "buffer_pool/buffer_impl.hpp"

#include <cstdint>
#include <optional>
#include <utility>

struct CUstream_st;

namespace holomq {

class DeviceBufferPoolImpl;

/// Non-owning view of one device buffer slot; borrows \c DeviceBufferPoolImpl for \ref
/// BufferHandle.
class DeviceBufferImpl final : public Buffer::Impl {
 public:
  using HandleData = detail::device::BufferHandleData;

  DeviceBufferImpl(DeviceBufferPoolImpl& pool, std::uint32_t index) noexcept;
  ~DeviceBufferImpl() override;

  DeviceBufferImpl(const DeviceBufferImpl&) = delete;
  DeviceBufferImpl& operator=(const DeviceBufferImpl&) = delete;
  DeviceBufferImpl(DeviceBufferImpl&&) noexcept;
  DeviceBufferImpl& operator=(DeviceBufferImpl&&) noexcept;

  [[nodiscard]] BufferMemoryType memory_type() const noexcept override;
  [[nodiscard]] std::optional<int32_t> cuda_device_ordinal() const noexcept override;
  [[nodiscard]] std::optional<BufferHandle> export_handle(CUstream_st* stream) noexcept override;
  [[nodiscard]] std::optional<BufferHandle> export_handle_unsynchronized() noexcept override;
  [[nodiscard]] bool enqueue_ready_set(CUstream_st* stream) noexcept override;
  [[nodiscard]] bool defer_release_on_stream(CUstream_st* stream) noexcept override;
  [[nodiscard]] bool prepare_for_reuse() noexcept override;

  friend void swap(DeviceBufferImpl& lhs, DeviceBufferImpl& rhs) noexcept;

 protected:
  [[nodiscard]] ShmRefCountMappingPtr mapping_owner() const noexcept override;
  [[nodiscard]] std::uint64_t next_export_generation() noexcept override;

 private:
  void release_slot_lease() noexcept;

  DeviceBufferPoolImpl* pool_{nullptr};
  CUstream_st* release_stream_{nullptr};
  bool defer_release_on_export_{false};
};

}  // namespace holomq
