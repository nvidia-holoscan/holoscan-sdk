/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/buffer.hpp>
#include <holomq/detail/buffer_pool/host/handle_data.hpp>
#include "buffer_pool/buffer_impl.hpp"

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

struct CUstream_st;

namespace holomq {

class HostBufferPoolImpl;

class HostBufferImpl final : public Buffer::Impl {
 public:
  using HandleData = detail::host::BufferHandleData;

  static_assert(std::is_same_v<HandleData, detail::host::BufferHandleData>);

  HostBufferImpl(HostBufferPoolImpl& pool, std::uint32_t index) noexcept;
  ~HostBufferImpl() override;

  HostBufferImpl(const HostBufferImpl&) = delete;
  HostBufferImpl& operator=(const HostBufferImpl&) = delete;
  HostBufferImpl(HostBufferImpl&&) noexcept;
  HostBufferImpl& operator=(HostBufferImpl&&) noexcept;

  [[nodiscard]] BufferMemoryType memory_type() const noexcept override;
  [[nodiscard]] std::optional<int32_t> cuda_device_ordinal() const noexcept override;
  [[nodiscard]] std::optional<BufferHandle> export_handle(CUstream_st* stream) noexcept override;
  [[nodiscard]] std::optional<BufferHandle> export_handle_unsynchronized() noexcept override;
  [[nodiscard]] bool enqueue_ready_set(CUstream_st* stream) noexcept override;
  [[nodiscard]] bool defer_release_on_stream(CUstream_st* stream) noexcept override;
  [[nodiscard]] bool prepare_for_reuse() noexcept override;

  friend void swap(HostBufferImpl& lhs, HostBufferImpl& rhs) noexcept;

 protected:
  [[nodiscard]] std::uint64_t next_export_generation() noexcept override;

 private:
  void release_slot_lease() noexcept;

  HostBufferPoolImpl* pool_{nullptr};
  CUstream_st* release_stream_{nullptr};
  bool defer_release_on_export_{false};
};

}  // namespace holomq
