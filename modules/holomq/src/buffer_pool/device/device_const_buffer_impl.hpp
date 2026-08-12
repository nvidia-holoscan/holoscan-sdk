/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/buffer_handle.hpp>
#include <holomq/const_buffer.hpp>
#include <holomq/detail/buffer_pool/device/handle_data.hpp>
#include "buffer_pool/const_buffer_impl.hpp"
#include "buffer_pool/ref_count/ref_count_segment_cache.hpp"
#include "buffer_pool/ref_count/shm_ref_count.hpp"
#include "vmm_pool_import_cache.hpp"

#include <optional>
#include <utility>

struct CUstream_st;

namespace holomq {

/// Subscriber-side \ref ConstBuffer impl: SHM ref-count lease + imported GPU slot from \ref
/// Context::import_buffer.
class DeviceConstBufferImpl final : public ConstBuffer::Impl {
 public:
  /// Maps ref-count SHM, imports the GPU slot, then constructs a \ref ConstBuffer that owns the
  /// subscriber ref-count lease. Does not enqueue a ready-generation wait; call \ref
  /// enqueue_ready_wait on the returned \ref ConstBuffer.
  [[nodiscard]] static auto import(const BufferHandle& handle, VmmPoolImportCache& import_cache,
                                   RefCountSegmentCache& ref_count_cache) noexcept
      -> std::optional<ConstBuffer>;

  /// Import plus export-fence wait on \p stream when the handle requires one.
  [[nodiscard]] static auto create(const BufferHandle& handle, CUstream_st* stream,
                                   VmmPoolImportCache& import_cache,
                                   RefCountSegmentCache& ref_count_cache) noexcept
      -> std::optional<ConstBuffer>;

  ~DeviceConstBufferImpl() override;

  DeviceConstBufferImpl(const DeviceConstBufferImpl&) = delete;
  DeviceConstBufferImpl& operator=(const DeviceConstBufferImpl&) = delete;
  DeviceConstBufferImpl(DeviceConstBufferImpl&&) noexcept;
  DeviceConstBufferImpl& operator=(DeviceConstBufferImpl&&) noexcept;

  [[nodiscard]] BufferMemoryType memory_type() const noexcept override;
  [[nodiscard]] std::optional<int32_t> cuda_device_ordinal() const noexcept override;
  [[nodiscard]] bool enqueue_ready_wait(CUstream_st* stream) noexcept override;
  /// Re-publish the imported handle without recording a new export fence.
  [[nodiscard]] std::optional<BufferHandle> export_handle() noexcept override;
  [[nodiscard]] bool active() const noexcept override;
  void release_for_reuse() noexcept override;
  [[nodiscard]] bool reacquire(const BufferHandle& handle) noexcept override;

  friend void swap(DeviceConstBufferImpl& lhs, DeviceConstBufferImpl& rhs) noexcept;

 private:
  void release_view_lease() noexcept;

  DeviceConstBufferImpl(ShmRefCountMappingPtr ref_count_mapping,
                        const detail::device::BufferHandleData& handle_data,
                        std::uint32_t slot_index, std::uint64_t ready_generation, void* pool_base,
                        int32_t cuda_device_ordinal) noexcept;

  ShmRefCountMappingPtr ref_count_mapping_;
  detail::device::BufferHandleData handle_data_{};
  std::uint32_t slot_index_{};
  std::uint64_t ready_generation_{};
  int32_t cuda_device_ordinal_{-1};
  CUstream_st* release_stream_{nullptr};
  bool lease_active_{true};
};

}  // namespace holomq
