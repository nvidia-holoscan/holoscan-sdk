/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/buffer_memory_type.hpp>
#include <holomq/const_buffer.hpp>
#include <holomq/detail/buffer_pool/host/handle_data.hpp>
#include "buffer_pool/const_buffer_impl.hpp"
#include "buffer_pool/host/host_backing_segment_cache.hpp"
#include "buffer_pool/ref_count/ref_count_segment_cache.hpp"
#include "buffer_pool/ref_count/shm_ref_count.hpp"
#include "buffer_pool/slot_region.hpp"

#include <cstdint>
#include <optional>
#include <utility>

struct CUstream_st;

namespace holomq {

class VmmPoolImportCache;
class Context;
class HostBufferImport;

class HostConstBufferImpl final : public ConstBuffer::Impl {
 public:
  using HandleData = detail::host::BufferHandleData;

  static_assert(std::is_same_v<HandleData, detail::host::BufferHandleData>);

  static auto import(const BufferHandle& handle, HostBackingSegmentCache& backing_cache,
                     RefCountSegmentCache& ref_count_cache,
                     VmmPoolImportCache& import_cache) noexcept -> std::optional<ConstBuffer>;

  static auto create(const BufferHandle& handle, CUstream_st* stream,
                     HostBackingSegmentCache& backing_cache, RefCountSegmentCache& ref_count_cache,
                     VmmPoolImportCache& import_cache) noexcept -> std::optional<ConstBuffer>;

  HostConstBufferImpl(const HostConstBufferImpl&) = delete;
  HostConstBufferImpl& operator=(const HostConstBufferImpl&) = delete;
  HostConstBufferImpl(HostConstBufferImpl&&) noexcept;
  HostConstBufferImpl& operator=(HostConstBufferImpl&&) noexcept;
  ~HostConstBufferImpl() override;

  [[nodiscard]] BufferMemoryType memory_type() const noexcept override;
  [[nodiscard]] std::optional<int32_t> cuda_device_ordinal() const noexcept override;
  [[nodiscard]] bool enqueue_ready_wait(CUstream_st* stream) noexcept override;
  [[nodiscard]] std::optional<BufferHandle> export_handle() noexcept override;
  [[nodiscard]] bool active() const noexcept override;
  void release_for_reuse() noexcept override;
  [[nodiscard]] bool reacquire(const BufferHandle& handle) noexcept override;

  friend void swap(HostConstBufferImpl& lhs, HostConstBufferImpl& rhs) noexcept;
  friend class Context;
  friend class HostBufferImport;

 private:
  HostConstBufferImpl(ShmRefCountMappingPtr ref_count_mapping, BufferMemoryType memory_type,
                      detail::host::BufferHandleData handle_data, std::uint32_t slot_index,
                      std::uint64_t ready_generation, SlotRegions regions) noexcept;

  void release_view_lease() noexcept;
  [[nodiscard]] bool reacquire(const BufferHandle& handle, bool require_same_slot) noexcept;

  ShmRefCountMappingPtr ref_count_mapping_;
  BufferMemoryType memory_type_{BufferMemoryType::None};
  detail::host::BufferHandleData handle_data_{};
  std::uint32_t slot_index_{};
  std::uint64_t ready_generation_{};
  CUstream_st* release_stream_{nullptr};
  bool lease_active_{true};
};

}  // namespace holomq
