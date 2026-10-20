/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include <holomq/buffer.hpp>
#include "buffer_pool/generation_slot_array.hpp"
#include "buffer_pool/payload_slot_array.hpp"
#include "buffer_pool/sync/stream_release.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

struct CUstream_st;

namespace holomq {

class Buffer::Impl {
 public:
  virtual ~Impl() = default;

  [[nodiscard]] virtual BufferMemoryType memory_type() const noexcept = 0;

  [[nodiscard]] virtual std::optional<int32_t> cuda_device_ordinal() const noexcept = 0;

  [[nodiscard]] virtual std::optional<BufferHandle> export_handle(CUstream_st* stream) noexcept = 0;

  [[nodiscard]] virtual std::optional<BufferHandle> export_handle_unsynchronized() noexcept = 0;

  [[nodiscard]] virtual bool enqueue_ready_set(CUstream_st* stream) noexcept = 0;

  [[nodiscard]] virtual bool defer_release_on_stream(CUstream_st* stream) noexcept = 0;

  [[nodiscard]] virtual bool prepare_for_reuse() noexcept = 0;

  [[nodiscard]] std::uint32_t index() const noexcept { return index_; }

  [[nodiscard]] std::size_t size_bytes() const noexcept {
    return payload_slots_ ? payload_slots_[index_].size_bytes() : 0U;
  }

  [[nodiscard]] void* ptr() noexcept {
    return payload_slots_ ? payload_slots_[index_].ptr() : nullptr;
  }

  [[nodiscard]] const void* ptr() const noexcept {
    return payload_slots_ ? payload_slots_[index_].ptr() : nullptr;
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(payload_slots_) && static_cast<bool>(generation_slots_) &&
           payload_slots_[index_].ptr() != nullptr;
  }

  [[nodiscard]] bool register_stream(
      CUstream_st* stream, const std::shared_ptr<const cuda::StreamLifetime>& lifetime) noexcept {
    return release_streams_.add(mapping_owner(), index_, stream, lifetime, retirement_owners_);
  }

  friend void swap(Impl& lhs, Impl& rhs) noexcept {
    using std::swap;
    swap(lhs.payload_slots_, rhs.payload_slots_);
    swap(lhs.generation_slots_, rhs.generation_slots_);
    swap(lhs.index_, rhs.index_);
    swap(lhs.release_streams_, rhs.release_streams_);
    swap(lhs.retirement_owners_, rhs.retirement_owners_);
    swap(lhs.lease_export_generation_, rhs.lease_export_generation_);
  }

 protected:
  [[nodiscard]] virtual ShmRefCountMappingPtr mapping_owner() const noexcept = 0;
  BufferStreamRelease release_streams_;
  BufferStreamRelease::Owners retirement_owners_;

  Impl() noexcept = default;

  [[nodiscard]] std::uint64_t record_ready(CUstream_st* stream) noexcept;

  [[nodiscard]] bool wait_ready(CUstream_st* stream, std::uint64_t generation) noexcept;

  /// Owner pools assign the next per-slot export generation for this lease.
  [[nodiscard]] virtual std::uint64_t next_export_generation() noexcept = 0;

  Impl(PayloadSlotArray payload_slots, GenerationSlotArray generation_slots,
       const std::uint32_t index) noexcept
      : payload_slots_(payload_slots), generation_slots_(generation_slots), index_(index) {}

  Impl(SlotRegions regions, const std::uint32_t index) noexcept
      : Impl(PayloadSlotArray(regions.payload), GenerationSlotArray(regions.generation), index) {}

  Impl(Impl&&) noexcept = default;
  Impl& operator=(Impl&&) noexcept = default;

  PayloadSlotArray payload_slots_{};
  GenerationSlotArray generation_slots_{};
  std::uint32_t index_{0U};
  /// G assigned by the last successful \ref record_ready on this lease (\c 0 = none).
  std::uint64_t lease_export_generation_{0U};
};

}  // namespace holomq
