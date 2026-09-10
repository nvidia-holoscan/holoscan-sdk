/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/const_buffer.hpp>
#include "buffer_pool/generation_slot_array.hpp"
#include "buffer_pool/payload_slot_array.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

struct CUstream_st;

namespace holomq {

class ConstBuffer::Impl {
 public:
  virtual ~Impl() = default;

  [[nodiscard]] virtual BufferMemoryType memory_type() const noexcept = 0;

  [[nodiscard]] virtual std::optional<int32_t> cuda_device_ordinal() const noexcept = 0;

  [[nodiscard]] virtual std::optional<BufferHandle> export_handle() noexcept = 0;

  [[nodiscard]] virtual bool active() const noexcept = 0;

  virtual void release_for_reuse() noexcept = 0;

  [[nodiscard]] virtual bool reacquire(const BufferHandle& handle) noexcept = 0;

  [[nodiscard]] std::uint32_t index() const noexcept { return index_; }

  [[nodiscard]] std::size_t size_bytes() const noexcept {
    return payload_slots_ ? payload_slots_[index_].size_bytes() : 0U;
  }

  [[nodiscard]] const void* ptr() const noexcept {
    return payload_slots_ ? payload_slots_[index_].ptr() : nullptr;
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(payload_slots_) && static_cast<bool>(generation_slots_) &&
           payload_slots_[index_].ptr() != nullptr;
  }

  /// Enqueue the pinned ready-generation wait on \p stream and arm deferred ref release on success.
  [[nodiscard]] virtual bool enqueue_ready_wait(CUstream_st* stream) noexcept = 0;

  [[nodiscard]] virtual bool has_producer_readiness_fence() const noexcept = 0;

  friend void swap(Impl& lhs, Impl& rhs) noexcept {
    using std::swap;
    swap(lhs.payload_slots_, rhs.payload_slots_);
    swap(lhs.generation_slots_, rhs.generation_slots_);
    swap(lhs.index_, rhs.index_);
  }

 protected:
  Impl() noexcept = default;

  [[nodiscard]] bool wait_ready(CUstream_st* stream, std::uint64_t generation) noexcept;

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
};

}  // namespace holomq
