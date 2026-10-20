/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "buffer_pool/slot_region.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

struct CUstream_st;

namespace holomq {

/// Array-like view over per-slot GPU generation words described by a \ref SlotRegion.
///
/// \c operator[] does not validate \p index against a slot count; out-of-range indices yield null
/// addresses.
class GenerationSlotArray {
 public:
  /// Non-owning (region, index) pair for one GPU generation word in this array.
  class Slot {
   public:
    [[nodiscard]] void* device_address() const noexcept { return region_.element_address(index_); }

    [[nodiscard]] bool write(CUstream_st* stream, std::uint64_t value) const noexcept;

    [[nodiscard]] bool wait(CUstream_st* stream, std::uint64_t generation) const noexcept;

   private:
    friend class GenerationSlotArray;

    Slot(SlotRegion region, const std::uint32_t index) noexcept : region_(region), index_(index) {}

    SlotRegion region_{};
    std::uint32_t index_{0U};
  };

  GenerationSlotArray() noexcept = default;

  explicit GenerationSlotArray(SlotRegion region) noexcept : region_(region) {}

  [[nodiscard]] Slot operator[](const std::uint32_t index) const noexcept {
    return Slot(region_, index);
  }

  [[nodiscard]] Slot at(const std::uint32_t index) const {
    if (!region_) {
      throw std::out_of_range("generation slot region is empty");
    }
    return Slot(region_, index);
  }

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(region_); }

 private:
  SlotRegion region_{};
};

using GenerationSlotRef = GenerationSlotArray::Slot;

}  // namespace holomq
