/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "buffer_pool/slot_region.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace holomq {

/// Array-like view over user payload slots described by a \ref SlotRegion.
///
/// \c operator[] does not validate \p index against a slot count; out-of-range indices yield null
/// \c ptr().
class PayloadSlotArray {
 public:
  /// Non-owning (region, index) pair for one user-payload slot in this array.
  class Slot {
   public:
    [[nodiscard]] void* ptr() noexcept { return region_.element_address(index_); }

    [[nodiscard]] const void* ptr() const noexcept { return region_.element_address(index_); }

    [[nodiscard]] std::size_t size_bytes() const noexcept { return region_.element_size; }

   private:
    friend class PayloadSlotArray;

    Slot(SlotRegion region, const std::uint32_t index) noexcept : region_(region), index_(index) {}

    SlotRegion region_{};
    std::uint32_t index_{0U};
  };

  PayloadSlotArray() noexcept = default;

  explicit PayloadSlotArray(SlotRegion region) noexcept : region_(region) {}

  [[nodiscard]] Slot operator[](const std::uint32_t index) const noexcept {
    return Slot(region_, index);
  }

  [[nodiscard]] Slot at(const std::uint32_t index) const {
    if (!region_) {
      throw std::out_of_range("payload slot region is empty");
    }
    return Slot(region_, index);
  }

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(region_); }

 private:
  SlotRegion region_{};
};

using PayloadSlotRef = PayloadSlotArray::Slot;

}  // namespace holomq
