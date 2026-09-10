/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/buffer_pool_impl.hpp"

#include <cassert>

namespace holomq {

std::uint64_t BufferPool::Impl::next_slot_generation(const std::uint32_t index) noexcept {
  if (index >= buffer_count_) {
    return 0U;
  }
  return ++slot_generations_[index];
}

auto BufferPool::Impl::acquire() noexcept -> std::optional<Buffer> {
  for (std::uint32_t offset = 0U; offset < buffer_count_; ++offset) {
    const std::uint32_t index = (acquire_cursor_ + offset) % buffer_count_;
    if (ref_count_mapping_->table().try_acquire_slot(index)) {
      acquire_cursor_ = (index + 1U) % buffer_count_;
      // A virtual function call is needed here to allow derived classes to customize the buffer
      // creation.
      return buffer_at_index(index);
    }
  }
  return std::nullopt;
}

}  // namespace holomq
