/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <span>

#include <flatbuffers/allocator.h>

namespace holoscan::schema::detail {

/// @brief FlatBuffers allocator that admits exactly one caller-owned bounded span.
class BoundedSpanAllocator final : public flatbuffers::Allocator {
 public:
  explicit BoundedSpanAllocator(std::span<std::byte> storage) noexcept : storage_(storage) {}

  std::uint8_t* allocate(std::size_t size) override {
    if (allocated_ || size == 0U || size > storage_.size()) {
      throw std::bad_alloc{};
    }
    allocated_ = true;
    return reinterpret_cast<std::uint8_t*>(storage_.data());
  }

  void deallocate(std::uint8_t* pointer, std::size_t size) override {
    static_cast<void>(pointer);
    static_cast<void>(size);
    allocated_ = false;
  }

  std::uint8_t* reallocate_downward(std::uint8_t* old_pointer, std::size_t old_size,
                                    std::size_t new_size, std::size_t in_use_back,
                                    std::size_t in_use_front) override {
    static_cast<void>(old_pointer);
    static_cast<void>(old_size);
    static_cast<void>(new_size);
    static_cast<void>(in_use_back);
    static_cast<void>(in_use_front);
    throw std::bad_alloc{};
  }

 private:
  std::span<std::byte> storage_;
  bool allocated_{};
};

}  // namespace holoscan::schema::detail
