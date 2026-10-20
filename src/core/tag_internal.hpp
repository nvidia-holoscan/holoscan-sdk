/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <limits>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/tag.hpp>

namespace holoscan::detail {

// The Dispatcher is the sole allocator for an Operator execution epoch. Reserving at candidate
// freeze, rather than at compute entry, ensures rejected and no-compute candidates cannot donate
// their identity to a later activation.
class TagAllocator {
 public:
  explicit constexpr TagAllocator(std::uint32_t generation = 1,
                                  std::uint64_t next_sequence = 1) noexcept
      : generation_(generation), next_sequence_(next_sequence) {}

  TagAllocator(const TagAllocator&) = delete;
  TagAllocator& operator=(const TagAllocator&) = delete;
  TagAllocator(TagAllocator&&) = delete;
  TagAllocator& operator=(TagAllocator&&) = delete;

  [[nodiscard]] expected<Tag, Error> reserve_candidate() noexcept {
    if (generation_ == 0) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument, .message = {}});
    }
    if (next_sequence_ == 0) {
      return make_unexpected(Error{.code = ErrorCode::kResourceExhausted, .message = {}});
    }

    const Tag result{.sequence = next_sequence_, .generation = generation_, .reserved = 0};
    next_sequence_ =
        next_sequence_ == std::numeric_limits<std::uint64_t>::max() ? 0 : next_sequence_ + 1;
    return result;
  }

  // Call only at the successful kStart publication barrier. Failed restarts retain the old
  // allocator untouched by not invoking this operation.
  [[nodiscard]] expected<void, Error> publish_restart() noexcept {
    if (generation_ == 0) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument, .message = {}});
    }
    if (generation_ == std::numeric_limits<std::uint32_t>::max()) {
      return make_unexpected(Error{.code = ErrorCode::kResourceExhausted, .message = {}});
    }

    ++generation_;
    next_sequence_ = 1;
    return {};
  }

  [[nodiscard]] constexpr std::uint32_t generation() const noexcept { return generation_; }

 private:
  std::uint32_t generation_;
  // Zero is an internal exhausted sentinel. It is never returned in a Tag and can be reset only by
  // publishing a later nonzero generation.
  std::uint64_t next_sequence_;
};

}  // namespace holoscan::detail
