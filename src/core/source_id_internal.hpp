/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <limits>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/source_id.hpp>

namespace holoscan::detail {

struct SourceIdAccess {
  [[nodiscard]] static constexpr SourceId make(std::uint32_t value) noexcept {
    return SourceId{value};
  }

  [[nodiscard]] static constexpr std::uint32_t value(SourceId id) noexcept { return id.value_; }
};

[[nodiscard]] constexpr SourceId make_source_id(std::uint32_t value) noexcept {
  return SourceIdAccess::make(value);
}

[[nodiscard]] constexpr std::uint32_t source_id_value(SourceId id) noexcept {
  return SourceIdAccess::value(id);
}

// Source acceptance is distinct from Dispatcher candidate identity: an accepted witness can be
// retired before candidate formation and therefore never receives a Tag.
struct SourceAcceptanceStamp {
  std::uint64_t sequence{};
  std::uint32_t generation{};
  std::uint32_t reserved{};
};

class SourceAcceptanceAllocator {
 public:
  explicit constexpr SourceAcceptanceAllocator(std::uint32_t generation = 1,
                                               std::uint64_t next_sequence = 1) noexcept
      : generation_(generation), next_sequence_(next_sequence) {}

  SourceAcceptanceAllocator(const SourceAcceptanceAllocator&) = delete;
  SourceAcceptanceAllocator& operator=(const SourceAcceptanceAllocator&) = delete;
  SourceAcceptanceAllocator(SourceAcceptanceAllocator&&) = delete;
  SourceAcceptanceAllocator& operator=(SourceAcceptanceAllocator&&) = delete;

  [[nodiscard]] expected<SourceAcceptanceStamp, Error> reserve_acceptance() noexcept {
    // The acquire pairs with successful restart publication. Lifecycle quiescence forbids a
    // concurrent restart, so a reservation that observes the new generation also observes its
    // sequence reset before joining the multi-producer acceptance order.
    const std::uint32_t generation = generation_.load(std::memory_order_acquire);
    if (generation == 0) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument, .message = {}});
    }

    // Multiple notification producers may race to reserve one accepted witness. The successful
    // compare-exchange is the acceptance ordering point. Zero is an exhausted sentinel, so the
    // maximum value is returned exactly once without ever wrapping to a reusable sequence.
    std::uint64_t current = next_sequence_.load(std::memory_order_relaxed);
    while (current != 0) {
      const std::uint64_t next =
          current == std::numeric_limits<std::uint64_t>::max() ? 0 : current + 1;
      if (next_sequence_.compare_exchange_weak(
              current, next, std::memory_order_relaxed, std::memory_order_relaxed)) {
        return SourceAcceptanceStamp{.sequence = current, .generation = generation, .reserved = 0};
      }
    }
    return make_unexpected(Error{.code = ErrorCode::kResourceExhausted, .message = {}});
  }

  // Lifecycle quiescence excludes concurrent acceptance while a new source generation is
  // published. That invariant lets the two fields remain independently lock-free on platforms
  // where their underlying scalar atomics are lock-free.
  [[nodiscard]] expected<void, Error> publish_restart() noexcept {
    const std::uint32_t generation = generation_.load(std::memory_order_relaxed);
    if (generation == 0) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument, .message = {}});
    }
    if (generation == std::numeric_limits<std::uint32_t>::max()) {
      return make_unexpected(Error{.code = ErrorCode::kResourceExhausted, .message = {}});
    }

    next_sequence_.store(1, std::memory_order_relaxed);
    generation_.store(generation + 1, std::memory_order_release);
    return {};
  }

  [[nodiscard]] std::uint32_t generation() const noexcept {
    return generation_.load(std::memory_order_acquire);
  }

 private:
  std::atomic<std::uint32_t> generation_;
  std::atomic<std::uint64_t> next_sequence_;
};

}  // namespace holoscan::detail
