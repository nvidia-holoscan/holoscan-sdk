/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <utility>

struct CUstream_st;

namespace holomq::cuda {

/// Process-local admission and completion evidence for one SDK-owned CUDA stream.
/// Sharing this control block never extends the native stream's lifetime. Its owner stops ordinary
/// submissions, closes admission, drains pins, synchronizes, and publishes the outcome before
/// destroying the stream. This internal header has no Core or CUDA library dependency.
class StreamLifetime final {
 public:
  enum class State : std::uint8_t { kOpen, kClosing, kRetired, kQuarantined };

  /// Embedded, single-use release node. The callback must not block, take the GIL, or use CUDA.
  /// On successful deferral, publication can invoke and destroy the node before defer() returns.
  struct DeferredRelease {
    void (*on_outcome)(DeferredRelease&, State) noexcept {};
    DeferredRelease* next{};
  };

  class Pin final {
   public:
    Pin() noexcept = default;
    Pin(const Pin&) = delete;
    Pin& operator=(const Pin&) = delete;
    Pin(Pin&& other) noexcept : lifetime_(std::exchange(other.lifetime_, nullptr)) {}
    Pin& operator=(Pin&& other) noexcept {
      if (this != &other) {
        reset();
        lifetime_ = std::exchange(other.lifetime_, nullptr);
      }
      return *this;
    }
    ~Pin() { reset(); }

    [[nodiscard]] explicit operator bool() const noexcept { return lifetime_ != nullptr; }
    [[nodiscard]] CUstream_st* stream() const noexcept {
      return lifetime_ != nullptr ? lifetime_->handle() : nullptr;
    }
    void reset() noexcept {
      if (const auto* lifetime = std::exchange(lifetime_, nullptr)) {
        lifetime->unpin();
      }
    }

   private:
    friend class StreamLifetime;
    explicit Pin(const StreamLifetime* lifetime) noexcept : lifetime_(lifetime) {}
    const StreamLifetime* lifetime_{};
  };

  struct PinResult {
    Pin pin;
    State state;
  };

  StreamLifetime(CUstream_st* stream, int device) noexcept : stream_(stream), device_(device) {}
  StreamLifetime(const StreamLifetime&) = delete;
  StreamLifetime& operator=(const StreamLifetime&) = delete;

  /// Pin for one non-blocking enqueue. No host waits, GIL acquisition, or owner locks while pinned.
  /// An empty Open result denotes count exhaustion and must fail closed, just like Quarantined.
  [[nodiscard]] PinResult try_pin() const noexcept {
    auto word = word_.load(std::memory_order_acquire);
    while (decode_state(word) == State::kOpen && (word & kCountMask) != kCountMask) {
      if (word_.compare_exchange_weak(
              word, word + 1U, std::memory_order_acquire, std::memory_order_acquire)) {
        return {Pin{this}, State::kOpen};
      }
    }
    return {Pin{}, decode_state(word)};
  }

  [[nodiscard]] State state() const noexcept {
    return decode_state(word_.load(std::memory_order_acquire));
  }

  /// Owner only. Separate closure from draining so a whole inventory can close before any wait.
  void close_admission() noexcept {
    auto word = word_.load(std::memory_order_acquire);
    while (decode_state(word) == State::kOpen) {
      if (word_.compare_exchange_weak(word,
                                      word | encode_state(State::kClosing),
                                      std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
        return;
      }
    }
  }

  void wait_for_pins() noexcept {
    auto word = word_.load(std::memory_order_acquire);
    assert(decode_state(word) != State::kOpen);
    while ((word & kCountMask) != 0U) {
      word_.wait(word, std::memory_order_acquire);
      word = word_.load(std::memory_order_acquire);
    }
  }

  void close() noexcept {
    close_admission();
    wait_for_pins();
  }

  /// Owner only, after draining. Retired additionally requires a successful final synchronize.
  /// False diagnoses an invalid transition; terminal states are immutable.
  [[nodiscard]] bool publish_retired() noexcept { return publish(State::kRetired); }
  [[nodiscard]] bool publish_quarantined() noexcept { return publish(State::kQuarantined); }

  /// Called after observing Closing. False means no registration took place: re-read state().
  /// Registration allocates nothing and never waits for publication. The caller must keep the
  /// block and node alive until delivery, and must not touch a successfully published node again.
  [[nodiscard]] bool defer(DeferredRelease& node) const noexcept {
    assert(node.on_outcome != nullptr);
    if (state() != State::kClosing) {
      return false;
    }
    auto* head = deferred_.load(std::memory_order_acquire);
    while (head != &sealed_) {
      node.next = head;
      if (deferred_.compare_exchange_weak(
              head, &node, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return true;
      }
    }
    return false;
  }

  /// Identity/diagnostics only. Use Pin::stream() for enqueues.
  [[nodiscard]] CUstream_st* handle() const noexcept { return stream_; }
  [[nodiscard]] int device() const noexcept { return device_; }

 private:
  static constexpr unsigned kStateShift = 62U;
  static constexpr std::uint64_t kCountMask = (std::uint64_t{1U} << kStateShift) - 1U;
  [[nodiscard]] static constexpr std::uint64_t encode_state(State state) noexcept {
    return static_cast<std::uint64_t>(state) << kStateShift;
  }
  [[nodiscard]] static constexpr State decode_state(std::uint64_t word) noexcept {
    return static_cast<State>(word >> kStateShift);
  }

  void unpin() const noexcept {
    const auto previous = word_.fetch_sub(1U, std::memory_order_release);
    assert((previous & kCountMask) != 0U);
    const bool last_closing_pin =
        (previous & kCountMask) == 1U && decode_state(previous) == State::kClosing;
    if (last_closing_pin) {
      word_.notify_all();
    }
  }

  [[nodiscard]] bool publish(State outcome) noexcept {
    auto expected = encode_state(State::kClosing);
    if (!word_.compare_exchange_strong(expected,
                                       encode_state(outcome),
                                       std::memory_order_release,
                                       std::memory_order_relaxed)) {
      return false;
    }
    auto* node = deferred_.exchange(&sealed_, std::memory_order_acq_rel);
    while (node != nullptr) {
      auto* const next = node->next;  // callback may destroy its embedded node
      node->on_outcome(*node, outcome);
      node = next;
    }
    return true;
  }

  mutable std::atomic<std::uint64_t> word_{};
  mutable std::atomic<DeferredRelease*> deferred_{};
  // A per-object sentinel has the same identity across Core and HoloMQ DSOs. Hidden inline static
  // sentinels could acquire distinct addresses in the two libraries.
  mutable DeferredRelease sealed_{};
  CUstream_st* const stream_;
  const int device_;
};

}  // namespace holomq::cuda
