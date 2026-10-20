/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/limits.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>

namespace holomq::detail {

/// @brief Per-publisher registry of `void()` callbacks keyed by sample payload address.
///
/// iceoryx2 invokes @ref dispatch from `SampleReleasedHandler` when a transport sample is
/// released. `on_released` callables must not throw because they may run on transport threads.
class PublisherSampleReleaseRegistry {
 public:
  /// @brief Construct a registry with reusable storage for the expected in-flight samples.
  /// @param initial_capacity Number of callback slots to prepare before the send hot path.
  explicit PublisherSampleReleaseRegistry(
      const std::size_t initial_capacity = static_cast<std::size_t>(k_max_in_flight) + 1U)
      : slots_(initial_capacity == 0U ? 1U : initial_capacity) {}

  /// @brief Store a release callback until @ref dispatch.
  /// @tparam Fn Callable type invocable with no arguments.
  /// @param payload_key Payload address used as the in-flight sample key.
  /// @param on_released Callback to invoke when the transport releases the sample.
  /// @return `false` when @p payload_key is already pending; otherwise `true`.
  template <typename Fn>
  [[nodiscard]] bool register_pending(const void* payload_key, Fn&& on_released) {
    using Callback = std::decay_t<Fn>;
    const std::lock_guard lock(mutex_);
    Slot* available_slot = nullptr;
    for (auto& slot : slots_) {
      if (slot.state == SlotState::kPending && slot.payload_key == payload_key) {
        return false;
      }
      if (available_slot == nullptr && slot.state == SlotState::kAvailable) {
        available_slot = &slot;
      }
    }
    if (available_slot == nullptr) {
      slots_.emplace_back();
      available_slot = &slots_.back();
    }

    if (available_slot->entry == nullptr ||
        available_slot->entry->callback_type() != callback_type_tag<Callback>()) {
      // Prepare every currently available slot for this call site's callback type. The first send
      // pays this setup cost; subsequent in-flight sends reuse the entries without allocating.
      for (auto& slot : slots_) {
        if (slot.state == SlotState::kAvailable &&
            (slot.entry == nullptr ||
             slot.entry->callback_type() != callback_type_tag<Callback>())) {
          slot.entry = std::make_unique<EntryImpl<Callback>>();
        }
      }
    }
    static_cast<EntryImpl<Callback>*>(available_slot->entry.get())
        ->replace(std::forward<Fn>(on_released));
    available_slot->payload_key = payload_key;
    available_slot->state = SlotState::kPending;
    pending_count_.fetch_add(1U, std::memory_order_release);
    return true;
  }

  /// @brief Dispatch and remove the release callback for a payload address.
  /// @param payload_ptr Payload address reported by the transport release notification.
  void dispatch(const void* payload_ptr) {
    Slot* matched_slot = nullptr;
    {
      const std::lock_guard lock(mutex_);
      if (pending_count_.load(std::memory_order_relaxed) == 0U) {
        return;
      }
      for (auto& slot : slots_) {
        if (slot.state == SlotState::kPending && slot.payload_key == payload_ptr) {
          matched_slot = &slot;
          break;
        }
      }
      if (matched_slot == nullptr) {
        return;
      }
      matched_slot->state = SlotState::kDispatching;
      pending_count_.fetch_sub(1U, std::memory_order_release);
    }

    try {
      matched_slot->entry->invoke();
    } catch (...) {
      release_slot(*matched_slot);
      throw;
    }
    release_slot(*matched_slot);
  }

  /// @brief Return the number of pending release callbacks.
  /// @return Count of callbacks currently registered in this publisher registry.
  [[nodiscard]] std::size_t pending_count() const noexcept {
    return pending_count_.load(std::memory_order_acquire);
  }

 private:
  enum class SlotState : std::uint8_t {
    kAvailable,
    kPending,
    kDispatching,
  };

  template <typename Fn>
  [[nodiscard]] static const void* callback_type_tag() noexcept {
    static constexpr std::byte tag{};
    return &tag;
  }

  struct Entry {
    virtual ~Entry() = default;
    [[nodiscard]] virtual const void* callback_type() const noexcept = 0;
    virtual void invoke() = 0;
  };

  template <typename Fn>
  struct EntryImpl final : Entry {
    std::optional<Fn> fn;

    EntryImpl() = default;

    template <typename Callable>
    void replace(Callable&& callable) {
      fn.reset();
      fn.emplace(std::forward<Callable>(callable));
    }

    [[nodiscard]] const void* callback_type() const noexcept override {
      return callback_type_tag<Fn>();
    }

    void invoke() override {
      // A pending slot owns an engaged callback; dispatch marks it in flight before invoking.
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      Fn callback = std::move(fn).value();
      fn.reset();
      callback();
    }
  };

  struct Slot {
    const void* payload_key{};
    std::unique_ptr<Entry> entry;
    SlotState state{SlotState::kAvailable};
  };

  void release_slot(Slot& slot) {
    const std::lock_guard lock(mutex_);
    slot.payload_key = nullptr;
    slot.state = SlotState::kAvailable;
  }

  std::mutex mutex_;
  std::deque<Slot> slots_;
  std::atomic<std::size_t> pending_count_{0U};
};

}  // namespace holomq::detail
