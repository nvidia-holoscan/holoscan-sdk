/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <utility>

namespace holoscan {

class NotificationSender;

namespace detail {

class ValueLifecycle {
 public:
  constexpr ValueLifecycle() noexcept = default;

  ValueLifecycle(const ValueLifecycle& other) noexcept
      : state_(other.state_), retain_(other.retain_), release_(other.release_) {
    if (state_ != nullptr) {
      retain_(state_);
    }
  }

  ValueLifecycle& operator=(const ValueLifecycle& other) noexcept {
    if (this == &other) {
      return *this;
    }
    auto replacement = ValueLifecycle(other);
    swap(replacement);
    return *this;
  }

  ValueLifecycle(ValueLifecycle&& other) noexcept
      : state_(std::exchange(other.state_, nullptr)),
        retain_(std::exchange(other.retain_, nullptr)),
        release_(std::exchange(other.release_, nullptr)) {}

  ValueLifecycle& operator=(ValueLifecycle&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    reset();
    state_ = std::exchange(other.state_, nullptr);
    retain_ = std::exchange(other.retain_, nullptr);
    release_ = std::exchange(other.release_, nullptr);
    return *this;
  }

  ~ValueLifecycle() noexcept { reset(); }

  [[nodiscard]] explicit constexpr operator bool() const noexcept { return state_ != nullptr; }

 private:
  using RetainFunction = void (*)(void*) noexcept;
  using ReleaseFunction = void (*)(void*) noexcept;

  explicit constexpr ValueLifecycle(void* state, RetainFunction retain,
                                    ReleaseFunction release) noexcept
      : state_(state), retain_(retain), release_(release) {}

  void reset() noexcept {
    void* const state = std::exchange(state_, nullptr);
    ReleaseFunction const release = std::exchange(release_, nullptr);
    retain_ = nullptr;
    if (state != nullptr) {
      release(state);
    }
  }

  void swap(ValueLifecycle& other) noexcept {
    std::swap(state_, other.state_);
    std::swap(retain_, other.retain_);
    std::swap(release_, other.release_);
  }

  void* state_{};
  RetainFunction retain_{};
  ReleaseFunction release_{};

  friend class ::holoscan::NotificationSender;
  friend struct ReadinessSourceAccess;
};

}  // namespace detail
}  // namespace holoscan
