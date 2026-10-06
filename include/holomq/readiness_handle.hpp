/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/holomq_export.h>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>

#include <memory>

namespace holomq {

template <typename Payload>
class Subscriber;

/// @brief Own one subscriber's externally pollable readiness descriptor.
///
/// A handle claims the subscriber's single wait-owner slot until explicit detach or destruction.
/// On Linux, its descriptor is suitable for level-triggered \c poll, \c select, or \c epoll
/// integration and becomes readable when the subscriber may have a sample. Readability is a hint:
/// a later receive may still report \ref ErrorCode::kNotReady. Other platforms retain this API for
/// source and link compatibility, but subscriber acquisition reports \ref ErrorCode::kNotSupported.
///
/// @note The safe consumer order after a readable result is \ref acknowledge followed by repeated
/// \ref Subscriber::try_take calls until \ref ErrorCode::kNotReady. A publication racing that
/// receive drain leaves a new wake pending for the next event-loop iteration.
/// @note Move-only. The handle retains the transport port, so it remains safe if the subscriber
/// shell is moved or destroyed. Destruction releases the wait-owner claim.
/// @warning Lifetime independence does not imply concurrent-operation safety. Acquisition,
/// acknowledgement, explicit detach, destruction, and operations on the associated subscriber must
/// be externally serialized.
class ReadinessHandle {
 public:
  HOLOMQ_EXPORT ReadinessHandle(ReadinessHandle&& other) noexcept;
  HOLOMQ_EXPORT ReadinessHandle& operator=(ReadinessHandle&& other) noexcept;
  HOLOMQ_EXPORT ~ReadinessHandle();

  ReadinessHandle(const ReadinessHandle&) = delete;
  ReadinessHandle& operator=(const ReadinessHandle&) = delete;

  /// @brief Return the native descriptor observed by an external event loop.
  /// @return A non-negative descriptor while live, or -1 after a move or detach.
  /// @warning Do not close the returned descriptor. It is borrowed from this handle.
  [[nodiscard]] HOLOMQ_EXPORT int descriptor() const noexcept;

  /// @brief Drain every wake that preceded this call.
  /// @return Empty success, or a checked listener error.
  /// @retval ErrorCode::kClosed This handle was moved from or detached.
  /// @note The descriptor is re-armed when a sample is still queued, so acknowledgement before
  /// the nonblocking receive drain cannot strand work behind a private control operation.
  /// @note A wake arriving concurrently after its source has been drained remains pending.
  [[nodiscard]] HOLOMQ_EXPORT expected<void, Error> acknowledge() noexcept;

  /// @brief Release the wait-owner claim and close this handle's descriptors.
  /// @note Idempotent. After detach, \ref descriptor returns -1 and \ref acknowledge returns
  /// \ref ErrorCode::kClosed. The associated subscriber may use another wait owner immediately.
  HOLOMQ_EXPORT void detach() noexcept;

 private:
  template <typename Payload>
  friend class Subscriber;

  class Impl;

  using AcknowledgeFn = expected<bool, Error> (*)(void*) noexcept;
  using DetachFn = void (*)(void*) noexcept;

  ReadinessHandle() noexcept = default;

  [[nodiscard]] HOLOMQ_EXPORT static expected<ReadinessHandle, Error> create(
      std::shared_ptr<void> owner, int source_descriptor, bool initially_ready,
      AcknowledgeFn acknowledge, DetachFn detach);

  std::unique_ptr<Impl> impl_;
};

}  // namespace holomq
