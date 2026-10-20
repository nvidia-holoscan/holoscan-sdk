/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include <memory>

namespace holoscan {
namespace channel_internal {
class ChannelReadinessHandleAccess;
}  // namespace channel_internal

/// @brief Own one Channel subscriber's externally pollable readiness descriptor.
///
/// A handle claims the subscriber's single wait-owner slot until explicit detach or destruction.
/// On Linux, its descriptor is suitable for level-triggered \c poll, \c select, or \c epoll
/// integration and becomes readable when any data, control, or terminal lane may have work.
/// Readability is a hint: a later receive may still report \ref ErrorCode::kNotReady after a lossy
/// replacement, private control operation, or competing consumer wins the race. Other platforms
/// retain this API for source and link compatibility, but subscriber acquisition reports \ref
/// ErrorCode::kNotSupported.
///
/// @note The safe consumer order after a readable result is \ref acknowledge followed by repeated
/// nonblocking receives until \ref ErrorCode::kNotReady. A wake arriving during that receive drain
/// remains pending for the next event-loop iteration.
/// @note Move-only. The handle retains its transport ports independently of the subscriber facade;
/// destruction detaches every lane and releases the subscriber for a later wait or handle.
/// @warning Lifetime independence does not imply concurrent-operation safety. Acquisition,
/// acknowledgement, explicit detach, destruction, and operations on the associated subscriber must
/// be externally serialized.
class ChannelReadinessHandle {
 public:
  HOLOSCAN_CORE_EXPORT ChannelReadinessHandle(ChannelReadinessHandle&& other) noexcept;
  HOLOSCAN_CORE_EXPORT ChannelReadinessHandle& operator=(ChannelReadinessHandle&& other) noexcept;
  HOLOSCAN_CORE_EXPORT ~ChannelReadinessHandle();

  ChannelReadinessHandle(const ChannelReadinessHandle&) = delete;
  ChannelReadinessHandle& operator=(const ChannelReadinessHandle&) = delete;

  /// @brief Return the native descriptor observed by an external event loop.
  /// @return A non-negative descriptor while live, or -1 after a move or detach.
  /// @warning Do not close the returned descriptor. It is borrowed from this handle.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT int descriptor() const noexcept;

  /// @brief Drain every lane wake that preceded this call.
  /// @return Empty success, or the first checked listener error after all lanes are drained.
  /// @retval ErrorCode::kClosed This handle was moved from or detached.
  /// @note A source that still has queued work remains readable as a level-triggered hint.
  /// @note A wake arriving after its lane has been drained remains pending.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> acknowledge() noexcept;

  /// @brief Release every retained lane and close this handle's descriptors.
  /// @note Idempotent. After detach, \ref descriptor returns -1 and \ref acknowledge returns
  /// \ref ErrorCode::kClosed. The associated subscriber may use another wait owner immediately.
  HOLOSCAN_CORE_EXPORT void detach() noexcept;

 private:
  friend class channel_internal::ChannelReadinessHandleAccess;

  class Impl;
  ChannelReadinessHandle() noexcept = default;

  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan
