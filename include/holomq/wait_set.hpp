/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/holomq_export.h>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/wait_set_handle.hpp>
#include <holomq/wait_strategy.hpp>
#include <holomq/wait_trigger.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>

#include <holomq/detail/transport/wait_set_backend.hpp>

#include <holomq/subscriber.hpp>

namespace holomq {

class Context;

namespace detail {
class WaitSetRuntimeAccess;
}

/// RAII detach token returned by \ref WaitSet::attach. Destroy to unregister the subscriber from
/// the wait set.
class WaitSetAttachment {
 public:
  HOLOMQ_EXPORT WaitSetAttachment(WaitSetAttachment&& other) noexcept;
  HOLOMQ_EXPORT WaitSetAttachment& operator=(WaitSetAttachment&& other) noexcept;
  HOLOMQ_EXPORT ~WaitSetAttachment();

  WaitSetAttachment(const WaitSetAttachment&) = delete;
  WaitSetAttachment& operator=(const WaitSetAttachment&) = delete;

  [[nodiscard]] WaitSetHandle handle() const noexcept { return handle_; }

 private:
  friend class WaitSet;

  WaitSetAttachment() = delete;

  HOLOMQ_EXPORT WaitSetAttachment(WaitSetHandle handle,
                                  std::function<void(WaitSetHandle)> detach_fn) noexcept;

  WaitSetHandle handle_{};
  std::function<void(WaitSetHandle)> detach_fn_;
};

/// Multiplexed blocking wait over multiple \ref Subscriber ports.
///
/// On the iceoryx2 backend, \ref WaitMode::KernelWait requires at least one attached subscriber
/// created with event wake enabled (default; kernel \c Listener registered on the wait set).
/// \ref WaitMode::Return and \ref WaitMode::TimedWait use userspace poll and sleep only.
///
/// \ref attach shares ownership of subscriber transport ports so an attached \ref Subscriber may be
/// moved.
///
/// \note \ref WaitSet may be moved or destroyed while \ref WaitSetAttachment objects remain;
/// attachments keep the backend alive until they are destroyed.
class WaitSet {
 public:
  WaitSet(WaitSet&&) noexcept = default;
  WaitSet& operator=(WaitSet&&) noexcept = default;
  ~WaitSet() = default;

  WaitSet(const WaitSet&) = delete;
  WaitSet& operator=(const WaitSet&) = delete;

  /// Attach \p subscriber for \ref wait / \ref ready_handles. The attachment shares the subscriber
  /// transport port until the returned \ref WaitSetAttachment is destroyed.
  template <typename Payload>
  [[nodiscard]] expected<WaitSetAttachment, Error> attach(Subscriber<Payload>& subscriber);

  /// @brief Create and attach a cross-thread wake trigger to this wait set.
  ///
  /// The returned trigger is the only supported way to end a \ref WaitMode::KernelWait that no
  /// publication will wake. It occupies one kernel-notification slot but is not a subscriber, so it
  /// is excluded from \ref attachment_count and cannot satisfy the event-wake capability that
  /// \ref WaitMode::KernelWait requires.
  ///
  /// @return An attachment retaining the trigger, or a checked setup error.
  /// @retval ErrorCode::kResourceExhausted The backend kernel-notification capacity is exhausted.
  /// @retval ErrorCode::kFailure The private kernel wake object could not be created or registered.
  /// @note The attachment retains the backend, exactly as \ref WaitSetAttachment does, so
  /// destroying this wait set first is safe and the attachment's detach still runs; the attachment
  /// is merely inert afterwards because nothing can wait on it. A fired trigger never appears in
  /// \ref ready_handles; \ref wait reports \ref WaitResult::cancelled and
  /// \ref WaitTrigger::is_set stays true, so check the latch before each block.
  /// @note This is HoloMQ's cancellation primitive and it is single-use by construction. Callers on
  /// the Holoscan Channel API do not use it directly: those waits take a \c std::stop_token, which
  /// is reusable and standard, and the Channel layer owns a trigger internally to translate a stop
  /// request into this wake. Reach for a trigger only when waiting on a \ref WaitSet directly.
  [[nodiscard]] HOLOMQ_EXPORT expected<WaitTriggerAttachment, Error> attach_trigger();

  /// Default \ref WaitStrategy for \ref wait and \ref wait_for_ready when no per-call strategy is
  /// passed.
  HOLOMQ_EXPORT void set_wait_strategy(WaitStrategy strategy) noexcept;

  /**
   * @brief Block using the default \ref WaitStrategy until an attachment may have work or the wait
   * fails.
   * @return `expected<WaitResult, Error>` holding \ref WaitResult::ready, \ref
   * WaitResult::timeout, or \ref WaitResult::cancelled on success, or a typed global wait error on
   * failure.
   * @note This function does not throw.
   * @note \ref WaitResult::cancelled means an attached \ref WaitTrigger was fired and
   * \ref ready_handles is empty. A dispatch loop must treat it as a reason to stop rather than
   * looping on the empty span.
   */
  [[nodiscard]] HOLOMQ_EXPORT auto wait() noexcept -> expected<WaitResult, Error>;

  /**
   * @brief Block using \p strategy until an attachment may have work or the wait fails.
   * @param strategy Wait behavior to use for this call.
   * @return `expected<WaitResult, Error>` holding \ref WaitResult::ready, \ref
   * WaitResult::timeout, or \ref WaitResult::cancelled on success, or a typed global wait error on
   * failure.
   * @note This function does not throw.
   * @note \ref WaitResult::cancelled is reported only for \ref WaitMode::KernelWait, which is the
   * only mode that blocks on the trigger's descriptor.
   */
  [[nodiscard]] HOLOMQ_EXPORT auto wait(WaitStrategy strategy) noexcept
      -> expected<WaitResult, Error>;

  /**
   * @brief Return handles with pending samples after the last successful \ref wait.
   * @return A span of ready attachment handles.
   * @note The span may be empty after a kernel spurious wake, and is always empty when \ref wait
   * reported \ref WaitResult::cancelled. A trigger never contributes a handle, so an empty span is
   * not by itself a reason to wait again -- check the \ref WaitResult first.
   */
  [[nodiscard]] HOLOMQ_EXPORT std::span<const WaitSetHandle> ready_handles() const noexcept;

  /**
   * @brief Wait using the default strategy.
   * @return The same `expected<WaitResult, Error>` result as \ref wait().
   * @see wait()
   * @note This function does not throw.
   */
  [[nodiscard]] auto wait_for_ready() noexcept -> expected<WaitResult, Error> { return wait(); }

  /**
   * @brief Wait using \p strategy.
   * @param strategy Wait behavior to use for this call.
   * @return The same `expected<WaitResult, Error>` result as \ref wait(WaitStrategy).
   * @see wait(WaitStrategy)
   * @note This function does not throw.
   */
  [[nodiscard]] auto wait_for_ready(WaitStrategy strategy) noexcept -> expected<WaitResult, Error> {
    return wait(strategy);
  }

  /// Maximum kernel-notification slots for the iceoryx2 WaitSet backend.
  /// @note Counts every kernel-notification slot, including any occupied by a cancellation
  /// trigger, whereas \ref attachment_count counts subscribers only. `capacity() -
  /// attachment_count()` therefore overstates the free slots by the number of attached triggers.
  [[nodiscard]] HOLOMQ_EXPORT std::size_t capacity() const noexcept;

  /// Number of subscribers currently attached.
  [[nodiscard]] HOLOMQ_EXPORT std::size_t attachment_count() const noexcept;

 private:
  friend class Context;
  friend class detail::WaitSetRuntimeAccess;

  HOLOMQ_EXPORT explicit WaitSet(
      std::shared_ptr<detail::transport::WaitSetBackend> backend) noexcept;

  // Keep the detach callback inside libholomq. The public attach template is instantiated by
  // downstream payload types, while detail transport methods intentionally remain hidden.
  [[nodiscard]] HOLOMQ_EXPORT WaitSetAttachment make_attachment(WaitSetHandle handle) const;

  [[nodiscard]] HOLOMQ_EXPORT auto wait_impl(const WaitStrategy& strategy) noexcept
      -> expected<WaitResult, Error>;

  [[nodiscard]] HOLOMQ_EXPORT auto wait_impl_assuming_no_ready(
      const WaitStrategy& strategy) noexcept -> expected<WaitResult, Error>;

  std::shared_ptr<detail::transport::WaitSetBackend> backend_;
  WaitStrategy default_strategy_{};
};

template <typename Payload>
expected<WaitSetAttachment, Error> WaitSet::attach(Subscriber<Payload>& subscriber) {
  auto handle_or = backend_->attach(subscriber.port_for_wait_set());
  if (!handle_or.has_value()) {
    return holomq::forward_error(handle_or);
  }

  const WaitSetHandle handle = handle_or.value();
  return make_attachment(handle);
}

}  // namespace holomq
