/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/holomq_export.h>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>

#include <atomic>
#include <functional>
#include <memory>

namespace holomq {

class WaitSet;

/// @brief Wake one \ref WaitSet from another thread through a kernel-visible monotonic latch.
///
/// A trigger owns a private kernel wake object registered with the wait set that created it. It is
/// the only way to end a \ref WaitMode::KernelWait that no publisher will ever wake, so it is what
/// makes an application-driven shutdown prompt instead of deadline-bound.
///
/// The latch is monotonic: \ref fire may be called any number of times from any thread, and once
/// \ref is_set returns \c true it never returns \c false again. A trigger is therefore single-use
/// by construction; a caller that must wait again after a cancellation builds a new trigger.
///
/// \ref fire is the only operation in this library that is safe to call on a live wait set from a
/// thread other than the waiting thread. \ref WaitSet::attach, \ref WaitSet::attach_trigger,
/// \ref WaitSet::wait, and \ref WaitSet::ready_handles remain externally serialized on the waiting
/// thread.
///
/// @note A fired trigger is deliberately absent from \ref WaitSet::ready_handles. Readiness there
/// means "this subscriber may have a sample"; cancellation is a different question, answered by
/// \ref is_set. Keeping the two apart means an existing handle-dispatch loop cannot receive a
/// handle it does not recognize. So that such a loop still terminates, the wake is reported on the
/// other axis instead: \ref WaitSet::wait returns \ref WaitResult::cancelled.
/// @note Holoscan's Channel API exposes cancellation as a \c std::stop_token rather than as a
/// trigger, because a token is reusable and standard where this latch is monotonic and single-use.
/// The Channel layer owns a trigger internally and fires it from a \c std::stop_callback, so the
/// two idioms are the same mechanism seen from two levels. Use a trigger when waiting on a
/// \ref WaitSet directly; use a \c std::stop_token when waiting on a Channel subscriber.
class WaitTrigger {
 public:
  WaitTrigger(const WaitTrigger&) = delete;
  WaitTrigger& operator=(const WaitTrigger&) = delete;
  WaitTrigger(WaitTrigger&&) = delete;
  WaitTrigger& operator=(WaitTrigger&&) = delete;

  /// @brief Release the private kernel wake object.
  HOLOMQ_EXPORT ~WaitTrigger();

  /// @brief Latch this trigger and wake a wait that is currently blocked on its wait set.
  /// @return Empty success once the latch is published.
  /// @retval ErrorCode::kFailure The kernel wake object rejected the post. The latch is still
  /// published, so a waiter that checks \ref is_set before blocking still observes the request.
  /// @note This function does not throw and performs no allocation. It is safe to call
  /// concurrently with \ref WaitSet::wait running on another thread. Repeated calls are idempotent
  /// -- but idempotent is not reusable: the latch is monotonic and there is no reset, so every wait
  /// on this wait set returns immediately from here on. A caller that must wait again attaches a
  /// replacement trigger; see the class documentation.
  /// @warning A failed post is not a failed cancellation. The latch is published before the post,
  /// so a waiter that rechecks \ref is_set still observes the request and only the prompt wake is
  /// lost. Treat \ref ErrorCode::kFailure here as degraded latency, not as grounds for terminating
  /// a shutdown that is in fact under way.
  [[nodiscard]] HOLOMQ_EXPORT expected<void, Error> fire() noexcept;

  /// @brief Return whether this trigger has been latched.
  /// @return \c true once \ref fire has been called; otherwise \c false.
  /// @note This function does not throw. It uses acquire ordering, so a waiting thread observes
  /// every write sequenced before the firing thread's \ref fire. Check this before every block: it
  /// is what makes a \ref fire that lands between two waits impossible to miss.
  [[nodiscard]] bool is_set() const noexcept { return latched_.load(std::memory_order_acquire); }

 private:
  friend class WaitSet;

  class Impl;

  HOLOMQ_EXPORT explicit WaitTrigger(std::unique_ptr<Impl> impl) noexcept;

  /// @brief Create the private kernel wake object backing one trigger.
  [[nodiscard]] HOLOMQ_EXPORT static expected<std::shared_ptr<WaitTrigger>, Error> create();

  /// @brief Return the native wake descriptor for backend registration.
  [[nodiscard]] HOLOMQ_EXPORT int descriptor() const noexcept;

  /// @brief Consume every pending wake posted to the kernel object.
  ///
  /// The latch, not the descriptor, carries cancellation state, so draining is unconditional. That
  /// keeps a level-triggered kernel attachment from reporting readiness forever after one \ref
  /// fire.
  HOLOMQ_EXPORT void drain() noexcept;

  std::unique_ptr<Impl> impl_;
  std::atomic<bool> latched_{false};
};

/// @brief Retain one attached \ref WaitTrigger and detach it on destruction.
///
/// Mirrors \ref WaitSetAttachment. The trigger itself is reference counted, so \ref trigger may be
/// retained and fired by another thread after this attachment is destroyed; the wake simply has no
/// wait set left to interrupt.
class WaitTriggerAttachment {
 public:
  HOLOMQ_EXPORT WaitTriggerAttachment(WaitTriggerAttachment&& other) noexcept;
  HOLOMQ_EXPORT WaitTriggerAttachment& operator=(WaitTriggerAttachment&& other) noexcept;
  HOLOMQ_EXPORT ~WaitTriggerAttachment();

  WaitTriggerAttachment(const WaitTriggerAttachment&) = delete;
  WaitTriggerAttachment& operator=(const WaitTriggerAttachment&) = delete;

  /// @brief Return shared ownership of the attached trigger.
  /// @return A retained trigger that any thread may \ref WaitTrigger::fire.
  [[nodiscard]] std::shared_ptr<WaitTrigger> trigger() const noexcept { return trigger_; }

 private:
  friend class WaitSet;

  HOLOMQ_EXPORT WaitTriggerAttachment(std::shared_ptr<WaitTrigger> trigger,
                                      std::function<void()> detach_fn) noexcept;

  std::shared_ptr<WaitTrigger> trigger_;
  std::function<void()> detach_fn_;
};

}  // namespace holomq
