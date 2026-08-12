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

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>

#include <holomq/detail/transport/wait_set_backend.hpp>

#include <holomq/subscriber.hpp>

namespace holomq {

class Context;

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

  /// Default \ref WaitStrategy for \ref wait and \ref wait_for_ready when no per-call strategy is
  /// passed.
  HOLOMQ_EXPORT void set_wait_strategy(WaitStrategy strategy) noexcept;

  /**
   * @brief Block using the default \ref WaitStrategy until an attachment may have work or the wait
   * fails.
   * @return `expected<WaitResult, Error>` holding \ref WaitResult::ready or \ref
   * WaitResult::timeout on success, or a typed global wait error on failure.
   * @note This function does not throw.
   */
  [[nodiscard]] HOLOMQ_EXPORT auto wait() noexcept -> expected<WaitResult, Error>;

  /**
   * @brief Block using \p strategy until an attachment may have work or the wait fails.
   * @param strategy Wait behavior to use for this call.
   * @return `expected<WaitResult, Error>` holding \ref WaitResult::ready or \ref
   * WaitResult::timeout on success, or a typed global wait error on failure.
   * @note This function does not throw.
   */
  [[nodiscard]] HOLOMQ_EXPORT auto wait(WaitStrategy strategy) noexcept
      -> expected<WaitResult, Error>;

  /**
   * @brief Return handles with pending samples after the last successful \ref wait.
   * @return A span of ready attachment handles.
   * @note The span may be empty after a kernel spurious wake.
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
  [[nodiscard]] HOLOMQ_EXPORT std::size_t capacity() const noexcept;

  /// Number of subscribers currently attached.
  [[nodiscard]] HOLOMQ_EXPORT std::size_t attachment_count() const noexcept;

 private:
  friend class Context;

  HOLOMQ_EXPORT explicit WaitSet(
      std::shared_ptr<detail::transport::WaitSetBackend> backend) noexcept;

  // Keep the detach callback inside libholomq. The public attach template is instantiated by
  // downstream payload types, while detail transport methods intentionally remain hidden.
  [[nodiscard]] HOLOMQ_EXPORT WaitSetAttachment make_attachment(WaitSetHandle handle) const;

  [[nodiscard]] HOLOMQ_EXPORT auto wait_impl(const WaitStrategy& strategy) noexcept
      -> expected<WaitResult, Error>;

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
