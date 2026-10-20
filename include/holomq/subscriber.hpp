/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/delivery_attribution.hpp>
#include <holomq/detail/transport/backend.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/inbound_message.hpp>
#include <holomq/readiness_handle.hpp>
#include <holomq/wait_strategy.hpp>

#include <cstdint>
#include <memory>
#include <utility>

namespace holomq {

template <typename Payload>
class SubscriberBuilder;

class WaitSet;

/// Subscribe-side port; the message type is the application payload only.
/// Adopt device memory via \ref Context::import_buffer on payload fields (e.g. \ref BufferHandle).
///
/// Move-only. The transport port is held in a \c std::shared_ptr so \ref WaitSet attachments keep
/// the port alive when this shell is moved or destroyed.
template <typename Payload>
class Subscriber {
  friend class SubscriberBuilder<Payload>;
  friend class WaitSet;

 public:
  Subscriber(Subscriber&&) noexcept = default;
  Subscriber& operator=(Subscriber&&) noexcept = default;
  Subscriber(const Subscriber&) = delete;
  Subscriber& operator=(const Subscriber&) = delete;

  /**
   * @brief Take the next queued sample without blocking.
   * @return An inbound message on success; \ref ErrorCode::kNotReady when no sample is queued,
   * \ref ErrorCode::kBackpressured when the borrowed-sample limit is reached, or a connection
   * error when the publisher cannot be reached or mapped.
   * @note Recoverable transport failures are returned as \ref Error, not thrown.
   */
  [[nodiscard]] auto try_take() -> expected<InboundMessage<Payload>, Error> {
    return take_from_port(port_->try_take());
  }

  /**
   * @brief Wait for and take a sample using the port default \ref WaitStrategy.
   * @return An inbound message on success, or a typed wait or transport error on failure.
   * @note Recoverable transport failures are returned as \ref Error, not thrown.
   */
  [[nodiscard]] auto wait_and_take() -> expected<InboundMessage<Payload>, Error> {
    return wait_and_take(port_->get_wait_strategy());
  }

  /**
   * @brief Wait for and take a sample using \p strategy.
   * @param strategy Wait behavior to use for this call.
   * @return An inbound message on success; \ref ErrorCode::kTimeout when the wait expires, \ref
   * ErrorCode::kInvalidArgument when this subscriber is already owned by a wait set or readiness
   * handle or when \ref WaitMode::KernelWait lacks event wake, or another typed transport error on
   * failure.
   * @note Retries internally after a spurious kernel wake with no queued sample. Recoverable
   * transport failures are returned as \ref Error, not thrown.
   */
  [[nodiscard]] auto wait_and_take(WaitStrategy strategy)
      -> expected<InboundMessage<Payload>, Error> {
    return take_from_port(port_->wait_and_take(strategy));
  }

  /**
   * @brief Set the default \ref WaitStrategy for \ref wait_and_take.
   * @param strategy Wait behavior used when no per-call strategy is passed.
   * @note This function does not throw.
   */
  void set_wait_strategy(const WaitStrategy strategy) noexcept {
    port_->set_wait_strategy(strategy);
  }

  /**
   * @brief Return whether the transport reports a sample available to \ref try_take.
   * @return `true` when a sample may be available, `false` when none is reported, or a connection
   * error when publisher connections cannot be refreshed.
   * @note Recoverable transport failures are returned as \ref Error, not thrown.
   */
  [[nodiscard]] auto has_samples() const -> expected<bool, Error> {
    auto ready = port_->has_samples();
    if (!ready.has_value()) {
      return make_unexpected(Error{ready.error()});
    }
    return ready.value();
  }

  /// @brief Return whether the service currently reports at least one publisher.
  /// @return `true` when a publisher is connected; otherwise `false`.
  /// @note This cold-path advisory observation does not prove that a sample is or will become
  /// available or that a publication will be delivered. It is intended for bounded
  /// peer-attachment checks, not receive readiness. A moved-from subscriber returns \c false.
  [[nodiscard]] bool has_publishers() const noexcept {
    return port_ != nullptr && port_->has_publishers();
  }

  /// @brief Return whether this subscriber was created with event wake enabled.
  /// @return True when the port holds a kernel listener, so \ref WaitMode::KernelWait is
  /// admissible for it.
  /// @note This function does not throw. Event wake is a frozen service-static fact, so callers
  /// can check admissibility before attempting a wait instead of interpreting a failure.
  [[nodiscard]] bool is_event_wake_enabled() const noexcept {
    return port_ != nullptr && port_->is_event_wake_enabled();
  }

  /// @brief Return whether a wait set or readiness handle currently owns this subscriber's
  /// listener.
  /// @return True while the port is attached to one exclusive wait owner.
  /// @note This is a snapshot for ownership checks, not receive readiness.
  [[nodiscard]] bool has_wait_owner() const noexcept {
    return port_ != nullptr && port_->is_attached_to_wait_set();
  }

  /// @brief Return whether a readiness handle specifically owns this subscriber's listener.
  /// @return True while one readiness handle retains the port.
  /// @note This ownership snapshot exists for composite Channel diagnostics.
  [[nodiscard]] bool has_readiness_owner() const noexcept {
    return port_ != nullptr && port_->is_attached_to_readiness_handle();
  }

  /// @brief Claim this subscriber's readiness descriptor for an external event loop.
  /// @return A move-only readiness handle, or a checked capability or ownership error.
  /// @retval ErrorCode::kClosed This subscriber was moved from.
  /// @retval ErrorCode::kNotSupported Native descriptor readiness is unavailable on this platform,
  /// or this subscriber was created without event wake.
  /// @retval ErrorCode::kInvalidArgument This subscriber is already attached to a wait set or owns
  /// another readiness handle.
  /// @note The handle retains the transport port and releases its claim on destruction. While it
  /// is live, blocking subscriber waits and \ref WaitSet::attach are rejected.
  [[nodiscard]] expected<ReadinessHandle, Error> readiness() {
    if (port_ == nullptr) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    auto source = port_->acquire_readiness();
    if (!source) {
      if (source.error() == ErrorCode::kInvalidArgument) {
        return make_unexpected_nothrow<ReadinessHandle>(
            source.error(),
            has_readiness_owner() ? "subscriber is already owned by a readiness handle"
                                  : "subscriber is already owned by a wait set");
      }
      return make_unexpected(Error{source.error()});
    }
    auto handle = ReadinessHandle::create(
        port_,
        source->descriptor,
        source->initially_ready,
        [](void* const owner) noexcept -> expected<bool, Error> {
          auto acknowledged = static_cast<detail::transport::SubscriberPort<Payload>*>(owner)
                                  ->acknowledge_readiness();
          if (!acknowledged) {
            return make_unexpected(Error{acknowledged.error()});
          }
          return *acknowledged;
        },
        [](void* const owner) noexcept {
          static_cast<detail::transport::SubscriberPort<Payload>*>(owner)->release_readiness();
        });
    return handle;
  }

  /// @brief Return the backend-independent recipient id used by per-recipient delivery
  /// callbacks.
  /// @return Recipient id for attribution-capable backends; otherwise a default id.
  [[nodiscard]] DeliveryRecipientId recipient_id() const { return port_->recipient_id(); }

 private:
  explicit Subscriber(std::shared_ptr<detail::transport::SubscriberPort<Payload>> port) noexcept
      : port_(std::move(port)) {}

  [[nodiscard]] static expected<InboundMessage<Payload>, Error> take_from_port(
      expected<detail::transport::InboundMessage<Payload>, Error> message_or) {
    if (!message_or.has_value()) {
      return holomq::forward_error(message_or);
    }
    return InboundMessage<Payload>(std::move(message_or).value());
  }

  [[nodiscard]] std::shared_ptr<detail::transport::SubscriberPort<Payload>> port_for_wait_set()
      const noexcept {
    return port_;
  }

  std::shared_ptr<detail::transport::SubscriberPort<Payload>> port_;
};

}  // namespace holomq
