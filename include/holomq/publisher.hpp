/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/byte_slice.hpp>
#include <holomq/detail/publisher_sample_release_registry.hpp>
#include <holomq/detail/transport/backend.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/logger.hpp>
#include <holomq/outbound_message.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace holomq {

template <typename Payload>
class PublisherBuilder;

namespace detail {

template <typename Payload>
struct PublisherImpl {
  transport::PublisherPort<Payload> port;
  std::shared_ptr<PublisherSampleReleaseRegistry> release_registry;
  std::uint32_t expected_recipient_count{};
};

/// Report an unexpected publisher failure without promoting ordinary flow-control outcomes.
inline void log_publisher_send_error(const Error& error) noexcept {
  try {
    if (error.code == ErrorCode::kNotReady || error.code == ErrorCode::kBackpressured) {
      HOLOMQ_LOG_TRACE("publisher send deferred: {}{}{}",
                       error_code_name(error.code),
                       error.message.empty() ? "" : ": ",
                       error.message);
      return;
    }
    HOLOMQ_LOG_ERROR("publisher send failed: {}{}{}",
                     error_code_name(error.code),
                     error.message.empty() ? "" : ": ",
                     error.message);
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Logging is diagnostic-only. A throwing user-installed sink must not escape the noexcept
    // publisher send path or mask the original transport error.
  }
}

}  // namespace detail

/// Publish–subscribe publisher; the message type is the application payload only.
/// GPU/device memory is managed externally via \ref BufferPool; embed \ref BufferHandle in \p
/// Payload.
///
/// Move-only. Transport state is held in a \c std::shared_ptr so dependent callbacks can outlive
/// shell moves.
template <typename Payload>
class Publisher {
  friend class PublisherBuilder<Payload>;

 public:
  Publisher(const Publisher&) = delete;
  Publisher(Publisher&&) noexcept = default;
  Publisher& operator=(const Publisher&) = delete;
  Publisher& operator=(Publisher&&) noexcept = default;
  ~Publisher() = default;

  /**
   * @brief Refresh publisher connection state.
   * @return Success when connections are refreshed, \ref ErrorCode::kNotReady when a connection
   * cannot be established, \ref ErrorCode::kStaleEndpoint when a peer data segment cannot be
   * mapped, or another typed connection error.
   * @note Recoverable transport failures are returned as \ref Error, not thrown.
   */
  [[nodiscard]] auto update_connections() noexcept -> expected<void, Error> {
    return impl_->port.update_connections();
  }

  /**
   * @brief Report whether the live publisher service currently contains subscribers.
   * @return `true` when at least one subscriber is connected; otherwise `false`.
   * @note The count comes from the backend's live dynamic service configuration. Call and check
   * \ref update_connections first when the publisher must map newly discovered peer segments
   * before publication. This function does not throw.
   */
  [[nodiscard]] bool has_subscribers() const noexcept { return number_of_subscribers() > 0U; }

  /**
   * @brief Return the subscriber count from the live publisher service configuration.
   * @return The number of currently connected subscribers.
   * @note Reading the count does not map newly discovered peer segments. Call and check
   * \ref update_connections first when preparing to publish to newly connected peers. This
   * function does not throw.
   */
  [[nodiscard]] std::uint64_t number_of_subscribers() const noexcept {
    return impl_->port.number_of_subscribers();
  }

  /**
   * @brief Borrow an outbound \ref OutboundMessage.
   * @return A message loan on success, \ref ErrorCode::kBackpressured when no loan slot or
   * transport memory is available, \ref ErrorCode::kInvalidArgument when the payload exceeds the
   * transport limit, or \ref ErrorCode::kFailure for an internal transport failure.
   * @note Failures are not logged and are returned as \ref Error, not thrown.
   */
  [[nodiscard]] auto loan() -> expected<OutboundMessage<Payload>, Error>
    requires(!kIsByteSlice<Payload>)
  {  // NOLINT(whitespace/braces)
    auto loaned = impl_->port.loan();
    if (!loaned.has_value()) {
      return holomq::forward_error(std::move(loaned));
    }
    return OutboundMessage<Payload>(std::move(loaned).value());
  }

  /**
   * @brief Borrow an outbound byte-slice frame of \p length bytes.
   * @param length Frame length; at most \ref PubSubServiceOptions::byte_slice_capacity.
   * @return A message loan whose payload is exactly \p length bytes on success, \ref
   * ErrorCode::kInvalidArgument when \p length exceeds the capacity, \ref
   * ErrorCode::kBackpressured when no loan slot is available, or \ref ErrorCode::kFailure for an
   * internal transport failure.
   * @note The frame occupies a full-capacity slot of the publisher's preallocated pool; the pool
   * never grows. Failures are not logged and are returned as \ref Error, not thrown.
   */
  [[nodiscard]] auto loan(const std::size_t length) -> expected<OutboundMessage<Payload>, Error>
    requires kIsByteSlice<Payload>
  {  // NOLINT(whitespace/braces)
    auto loaned = impl_->port.loan(static_cast<std::uint64_t>(length));
    if (!loaned.has_value()) {
      return holomq::forward_error(std::move(loaned));
    }
    return OutboundMessage<Payload>(std::move(loaned).value());
  }

  /**
   * @brief Publish \p message.
   * @param message Message loan to publish. The loan is consumed by this call.
   * @return The number of subscribers that received the sample, \ref
   * ErrorCode::kIncompleteFanout before submission when required recipients are absent, or a typed
   * transport error.
   * @warning When the payload contains a \ref BufferHandle, the caller must keep the owner \ref
   * Buffer alive until the transport releases the sample. Prefer the callback overload of \ref
   * send and move the owner into \c on_released unless its lifetime is managed separately.
   * @note This function does not throw.
   */
  [[nodiscard]] auto send(OutboundMessage<Payload>&& message) noexcept -> expected<int, Error> {
    if (!has_expected_recipients()) {
      return make_unexpected(Error{ErrorCode::kIncompleteFanout});
    }
    auto sent = impl_->port.send(std::move(message).release_loan());
    if (!sent.has_value()) {
      detail::log_publisher_send_error(sent.error());
    }
    return sent;
  }

  /**
   * @brief Publish \p message and run \p on_released when the transport releases its sample slot.
   * @tparam Fn Callable invoked when the transport releases the sample slot.
   * @param message Message loan to publish. The loan is consumed by this call.
   * @param on_released Callback retained until the transport releases the sample slot.
   * @return The number of subscribers that received the sample, \ref
   * ErrorCode::kIncompleteFanout before submission when required recipients are absent, \ref
   * ErrorCode::kNotImplemented when release callbacks were disabled at publisher creation, or
   * another typed transport error.
   * @throws std::exception if registering the release callback propagates an exception.
   * @note Move \ref Buffer and other state into \p on_released to extend owner leases. The callback
   * remains registered after a send error because recipients may already hold the sample. When
   * required recipients are absent, the unsent loan is retired and the callback is invoked
   * synchronously before this function returns \ref ErrorCode::kIncompleteFanout.
   */
  template <typename Fn>
  [[nodiscard]] auto send(OutboundMessage<Payload>&& message, Fn&& on_released)
      -> expected<int, Error> {
    if (!impl_->release_registry) {
      if (!has_expected_recipients()) {
        return make_unexpected(Error{ErrorCode::kIncompleteFanout});
      }
      HOLOMQ_LOG_ERROR("publisher send failed: sample release callbacks are disabled");
      return make_unexpected(
          Error{ErrorCode::kNotImplemented, "sample release callbacks are disabled"});
    }
    const void* const payload_key = message.release_key();
    if (!impl_->release_registry->register_pending(payload_key, std::forward<Fn>(on_released))) {
      HOLOMQ_LOG_ERROR("publisher send failed: release callback already pending for payload");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!has_expected_recipients()) {
      {
        // Retire the unsent transport loan before releasing the callback-owned backing. This
        // preserves the same callback handoff invariant as every post-submit send result.
        [[maybe_unused]] auto rejected_loan = std::move(message).release_loan();
      }
      impl_->release_registry->dispatch(payload_key);
      return make_unexpected(Error{ErrorCode::kIncompleteFanout});
    }
    auto sent = impl_->port.send(std::move(message).release_loan());
    if (!sent.has_value()) {
      detail::log_publisher_send_error(sent.error());
    }
    return sent;
  }

 private:
  [[nodiscard]] bool has_expected_recipients() const noexcept {
    return impl_->expected_recipient_count == 0U ||
           number_of_subscribers() >= impl_->expected_recipient_count;
  }

  explicit Publisher(std::shared_ptr<detail::PublisherImpl<Payload>> impl) noexcept
      : impl_(std::move(impl)) {}

  std::shared_ptr<detail::PublisherImpl<Payload>> impl_;
};

}  // namespace holomq
