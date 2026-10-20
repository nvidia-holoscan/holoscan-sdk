/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/detail/transport/backend.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/logger.hpp>
#include <holomq/pubsub_options.hpp>
#include <holomq/subscriber.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace holomq {

class Context;

class SubscriberServiceBuilder;

/// @brief Configure one typed subscriber before creating its backend endpoint.
///
/// The move-only builder borrows the parent \ref Context transport and must be used while that
/// context is alive. Builder mutation and \ref create are setup-only and not thread-safe.
/// @tparam Payload Standard-layout, trivially copyable transport payload, or \ref ByteSlice for a
/// bounded byte-slice service whose capacity is \ref PubSubServiceOptions::byte_slice_capacity.
template <typename Payload>
class SubscriberBuilder {
  friend class SubscriberServiceBuilder;

 public:
  SubscriberBuilder(const SubscriberBuilder&) = delete;
  SubscriberBuilder& operator=(const SubscriberBuilder&) = delete;
  SubscriberBuilder(SubscriberBuilder&&) noexcept = default;
  SubscriberBuilder& operator=(SubscriberBuilder&&) noexcept = default;

  /// @brief Enable or disable event wake for this subscriber's service on an rvalue builder.
  /// @param enable Whether to create the event listener used by \ref WaitSet kernel waits.
  /// @return Moved builder with \ref PubSubServiceOptions::enable_event_wake updated.
  /// @note Every endpoint of the service must use the same setting. The default is \c true.
  [[nodiscard]] SubscriberBuilder enable_event_wake(const bool enable) && noexcept {
    service_options_.enable_event_wake = enable;
    return std::move(*this);
  }

  /// @brief Enable or disable event wake for this subscriber's service.
  /// @param enable Whether to create the event listener used by \ref WaitSet kernel waits.
  /// @return Reference to this builder.
  /// @note Every endpoint of the service must use the same setting. The default is \c true.
  [[nodiscard]] SubscriberBuilder& enable_event_wake(const bool enable) & noexcept {
    service_options_.enable_event_wake = enable;
    return *this;
  }

  /// @brief Set the static pub/sub service contract on an rvalue builder.
  /// @param options Complete service-static option set shared by every endpoint.
  /// @return Moved builder with the updated service contract.
  [[nodiscard]] SubscriberBuilder service_options(PubSubServiceOptions options) && noexcept {
    service_options_ = options;
    return std::move(*this);
  }

  /// @brief Set the static pub/sub service contract on this builder.
  /// @param options Complete service-static option set shared by every endpoint.
  /// @return Reference to this builder.
  [[nodiscard]] SubscriberBuilder& service_options(PubSubServiceOptions options) & noexcept {
    service_options_ = options;
    return *this;
  }

  /// @brief Set subscriber-local typed pub/sub options on an rvalue builder.
  /// @param options Subscriber queue and borrow option set to copy into the builder.
  /// @return Moved builder with updated options.
  [[nodiscard]] SubscriberBuilder subscriber_options(SubscriberPubSubOptions options) && noexcept {
    options_ = options;
    return std::move(*this);
  }

  /// @brief Set subscriber-local typed pub/sub options on this builder.
  /// @param options Subscriber queue and borrow option set to copy into the builder.
  /// @return Reference to this builder.
  [[nodiscard]] SubscriberBuilder& subscriber_options(SubscriberPubSubOptions options) & noexcept {
    options_ = options;
    return *this;
  }

  /// @brief Set full-queue behavior for the typed pub/sub service on an rvalue builder.
  /// @param policy Queue policy to store in \ref PubSubServiceOptions::queue_policy.
  /// @return Moved builder with updated queue policy.
  [[nodiscard]] SubscriberBuilder queue_policy(const QueuePolicy policy) && noexcept {
    service_options_.queue_policy = policy;
    return std::move(*this);
  }

  /// @brief Set full-queue behavior for the typed pub/sub service on this builder.
  /// @param policy Queue policy to store in \ref PubSubServiceOptions::queue_policy.
  /// @return Reference to this builder.
  [[nodiscard]] SubscriberBuilder& queue_policy(const QueuePolicy policy) & noexcept {
    service_options_.queue_policy = policy;
    return *this;
  }

  /// @brief Set this subscriber's requested queue depth on an rvalue builder.
  /// @param depth Requested queue depth; must be greater than zero before \ref create.
  /// @return Moved builder with updated subscriber queue depth.
  [[nodiscard]] SubscriberBuilder queue_depth(const std::uint32_t depth) && noexcept {
    options_.queue_depth = depth;
    return std::move(*this);
  }

  /// @brief Set this subscriber's requested queue depth on this builder.
  /// @param depth Requested queue depth; must be greater than zero before \ref create.
  /// @return Reference to this builder.
  [[nodiscard]] SubscriberBuilder& queue_depth(const std::uint32_t depth) & noexcept {
    options_.queue_depth = depth;
    return *this;
  }

  /// @brief Set the borrowed-sample allowance required by this subscriber.
  /// @param count Required borrowed-sample allowance; zero derives from the configured queue depth.
  /// @return Moved builder with updated borrowed-sample requirement.
  [[nodiscard]] SubscriberBuilder max_borrowed_samples(const std::uint32_t count) && noexcept {
    options_.max_borrowed_samples = count;
    return std::move(*this);
  }

  /// @brief Set the borrowed-sample allowance required by this subscriber.
  /// @param count Required borrowed-sample allowance; zero derives from the configured queue depth.
  /// @return Reference to this builder.
  [[nodiscard]] SubscriberBuilder& max_borrowed_samples(const std::uint32_t count) & noexcept {
    options_.max_borrowed_samples = count;
    return *this;
  }

  /**
   * @brief Return a subscriber or a typed service, resource, permission, or configuration error.
   *
   * Failures are logged with setup detail.
   * @return A subscriber on success, or the typed setup error on failure.
   * @throws std::exception if backend setup or resource allocation propagates an exception.
   */
  [[nodiscard]] auto create() && -> expected<Subscriber<Payload>, Error> {
    static_assert(std::is_standard_layout_v<Payload> && std::is_trivially_copyable_v<Payload>,
                  "subscriber_builder: Payload must be standard-layout and trivially copyable");

    if (transport_ == nullptr) {
      HOLOMQ_LOG_ERROR("SubscriberBuilder::create failed: null transport");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    if (!detail::is_valid_queue_policy(service_options_.queue_policy)) {
      HOLOMQ_LOG_ERROR("SubscriberBuilder::create failed: unsupported queue policy");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (options_.queue_depth == 0U) {
      HOLOMQ_LOG_ERROR("SubscriberBuilder::create failed: queue depth must be greater than zero");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (service_options_.max_subscriber_queue_depth == 0U) {
      HOLOMQ_LOG_ERROR(
          "SubscriberBuilder::create failed: max subscriber queue depth must be greater than zero");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!detail::is_valid_byte_slice_contract(kIsByteSlice<Payload>,
                                              service_options_.byte_slice_capacity,
                                              service_options_.byte_slice_profile)) {
      HOLOMQ_LOG_ERROR("SubscriberBuilder::create failed: invalid byte-slice capacity/profile");
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "invalid byte-slice capacity/profile"});
    }
    if constexpr (kIsByteSlice<Payload>) {
      const std::uint64_t subscribers = service_options_.max_subscribers == 0U
                                            ? k_default_max_subscribers
                                            : service_options_.max_subscribers;
      if (!detail::is_valid_byte_slice_pool_geometry(service_options_, 0U, subscribers)) {
        HOLOMQ_LOG_ERROR(
            "SubscriberBuilder::create failed: byte-slice pool geometry exceeds its limit");
        return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                     "byte-slice sample pool geometry exceeds its limit"});
      }
    }
    auto port_or =
        transport_->create_subscriber<Payload>(service_name_, service_options_, options_);
    if (!port_or.has_value()) {
      HOLOMQ_LOG_ERROR("SubscriberBuilder::create failed: {}", error_message(port_or.error()));
      return holomq::forward_error(std::move(port_or));
    }
    return Subscriber<Payload>(
        std::make_shared<detail::transport::SubscriberPort<Payload>>(std::move(port_or).value()));
  }

 private:
  explicit SubscriberBuilder(detail::transport::Runtime* transport, std::string service_name)
      : transport_(transport), service_name_(std::move(service_name)) {}

  detail::transport::Runtime* transport_{nullptr};
  std::string service_name_;
  PubSubServiceOptions service_options_{};
  SubscriberPubSubOptions options_{};
};

/// @brief Configure a named subscriber service before selecting its payload type.
///
/// Call \c payload_type<Payload>() to create the typed builder. This move-only builder borrows the
/// parent \ref Context transport and must be used while that context is alive. Builder mutation is
/// setup-only and not thread-safe.
class SubscriberServiceBuilder {
  friend class Context;

 public:
  SubscriberServiceBuilder(const SubscriberServiceBuilder&) = delete;
  SubscriberServiceBuilder& operator=(const SubscriberServiceBuilder&) = delete;
  SubscriberServiceBuilder(SubscriberServiceBuilder&&) noexcept = default;
  SubscriberServiceBuilder& operator=(SubscriberServiceBuilder&&) noexcept = default;
  ~SubscriberServiceBuilder() = default;

  /// @brief Enable or disable event wake for this subscriber's service on an rvalue builder.
  /// @param enable Whether to create the event listener used by \ref WaitSet kernel waits.
  /// @return Moved service builder with \ref PubSubServiceOptions::enable_event_wake updated.
  /// @note Every endpoint of the service must use the same setting. The default is \c true.
  [[nodiscard]] SubscriberServiceBuilder enable_event_wake(const bool enable) && noexcept {
    service_options_.enable_event_wake = enable;
    return std::move(*this);
  }

  /// @brief Enable or disable event wake for this subscriber's service.
  /// @param enable Whether to create the event listener used by \ref WaitSet kernel waits.
  /// @return Reference to this service builder.
  /// @note Every endpoint of the service must use the same setting. The default is \c true.
  [[nodiscard]] SubscriberServiceBuilder& enable_event_wake(const bool enable) & noexcept {
    service_options_.enable_event_wake = enable;
    return *this;
  }

  /// @brief Set the static pub/sub service contract on an rvalue service builder.
  /// @param options Complete service-static option set shared by every endpoint.
  /// @return Moved service builder with the updated service contract.
  [[nodiscard]] SubscriberServiceBuilder service_options(PubSubServiceOptions options) && noexcept {
    service_options_ = options;
    return std::move(*this);
  }

  /// @brief Set the static pub/sub service contract on this service builder.
  /// @param options Complete service-static option set shared by every endpoint.
  /// @return Reference to this service builder.
  [[nodiscard]] SubscriberServiceBuilder& service_options(PubSubServiceOptions options) & noexcept {
    service_options_ = options;
    return *this;
  }

  /// @brief Set subscriber-local typed pub/sub options on an rvalue service builder.
  /// @param options Subscriber queue and borrow option set to copy into the builder.
  /// @return Moved service builder with updated options.
  [[nodiscard]] SubscriberServiceBuilder subscriber_options(
      SubscriberPubSubOptions options) && noexcept {
    options_ = options;
    return std::move(*this);
  }

  /// @brief Set subscriber-local typed pub/sub options on this service builder.
  /// @param options Subscriber queue and borrow option set to copy into the builder.
  /// @return Reference to this service builder.
  [[nodiscard]] SubscriberServiceBuilder& subscriber_options(
      SubscriberPubSubOptions options) & noexcept {
    options_ = options;
    return *this;
  }

  /// @brief Set full-queue behavior for the typed pub/sub service on an rvalue service builder.
  /// @param policy Queue policy to store in \ref PubSubServiceOptions::queue_policy.
  /// @return Moved service builder with updated queue policy.
  [[nodiscard]] SubscriberServiceBuilder queue_policy(const QueuePolicy policy) && noexcept {
    service_options_.queue_policy = policy;
    return std::move(*this);
  }

  /// @brief Set full-queue behavior for the typed pub/sub service on this service builder.
  /// @param policy Queue policy to store in \ref PubSubServiceOptions::queue_policy.
  /// @return Reference to this service builder.
  [[nodiscard]] SubscriberServiceBuilder& queue_policy(const QueuePolicy policy) & noexcept {
    service_options_.queue_policy = policy;
    return *this;
  }

  /// @brief Set this subscriber's requested queue depth on an rvalue service builder.
  /// @param depth Requested queue depth; must be greater than zero before \ref create.
  /// @return Moved service builder with updated subscriber queue depth.
  [[nodiscard]] SubscriberServiceBuilder queue_depth(const std::uint32_t depth) && noexcept {
    options_.queue_depth = depth;
    return std::move(*this);
  }

  /// @brief Set this subscriber's requested queue depth on this service builder.
  /// @param depth Requested queue depth; must be greater than zero before \ref create.
  /// @return Reference to this service builder.
  [[nodiscard]] SubscriberServiceBuilder& queue_depth(const std::uint32_t depth) & noexcept {
    options_.queue_depth = depth;
    return *this;
  }

  /// @brief Set the borrowed-sample allowance required by this subscriber.
  /// @param count Required borrowed-sample allowance; zero derives from the configured queue depth.
  /// @return Moved service builder with updated borrowed-sample requirement.
  [[nodiscard]] SubscriberServiceBuilder max_borrowed_samples(
      const std::uint32_t count) && noexcept {
    options_.max_borrowed_samples = count;
    return std::move(*this);
  }

  /// @brief Set the borrowed-sample allowance required by this subscriber.
  /// @param count Required borrowed-sample allowance; zero derives from the configured queue depth.
  /// @return Reference to this service builder.
  [[nodiscard]] SubscriberServiceBuilder& max_borrowed_samples(
      const std::uint32_t count) & noexcept {
    options_.max_borrowed_samples = count;
    return *this;
  }

  /// @brief Fix the payload type on an rvalue service builder.
  /// @tparam P Standard-layout, trivially copyable transport payload.
  /// @return Typed builder carrying the complete service-static and subscriber-local option sets.
  template <typename P>
  SubscriberBuilder<P> payload_type() && {
    return SubscriberBuilder<P>(transport_, std::move(service_name_))
        .service_options(service_options_)
        .subscriber_options(options_);
  }

  /// @brief Fix the payload type on this service builder.
  /// @tparam P Standard-layout, trivially copyable transport payload.
  /// @return Typed builder carrying the complete service-static and subscriber-local option sets.
  template <typename P>
  SubscriberBuilder<P> payload_type() & {
    return SubscriberBuilder<P>(transport_, service_name_)
        .service_options(service_options_)
        .subscriber_options(options_);
  }

 private:
  explicit SubscriberServiceBuilder(detail::transport::Runtime* transport, std::string service_name)
      : transport_(transport), service_name_(std::move(service_name)) {}

  detail::transport::Runtime* transport_{nullptr};
  std::string service_name_;
  PubSubServiceOptions service_options_{};
  SubscriberPubSubOptions options_{};
};

}  // namespace holomq
