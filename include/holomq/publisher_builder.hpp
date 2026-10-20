/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/byte_slice.hpp>
#include <holomq/delivery_attribution.hpp>
#include <holomq/detail/transport/backend.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/logger.hpp>
#include <holomq/publisher.hpp>
#include <holomq/pubsub_options.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace holomq {

class Context;

class PublisherServiceBuilder;

/// @brief Configure one typed publisher before creating its backend endpoint.
///
/// The move-only builder borrows the parent \ref Context transport and must be used while that
/// context is alive. Builder mutation and \ref create are setup-only and not thread-safe.
/// @tparam Payload Standard-layout, trivially copyable transport payload, or \ref ByteSlice for a
/// bounded byte-slice service whose capacity is \ref PubSubServiceOptions::byte_slice_capacity.
template <typename Payload>
class PublisherBuilder {
  friend class PublisherServiceBuilder;

 public:
  PublisherBuilder(const PublisherBuilder&) = delete;
  PublisherBuilder& operator=(const PublisherBuilder&) = delete;
  PublisherBuilder(PublisherBuilder&&) noexcept = default;
  PublisherBuilder& operator=(PublisherBuilder&&) noexcept = default;

  /// @brief Enable or disable event wake for this publisher's service on an rvalue builder.
  /// @param enable Whether to create the event notifier used by \ref WaitSet kernel waits.
  /// @return Moved builder with \ref PubSubServiceOptions::enable_event_wake updated.
  /// @note Every endpoint of the service must use the same setting. The default is \c true.
  [[nodiscard]] PublisherBuilder enable_event_wake(const bool enable) && noexcept {
    service_options_.enable_event_wake = enable;
    return std::move(*this);
  }

  /// @brief Enable or disable event wake for this publisher's service.
  /// @param enable Whether to create the event notifier used by \ref WaitSet kernel waits.
  /// @return Reference to this builder.
  /// @note Every endpoint of the service must use the same setting. The default is \c true.
  [[nodiscard]] PublisherBuilder& enable_event_wake(const bool enable) & noexcept {
    service_options_.enable_event_wake = enable;
    return *this;
  }

  /// @brief Set the static pub/sub service contract on an rvalue builder.
  /// @param options Complete service-static option set shared by every endpoint.
  /// @return Moved builder with the updated service contract.
  [[nodiscard]] PublisherBuilder service_options(PubSubServiceOptions options) && noexcept {
    service_options_ = options;
    return std::move(*this);
  }

  /// @brief Set the static pub/sub service contract on this builder.
  /// @param options Complete service-static option set shared by every endpoint.
  /// @return Reference to this builder.
  [[nodiscard]] PublisherBuilder& service_options(PubSubServiceOptions options) & noexcept {
    service_options_ = options;
    return *this;
  }

  /// @brief Set publisher-local typed pub/sub options on an rvalue builder.
  /// @param options Publisher loan and callback option set to copy into the builder.
  /// @return Moved builder with updated options.
  [[nodiscard]] PublisherBuilder publisher_options(PublisherPubSubOptions options) && noexcept {
    options_ = options;
    return std::move(*this);
  }

  /// @brief Set publisher-local typed pub/sub options on this builder.
  /// @param options Publisher loan and callback option set to copy into the builder.
  /// @return Reference to this builder.
  [[nodiscard]] PublisherBuilder& publisher_options(PublisherPubSubOptions options) & noexcept {
    options_ = options;
    return *this;
  }

  /// @brief Set full-queue behavior for the typed pub/sub service on an rvalue builder.
  /// @param policy Queue policy to store in \ref PubSubServiceOptions::queue_policy.
  /// @return Moved builder with updated queue policy.
  [[nodiscard]] PublisherBuilder queue_policy(const QueuePolicy policy) && noexcept {
    service_options_.queue_policy = policy;
    return std::move(*this);
  }

  /// @brief Set full-queue behavior for the typed pub/sub service on this builder.
  /// @param policy Queue policy to store in \ref PubSubServiceOptions::queue_policy.
  /// @return Reference to this builder.
  [[nodiscard]] PublisherBuilder& queue_policy(const QueuePolicy policy) & noexcept {
    service_options_.queue_policy = policy;
    return *this;
  }

  /// @brief Set the maximum per-subscriber queue depth supported by the service.
  /// @param depth Maximum queue depth; must be greater than zero before \ref create.
  /// @return Moved builder with updated service maximum queue depth.
  [[nodiscard]] PublisherBuilder max_subscriber_queue_depth(const std::uint32_t depth) && noexcept {
    service_options_.max_subscriber_queue_depth = depth;
    return std::move(*this);
  }

  /// @brief Set the maximum per-subscriber queue depth supported by the service.
  /// @param depth Maximum queue depth; must be greater than zero before \ref create.
  /// @return Reference to this builder.
  [[nodiscard]] PublisherBuilder& max_subscriber_queue_depth(const std::uint32_t depth) & noexcept {
    service_options_.max_subscriber_queue_depth = depth;
    return *this;
  }

  /// @brief Set the maximum samples each subscriber may borrow concurrently.
  /// @param count Maximum borrowed samples; zero derives from the configured service queue depth.
  /// @return Moved builder with updated service maximum borrowed-sample count.
  [[nodiscard]] PublisherBuilder max_subscriber_borrowed_samples(
      const std::uint32_t count) && noexcept {
    service_options_.max_subscriber_borrowed_samples = count;
    return std::move(*this);
  }

  /// @brief Set the maximum samples each subscriber may borrow concurrently.
  /// @param count Maximum borrowed samples; zero derives from the configured service queue depth.
  /// @return Reference to this builder.
  [[nodiscard]] PublisherBuilder& max_subscriber_borrowed_samples(
      const std::uint32_t count) & noexcept {
    service_options_.max_subscriber_borrowed_samples = count;
    return *this;
  }

  /// @brief Set the maximum unsent samples this publisher may loan concurrently.
  /// @param count Maximum loaned samples; zero requests the derived compatibility default.
  /// @return Moved builder with updated publisher loan limit.
  [[nodiscard]] PublisherBuilder max_loaned_samples(const std::uint32_t count) && noexcept {
    options_.max_loaned_samples = count;
    return std::move(*this);
  }

  /// @brief Set the maximum unsent samples this publisher may loan concurrently.
  /// @param count Maximum loaned samples; zero requests the derived compatibility default.
  /// @return Reference to this builder.
  [[nodiscard]] PublisherBuilder& max_loaned_samples(const std::uint32_t count) & noexcept {
    options_.max_loaned_samples = count;
    return *this;
  }

  /**
   * @brief Set a callback for samples replaced by a lossy queue policy on an rvalue builder.
   * @tparam Fn Callable accepting `const Payload&` (or `std::span<const std::byte>` for
   * \ref ByteSlice) and `const SampleReplacedInfo&`.
   * @param callback Callback registered during publisher setup.
   * @return Moved builder with replacement attribution enabled.
   * @throws std::bad_alloc if storing the callback allocates and allocation fails.
   * @note The callback receives a read-only view of the replaced payload and avoids the per-send
   * callback registry for descriptor-backed payloads whose cleanup derives from the payload.
   * Calling this method requests \ref PublisherPubSubOptions::replacement_attribution_required.
   */
  template <typename Fn>
  [[nodiscard]] PublisherBuilder on_sample_replaced(Fn&& callback) && {
    static_assert(std::is_invocable_v<Fn&, ConstPayloadView<Payload>, const SampleReplacedInfo&>,
                  "on_sample_replaced callback must accept "
                  "(ConstPayloadView<Payload>, const SampleReplacedInfo&)");
    sample_replaced_callback_ = std::forward<Fn>(callback);
    options_.replacement_attribution_required = true;
    return std::move(*this);
  }

  /**
   * @brief Set a callback for samples replaced by a lossy queue policy on this builder.
   * @tparam Fn Callable accepting `const Payload&` (or `std::span<const std::byte>` for
   * \ref ByteSlice) and `const SampleReplacedInfo&`.
   * @param callback Callback registered during publisher setup.
   * @return Reference to this builder with replacement attribution enabled.
   * @throws std::bad_alloc if storing the callback allocates and allocation fails.
   * @note The callback receives a read-only view of the replaced payload and avoids the per-send
   * callback registry for descriptor-backed payloads whose cleanup derives from the payload.
   * Calling this method requests \ref PublisherPubSubOptions::replacement_attribution_required.
   */
  template <typename Fn>
  [[nodiscard]] PublisherBuilder& on_sample_replaced(Fn&& callback) & {
    static_assert(std::is_invocable_v<Fn&, ConstPayloadView<Payload>, const SampleReplacedInfo&>,
                  "on_sample_replaced callback must accept "
                  "(ConstPayloadView<Payload>, const SampleReplacedInfo&)");
    sample_replaced_callback_ = std::forward<Fn>(callback);
    options_.replacement_attribution_required = true;
    return *this;
  }

  /**
   * @brief Return a publisher or a typed service, resource, permission, or configuration error.
   *
   * Failures are logged with setup detail.
   * @return A publisher on success, or the typed setup error on failure.
   * @throws std::exception if backend setup or resource allocation propagates an exception.
   */
  [[nodiscard]] auto create() && -> expected<Publisher<Payload>, Error> {
    static_assert(std::is_standard_layout_v<Payload> && std::is_trivially_copyable_v<Payload>,
                  "publisher_builder: Payload must be standard-layout and trivially copyable");

    auto validated = validate_options();
    if (!validated) {
      return make_unexpected(std::move(validated).error());
    }

    std::shared_ptr<detail::PublisherSampleReleaseRegistry> release_registry;
    if (options_.enable_sample_release_callbacks) {
      const auto callback_capacity =
          options_.effective_sample_release_callback_capacity(service_options_);
      if (!callback_capacity.has_value()) {
        HOLOMQ_LOG_ERROR(
            "PublisherBuilder::create failed: sample release callback capacity exceeds size "
            "range");
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      release_registry =
          std::make_shared<detail::PublisherSampleReleaseRegistry>(*callback_capacity);
    }
    auto port_or = transport_->create_publisher<Payload>(service_name_,
                                                         release_registry,
                                                         std::move(sample_replaced_callback_),
                                                         service_options_,
                                                         options_);
    if (!port_or.has_value()) {
      HOLOMQ_LOG_ERROR("PublisherBuilder::create failed: {}", error_message(port_or.error()));
      return holomq::forward_error(std::move(port_or));
    }

    return Publisher<Payload>(std::make_shared<detail::PublisherImpl<Payload>>(
        detail::PublisherImpl<Payload>{std::move(port_or).value(),
                                       std::move(release_registry),
                                       options_.expected_recipient_count}));
  }

 private:
  [[nodiscard]] expected<void, Error> validate_options() const {
    if (transport_ == nullptr) {
      HOLOMQ_LOG_ERROR("PublisherBuilder::create failed: null transport");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    if (!detail::is_valid_queue_policy(service_options_.queue_policy)) {
      HOLOMQ_LOG_ERROR("PublisherBuilder::create failed: unsupported queue policy");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!detail::is_valid_publisher_pressure_policy(options_.pressure_policy)) {
      HOLOMQ_LOG_ERROR("PublisherBuilder::create failed: unsupported publisher pressure policy");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if ((options_.pressure_policy == PublisherPressurePolicy::kRejectWhenFull ||
         options_.pressure_policy == PublisherPressurePolicy::kBoundedRetry) &&
        service_options_.queue_policy != QueuePolicy::kBlocking) {
      HOLOMQ_LOG_ERROR(
          "PublisherBuilder::create failed: bounded publisher pressure policy requires kBlocking "
          "queue policy");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (options_.pressure_policy == PublisherPressurePolicy::kBoundedRetry &&
        options_.bounded_retry_budget_ns == 0U) {
      HOLOMQ_LOG_ERROR(
          "PublisherBuilder::create failed: bounded-retry publisher requires a non-zero budget");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (options_.pressure_policy != PublisherPressurePolicy::kBoundedRetry &&
        (options_.bounded_retry_budget_ns != 0U ||
         options_.bounded_retry_cancellation != nullptr)) {
      HOLOMQ_LOG_ERROR(
          "PublisherBuilder::create failed: retry budget or cancellation requires bounded-retry "
          "pressure policy");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (service_options_.max_subscribers != 0U &&
        options_.expected_recipient_count > service_options_.max_subscribers) {
      HOLOMQ_LOG_ERROR(
          "PublisherBuilder::create failed: expected recipient count exceeds subscriber capacity");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (service_options_.max_subscriber_queue_depth == 0U) {
      HOLOMQ_LOG_ERROR(
          "PublisherBuilder::create failed: max subscriber queue depth must be "
          "greater than zero");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (options_.replacement_attribution_required &&
        service_options_.queue_policy != QueuePolicy::kDropOldest) {
      HOLOMQ_LOG_ERROR(
          "PublisherBuilder::create failed: replacement attribution requires "
          "kDropOldest queue policy");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (options_.replacement_attribution_required && !sample_replaced_callback_) {
      HOLOMQ_LOG_ERROR(
          "PublisherBuilder::create failed: replacement attribution requires "
          "on_sample_replaced callback");
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    if (!detail::is_valid_byte_slice_contract(kIsByteSlice<Payload>,
                                              service_options_.byte_slice_capacity,
                                              service_options_.byte_slice_profile)) {
      HOLOMQ_LOG_ERROR("PublisherBuilder::create failed: invalid byte-slice capacity/profile");
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "invalid byte-slice capacity/profile"});
    }
    if constexpr (kIsByteSlice<Payload>) {
      const std::uint64_t subscribers = service_options_.max_subscribers == 0U
                                            ? k_default_max_subscribers
                                            : service_options_.max_subscribers;
      if (!detail::is_valid_byte_slice_pool_geometry(
              service_options_,
              options_.effective_max_loaned_samples(service_options_),
              std::max(subscribers,
                       static_cast<std::uint64_t>(options_.expected_recipient_count)))) {
        HOLOMQ_LOG_ERROR(
            "PublisherBuilder::create failed: byte-slice pool geometry exceeds its limit");
        return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                     "byte-slice sample pool geometry exceeds its limit"});
      }
    }

    return {};
  }

  explicit PublisherBuilder(detail::transport::Runtime* transport, std::string service_name)
      : transport_(transport), service_name_(std::move(service_name)) {}

  detail::transport::Runtime* transport_{nullptr};
  std::string service_name_;
  PubSubServiceOptions service_options_{};
  PublisherPubSubOptions options_{};
  SampleReplacedCallback<Payload> sample_replaced_callback_{};
};

/// @brief Configure a named publisher service before selecting its payload type.
///
/// Call \c payload_type<Payload>() to create the typed builder. This move-only builder borrows the
/// parent \ref Context transport and must be used while that context is alive. Builder mutation is
/// setup-only and not thread-safe.
class PublisherServiceBuilder {
  friend class Context;

 public:
  PublisherServiceBuilder(const PublisherServiceBuilder&) = delete;
  PublisherServiceBuilder& operator=(const PublisherServiceBuilder&) = delete;
  PublisherServiceBuilder(PublisherServiceBuilder&&) noexcept = default;
  PublisherServiceBuilder& operator=(PublisherServiceBuilder&&) noexcept = default;
  ~PublisherServiceBuilder() = default;

  /// @brief Enable or disable event wake for this publisher's service on an rvalue builder.
  /// @param enable Whether to create the event notifier used by \ref WaitSet kernel waits.
  /// @return Moved service builder with \ref PubSubServiceOptions::enable_event_wake updated.
  /// @note Every endpoint of the service must use the same setting. The default is \c true.
  [[nodiscard]] PublisherServiceBuilder enable_event_wake(const bool enable) && noexcept {
    service_options_.enable_event_wake = enable;
    return std::move(*this);
  }

  /// @brief Enable or disable event wake for this publisher's service.
  /// @param enable Whether to create the event notifier used by \ref WaitSet kernel waits.
  /// @return Reference to this service builder.
  /// @note Every endpoint of the service must use the same setting. The default is \c true.
  [[nodiscard]] PublisherServiceBuilder& enable_event_wake(const bool enable) & noexcept {
    service_options_.enable_event_wake = enable;
    return *this;
  }

  /// @brief Set the static pub/sub service contract on an rvalue service builder.
  /// @param options Complete service-static option set shared by every endpoint.
  /// @return Moved service builder with the updated service contract.
  [[nodiscard]] PublisherServiceBuilder service_options(PubSubServiceOptions options) && noexcept {
    service_options_ = options;
    return std::move(*this);
  }

  /// @brief Set the static pub/sub service contract on this service builder.
  /// @param options Complete service-static option set shared by every endpoint.
  /// @return Reference to this service builder.
  [[nodiscard]] PublisherServiceBuilder& service_options(PubSubServiceOptions options) & noexcept {
    service_options_ = options;
    return *this;
  }

  /// @brief Set publisher-local typed pub/sub options on an rvalue service builder.
  /// @param options Publisher loan and callback option set to copy into the builder.
  /// @return Moved service builder with updated options.
  [[nodiscard]] PublisherServiceBuilder publisher_options(
      PublisherPubSubOptions options) && noexcept {
    options_ = options;
    return std::move(*this);
  }

  /// @brief Set publisher-local typed pub/sub options on this service builder.
  /// @param options Publisher loan and callback option set to copy into the builder.
  /// @return Reference to this service builder.
  [[nodiscard]] PublisherServiceBuilder& publisher_options(
      PublisherPubSubOptions options) & noexcept {
    options_ = options;
    return *this;
  }

  /// @brief Set full-queue behavior for the typed pub/sub service on an rvalue service builder.
  /// @param policy Queue policy to store in \ref PubSubServiceOptions::queue_policy.
  /// @return Moved service builder with updated queue policy.
  [[nodiscard]] PublisherServiceBuilder queue_policy(const QueuePolicy policy) && noexcept {
    service_options_.queue_policy = policy;
    return std::move(*this);
  }

  /// @brief Set full-queue behavior for the typed pub/sub service on this service builder.
  /// @param policy Queue policy to store in \ref PubSubServiceOptions::queue_policy.
  /// @return Reference to this service builder.
  [[nodiscard]] PublisherServiceBuilder& queue_policy(const QueuePolicy policy) & noexcept {
    service_options_.queue_policy = policy;
    return *this;
  }

  /// @brief Set the maximum per-subscriber queue depth supported by the service.
  /// @param depth Maximum queue depth; must be greater than zero before \ref create.
  /// @return Moved service builder with updated service maximum queue depth.
  [[nodiscard]] PublisherServiceBuilder max_subscriber_queue_depth(
      const std::uint32_t depth) && noexcept {
    service_options_.max_subscriber_queue_depth = depth;
    return std::move(*this);
  }

  /// @brief Set the maximum per-subscriber queue depth supported by the service.
  /// @param depth Maximum queue depth; must be greater than zero before \ref create.
  /// @return Reference to this service builder.
  [[nodiscard]] PublisherServiceBuilder& max_subscriber_queue_depth(
      const std::uint32_t depth) & noexcept {
    service_options_.max_subscriber_queue_depth = depth;
    return *this;
  }

  /// @brief Set the maximum samples each subscriber may borrow concurrently.
  /// @param count Maximum borrowed samples; zero derives from the configured service queue depth.
  /// @return Moved service builder with updated service maximum borrowed-sample count.
  [[nodiscard]] PublisherServiceBuilder max_subscriber_borrowed_samples(
      const std::uint32_t count) && noexcept {
    service_options_.max_subscriber_borrowed_samples = count;
    return std::move(*this);
  }

  /// @brief Set the maximum samples each subscriber may borrow concurrently.
  /// @param count Maximum borrowed samples; zero derives from the configured service queue depth.
  /// @return Reference to this service builder.
  [[nodiscard]] PublisherServiceBuilder& max_subscriber_borrowed_samples(
      const std::uint32_t count) & noexcept {
    service_options_.max_subscriber_borrowed_samples = count;
    return *this;
  }

  /// @brief Set the maximum unsent samples this publisher may loan concurrently.
  /// @param count Maximum loaned samples; zero requests the derived compatibility default.
  /// @return Moved service builder with updated publisher loan limit.
  [[nodiscard]] PublisherServiceBuilder max_loaned_samples(const std::uint32_t count) && noexcept {
    options_.max_loaned_samples = count;
    return std::move(*this);
  }

  /// @brief Set the maximum unsent samples this publisher may loan concurrently.
  /// @param count Maximum loaned samples; zero requests the derived compatibility default.
  /// @return Reference to this service builder.
  [[nodiscard]] PublisherServiceBuilder& max_loaned_samples(const std::uint32_t count) & noexcept {
    options_.max_loaned_samples = count;
    return *this;
  }

  /// @brief Fix the payload type on an rvalue service builder.
  /// @tparam P Standard-layout, trivially copyable transport payload.
  /// @return Typed builder carrying the complete service-static and publisher-local option sets.
  template <typename P>
  PublisherBuilder<P> payload_type() && {
    return PublisherBuilder<P>(transport_, std::move(service_name_))
        .service_options(service_options_)
        .publisher_options(options_);
  }

  /// @brief Fix the payload type on this service builder.
  /// @tparam P Standard-layout, trivially copyable transport payload.
  /// @return Typed builder carrying the complete service-static and publisher-local option sets.
  template <typename P>
  PublisherBuilder<P> payload_type() & {
    return PublisherBuilder<P>(transport_, service_name_)
        .service_options(service_options_)
        .publisher_options(options_);
  }

 private:
  explicit PublisherServiceBuilder(detail::transport::Runtime* transport, std::string service_name)
      : transport_(transport), service_name_(std::move(service_name)) {}

  detail::transport::Runtime* transport_{nullptr};
  std::string service_name_;
  PubSubServiceOptions service_options_{};
  PublisherPubSubOptions options_{};
};

}  // namespace holomq
