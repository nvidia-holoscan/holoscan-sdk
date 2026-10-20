/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/detail/transport/iox2/config.hpp>
#include <holomq/detail/transport/iox2/id_conversion.hpp>
#include <holomq/detail/transport/iox2/iox_payload.hpp>
#include <holomq/detail/transport/iox2/setup_error.hpp>
#include <holomq/detail/transport/iox2/unable_to_deliver_evidence.hpp>

#include <holomq/byte_slice.hpp>
#include <holomq/delivery_attribution.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/logger.hpp>
#include <holomq/pubsub_options.hpp>

#include <iox2/allocation_strategy.hpp>
#include <iox2/attribute_verifier.hpp>
#include <iox2/backpressure_handler.hpp>
#include <iox2/backpressure_strategy.hpp>
#include <iox2/log.hpp>
#include <iox2/port_error.hpp>
#include <iox2/port_factory_publish_subscribe.hpp>
#include <iox2/sample_released_handler.hpp>
#include <iox2/sample_replaced_handler.hpp>
#include <iox2/service_name.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

namespace holomq::detail::transport {

namespace {

/// @brief Add one HoloMQ contract field as a required iceoryx2 service attribute.
///
/// Required attributes make iceoryx2 reject a peer whose stored value differs. Keeping attribute
/// construction here also gives every HoloMQ contract field consistent setup-error diagnostics.
[[nodiscard]] inline auto require_service_attribute(iox2::AttributeVerifier& verifier,
                                                    const char* key_text,
                                                    const std::string& value_text,
                                                    const char* field)
    -> holomq::expected<void, holomq::Error> {
  const auto key_or = iox2::Attribute::Key::from_utf8_null_terminated_unchecked(key_text);
  const auto value_or =
      iox2::Attribute::Value::from_utf8_null_terminated_unchecked(value_text.c_str());
  if (!key_or.has_value() || !value_or.has_value()) {
    return holomq::make_unexpected(holomq::Error{
        holomq::ErrorCode::kFailure,
        std::string("transport: failed to construct the HoloMQ ") + field + " service attribute"});
  }
  const auto required_or = verifier.require(key_or.value(), value_or.value());
  if (!required_or.has_value()) {
    return holomq::make_unexpected(holomq::Error{
        holomq::ErrorCode::kFailure,
        std::string("transport: failed to define the HoloMQ ") + field + " service attribute"});
  }
  return {};
}

/// @brief Return the stable service-attribute spelling for a queue policy.
///
/// Exact policy identity distinguishes lossless blocking from lossy safe-overflow services.
[[nodiscard]] inline const char* queue_policy_attribute_value(
    const holomq::QueuePolicy policy) noexcept {
  switch (policy) {
    case holomq::QueuePolicy::kBlocking:
      return "blocking";
    case holomq::QueuePolicy::kDropOldest:
      return "drop_oldest";
    default:
      return "unsupported";
  }
}

/// @brief Return the process-lifetime handler for nonblocking publisher pressure rejection.
///
/// iceoryx2 borrows the handler address for the publisher lifetime. The callback is stateless and
/// its invocation operator is const, so one immutable process-lifetime instance is safe to share
/// across publishers and keeps optional pressure state out of the default publication hot object.
[[nodiscard]] inline iox2::BackpressureHandler& reject_when_full_handler() noexcept {
  static iox2::BackpressureHandler handler{[](const iox2::BackpressureInfo& info) noexcept {
    record_unable_to_deliver_recipient(
        to_delivery_recipient_id_from_big_endian(info.receiver_port_id()));
    return iox2::BackpressureAction::DiscardDataAndFail;
  }};
  return handler;
}

/// @brief Build a verifier for HoloMQ fields not fully represented by iceoryx2 static config.
///
/// iceoryx2 does not expose event-service intent or HoloMQ's queue-policy spelling. Required
/// attributes preserve that intent regardless of which endpoint creates the service first.
[[nodiscard]] inline auto make_service_attribute_verifier(
    const holomq::PubSubServiceOptions& options)
    -> holomq::expected<iox2::AttributeVerifier, holomq::Error> {
  iox2::AttributeVerifier verifier;
  auto event_wake_or = require_service_attribute(verifier,
                                                 "holomq.event_wake",
                                                 options.enable_event_wake ? "enabled" : "disabled",
                                                 "event-wake");
  if (!event_wake_or.has_value()) {
    return holomq::make_unexpected(std::move(event_wake_or).error());
  }
  auto queue_policy_or =
      require_service_attribute(verifier,
                                "holomq.queue_policy",
                                queue_policy_attribute_value(options.queue_policy),
                                "queue-policy");
  if (!queue_policy_or.has_value()) {
    return holomq::make_unexpected(std::move(queue_policy_or).error());
  }
  // iceoryx2 identifies every byte-slice service as a u8 slice, so capacity and frame layout would
  // otherwise be invisible when a peer opens the service.
  if (options.byte_slice_capacity != 0U) {
    auto capacity_or = require_service_attribute(verifier,
                                                 "holomq.byte_slice_capacity",
                                                 std::to_string(options.byte_slice_capacity),
                                                 "byte-slice capacity");
    if (!capacity_or.has_value()) {
      return holomq::make_unexpected(std::move(capacity_or).error());
    }
    auto profile_or = require_service_attribute(verifier,
                                                "holomq.byte_slice_profile",
                                                std::to_string(options.byte_slice_profile),
                                                "byte-slice profile");
    if (!profile_or.has_value()) {
      return holomq::make_unexpected(std::move(profile_or).error());
    }
  }
  return verifier;
}

template <typename Payload>
using PubSubPortFactory =
    iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, IoxPayload<Payload>, void>;

/// Return a pub/sub service builder for \p Payload; byte slices start at a scalar-aligned address.
template <typename Payload>
[[nodiscard]] auto publish_subscribe_builder(iox2::Node<iox2::ServiceType::Ipc>& node,
                                             const iox2::ServiceName& name) {
  if constexpr (holomq::kIsByteSlice<Payload>) {
    return node.service_builder(name)
        .template publish_subscribe<IoxPayload<Payload>>()
        .payload_alignment(holomq::kByteSliceAlignment);
  } else {
    return node.service_builder(name).template publish_subscribe<IoxPayload<Payload>>();
  }
}

/// Reject byte-slice fields that do not match the payload kind before touching the backend.
template <typename Payload>
[[nodiscard]] auto validate_byte_slice_contract(const holomq::PubSubServiceOptions& options,
                                                const std::string& service_name,
                                                const char* endpoint_role,
                                                const std::uint64_t loans = 0U,
                                                const std::uint64_t expected_subscribers = 0U)
    -> holomq::expected<void, holomq::Error> {
  if (!holomq::detail::is_valid_byte_slice_contract(
          holomq::kIsByteSlice<Payload>, options.byte_slice_capacity, options.byte_slice_profile)) {
    return holomq::make_unexpected(holomq::Error{
        holomq::ErrorCode::kInvalidArgument,
        std::string(endpoint_role) + " service '" + service_name +
            (holomq::kIsByteSlice<Payload>
                 ? "': a byte-slice service requires a non-zero, representable capacity"
                 : "': a fixed-payload service cannot set a byte-slice capacity or profile")});
  }
  if constexpr (holomq::kIsByteSlice<Payload>) {
    const std::uint64_t subscribers =
        options.max_subscribers == 0U ? holomq::k_default_max_subscribers : options.max_subscribers;
    if (!holomq::detail::is_valid_byte_slice_pool_geometry(
            options, loans, std::max(subscribers, expected_subscribers))) {
      return holomq::make_unexpected(
          holomq::Error{holomq::ErrorCode::kInvalidArgument,
                        std::string(endpoint_role) + " service '" + service_name +
                            "': byte-slice sample pool geometry exceeds its limit"});
    }
  }
  return {};
}

constexpr std::uint32_t kEventServiceSetupAttemptLimit = 5U;
constexpr std::chrono::milliseconds kEventServiceSetupRetryDelay{1};

[[nodiscard]] inline auto open_or_create_event_service(iox2::Node<iox2::ServiceType::Ipc>& node,
                                                       const iox2::ServiceName& name,
                                                       const std::uint32_t max_listeners) {
  for (std::uint32_t attempt = 1U;; ++attempt) {
    auto event_builder = node.service_builder(name).event();
    if (max_listeners != 0U) {
      event_builder = std::move(event_builder).max_listeners(max_listeners);
    }
    auto event_factory_or = std::move(event_builder).open_or_create();
    if (event_factory_or.has_value() || attempt == kEventServiceSetupAttemptLimit ||
        !is_retryable_event_open_or_create_error(event_factory_or.error())) {
      return event_factory_or;
    }
    std::this_thread::sleep_for(kEventServiceSetupRetryDelay);
  }
}

[[nodiscard]] inline holomq::Error service_contract_mismatch(const std::string& service_name,
                                                             const char* endpoint_role,
                                                             const char* field,
                                                             const std::uint64_t expected,
                                                             const std::uint64_t actual) {
  return holomq::Error{holomq::ErrorCode::kIncompatible,
                       std::string(endpoint_role) + " service '" + service_name + "': static " +
                           field + " mismatch (expected " + std::to_string(expected) + ", actual " +
                           std::to_string(actual) + ")"};
}

template <typename Payload>
[[nodiscard]] auto verify_pub_sub_service_contract(const PubSubPortFactory<Payload>& port_factory,
                                                   const std::string& service_name,
                                                   const char* endpoint_role,
                                                   const holomq::PubSubServiceOptions& options)
    -> holomq::expected<void, holomq::Error> {
  const auto static_config = port_factory.static_config();
  const std::uint64_t expected_safe_overflow = options.uses_safe_overflow() ? 1U : 0U;
  const std::uint64_t actual_safe_overflow = static_config.has_safe_overflow() ? 1U : 0U;
  if (actual_safe_overflow != expected_safe_overflow) {
    return holomq::make_unexpected(service_contract_mismatch(service_name,
                                                             endpoint_role,
                                                             "safe-overflow policy",
                                                             expected_safe_overflow,
                                                             actual_safe_overflow));
  }
  if (options.max_subscribers != 0U && static_config.max_subscribers() != options.max_subscribers) {
    return holomq::make_unexpected(service_contract_mismatch(service_name,
                                                             endpoint_role,
                                                             "maximum subscriber count",
                                                             options.max_subscribers,
                                                             static_config.max_subscribers()));
  }
  if (static_config.subscriber_max_buffer_size() !=
      options.effective_max_subscriber_queue_depth()) {
    return holomq::make_unexpected(
        service_contract_mismatch(service_name,
                                  endpoint_role,
                                  "maximum subscriber queue depth",
                                  options.effective_max_subscriber_queue_depth(),
                                  static_config.subscriber_max_buffer_size()));
  }
  if (static_config.subscriber_max_borrowed_samples() !=
      options.effective_max_subscriber_borrowed_samples()) {
    return holomq::make_unexpected(
        service_contract_mismatch(service_name,
                                  endpoint_role,
                                  "maximum borrowed-sample count",
                                  options.effective_max_subscriber_borrowed_samples(),
                                  static_config.subscriber_max_borrowed_samples()));
  }
  if (static_config.history_size() != options.publisher_history_depth) {
    return holomq::make_unexpected(service_contract_mismatch(service_name,
                                                             endpoint_role,
                                                             "publisher history depth",
                                                             options.publisher_history_depth,
                                                             static_config.history_size()));
  }
  return {};
}

template <typename Payload>
[[nodiscard]] auto open_or_create_pub_sub_service(iox2::Node<iox2::ServiceType::Ipc>& node,
                                                  const iox2::ServiceName& name,
                                                  const std::string& service_name,
                                                  const char* endpoint_role,
                                                  const holomq::PubSubServiceOptions& options)
    -> holomq::expected<PubSubPortFactory<Payload>, holomq::Error> {
  auto attributes_or = make_service_attribute_verifier(options);
  if (!attributes_or.has_value()) {
    return holomq::make_unexpected(std::move(attributes_or).error());
  }
  auto attributes = std::move(attributes_or).value();

  auto open_or = publish_subscribe_builder<Payload>(node, name).open_with_attributes(attributes);
  if (open_or.has_value()) {
    auto port_factory = std::move(open_or).value();
    auto compatible_or = verify_pub_sub_service_contract<Payload>(
        port_factory, service_name, endpoint_role, options);
    if (!compatible_or.has_value()) {
      return holomq::make_unexpected(std::move(compatible_or).error());
    }
    return port_factory;
  }
  if (!is_missing_publish_subscribe_service(open_or.error())) {
    return holomq::make_unexpected(
        prepend_error_context(map_publish_subscribe_open_error(open_or.error()),
                              std::string(endpoint_role) + " service '" + service_name + "'"));
  }

  auto pub_sub_builder =
      publish_subscribe_builder<Payload>(node, name)
          .history_size(static_cast<std::uint64_t>(options.publisher_history_depth))
          .subscriber_max_buffer_size(
              static_cast<std::uint64_t>(options.effective_max_subscriber_queue_depth()))
          .subscriber_max_borrowed_samples(
              static_cast<std::uint64_t>(options.effective_max_subscriber_borrowed_samples()))
          .enable_safe_overflow(options.uses_safe_overflow());
  if (options.max_subscribers != 0U) {
    pub_sub_builder = std::move(pub_sub_builder).max_subscribers(options.max_subscribers);
  }

  auto port_factory_or = std::move(pub_sub_builder).open_or_create_with_attributes(attributes);
  if (!port_factory_or.has_value()) {
    return holomq::make_unexpected(
        prepend_error_context(map_publish_subscribe_open_or_create_error(port_factory_or.error()),
                              std::string(endpoint_role) + " service '" + service_name + "'"));
  }
  auto port_factory = std::move(port_factory_or).value();
  auto compatible_or =
      verify_pub_sub_service_contract<Payload>(port_factory, service_name, endpoint_role, options);
  if (!compatible_or.has_value()) {
    return holomq::make_unexpected(std::move(compatible_or).error());
  }
  return port_factory;
}

[[nodiscard]] inline std::unique_ptr<iox2::BackpressureHandler> make_bounded_retry_handler(
    const holomq::PublisherPubSubOptions& options) {
  const iox2::bb::Duration budget = iox2::bb::Duration::from_nanos(options.bounded_retry_budget_ns);
  const std::atomic<bool>* const cancellation = options.bounded_retry_cancellation;
  return std::make_unique<iox2::BackpressureHandler>(
      [budget, cancellation](const iox2::BackpressureInfo& info) noexcept {
        // An expired budget rejects as pressure even when cancellation is also set, so only a
        // wait the flag actually ended is reported as cancelled.
        const bool expired = info.elapsed_time() >= budget;
        const bool cancelled =
            !expired && cancellation != nullptr && cancellation->load(std::memory_order_acquire);
        if (!expired && !cancelled) {
          return iox2::BackpressureAction::Retry;
        }
        // iceoryx2 keeps delivering to later recipients after this one fails, so only a
        // recipient that actually rejected the sample may overwrite the send's evidence.
        record_unable_to_deliver_recipient(
            to_delivery_recipient_id_from_big_endian(info.receiver_port_id()), cancelled);
        return iox2::BackpressureAction::DiscardDataAndFail;
      });
}

// Build the setup-time callback without changing per-sample replacement attribution.
template <typename Payload>
[[nodiscard]] iox2::SampleReplacedHandler make_sample_replaced_handler(
    holomq::SampleReplacedCallback<Payload> sample_replaced_callback) {
  return iox2::SampleReplacedHandler([callback = std::move(sample_replaced_callback)](
                                         const iox2::SampleReplacedInfo& info) noexcept {
    try {
      holomq::SampleReplacedInfo replaced_info{};
      replaced_info.receiver_id = to_delivery_recipient_id(info.receiver_port_id());
      replaced_info.payload_ptr = info.payload_ptr();
      replaced_info.payload_size = info.payload_len();
      if constexpr (holomq::kIsByteSlice<Payload>) {
        if (callback && replaced_info.payload_ptr != nullptr) {
          callback(
              std::span<const std::byte>(static_cast<const std::byte*>(replaced_info.payload_ptr),
                                         replaced_info.payload_size),
              replaced_info);
        }
      } else if (callback && replaced_info.payload_ptr != nullptr &&
                 replaced_info.payload_size == sizeof(Payload)) {
        callback(*static_cast<const Payload*>(replaced_info.payload_ptr), replaced_info);
      }
    } catch (const std::exception& ex) {
      HOLOMQ_LOG_ERROR("iox2 sample replaced callback threw: {}", ex.what());
    } catch (...) {
      HOLOMQ_LOG_ERROR("iox2 sample replaced callback threw an unknown exception");
    }
  });
}

}  // namespace

template <typename Payload>
auto Runtime::create_publisher(
    const std::string& service_name,
    const std::shared_ptr<holomq::detail::PublisherSampleReleaseRegistry>& release_registry,
    holomq::SampleReplacedCallback<Payload> sample_replaced_callback,
    holomq::PubSubServiceOptions service_options, holomq::PublisherPubSubOptions options)
    -> holomq::expected<PublisherPort<Payload>, holomq::Error> {
  static_assert(std::is_standard_layout_v<Payload> && std::is_trivially_copyable_v<Payload>,
                "transport: Payload must be standard-layout and trivially copyable");

  if (options.enable_sample_release_callbacks != static_cast<bool>(release_registry)) {
    return holomq::make_unexpected(
        holomq::Error{holomq::ErrorCode::kInvalidArgument,
                      "publisher service '" + service_name +
                          "': sample release callback configuration does not match its registry"});
  }
  auto byte_slice_or =
      validate_byte_slice_contract<Payload>(service_options,
                                            service_name,
                                            "publisher",
                                            options.effective_max_loaned_samples(service_options),
                                            options.expected_recipient_count);
  if (!byte_slice_or.has_value()) {
    return holomq::make_unexpected(std::move(byte_slice_or).error());
  }

  const auto name_or = iox2::ServiceName::create(service_name.c_str());
  if (!name_or.has_value()) {
    return holomq::make_unexpected(prepend_error_context(
        map_service_name_error(name_or.error()), "publisher service '" + service_name + "'"));
  }
  const iox2::ServiceName name = std::move(name_or).value();

  auto port_factory_or = open_or_create_pub_sub_service<Payload>(
      node_, name, service_name, "publisher", service_options);
  if (!port_factory_or.has_value()) {
    return holomq::make_unexpected(std::move(port_factory_or).error());
  }
  auto port_factory = std::move(port_factory_or).value();

  if constexpr (holomq::kIsByteSlice<Payload>) {
    // Zero requested subscribers leaves the backend default or an existing service's actual
    // capacity unchanged. Revalidate that negotiated count before creating its static pool.
    if (!holomq::detail::is_valid_byte_slice_pool_geometry(
            service_options,
            options.effective_max_loaned_samples(service_options),
            port_factory.static_config().max_subscribers())) {
      return holomq::make_unexpected(
          holomq::Error{holomq::ErrorCode::kInvalidArgument,
                        "publisher service '" + service_name +
                            "': byte-slice sample pool geometry exceeds its limit"});
    }
  }
  auto builder = port_factory.publisher_builder();
  builder = std::move(builder).max_loaned_samples(
      static_cast<std::uint64_t>(options.effective_max_loaned_samples(service_options)));
  if constexpr (holomq::kIsByteSlice<Payload>) {
    // A static pool is sized once for the capacity; an oversized loan fails instead of growing it.
    builder = std::move(builder)
                  .initial_max_slice_len(service_options.byte_slice_capacity)
                  .allocation_strategy(iox2::AllocationStrategy::Static);
  }
  if (service_options.queue_policy == holomq::QueuePolicy::kBlocking) {
    builder =
        std::move(builder).backpressure_strategy(iox2::BackpressureStrategy::RetryUntilDelivered);
  } else {
    builder = std::move(builder).backpressure_strategy(iox2::BackpressureStrategy::DiscardData);
  }

  if (options.pressure_policy == holomq::PublisherPressurePolicy::kRejectWhenFull) {
    builder = std::move(builder).set_backpressure_handler(&reject_when_full_handler());
  }
  std::unique_ptr<iox2::BackpressureHandler> bounded_retry_handler;
  if (options.pressure_policy == holomq::PublisherPressurePolicy::kBoundedRetry) {
    bounded_retry_handler = make_bounded_retry_handler(options);
    builder = std::move(builder).set_backpressure_handler(bounded_retry_handler.get());
  }

  if (release_registry) {
    builder = std::move(builder).set_sample_released_handler(
        iox2::SampleReleasedHandler([release_registry](const iox2::SampleReleasedInfo& info) {
          release_registry->dispatch(info.payload_ptr());
        }));
  }
  if (service_options.queue_policy == holomq::QueuePolicy::kDropOldest &&
      options.replacement_attribution_required && sample_replaced_callback) {
    // HoloMQ reports queue replacement directly through the setup-time publisher callback.
    // Per-send replacement callbacks are intentionally not part of this hot path.
    builder = std::move(builder).set_sample_replaced_handler(
        make_sample_replaced_handler<Payload>(std::move(sample_replaced_callback)));
  }

  auto iox_publisher_or = std::move(builder).create();
  if (!iox_publisher_or.has_value()) {
    return holomq::make_unexpected(
        prepend_error_context(map_publisher_create_error(iox_publisher_or.error()),
                              "publisher service '" + service_name + "'"));
  }

  std::optional<iox2::Notifier<iox2::ServiceType::Ipc>> wake_notifier;
  if (service_options.enable_event_wake) {
    auto event_factory_or =
        open_or_create_event_service(node_, name, service_options.max_subscribers);
    if (!event_factory_or.has_value()) {
      return holomq::make_unexpected(
          prepend_error_context(map_event_open_or_create_error(event_factory_or.error()),
                                "publisher wake service '" + service_name + "'"));
    }
    auto notifier_or = std::move(event_factory_or).value().notifier_builder().create();
    if (!notifier_or.has_value()) {
      return holomq::make_unexpected(
          prepend_error_context(map_notifier_create_error(notifier_or.error()),
                                "publisher wake service '" + service_name + "'"));
    }
    wake_notifier.emplace(std::move(notifier_or).value());
  }

  return PublisherPort<Payload>(
      std::move(iox_publisher_or).value(),
      std::move(port_factory),
      std::move(wake_notifier),
      std::move(bounded_retry_handler),
      options.pressure_policy == holomq::PublisherPressurePolicy::kRejectWhenFull ||
          options.pressure_policy == holomq::PublisherPressurePolicy::kBoundedRetry);
}

template <typename Payload>
auto Runtime::create_subscriber(const std::string& service_name,
                                holomq::PubSubServiceOptions service_options,
                                holomq::SubscriberPubSubOptions options)
    -> holomq::expected<SubscriberPort<Payload>, holomq::Error> {
  static_assert(std::is_standard_layout_v<Payload> && std::is_trivially_copyable_v<Payload>,
                "transport: Payload must be standard-layout and trivially copyable");

  const auto name_or = iox2::ServiceName::create(service_name.c_str());
  if (!name_or.has_value()) {
    return holomq::make_unexpected(prepend_error_context(
        map_service_name_error(name_or.error()), "subscriber service '" + service_name + "'"));
  }
  const iox2::ServiceName name = std::move(name_or).value();

  auto byte_slice_or =
      validate_byte_slice_contract<Payload>(service_options, service_name, "subscriber");
  if (!byte_slice_or.has_value()) {
    return holomq::make_unexpected(std::move(byte_slice_or).error());
  }

  const std::uint32_t effective_queue_depth = options.effective_queue_depth(service_options);
  if (effective_queue_depth > service_options.effective_max_subscriber_queue_depth()) {
    return holomq::make_unexpected(
        holomq::Error{holomq::ErrorCode::kIncompatible,
                      "subscriber service '" + service_name +
                          "': requested queue depth exceeds the service maximum"});
  }
  if (options.effective_max_borrowed_samples() >
      service_options.effective_max_subscriber_borrowed_samples()) {
    return holomq::make_unexpected(
        holomq::Error{holomq::ErrorCode::kIncompatible,
                      "subscriber service '" + service_name +
                          "': requested borrowed-sample allowance exceeds the service maximum"});
  }

  auto create_port_from_factory =
      [&](auto port_factory) -> holomq::expected<SubscriberPort<Payload>, holomq::Error> {
    if constexpr (holomq::kIsByteSlice<Payload>) {
      if (!holomq::detail::is_valid_byte_slice_pool_geometry(
              service_options, 0U, port_factory.static_config().max_subscribers())) {
        return holomq::make_unexpected(
            holomq::Error{holomq::ErrorCode::kInvalidArgument,
                          "subscriber service '" + service_name +
                              "': byte-slice sample pool geometry exceeds its limit"});
      }
    }
    auto sub_builder = port_factory.subscriber_builder();
    auto iox_subscriber_or = std::move(sub_builder)
                                 .buffer_size(static_cast<std::uint64_t>(effective_queue_depth))
                                 .create();
    if (!iox_subscriber_or.has_value()) {
      return holomq::make_unexpected(
          prepend_error_context(map_subscriber_create_error(iox_subscriber_or.error()),
                                "subscriber service '" + service_name + "'"));
    }

    std::optional<iox2::Listener<iox2::ServiceType::Ipc>> wake_listener;
    if (service_options.enable_event_wake) {
      auto event_factory_or =
          open_or_create_event_service(node_, name, service_options.max_subscribers);
      if (!event_factory_or.has_value()) {
        return holomq::make_unexpected(
            prepend_error_context(map_event_open_or_create_error(event_factory_or.error()),
                                  "subscriber wake service '" + service_name + "'"));
      }
      auto listener_or = std::move(event_factory_or).value().listener_builder().create();
      if (!listener_or.has_value()) {
        return holomq::make_unexpected(
            prepend_error_context(map_listener_create_error(listener_or.error()),
                                  "subscriber wake service '" + service_name + "'"));
      }
      wake_listener.emplace(std::move(listener_or).value());
    }

    return SubscriberPort<Payload>(
        std::move(iox_subscriber_or).value(), std::move(port_factory), std::move(wake_listener));
  };

  auto port_factory_or = open_or_create_pub_sub_service<Payload>(
      node_, name, service_name, "subscriber", service_options);
  if (!port_factory_or.has_value()) {
    return holomq::make_unexpected(std::move(port_factory_or).error());
  }
  return create_port_from_factory(std::move(port_factory_or).value());
}

}  // namespace holomq::detail::transport
