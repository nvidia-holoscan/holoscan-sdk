/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/detail/native_descriptor_readiness.hpp>
#include <holomq/detail/transport/iox2/connection_error.hpp>
#include <holomq/detail/transport/iox2/id_conversion.hpp>
#include <holomq/detail/transport/iox2/inbound_message.hpp>
#include <holomq/detail/transport/iox2/listener_error.hpp>
#include <holomq/detail/transport/iox2/pubsub_wake_event.hpp>
#include <holomq/detail/transport/iox2/receive_error.hpp>

#include <iox2/event_id.hpp>
#include <iox2/listener.hpp>
#include <iox2/port_factory_publish_subscribe.hpp>
#include <iox2/service_type.hpp>
#include <iox2/subscriber.hpp>

#include <holomq/detail/wait_strategy_utils.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/logger.hpp>
#include <holomq/wait_strategy.hpp>

#include <chrono>
#include <cstdint>
#include <optional>
#include <thread>
#include <utility>

namespace holomq::detail::transport {

class Iox2WaitSetBackend;

template <typename Payload>
class SubscriberPort {
 public:
  struct ReadinessSource {
    int descriptor{-1};
    bool initially_ready{false};
  };

  SubscriberPort() = default;
  SubscriberPort(SubscriberPort&&) noexcept = default;
  SubscriberPort& operator=(SubscriberPort&&) noexcept = default;
  ~SubscriberPort() = default;

  SubscriberPort(const SubscriberPort&) = delete;
  SubscriberPort& operator=(const SubscriberPort&) = delete;

  /// Non-blocking receive. An empty value means no sample is queued; an error reports transport
  /// failure; otherwise the value contains a message loan.
  [[nodiscard]] auto receive()
      -> holomq::expected<std::optional<InboundMessage<Payload>>, holomq::ErrorCode>;

  /// Non-blocking take. Returns typed queue, backpressure, and connection errors.
  [[nodiscard]] holomq::expected<InboundMessage<Payload>, holomq::Error> try_take();

  /// Return whether the iceoryx2 subscriber reports a sample available to take.
  [[nodiscard]] auto has_samples() const -> holomq::expected<bool, holomq::ErrorCode>;

  /// Return whether the service currently reports at least one publisher.
  [[nodiscard]] bool has_publishers() const noexcept {
    return port_factory_.dynamic_config().number_of_publishers() > 0U;
  }

  /// @brief Return the backend-independent recipient id for this subscriber.
  /// @return Converted iceoryx2 unique subscriber port id, or a default id when the backend
  /// does not expose id bytes.
  [[nodiscard]] holomq::DeliveryRecipientId recipient_id() const {
    const auto id = inner_.id();
    const auto& bytes = id.bytes();
    if (!bytes.has_value()) {
      return {};
    }
    return to_delivery_recipient_id(bytes.value());
  }

  /// \c true when the subscriber was created with event wake enabled and has a \c Listener.
  [[nodiscard]] bool is_event_wake_enabled() const noexcept { return listener_.has_value(); }

  /// Default \ref holomq::WaitStrategy for \ref wait_for_sample when none is passed explicitly.
  void set_wait_strategy(holomq::WaitStrategy strategy) noexcept { wait_strategy_ = strategy; }

  [[nodiscard]] holomq::WaitStrategy get_wait_strategy() const noexcept { return wait_strategy_; }

  /// Drain pending publisher wake events and return \c true if a sample may be available.
  [[nodiscard]] auto drain_wake_events() -> holomq::expected<bool, holomq::ErrorCode>;

  /// Block until a sample may be available or the strategy times out. \ref WaitMode::TimedWait
  /// always sleeps in userspace; \ref WaitMode::KernelWait uses the kernel listener when event wake
  /// is enabled. After a successful kernel wake, returns \ref holomq::WaitResult::ready even when
  /// \ref has_samples is still false (spurious wake); callers that need a loan must retry \ref
  /// receive.
  [[nodiscard]] auto wait_for_sample(const holomq::WaitStrategy& strategy)
      -> holomq::expected<holomq::WaitResult, holomq::ErrorCode>;

  /// Block until a sample is available; loops on spurious wake (ready wait but empty \ref receive).
  [[nodiscard]] holomq::expected<InboundMessage<Payload>, holomq::Error> wait_and_take(
      const holomq::WaitStrategy& strategy);

  [[nodiscard]] bool is_attached_to_wait_set() const noexcept {
    return wait_owner_ != WaitOwner::kNone;
  }

  [[nodiscard]] bool is_attached_to_readiness_handle() const noexcept {
    return wait_owner_ == WaitOwner::kReadinessHandle;
  }

  /// Claim the listener descriptor for one externally owned readiness handle.
  [[nodiscard]] auto acquire_readiness() -> holomq::expected<ReadinessSource, holomq::ErrorCode>;

  /// Drain pending listener notifications and report whether a sample remains queued.
  [[nodiscard]] auto acknowledge_readiness() -> holomq::expected<bool, holomq::ErrorCode>;

  /// Release an external readiness-handle claim.
  void release_readiness() noexcept { wait_owner_ = WaitOwner::kNone; }

 private:
  friend class Runtime;
  friend class Iox2WaitSetBackend;

  void set_attached_to_wait_set(const bool attached) noexcept {
    wait_owner_ = attached ? WaitOwner::kWaitSet : WaitOwner::kNone;
  }

  [[nodiscard]] iox2::Listener<iox2::ServiceType::Ipc>* get_listener_if_ready() noexcept {
    return listener_.has_value() ? &listener_.value() : nullptr;
  }

  explicit SubscriberPort(
      iox2::Subscriber<iox2::ServiceType::Ipc, Payload, void> inner,
      iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, Payload, void> port_factory,
      std::optional<iox2::Listener<iox2::ServiceType::Ipc>> wake_listener) noexcept
      : inner_(std::move(inner)),
        port_factory_(std::move(port_factory)),
        listener_(std::move(wake_listener)) {}

  using IoxSubscriber = iox2::Subscriber<iox2::ServiceType::Ipc, Payload, void>;
  using IoxPortFactory = iox2::PortFactoryPublishSubscribe<iox2::ServiceType::Ipc, Payload, void>;

  IoxSubscriber inner_;
  IoxPortFactory port_factory_;
  holomq::WaitStrategy wait_strategy_{};
  std::optional<iox2::Listener<iox2::ServiceType::Ipc>> listener_;
  enum class WaitOwner : std::uint8_t {
    kNone,
    kWaitSet,
    kReadinessHandle,
  };

  WaitOwner wait_owner_{WaitOwner::kNone};
};

template <typename Payload>
auto SubscriberPort<Payload>::receive()
    -> holomq::expected<std::optional<InboundMessage<Payload>>, holomq::ErrorCode> {
  auto received = inner_.receive();
  if (!received.has_value()) {
    return holomq::make_unexpected(map_receive_error_code(received.error()));
  }
  auto& optional_sample = received.value();
  if (!optional_sample.has_value()) {
    return std::optional<InboundMessage<Payload>>{};
  }
  return std::optional<InboundMessage<Payload>>(
      InboundMessage<Payload>(std::move(optional_sample).value()));
}

template <typename Payload>
holomq::expected<InboundMessage<Payload>, holomq::Error> SubscriberPort<Payload>::try_take() {
  auto received = receive();
  if (!received.has_value()) {
    HOLOMQ_LOG_ERROR("subscriber receive failed: {}", holomq::error_code_name(received.error()));
    return holomq::make_unexpected(holomq::Error{received.error()});
  }
  auto& optional_message = received.value();
  if (!optional_message.has_value()) {
    return holomq::make_unexpected(holomq::Error{holomq::ErrorCode::kNotReady});
  }
  return InboundMessage<Payload>(std::move(optional_message).value());
}

template <typename Payload>
auto SubscriberPort<Payload>::has_samples() const -> holomq::expected<bool, holomq::ErrorCode> {
  auto ready = inner_.has_samples();
  if (!ready.has_value()) {
    return holomq::make_unexpected(map_connection_error_code(ready.error()));
  }
  return ready.value();
}

template <typename Payload>
auto SubscriberPort<Payload>::drain_wake_events() -> holomq::expected<bool, holomq::ErrorCode> {
  if (!listener_.has_value()) {
    return has_samples();
  }

  const auto event_or = listener_->try_wait_one();
  if (!event_or.has_value()) {
    return holomq::make_unexpected(map_listener_wait_error_code(event_or.error()));
  }
  const bool saw_sent_sample =
      event_or.value().has_value() &&
      event_or.value()->as_value() == static_cast<std::size_t>(PubSubWakeEvent::SentSample);
  auto ready = has_samples();
  if (!ready.has_value()) {
    return holomq::forward_error(std::move(ready));
  }
  return saw_sent_sample || ready.value();
}

template <typename Payload>
auto SubscriberPort<Payload>::acquire_readiness()
    -> holomq::expected<ReadinessSource, holomq::ErrorCode> {
#if !HOLOMQ_HAS_NATIVE_DESCRIPTOR_READINESS
  return holomq::make_unexpected(holomq::ErrorCode::kNotSupported);
#else
  if (!listener_.has_value()) {
    return holomq::make_unexpected(holomq::ErrorCode::kNotSupported);
  }
  if (is_attached_to_wait_set()) {
    return holomq::make_unexpected(holomq::ErrorCode::kInvalidArgument);
  }

  auto ready = has_samples();
  if (!ready) {
    return holomq::make_unexpected(ready.error());
  }
  const int descriptor = listener_->file_descriptor().unsafe_native_handle();
  if (descriptor < 0) {
    return holomq::make_unexpected(holomq::ErrorCode::kFailure);
  }
  wait_owner_ = WaitOwner::kReadinessHandle;
  return ReadinessSource{.descriptor = descriptor, .initially_ready = *ready};
#endif
}

template <typename Payload>
auto SubscriberPort<Payload>::acknowledge_readiness() -> holomq::expected<bool, holomq::ErrorCode> {
#if !HOLOMQ_HAS_NATIVE_DESCRIPTOR_READINESS
  return holomq::make_unexpected(holomq::ErrorCode::kNotSupported);
#else
  if (wait_owner_ != WaitOwner::kReadinessHandle) {
    return holomq::make_unexpected(holomq::ErrorCode::kInvalidArgument);
  }
  if (!listener_.has_value()) {
    return holomq::make_unexpected(holomq::ErrorCode::kNotSupported);
  }

  while (true) {
    const auto event = listener_->try_wait_one();
    if (!event.has_value()) {
      return holomq::make_unexpected(map_listener_wait_error_code(event.error()));
    }
    if (!event->has_value()) {
      return has_samples();
    }
  }
#endif
}

template <typename Payload>
auto SubscriberPort<Payload>::wait_for_sample(const holomq::WaitStrategy& strategy)
    -> holomq::expected<holomq::WaitResult, holomq::ErrorCode> {
  if (is_attached_to_wait_set()) {
    return holomq::make_unexpected(holomq::ErrorCode::kInvalidArgument);
  }
  holomq::detail::warn_if_ignored_timeout_on_return(strategy);

  const auto spin_until = std::chrono::steady_clock::now() + strategy.spin_duration;

  while (std::chrono::steady_clock::now() < spin_until) {
    auto ready = has_samples();
    if (!ready.has_value()) {
      return holomq::forward_error(std::move(ready));
    }
    if (ready.value()) {
      return holomq::WaitResult::ready;
    }
    std::this_thread::yield();
  }

  if (strategy.mode == holomq::WaitMode::Return) {
    auto ready = has_samples();
    if (!ready.has_value()) {
      return holomq::forward_error(std::move(ready));
    }
    return ready.value() ? holomq::WaitResult::ready : holomq::WaitResult::timeout;
  }

  if (strategy.mode == holomq::WaitMode::TimedWait) {
    std::this_thread::sleep_for(holomq::detail::timed_wait_sleep_duration(strategy));
    auto ready = has_samples();
    if (!ready.has_value()) {
      return holomq::forward_error(std::move(ready));
    }
    return ready.value() ? holomq::WaitResult::ready : holomq::WaitResult::timeout;
  }

  if (!listener_.has_value()) {
    return holomq::make_unexpected(holomq::ErrorCode::kInvalidArgument);
  }

  if (strategy.timeout.has_value()) {
    const iox2::bb::Duration timeout =
        iox2::bb::Duration::from_nanos(static_cast<std::uint64_t>(strategy.timeout->count()));
    const auto event_or = listener_->timed_wait_one(timeout);
    if (!event_or.has_value()) {
      return holomq::make_unexpected(map_listener_wait_error_code(event_or.error()));
    }
    if (!event_or.value().has_value()) {
      return holomq::WaitResult::timeout;
    }
  } else {
    const auto event_or = listener_->blocking_wait_one();
    if (!event_or.has_value()) {
      return holomq::make_unexpected(map_listener_wait_error_code(event_or.error()));
    }
  }
  auto drained = drain_wake_events();
  if (!drained.has_value()) {
    return holomq::forward_error(std::move(drained));
  }
  return holomq::WaitResult::ready;
}

template <typename Payload>
holomq::expected<InboundMessage<Payload>, holomq::Error> SubscriberPort<Payload>::wait_and_take(
    const holomq::WaitStrategy& strategy) {
  for (;;) {
    auto wait_result = wait_for_sample(strategy);
    if (!wait_result.has_value()) {
      HOLOMQ_LOG_ERROR("subscriber wait failed: {}", holomq::error_code_name(wait_result.error()));
      if (wait_result.error() == holomq::ErrorCode::kInvalidArgument && is_attached_to_wait_set()) {
        return holomq::make_unexpected_nothrow<InboundMessage<Payload>>(
            wait_result.error(),
            is_attached_to_readiness_handle() ? "subscriber is already owned by a readiness handle"
                                              : "subscriber is already owned by a wait set");
      }
      return holomq::make_unexpected(holomq::Error{wait_result.error()});
    }
    if (wait_result.value() == holomq::WaitResult::timeout) {
      return holomq::make_unexpected(holomq::Error{holomq::ErrorCode::kTimeout});
    }
    if (wait_result.value() != holomq::WaitResult::ready) {
      return holomq::make_unexpected(holomq::Error{holomq::ErrorCode::kReceiveError});
    }

    auto received = receive();
    if (!received.has_value()) {
      HOLOMQ_LOG_ERROR("subscriber receive failed: {}", holomq::error_code_name(received.error()));
      return holomq::make_unexpected(holomq::Error{received.error()});
    }
    if (received->has_value()) {
      return InboundMessage<Payload>(std::move(received->value()));
    }
  }
}

}  // namespace holomq::detail::transport
