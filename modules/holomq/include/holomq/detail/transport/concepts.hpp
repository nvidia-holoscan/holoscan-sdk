/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/delivery_attribution.hpp>
#include <holomq/detail/publisher_sample_release_registry.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/pubsub_options.hpp>
#include <holomq/wait_strategy.hpp>

#include <concepts>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace holomq::detail::transport::concepts {

template <typename Loan, typename Payload>
concept OutboundLoanLike =
    std::movable<Loan> && !std::copy_constructible<Loan> && requires(Loan loan) {
      { loan.payload() } -> std::same_as<Payload&>;
      { std::as_const(loan).payload() } -> std::same_as<const Payload&>;
      { loan.release_key() } -> std::same_as<const void*>;
    };

template <typename Message, typename Payload>
concept InboundMessageLike =
    std::movable<Message> && !std::copy_constructible<Message> && requires(const Message message) {
      { message.payload() } -> std::same_as<const Payload&>;
    };

template <typename Port, typename Payload, typename Loan>
concept PublisherPortLike = std::move_constructible<Port> && !std::copy_constructible<Port> &&
                            requires(Port port, Loan loan) {
                              {
                                port.loan()
                              } -> std::same_as<holomq::expected<Loan, holomq::Error>>;
                              {
                                port.send(std::move(loan))
                              } -> std::same_as<holomq::expected<int, holomq::Error>>;
                              {
                                port.update_connections()
                              } -> std::same_as<holomq::expected<void, holomq::Error>>;
                              { port.number_of_subscribers() } -> std::same_as<std::uint64_t>;
                            };

template <typename Port, typename InMsg>
concept SubscriberPortLike =
    std::movable<Port> && !std::copy_constructible<Port> &&
    requires(Port port, const Port& const_port) {
      { port.receive() } -> std::same_as<holomq::expected<std::optional<InMsg>, holomq::ErrorCode>>;
      { port.try_take() } -> std::same_as<holomq::expected<InMsg, holomq::Error>>;
      { port.has_samples() } -> std::same_as<holomq::expected<bool, holomq::ErrorCode>>;
      { const_port.has_publishers() } -> std::same_as<bool>;
      { const_port.recipient_id() } -> std::same_as<holomq::DeliveryRecipientId>;
    };

template <typename Port, typename InMsg>
concept WaitableSubscriberPortLike = requires(Port port, holomq::WaitStrategy strategy) {
  { port.has_samples() } -> std::same_as<holomq::expected<bool, holomq::ErrorCode>>;
  { port.is_event_wake_enabled() } -> std::same_as<bool>;
  { port.drain_wake_events() } -> std::same_as<holomq::expected<bool, holomq::ErrorCode>>;
  {
    port.wait_for_sample(strategy)
  } -> std::same_as<holomq::expected<holomq::WaitResult, holomq::ErrorCode>>;
  { port.wait_and_take(strategy) } -> std::same_as<holomq::expected<InMsg, holomq::Error>>;
};

template <typename RuntimeT, typename Payload, typename PubPort, typename SubPort>
concept RuntimeLike =
    std::movable<RuntimeT> && !std::copy_constructible<RuntimeT> &&
    requires(
        RuntimeT runtime, const std::string& service_name,
        const std::shared_ptr<holomq::detail::PublisherSampleReleaseRegistry>& release_registry,
        holomq::PubSubServiceOptions service_options,
        holomq::PublisherPubSubOptions publisher_options,
        holomq::SubscriberPubSubOptions subscriber_options) {
      {
        runtime.template create_publisher<Payload>(service_name,
                                                   release_registry,
                                                   holomq::SampleReplacedCallback<Payload>{},
                                                   service_options,
                                                   publisher_options)
      } -> std::same_as<holomq::expected<PubPort, holomq::Error>>;
      {
        runtime.template create_subscriber<Payload>(
            service_name, service_options, subscriber_options)
      } -> std::same_as<holomq::expected<SubPort, holomq::Error>>;
    };

}  // namespace holomq::detail::transport::concepts
