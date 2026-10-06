/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/detail/transport/iox2/publisher_port.hpp>
#include <holomq/detail/transport/iox2/subscriber_port.hpp>

#include <iox2/node.hpp>
#include <iox2/service_type.hpp>

#include <holomq/delivery_attribution.hpp>
#include <holomq/detail/publisher_sample_release_registry.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/pubsub_options.hpp>

#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace holomq::detail::transport {

class Iox2WaitSetBackend;

class Runtime {
 public:
  Runtime() = default;
  Runtime(Runtime&&) noexcept = default;
  Runtime& operator=(Runtime&&) noexcept = default;
  ~Runtime() = default;

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  /// @brief Create an iceoryx2 publisher port for a typed service.
  /// @tparam Payload Standard-layout, trivially-copyable payload type.
  /// @param service_name HoloMQ service name used to join publishers and subscribers.
  /// @param release_registry Registry used for sample-release callbacks, or null when callbacks
  /// are disabled in @p options.
  /// @param sample_replaced_callback Optional setup-time callback for lossy-queue replacement
  /// attribution.
  /// @param service_options Static pub/sub service contract shared by both endpoint roles.
  /// @param options Publisher-local loan and attribution options.
  /// @return Publisher port or a typed service, resource, permission, or configuration error.
  template <typename Payload>
  [[nodiscard]] auto create_publisher(
      const std::string& service_name,
      const std::shared_ptr<holomq::detail::PublisherSampleReleaseRegistry>& release_registry,
      holomq::SampleReplacedCallback<Payload> sample_replaced_callback,
      holomq::PubSubServiceOptions service_options = {},
      holomq::PublisherPubSubOptions options = {})
      -> holomq::expected<PublisherPort<Payload>, holomq::Error>;

  /// @brief Create an iceoryx2 subscriber port for a typed service.
  /// @tparam Payload Standard-layout, trivially-copyable payload type.
  /// @param service_name HoloMQ service name used to join publishers and subscribers.
  /// @param service_options Static pub/sub service contract shared by both endpoint roles.
  /// @param options Subscriber-local queue and borrow options.
  /// @return Subscriber port or a typed service, resource, permission, or configuration error.
  template <typename Payload>
  [[nodiscard]] auto create_subscriber(const std::string& service_name,
                                       holomq::PubSubServiceOptions service_options = {},
                                       holomq::SubscriberPubSubOptions options = {})
      -> holomq::expected<SubscriberPort<Payload>, holomq::Error>;

 private:
  friend holomq::expected<Runtime, holomq::Error> create_runtime();
  friend class Iox2WaitSetBackend;

  [[nodiscard]] iox2::Node<iox2::ServiceType::Ipc>& node_for_waitset() noexcept { return node_; }

  explicit Runtime(iox2::Node<iox2::ServiceType::Ipc> node) noexcept : node_(std::move(node)) {}

  iox2::Node<iox2::ServiceType::Ipc> node_;
};

[[nodiscard]] holomq::expected<Runtime, holomq::Error> create_runtime();

}  // namespace holomq::detail::transport

#include <holomq/detail/transport/iox2/runtime_impl.hpp>
