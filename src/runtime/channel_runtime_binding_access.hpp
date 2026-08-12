/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../channel/channel_publisher_pressure.hpp"
#include "../channel/protected_terminal_channel.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/channel/manual_tensor_publisher.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/channel/tensor_endpoints.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::detail {

/// @brief Expose only the private inline endpoints needed by the Core session binder.
///
/// This source-local access class keeps ordinary Channel factories public while preventing
/// Runtime from fabricating a typed C++ endpoint through RTTI or a size-only public cast.
class RuntimeChannelBindingAccess {
 public:
  [[nodiscard]] static expected<std::unique_ptr<channel_internal::InlinePayloadPublisher>, Error>
  create_publisher(ChannelRuntime& runtime, std::string_view service, std::size_t payload_size,
                   const ChannelOptions& options) {
    return runtime.create_inline_payload_publisher(
        service, payload_size, options, graph_pressure_options(options));
  }

  [[nodiscard]] static expected<std::unique_ptr<channel_internal::InlinePayloadSubscriber>, Error>
  create_subscriber(ChannelRuntime& runtime, std::string_view service, std::size_t payload_size,
                    const ChannelOptions& options) {
    return runtime.create_inline_payload_subscriber(service, payload_size, options);
  }

  [[nodiscard]] static expected<ManualTensorPublisher, Error> create_tensor_publisher(
      ChannelRuntime& runtime, std::string_view service, const ChannelOptions& options) {
    return runtime.create_tensor_publisher(service, options, graph_pressure_options(options));
  }

  [[nodiscard]] static expected<ChannelPublisher<Tensor>, Error> create_catalog_tensor_publisher(
      ChannelRuntime& runtime, ChannelEndpointCatalog::EndpointView endpoint,
      const std::uint32_t publisher_loan_bound, const std::optional<DeviceId> planned_device) {
    return runtime.create_catalog_tensor_publisher(endpoint,
                                                   {},
                                                   publisher_loan_bound,
                                                   TopicDirection::kExport,
                                                   false,
                                                   planned_device,
                                                   graph_blocking_pressure_options());
  }

  [[nodiscard]] static expected<ChannelSubscriber<Tensor>, Error> create_tensor_subscriber(
      ChannelRuntime& runtime, std::string_view service, const ChannelOptions& options,
      std::optional<LogicalTensorRequirementsRecord> receive_contract = std::nullopt,
      const std::uint32_t receive_control_capacity = 0U) {
    if (!receive_contract.has_value() && receive_control_capacity == 0U) {
      return runtime.create_tensor_subscriber(service, options);
    }
    return runtime.create_tensor_subscriber(
        service, options, std::move(receive_contract), receive_control_capacity);
  }

  [[nodiscard]] static expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>,
                                Error>
  create_tensor_schema_publisher(
      ChannelRuntime& runtime, std::string_view service, std::size_t payload_size,
      std::span<const std::size_t> descriptor_offsets, const ChannelOptions& options,
      std::span<const channel_internal::TensorSchemaStagingOptions> staging = {}) {
    return runtime.create_tensor_schema_payload_publisher(service,
                                                          payload_size,
                                                          descriptor_offsets,
                                                          options,
                                                          graph_pressure_options(options),
                                                          staging);
  }

  [[nodiscard]] static expected<std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber>,
                                Error>
  create_tensor_schema_subscriber(ChannelRuntime& runtime, std::string_view service,
                                  std::size_t payload_size, std::uint32_t tensor_field_count,
                                  const ChannelOptions& options) {
    return runtime.create_tensor_schema_payload_subscriber(
        service, payload_size, tensor_field_count, options);
  }

  [[nodiscard]] static expected<std::unique_ptr<channel_internal::InlinePayloadWaitSet>, Error>
  create_wait_set(ChannelRuntime& runtime) {
    return runtime.create_inline_payload_wait_set();
  }

  [[nodiscard]] static expected<std::uint64_t, Error> attach_wait_set(
      channel_internal::InlinePayloadWaitSet& wait_set, ChannelSubscriber<Tensor>& subscriber) {
    auto* const waitable = subscriber.wait_set_subscriber();
    if (waitable == nullptr) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return wait_set.attach(*waitable);
  }

  [[nodiscard]] static std::optional<std::uint64_t> secondary_wait_set_handle(
      ChannelSubscriber<Tensor>& subscriber) noexcept {
    auto* const waitable = subscriber.wait_set_subscriber();
    return waitable != nullptr ? channel_internal::protected_terminal_wait_handle(*waitable)
                               : std::nullopt;
  }

  [[nodiscard]] static std::uint64_t accepted_tensor_end_of_stream_sequence(
      const ChannelSubscriber<Tensor>& subscriber) noexcept {
    return subscriber.accepted_end_of_stream_sequence();
  }

  [[nodiscard]] static expected<Sample<Tensor>, Error> receive_tensor_sample(
      ChannelSubscriber<Tensor>& subscriber, const bool accept_data) {
    return subscriber.receive_sample_if_data_accepted(accept_data);
  }

  /// @brief Return the finite graph-owned policy for one frozen service queue.
  ///
  /// The five-millisecond budget applies independently to each intended recipient. A publication
  /// to N pressured recipients therefore has an aggregate upper bound of N times this value. The
  /// backend consults the budget only after queue pressure; accepted sends do not read a clock or
  /// enter a retry loop.
  [[nodiscard]] static constexpr channel_internal::PublisherPressureOptions graph_pressure_options(
      const ChannelOptions& options) noexcept {
    return options.queue_policy == ChannelQueuePolicy::kBlocking
               ? graph_blocking_pressure_options()
               : channel_internal::PublisherPressureOptions::follow_service();
  }

  [[nodiscard]] static constexpr channel_internal::PublisherPressureOptions
  graph_blocking_pressure_options() noexcept {
    return channel_internal::PublisherPressureOptions::bounded_retry(kGraphBlockingRetryBudgetNs);
  }

 private:
  static constexpr std::uint64_t kGraphBlockingRetryBudgetNs = 5'000'000U;
};

}  // namespace holoscan::detail
