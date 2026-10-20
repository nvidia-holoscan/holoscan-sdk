/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../core/domain/tensor_access.hpp"

#include "../channel/catalog_received_operation.hpp"
#include "../channel/channel_publisher_pressure.hpp"
#include "../channel/protected_terminal_channel.hpp"

#include <atomic>
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
                   const ChannelOptions& options,
                   const std::atomic<bool>* const stop_latch = nullptr,
                   const channel_internal::ReplacementObserver replacement_observer = {}) {
    return runtime.create_inline_payload_publisher(
        service,
        payload_size,
        options,
        graph_pressure_options(options, stop_latch, replacement_observer));
  }

  [[nodiscard]] static expected<std::unique_ptr<channel_internal::InlinePayloadSubscriber>, Error>
  create_subscriber(ChannelRuntime& runtime, std::string_view service, std::size_t payload_size,
                    const ChannelOptions& options) {
    return runtime.create_inline_payload_subscriber(service, payload_size, options);
  }

  [[nodiscard]] static expected<ManualTensorPublisher, Error> create_tensor_publisher(
      ChannelRuntime& runtime, std::string_view service, const ChannelOptions& options,
      const std::atomic<bool>* const stop_latch = nullptr,
      const channel_internal::ReplacementObserver replacement_observer = {}) {
    return runtime.create_tensor_publisher(
        service, options, graph_pressure_options(options, stop_latch, replacement_observer));
  }

  [[nodiscard]] static expected<ChannelPublisher<Tensor>, Error> create_catalog_tensor_publisher(
      ChannelRuntime& runtime, ChannelEndpointCatalog::EndpointView endpoint,
      const std::uint32_t publisher_loan_bound, const std::optional<DeviceId> planned_device,
      const std::atomic<bool>* const stop_latch = nullptr) {
    return runtime.create_catalog_tensor_publisher(endpoint,
                                                   {},
                                                   publisher_loan_bound,
                                                   TopicDirection::kExport,
                                                   false,
                                                   planned_device,
                                                   graph_blocking_pressure_options(stop_latch));
  }

  [[nodiscard]] static expected<ChannelSubscriber<Tensor>, Error> create_tensor_subscriber(
      ChannelRuntime& runtime, std::string_view service, const ChannelOptions& options,
      std::optional<TensorBoundaryContractRecord> receive_contract = std::nullopt,
      const std::uint32_t receive_control_capacity = 0U) {
    if (!receive_contract.has_value() && receive_control_capacity == 0U) {
      return runtime.create_tensor_subscriber(service, options);
    }
    return runtime.create_tensor_subscriber(
        service, options, std::move(receive_contract), receive_control_capacity);
  }

  [[nodiscard]] static expected<ChannelSubscriber<Tensor>, Error> create_catalog_tensor_subscriber(
      ChannelRuntime& runtime, ChannelEndpointCatalog::EndpointView endpoint,
      const std::uint32_t receive_control_capacity) {
    return runtime.create_catalog_tensor_subscriber(
        endpoint, {}, TopicDirection::kImport, false, receive_control_capacity);
  }

  [[nodiscard]] static expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>,
                                Error>
  create_tensor_schema_publisher(
      ChannelRuntime& runtime, std::string_view service, std::size_t payload_size,
      std::span<const std::size_t> descriptor_offsets, const ChannelOptions& options,
      std::span<const channel_internal::TensorSchemaStagingOptions> staging = {},
      cudaStream_t staging_stream = nullptr, const std::atomic<bool>* const stop_latch = nullptr,
      const channel_internal::ReplacementObserver replacement_observer = {},
      const std::shared_ptr<const holomq::cuda::StreamLifetime>& lifetime = {}) {
    const TensorConsumerScope scope{staging_stream, &lifetime};
    return runtime.create_tensor_schema_payload_publisher(
        service,
        payload_size,
        descriptor_offsets,
        options,
        graph_pressure_options(options, stop_latch, replacement_observer),
        staging,
        staging_stream);
  }

  [[nodiscard]] static expected<std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber>,
                                Error>
  create_tensor_schema_subscriber(ChannelRuntime& runtime, std::string_view service,
                                  std::size_t payload_size, std::uint32_t tensor_field_count,
                                  const ChannelOptions& options,
                                  const std::uint32_t receive_capacity = 0U) {
    return runtime.create_tensor_schema_payload_subscriber(service,
                                                           payload_size,
                                                           tensor_field_count,
                                                           options,
                                                           nullptr,
                                                           false,
                                                           {},
                                                           0U,
                                                           receive_capacity);
  }

  [[nodiscard]] static expected<std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber>,
                                Error>
  create_catalog_tensor_schema_subscriber(ChannelRuntime& runtime,
                                          ChannelEndpointCatalog::EndpointView endpoint,
                                          const SchemaIdentityRecord& expected_schema,
                                          const std::size_t payload_size,
                                          const std::uint32_t tensor_field_count) {
    return runtime.create_catalog_tensor_schema_payload_subscriber(
        endpoint,
        SchemaIdentity{.canonical_name = expected_schema.canonical_name,
                       .compatibility_epoch = expected_schema.compatibility_epoch,
                       .family_id = expected_schema.family_id,
                       .file_identifier = expected_schema.file_identifier},
        payload_size,
        tensor_field_count,
        {},
        TopicDirection::kImport,
        false);
  }

  [[nodiscard]] static expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>,
                                Error>
  create_catalog_tensor_schema_publisher(
      ChannelRuntime& runtime, ChannelEndpointCatalog::EndpointView endpoint,
      const SchemaIdentityRecord& expected_schema, const std::size_t payload_size,
      const std::span<const std::size_t> descriptor_offsets,
      const std::span<const channel_internal::TensorSchemaStagingOptions> staging,
      cudaStream_t staging_stream, const std::atomic<bool>* const stop_latch = nullptr,
      const std::shared_ptr<const holomq::cuda::StreamLifetime>& lifetime = {}) {
    const TensorConsumerScope scope{staging_stream, &lifetime};
    return runtime.create_catalog_tensor_schema_payload_publisher(
        endpoint,
        SchemaIdentity{.canonical_name = expected_schema.canonical_name,
                       .compatibility_epoch = expected_schema.compatibility_epoch,
                       .family_id = expected_schema.family_id,
                       .file_identifier = expected_schema.file_identifier},
        payload_size,
        descriptor_offsets,
        {},
        staging,
        staging_stream,
        graph_blocking_pressure_options(stop_latch),
        TopicDirection::kExport,
        false);
  }

  template <ChannelDescriptorFlatBufferPayload Payload>
  [[nodiscard]] static ChannelTensorSchemaPublisher<Payload> bind_tensor_schema_publisher(
      std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher> publisher,
      const bool runtime_envelope = false) {
    return ChannelTensorSchemaPublisher<Payload>(std::move(publisher), runtime_envelope);
  }

  template <ChannelDescriptorFlatBufferPayload Payload>
  [[nodiscard]] static ChannelTensorSchemaSubscriber<Payload> bind_tensor_schema_subscriber(
      std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber> subscriber,
      const bool runtime_envelope) {
    return ChannelTensorSchemaSubscriber<Payload>(std::move(subscriber), runtime_envelope);
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

  [[nodiscard]] static channel_internal::StagedDataWaitSetSubscriber* staged_tensor_subscriber(
      ChannelSubscriber<Tensor>& subscriber) noexcept {
    auto* const waitable = subscriber.wait_set_subscriber();
    return waitable != nullptr
               ? dynamic_cast<channel_internal::StagedDataWaitSetSubscriber*>(waitable)
               : nullptr;
  }

  [[nodiscard]] static std::uint64_t accepted_tensor_end_of_stream_sequence(
      const ChannelSubscriber<Tensor>& subscriber) noexcept {
    return subscriber.accepted_end_of_stream_sequence();
  }

  [[nodiscard]] static expected<Sample<Tensor>, Error> receive_tensor_sample(
      ChannelSubscriber<Tensor>& subscriber, const bool accept_data) {
    return subscriber.receive_sample_if_data_accepted(accept_data);
  }

  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<channel_internal::CatalogReceivedOperation,
                                                     Error>
  receive_catalog_tensor_operation(ChannelSubscriber<Tensor>& subscriber,
                                   bool accept_data) noexcept;

  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<bool, Error>
  has_pending_catalog_tensor_operation(const ChannelSubscriber<Tensor>& subscriber) noexcept;

  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<void, Error> detach_catalog_tensor_wait_set(
      ChannelSubscriber<Tensor>& subscriber) noexcept;

  [[nodiscard]] HOLOSCAN_CORE_EXPORT static std::uint64_t
  latched_catalog_tensor_producer_generation(const ChannelSubscriber<Tensor>& subscriber) noexcept;

  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<void, Error>
  supersede_catalog_tensor_producer_generation(ChannelSubscriber<Tensor>& subscriber,
                                               std::uint64_t generation) noexcept;

  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<channel_internal::CatalogImportQuiescence,
                                                     Error>
  catalog_tensor_import_quiescence(ChannelSubscriber<Tensor>& subscriber) noexcept;

  [[nodiscard]] static expected<channel_internal::CatalogReceivedOperation, Error>
  receive_catalog_tensor_schema_operation(
      channel_internal::TensorSchemaPayloadSubscriber& subscriber, bool accept_data) noexcept {
    auto* const catalog = catalog_tensor_schema_subscriber(subscriber);
    if (catalog == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    return catalog->receive_catalog_operation(accept_data);
  }

  template <ChannelDescriptorFlatBufferPayload Payload>
  [[nodiscard]] static expected<channel_internal::CatalogReceivedOperation, Error>
  receive_catalog_tensor_schema_operation(ChannelTensorSchemaSubscriber<Payload>& subscriber,
                                          const bool accept_data) noexcept {
    if (!subscriber.subscriber_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return receive_catalog_tensor_schema_operation(*subscriber.subscriber_, accept_data);
  }

  template <ChannelDescriptorFlatBufferPayload Payload>
  static void release_catalog_tensor_schema_payload_loan(
      ChannelTensorSchemaSubscriber<Payload>& subscriber) noexcept {
    if (subscriber.subscriber_) {
      subscriber.subscriber_->release_payload_loan();
    }
  }

  [[nodiscard]] static expected<bool, Error> has_pending_catalog_tensor_schema_operation(
      const channel_internal::TensorSchemaPayloadSubscriber& subscriber) noexcept {
    const auto* const catalog = catalog_tensor_schema_subscriber(subscriber);
    if (catalog == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    return catalog->has_pending_catalog_operation();
  }

  [[nodiscard]] static expected<void, Error> detach_catalog_tensor_schema_wait_set(
      channel_internal::TensorSchemaPayloadSubscriber& subscriber) noexcept {
    auto* const catalog = catalog_tensor_schema_subscriber(subscriber);
    if (catalog == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    return catalog->detach_catalog_wait_set();
  }

  [[nodiscard]] static expected<std::uint64_t, Error>
  latched_catalog_tensor_schema_producer_generation(
      const channel_internal::TensorSchemaPayloadSubscriber& subscriber) noexcept {
    const auto* const catalog = catalog_tensor_schema_subscriber(subscriber);
    if (catalog == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    return catalog->latched_producer_generation();
  }

  [[nodiscard]] static expected<void, Error> supersede_catalog_tensor_schema_producer_generation(
      channel_internal::TensorSchemaPayloadSubscriber& subscriber,
      const std::uint64_t generation) noexcept {
    auto* const catalog = catalog_tensor_schema_subscriber(subscriber);
    if (catalog == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    return catalog->supersede_producer_generation(generation);
  }

  [[nodiscard]] static expected<channel_internal::CatalogImportQuiescence, Error>
  catalog_tensor_schema_import_quiescence(
      channel_internal::TensorSchemaPayloadSubscriber& subscriber) noexcept {
    auto* const catalog = catalog_tensor_schema_subscriber(subscriber);
    if (catalog == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    return catalog->import_quiescence();
  }

  /// Per-recipient budget for a graph-owned blocking publication to wait on a full recipient. A
  /// transported send retries in the backend; a direct-retained publication to a consumer on
  /// another execution lane waits for typed capacity in Runtime.
  static constexpr std::uint64_t kGraphBlockingRetryBudgetNs = 5'000'000U;

  /// @brief Return the finite graph-owned policy for one frozen service queue.
  ///
  /// The five-millisecond budget applies independently to each intended recipient. A publication
  /// to N pressured recipients therefore has an aggregate upper bound of N times this value. The
  /// backend consults the budget only after queue pressure; accepted sends do not read a clock or
  /// enter a retry loop.
  ///
  /// `stop_latch` is the session's monotonic stop latch. Once it is set, a full recipient is
  /// rejected at once instead of retried, so a stop never waits out the budget.
  /// `replacement_observer` counts deliveries that a drop-oldest queue replaced unreceived.
  [[nodiscard]] static constexpr channel_internal::PublisherPressureOptions graph_pressure_options(
      const ChannelOptions& options, const std::atomic<bool>* const stop_latch = nullptr,
      const channel_internal::ReplacementObserver replacement_observer = {}) noexcept {
    auto pressure = options.queue_policy == ChannelQueuePolicy::kBlocking
                        ? graph_blocking_pressure_options(stop_latch)
                        : channel_internal::PublisherPressureOptions::follow_service();
    pressure.replacement_observer = replacement_observer;
    return pressure;
  }

  [[nodiscard]] static constexpr channel_internal::PublisherPressureOptions
  graph_blocking_pressure_options(const std::atomic<bool>* const stop_latch = nullptr) noexcept {
    return channel_internal::PublisherPressureOptions::bounded_retry(kGraphBlockingRetryBudgetNs,
                                                                     stop_latch);
  }

 private:
  [[nodiscard]] static channel_internal::CatalogTensorSchemaPayloadSubscriber*
  catalog_tensor_schema_subscriber(
      channel_internal::TensorSchemaPayloadSubscriber& subscriber) noexcept {
    return dynamic_cast<channel_internal::CatalogTensorSchemaPayloadSubscriber*>(&subscriber);
  }

  [[nodiscard]] static const channel_internal::CatalogTensorSchemaPayloadSubscriber*
  catalog_tensor_schema_subscriber(
      const channel_internal::TensorSchemaPayloadSubscriber& subscriber) noexcept {
    return dynamic_cast<const channel_internal::CatalogTensorSchemaPayloadSubscriber*>(&subscriber);
  }
};

}  // namespace holoscan::detail
