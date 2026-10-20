/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <holomq/context_builder.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/memory/block_memory_pool.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../common/cuda_test_utils.hpp"
#include "image_generated.h"

#include "channel/channel_common.hpp"
#include "channel/channel_payload_frame.hpp"
#include "channel/channel_payload_limits.hpp"
#include "channel/channel_tensor_test_access.hpp"
#include "channel/endpoint_catalog_state.hpp"
#include "core/graph_compiler_access.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/endpoint_clock_health_runtime.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding_test_access.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_external_channel_test_support.hpp"
#include "runtime_test_graph.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

using namespace test::external_channel;

[[nodiscard]] expected<void, Error> wait_for_catalog_tensor_subscriber(
    ChannelPublisher<Tensor>& publisher, ChannelSubscriber<Tensor>& subscriber,
    std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::uint32_t consecutive_observations = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto receiver_refreshed = subscriber.has_samples();
    if (!receiver_refreshed) {
      return make_unexpected(std::move(receiver_refreshed).error());
    }
    auto publisher_refreshed = publisher.update_connections();
    if (!publisher_refreshed) {
      return make_unexpected(std::move(publisher_refreshed).error());
    }
    if (subscriber.has_publishers() && publisher.has_subscribers()) {
      if (++consecutive_observations == 2U) {
        return {};
      }
    } else {
      consecutive_observations = 0U;
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

template <ChannelDescriptorFlatBufferPayload Payload>
[[nodiscard]] expected<void, Error> wait_for_catalog_schema_subscriber(
    ChannelTensorSchemaPublisher<Payload>& publisher,
    channel_internal::TensorSchemaPayloadSubscriber& subscriber,
    std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::uint32_t consecutive_observations = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto receiver_refreshed = subscriber.has_samples();
    if (!receiver_refreshed) {
      return make_unexpected(std::move(receiver_refreshed).error());
    }
    auto publisher_refreshed = publisher.update_connections();
    if (!publisher_refreshed) {
      return make_unexpected(std::move(publisher_refreshed).error());
    }
    if (subscriber.has_publishers() && publisher.has_subscribers()) {
      if (++consecutive_observations == 2U) {
        return {};
      }
    } else {
      consecutive_observations = 0U;
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

template <typename Publisher>
[[nodiscard]] expected<void, Error> wait_for_no_subscribers(
    Publisher& publisher, std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    if (!publisher.has_subscribers()) {
      return {};
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

[[nodiscard]] expected<channel_internal::CatalogReceivedOperation, Error>
wait_for_catalog_tensor_operation(ChannelSubscriber<Tensor>& subscriber,
                                  std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto operation =
        detail::RuntimeChannelBindingAccess::receive_catalog_tensor_operation(subscriber, true);
    if (!operation || operation->kind != channel_internal::CatalogReceivedOperationKind::kNone) {
      return operation;
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

// Descriptor-backed schema imports share the catalog's ordered health stream with scalar and raw
// Tensor imports. A preceding unhealthy report suppresses the following sample without exposing
// its schema value to compute or failing the run.
// REQ: CHAN-069
// REQ: EXTERNAL-010
// REQ: RUNTIME-016
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest,
     DescriptorTensorSchemaImportAppliesPrecedingClockHealthBeforeCompute) {
  ExternalImageImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-health"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  const std::int64_t capture_offset = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count();
  graph.add_clock_conversion(
      capture_clock, runtime_clock, AffineClockConversion{.offset_ns = capture_offset});
  TopicOptions options = external_image_import_options(endpoint_name);
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(options), external_image_contract());
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  // Key setup: place a healthy report before the first descriptor Tensor and an unhealthy report
  // before the second within the catalog's shared ordered operation stream.
  RunSession session = run_async(plan);
  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto first_tensor = make_external_image_tensor(0x31U);
  ASSERT_TRUE(first_tensor) << runtime_error_message(first_tensor.error());
  auto first_metadata = schema::build_image_metadata(
      *first_tensor,
      schema::ImageMetadataOptions{
          .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
  ASSERT_TRUE(first_metadata) << runtime_error_message(first_metadata.error());
  ASSERT_TRUE(publisher->emit_captured(*first_metadata, std::move(*first_tensor), 1));
  ASSERT_TRUE(ExternalImageImportProbe::wait_for_count(1U));

  ASSERT_TRUE(publisher->report_clock_health(
      ClockHealthReport::unhealthy(ClockHealthReason::kSynchronizationLockLost)));
  auto suppressed_tensor = make_external_image_tensor(0x62U);
  ASSERT_TRUE(suppressed_tensor) << runtime_error_message(suppressed_tensor.error());
  auto suppressed_metadata = schema::build_image_metadata(
      *suppressed_tensor,
      schema::ImageMetadataOptions{
          .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
  ASSERT_TRUE(suppressed_metadata) << runtime_error_message(suppressed_metadata.error());
  ASSERT_TRUE(publisher->emit_captured(*suppressed_metadata, std::move(*suppressed_tensor), 2));
  std::this_thread::sleep_for(100ms);

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  // Key assertion: only the Tensor preceded by healthy state reaches typed compute; suppression
  // of the second sample is nonterminal.
  EXPECT_EQ(ExternalImageImportProbe::count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(ExternalImageImportProbe::first_byte.load(std::memory_order_relaxed), 0x31U);
}

// The descriptor-schema readiness facade must expose control-only clock-health traffic so a
// standalone consumer can decide to drain the next ordered catalog operation without waiting for
// unrelated Tensor data.
// REQ: CHAN-069
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest, DescriptorTensorSchemaHasSamplesIncludesClockHealthLane) {
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-health-readiness"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  graph.add_clock_conversion(capture_clock, runtime_clock, AffineClockConversion{});
  TopicOptions options = external_image_import_options(endpoint_name);
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(options), external_image_contract());
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << (plan.diagnostics().empty() ? "<no diagnostic>"
                                                        : plan.diagnostics().front().token);
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
  const auto* const endpoint_record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
  const auto* const type_ops = detail::runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(endpoint_record, nullptr);
  ASSERT_NE(type_ops, nullptr);
  ASSERT_TRUE(endpoint_record->payload_identity.schema);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  const auto& payload_schema = *endpoint_record->payload_identity.schema;

  auto graph_runtime = ChannelRuntime::create();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime) << runtime_error_message(graph_runtime.error());
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto subscriber = detail::RuntimeChannelBindingAccess::create_catalog_tensor_schema_subscriber(
      *graph_runtime,
      *endpoint,
      payload_schema,
      type_ops->wire_capacity,
      type_ops->descriptor_tensor_static_count);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  // Key setup: probe the normalized field contract retained by the actual catalog subscriber
  // with otherwise structural metadata that independently violates each constrained dimension.
  const PayloadDescriptor valid_descriptor{
      .payload_kind = ChannelPayloadKind::kLocalBufferDescriptor,
      .schema_id = schema_identity<schema::Tensor>().family_id,
      .sequence = 1U,
      .byte_size = 8U,
      .local =
          LocalPayloadReference{
              .process_id = 1U, .token = 1U, .generation = 1U, .expected_recipient_count = 1U},
  };
  const TensorViewMetadata valid_view{
      .rank = 2U,
      .dtype = kExternalTensorDtype,
      .device = DLDevice{.device_type = kDLCPU, .device_id = 0},
      .byte_size = 8U,
      .shape = {1, 8},
  };
  const auto expect_field_rejection = [&](TensorViewMetadata candidate,
                                          PayloadDescriptor descriptor) {
    auto validated = (*subscriber)->validate_field_descriptor(0U, candidate, descriptor);
    ASSERT_FALSE(validated);
    EXPECT_EQ(validated.error().code, ErrorCode::kInvalidArgument);
  };
  auto wrong_dtype = valid_view;
  wrong_dtype.dtype.bits = 16U;
  expect_field_rejection(wrong_dtype, valid_descriptor);
  auto wrong_rank = valid_view;
  wrong_rank.rank = 1U;
  expect_field_rejection(wrong_rank, valid_descriptor);
  auto wrong_memory = valid_view;
  wrong_memory.device = DLDevice{.device_type = kDLCUDA, .device_id = 0};
  expect_field_rejection(wrong_memory, valid_descriptor);
  auto wrong_strides = valid_view;
  wrong_strides.has_explicit_byte_strides = true;
  wrong_strides.byte_strides = {16, 1};
  expect_field_rejection(wrong_strides, valid_descriptor);
  auto wrong_span = valid_view;
  wrong_span.byte_size = 9U;
  auto oversized_descriptor = valid_descriptor;
  oversized_descriptor.byte_size = 9U;
  expect_field_rejection(wrong_span, oversized_descriptor);
  auto mismatched_descriptor = valid_descriptor;
  mismatched_descriptor.byte_size = 7U;
  expect_field_rejection(valid_view, mismatched_descriptor);

  // Key assertion: all constrained dimensions reject before import, while the exact field view
  // remains admissible through the same contract hook used by standalone and Runtime facades.
  auto valid_field = (*subscriber)->validate_field_descriptor(0U, valid_view, valid_descriptor);
  ASSERT_TRUE(valid_field) << runtime_error_message(valid_field.error());
  auto publisher = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_schema_subscriber(*publisher, **subscriber));

  // Key setup: publish only a healthy clock-control operation; no descriptor Tensor data is
  // available to make the data or terminal lanes ready.
  auto reported = publisher->report_clock_health(ClockHealthReport::healthy());
  ASSERT_TRUE(reported) << runtime_error_message(reported.error());
  const auto readiness_deadline = std::chrono::steady_clock::now() + 5s;
  expected<bool, Error> ready{false};
  while (std::chrono::steady_clock::now() < readiness_deadline) {
    ready = (*subscriber)->has_samples();
    ASSERT_TRUE(ready) << runtime_error_message(ready.error());
    if (*ready) {
      break;
    }
    std::this_thread::sleep_for(1ms);
  }

  // Key assertion: has_samples reports the queued health lane and the same operation drains as
  // the first ordered clock-health record.
  ASSERT_TRUE(*ready);
  auto operation = wait_for_catalog_tensor_schema_operation(**subscriber);
  ASSERT_TRUE(operation) << runtime_error_message(operation.error());
  ASSERT_EQ(operation->kind, channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(operation->clock_health.publisher_operation_sequence, 1U);
  EXPECT_EQ(operation->clock_health.report_sequence, 1U);
}

// Descriptor-schema EOS is authoritative lifecycle control only after the complete terminal
// record, including its protected lane, has passed canonical validation.
// REQ: EXTERNAL-009
// REQ: SYSTEM-024
TEST(RuntimeExternalChannelTest, MalformedDescriptorSchemaEndOfStreamCannotAuthorizeRebind) {
  enum class Malformation : std::uint8_t {
    kMetadata,
    kPayload,
    kLane,
  };
  constexpr std::array kMalformations{
      Malformation::kMetadata, Malformation::kPayload, Malformation::kLane};
  using Envelope = channel_internal::TensorSchemaRuntimeEnvelope<ExternalImagePayload>;
  constexpr std::size_t kPayloadCapacity =
      channel_internal::inline_payload_capacity(sizeof(Envelope)).value_or(0U);
  static_assert(kPayloadCapacity != 0U);
  constexpr auto kProfile = channel_internal::PayloadFrameProfile::kTensorSchema;

  for (const Malformation malformation : kMalformations) {
    SCOPED_TRACE(static_cast<std::uint32_t>(malformation));
    const std::string endpoint_name = unique_endpoint_name();
    Graph graph{"runtime-external-malformed-image-eos"};
    const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
    graph.set_default_clock(runtime_clock);
    const auto imported =
        graph.import_topic<schema::ImageT>("holomq://" + endpoint_name,
                                           external_image_import_options(endpoint_name),
                                           external_image_contract());
    const auto sink = graph.op<ExternalImageImportSink>("sink");
    graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto compiled_endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
    ASSERT_TRUE(compiled_endpoint) << runtime_error_message(compiled_endpoint.error());
    const auto* const compiled_record =
        detail::ChannelEndpointCatalogAccess::record(*compiled_endpoint);
    ASSERT_NE(compiled_record, nullptr);
    auto lossy_record = *compiled_record;
    lossy_record.queue_policy = QueuePolicy::kDropOldest;
    lossy_record.service_static.queue_policy = QueuePolicy::kDropOldest;
    auto lossy_catalog = detail::ChannelEndpointCatalogAccess::normalize(
        "runtime-external-malformed-image-eos", {std::move(lossy_record)});
    ASSERT_TRUE(lossy_catalog) << runtime_error_message(lossy_catalog.error());
    const auto endpoint = lossy_catalog->endpoint(endpoint_name);
    ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
    const auto* const endpoint_record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
    const auto* const type_ops = detail::runtime_port_type_ops<schema::ImageT>();
    ASSERT_NE(endpoint_record, nullptr);
    ASSERT_NE(type_ops, nullptr);
    ASSERT_TRUE(endpoint_record->payload_identity.schema);
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
    const auto& payload_schema = *endpoint_record->payload_identity.schema;
    ASSERT_EQ(endpoint_record->service_static.queue_policy, QueuePolicy::kDropOldest);

    auto subscriber_runtime = ChannelRuntime::create();
    ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());
    auto subscriber = detail::RuntimeChannelBindingAccess::create_catalog_tensor_schema_subscriber(
        *subscriber_runtime,
        *endpoint,
        payload_schema,
        type_ops->wire_capacity,
        type_ops->descriptor_tensor_static_count);
    ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
    auto publisher_context = holomq::ContextBuilder().with_vmm_fd_export(true).create();
    ASSERT_TRUE(publisher_context) << publisher_context.error().message;

    const std::string service =
        detail::ChannelEndpointCatalogAccess::backend_service_binding(*endpoint);
    const holomq::PubSubServiceOptions data_service_options{
        .queue_policy = holomq::QueuePolicy::kDropOldest,
        .max_subscribers = 1U,
        .max_subscriber_queue_depth = endpoint_record->service_static.max_subscriber_depth,
        .max_subscriber_borrowed_samples = endpoint_record->service_static.max_subscriber_borrowed,
        .publisher_history_depth = endpoint_record->service_static.publisher_history_depth,
        .enable_event_wake = endpoint_record->service_static.event_wake,
    };
    auto data_publisher = publisher_context->publisher_builder(service)
                              .service_options(channel_internal::payload_frame_service_options(
                                  data_service_options, kProfile, kPayloadCapacity))
                              .publisher_options(holomq::PublisherPubSubOptions{
                                  .max_loaned_samples = 1U,
                                  .expected_recipient_count = 1U,
                              })
                              .payload_type<holomq::ByteSlice>()
                              .create();
    ASSERT_TRUE(data_publisher) << data_publisher.error().message;
    const holomq::PubSubServiceOptions terminal_service_options{
        .queue_policy = holomq::QueuePolicy::kBlocking,
        .max_subscribers = 1U,
        .max_subscriber_queue_depth = 1U,
        .max_subscriber_borrowed_samples = 1U,
        .publisher_history_depth = 0U,
        .enable_event_wake = endpoint_record->service_static.event_wake,
    };
    auto terminal_publisher = publisher_context->publisher_builder(service + ".hsdk-terminal-v1")
                                  .service_options(channel_internal::payload_frame_service_options(
                                      terminal_service_options, kProfile, kPayloadCapacity))
                                  .publisher_options(holomq::PublisherPubSubOptions{
                                      .max_loaned_samples = 1U,
                                      .expected_recipient_count = 1U,
                                  })
                                  .payload_type<holomq::ByteSlice>()
                                  .create();
    ASSERT_TRUE(terminal_publisher) << terminal_publisher.error().message;

    const auto connection_deadline = std::chrono::steady_clock::now() + 2s;
    bool connected = false;
    while (!connected && std::chrono::steady_clock::now() < connection_deadline) {
      ASSERT_TRUE((*subscriber)->has_samples());
      ASSERT_TRUE(data_publisher->update_connections());
      ASSERT_TRUE(terminal_publisher->update_connections());
      connected = data_publisher->has_subscribers() && terminal_publisher->has_subscribers();
      if (!connected) {
        std::this_thread::sleep_for(1ms);
      }
    }
    ASSERT_TRUE(connected);

    // Key setup: forge one EOS with either noncanonical metadata, nonempty payload bytes, or the
    // lossy data lane instead of the protected terminal lane.
    auto* selected_publisher =
        malformation == Malformation::kLane ? &*data_publisher : &*terminal_publisher;
    const std::uint32_t payload_size = malformation == Malformation::kPayload ? 1U : 0U;
    auto terminal = selected_publisher->loan(channel_internal::kTensorSchemaPayloadFrameDataOffset +
                                             payload_size);
    ASSERT_TRUE(terminal) << terminal.error().message;
    channel_internal::write_payload_frame_header(
        kProfile,
        terminal->payload(),
        channel_internal::PayloadFrameFields{
            .operation = channel_internal::InlinePayloadOperation::kEndOfStream,
            .metadata =
                SampleMetadata{
                    .sequence = 1U,
                    .trace_id = malformation == Malformation::kMetadata ? 7U : 0U,
                    .payload_kind = PayloadKind::kDescriptorBacked,
                },
            .producer_generation = 1U,
            .payload_size = payload_size,
        });
    auto sent = selected_publisher->send(std::move(*terminal));
    ASSERT_TRUE(sent) << sent.error().message;
    auto received = wait_for_catalog_tensor_schema_operation(**subscriber);

    // Key assertion: validation rejects before generation/EOS authority is latched, so lifecycle
    // replacement cannot mistake the malformed terminal record for a completed stream.
    ASSERT_FALSE(received);
    EXPECT_EQ(received.error().code, ErrorCode::kProtocolError);
    auto generation =
        detail::RuntimeChannelBindingAccess::latched_catalog_tensor_schema_producer_generation(
            **subscriber);
    ASSERT_TRUE(generation) << runtime_error_message(generation.error());
    EXPECT_EQ(*generation, 0U);
    auto quiescence =
        detail::RuntimeChannelBindingAccess::catalog_tensor_schema_import_quiescence(**subscriber);
    ASSERT_TRUE(quiescence) << runtime_error_message(quiescence.error());
    EXPECT_EQ(quiescence->latched_producer_generation, 0U);
  }
}

// Raw Tensor imports use the same ordered catalog health protocol as descriptor-backed schemas.
// A healthy report authorizes the following sample, while a preceding unhealthy report suppresses
// data before the Tensor reaches compute.
// REQ: CHAN-069
// REQ: EXTERNAL-010
// REQ: RUNTIME-016
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest, RawTensorImportAppliesPrecedingClockHealthBeforeCompute) {
  ExternalTensorExportSink::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-tensor-health"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  const std::int64_t capture_offset = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch())
                                          .count();
  graph.add_clock_conversion(
      capture_clock, runtime_clock, AffineClockConversion{.offset_ns = capture_offset});
  TopicOptions options = optional_tensor_export_options(endpoint_name);
  options.bounds.subscriber.reset();
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalTensorExportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  // Key setup: place healthy and unhealthy health operations immediately before two raw Tensor
  // publications on the same catalog sequence.
  RunSession session = run_async(plan);
  ASSERT_TRUE(wait_for_subscriber(*publisher));
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto first_capture = publisher->make_capture_time(1);
  ASSERT_TRUE(first_capture) << runtime_error_message(first_capture.error());
  ASSERT_TRUE(publish_external_host_tensor(*publisher, *first_capture));
  ASSERT_TRUE(ExternalTensorExportSink::wait_for_sample());

  ASSERT_TRUE(publisher->report_clock_health(
      ClockHealthReport::unhealthy(ClockHealthReason::kSynchronizationLockLost)));
  auto suppressed_capture = publisher->make_capture_time(2);
  ASSERT_TRUE(suppressed_capture) << runtime_error_message(suppressed_capture.error());
  ASSERT_TRUE(publish_external_host_tensor(*publisher, *suppressed_capture));
  std::this_thread::sleep_for(100ms);

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  // Key assertion: raw Tensor ingress exposes only the healthy-authorized sample without failing
  // the session on the deliberately suppressed publication.
  EXPECT_EQ(ExternalTensorExportSink::received.load(std::memory_order_acquire), 1U);
}

// A replacement descriptor-schema attachment starts after an earlier subscriber consumed data
// and health operations. Its first visible data and first visible health operation establish the
// new attachment's sequence baseline without weakening strict ordering after either baseline.
// REQ: CHAN-069
// REQ: RUNTIME-008
TEST(RuntimeExternalChannelTest,
     DescriptorTensorSchemaReplacementAttachmentBaselinesDataAndClockHealth) {
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-attachment-baseline"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  graph.add_clock_conversion(capture_clock, runtime_clock, AffineClockConversion{});
  TopicOptions options = external_image_import_options(endpoint_name);
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(options), external_image_contract());
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  const auto* const endpoint_record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
  const auto* const type_ops = detail::runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(endpoint_record, nullptr);
  ASSERT_NE(type_ops, nullptr);
  ASSERT_TRUE(endpoint_record->payload_identity.schema);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  const auto& payload_schema = *endpoint_record->payload_identity.schema;

  auto graph_runtime = ChannelRuntime::create();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime) << runtime_error_message(graph_runtime.error());
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  const auto create_subscriber = [&]() {
    return detail::RuntimeChannelBindingAccess::create_catalog_tensor_schema_subscriber(
        *graph_runtime,
        *endpoint,
        payload_schema,
        type_ops->wire_capacity,
        type_ops->descriptor_tensor_static_count);
  };
  const auto emit = [&](const std::uint8_t value,
                        const std::int64_t capture_timestamp_ns) -> expected<void, Error> {
    auto tensor = make_external_image_tensor(value);
    if (!tensor) {
      return make_unexpected(std::move(tensor).error());
    }
    auto metadata = schema::build_image_metadata(
        *tensor,
        schema::ImageMetadataOptions{
            .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
    if (!metadata) {
      return make_unexpected(std::move(metadata).error());
    }
    auto emitted = publisher->emit_captured(*metadata, std::move(*tensor), capture_timestamp_ns);
    return emitted ? expected<void, Error>{}
                   : expected<void, Error>{make_unexpected(std::move(emitted).error())};
  };

  auto first = create_subscriber();
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));
  ASSERT_TRUE(emit(0x11U, 1));
  auto first_data = wait_for_catalog_tensor_schema_operation(**first);
  ASSERT_TRUE(first_data) << runtime_error_message(first_data.error());
  ASSERT_EQ(first_data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(first_data->metadata.sequence, 1U);
  (*first)->release_payload_loan();
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto first_health = wait_for_catalog_tensor_schema_operation(**first);
  ASSERT_TRUE(first_health) << runtime_error_message(first_health.error());
  ASSERT_EQ(first_health->kind, channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(first_health->clock_health.publisher_operation_sequence, 2U);
  EXPECT_EQ(first_health->clock_health.report_sequence, 1U);
  first->reset();
  ASSERT_TRUE(wait_for_no_subscribers(*publisher));

  // Key setup: replace the receive attachment after the publisher has already consumed one data
  // and one health operation, so neither sequence domain starts at one for this subscriber.
  auto data_baseline = create_subscriber();
  ASSERT_TRUE(data_baseline) << runtime_error_message(data_baseline.error());
  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));
  ASSERT_TRUE(emit(0x22U, 2));
  auto later_data = wait_for_catalog_tensor_schema_operation(**data_baseline);
  ASSERT_TRUE(later_data) << runtime_error_message(later_data.error());
  // Key assertion: the first visible data operation establishes both the operation-sequence and
  // carried health-report baselines; the immediately following health report remains contiguous.
  ASSERT_EQ(later_data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(later_data->metadata.sequence, 3U);
  (*data_baseline)->release_payload_loan();
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto later_health = wait_for_catalog_tensor_schema_operation(**data_baseline);
  ASSERT_TRUE(later_health) << runtime_error_message(later_health.error());
  ASSERT_EQ(later_health->kind, channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(later_health->clock_health.publisher_operation_sequence, 4U);
  EXPECT_EQ(later_health->clock_health.report_sequence, 2U);
  data_baseline->reset();
  ASSERT_TRUE(wait_for_no_subscribers(*publisher));

  auto health_baseline = create_subscriber();
  ASSERT_TRUE(health_baseline) << runtime_error_message(health_baseline.error());
  ASSERT_TRUE(wait_for_schema_subscriber(*publisher));
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto first_visible_health = wait_for_catalog_tensor_schema_operation(**health_baseline);
  ASSERT_TRUE(first_visible_health) << runtime_error_message(first_visible_health.error());
  // Key assertion: a health-first replacement independently adopts both nonzero sequence values.
  ASSERT_EQ(first_visible_health->kind,
            channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(first_visible_health->clock_health.publisher_operation_sequence, 5U);
  EXPECT_EQ(first_visible_health->clock_health.report_sequence, 3U);
}

// Catalog descriptor-schema sequencing must arbitrate a final data record ahead of protected EOS
// when the terminal becomes visible after the receiver's first empty data probe.
// REQ: CHAN-136
TEST(RuntimeExternalChannelTest,
     DescriptorTensorSchemaCatalogProtectedTerminalRechecksFinalDataAfterInitialProbe) {
  const std::string endpoint_name = unique_endpoint_name();
  Graph graph{"runtime-external-image-terminal-recheck"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  const auto imported =
      graph.import_topic<schema::ImageT>("holomq://" + endpoint_name,
                                         external_image_import_options(endpoint_name),
                                         external_image_contract());
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(),
                 sink->input,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
  const auto* const endpoint_record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
  const auto* const type_ops = detail::runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(endpoint_record, nullptr);
  ASSERT_NE(type_ops, nullptr);
  ASSERT_TRUE(endpoint_record->payload_identity.schema);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  const auto& payload_schema = *endpoint_record->payload_identity.schema;

  auto subscriber_runtime = ChannelRuntime::create();
  auto publisher_runtime = ChannelRuntime::create();
  ASSERT_TRUE(subscriber_runtime) << runtime_error_message(subscriber_runtime.error());
  ASSERT_TRUE(publisher_runtime) << runtime_error_message(publisher_runtime.error());
  channel_internal::TensorPublisherTestAccess::ReceiveProbeBarrier barrier;
  channel_internal::TensorPublisherTestAccess::
      attach_receive_probe_barrier_to_next_subscriber_for_test(barrier);
  auto subscriber = detail::RuntimeChannelBindingAccess::create_catalog_tensor_schema_subscriber(
      *subscriber_runtime,
      *endpoint,
      payload_schema,
      type_ops->wire_capacity,
      type_ops->descriptor_tensor_static_count);
  auto publisher = publisher_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_schema_subscriber(*publisher, **subscriber));

  auto receive_future = std::async(std::launch::async, [&subscriber] {
    return detail::RuntimeChannelBindingAccess::receive_catalog_tensor_schema_operation(
        **subscriber, true);
  });
  const bool first_data_probe_completed = barrier.reached.try_acquire_for(2s);
  std::optional<expected<std::uint32_t, Error>> emitted;
  std::optional<expected<void, Error>> closed;
  if (first_data_probe_completed) {
    auto tensor = make_external_image_tensor(0x39U);
    if (tensor) {
      auto metadata = schema::build_image_metadata(
          *tensor,
          schema::ImageMetadataOptions{
              .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
      if (metadata) {
        emitted.emplace(publisher->emit(*metadata, std::move(*tensor)));
      }
    }
    closed.emplace(publisher->close());
  }
  barrier.resume.release();

  auto data = receive_future.get();
  ASSERT_TRUE(first_data_probe_completed);
  ASSERT_TRUE(emitted.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  ASSERT_TRUE(*emitted) << runtime_error_message(emitted->error());
  ASSERT_TRUE(closed.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  ASSERT_TRUE(*closed) << runtime_error_message(closed->error());
  ASSERT_TRUE(data) << runtime_error_message(data.error());
  ASSERT_EQ(data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(data->metadata.sequence, 1U);
  (*subscriber)->release_payload_loan();

  auto terminal = detail::RuntimeChannelBindingAccess::receive_catalog_tensor_schema_operation(
      **subscriber, true);
  ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
  EXPECT_EQ(terminal->kind, channel_internal::CatalogReceivedOperationKind::kEndOfStream);
  EXPECT_EQ(terminal->metadata.sequence, 2U);
}

// Raw Tensor attachments use the same sequence-baseline rule as descriptor-schema attachments.
// Exercise both a first-visible data sample carrying an established health watermark and a
// first-visible health report after replacing an earlier subscriber.
// REQ: CHAN-069
// REQ: RUNTIME-008
TEST(RuntimeExternalChannelTest, RawTensorReplacementAttachmentBaselinesDataAndClockHealth) {
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-tensor-attachment-baseline"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  graph.add_clock_conversion(capture_clock, runtime_clock, AffineClockConversion{});
  TopicOptions options = optional_tensor_export_options(endpoint_name);
  options.bounds.subscriber.reset();
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalTensorExportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  auto graph_runtime = ChannelRuntime::create();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime) << runtime_error_message(graph_runtime.error());
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());
  const auto create_subscriber = [&]() {
    return detail::RuntimeChannelBindingAccess::create_catalog_tensor_subscriber(
        *graph_runtime, *endpoint, 0U);
  };
  const auto publish = [&](const std::int64_t capture_timestamp_ns) -> expected<void, Error> {
    auto capture_time = publisher->make_capture_time(capture_timestamp_ns);
    if (!capture_time) {
      return make_unexpected(std::move(capture_time).error());
    }
    return publish_external_host_tensor(*publisher, *capture_time);
  };

  auto created_first = create_subscriber();
  ASSERT_TRUE(created_first) << runtime_error_message(created_first.error());
  std::optional<ChannelSubscriber<Tensor>> first{std::move(*created_first)};
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*publisher, *first));
  ASSERT_TRUE(publish(1));
  auto first_data = wait_for_catalog_tensor_operation(*first);
  ASSERT_TRUE(first_data) << runtime_error_message(first_data.error());
  ASSERT_EQ(first_data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(first_data->metadata.sequence, 1U);
  first_data->tensor_sample.reset();
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto first_health = wait_for_catalog_tensor_operation(*first);
  ASSERT_TRUE(first_health) << runtime_error_message(first_health.error());
  ASSERT_EQ(first_health->kind, channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(first_health->clock_health.publisher_operation_sequence, 2U);
  EXPECT_EQ(first_health->clock_health.report_sequence, 1U);
  first.reset();
  ASSERT_TRUE(wait_for_no_subscribers(*publisher));

  // Key setup: replace the receive attachment after one raw Tensor and one health report have
  // advanced the publisher's independent operation and report sequence domains.
  auto created_data_baseline = create_subscriber();
  ASSERT_TRUE(created_data_baseline) << runtime_error_message(created_data_baseline.error());
  std::optional<ChannelSubscriber<Tensor>> data_baseline{std::move(*created_data_baseline)};
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*publisher, *data_baseline));
  ASSERT_TRUE(publish(2));
  auto later_data = wait_for_catalog_tensor_operation(*data_baseline);
  ASSERT_TRUE(later_data) << runtime_error_message(later_data.error());
  // Key assertion: first-visible raw data adopts the carried health watermark and later health
  // still has to be exactly contiguous in both sequence domains.
  ASSERT_EQ(later_data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(later_data->metadata.sequence, 3U);
  later_data->tensor_sample.reset();
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto later_health = wait_for_catalog_tensor_operation(*data_baseline);
  ASSERT_TRUE(later_health) << runtime_error_message(later_health.error());
  ASSERT_EQ(later_health->kind, channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(later_health->clock_health.publisher_operation_sequence, 4U);
  EXPECT_EQ(later_health->clock_health.report_sequence, 2U);
  data_baseline.reset();
  ASSERT_TRUE(wait_for_no_subscribers(*publisher));

  auto created_health_baseline = create_subscriber();
  ASSERT_TRUE(created_health_baseline) << runtime_error_message(created_health_baseline.error());
  std::optional<ChannelSubscriber<Tensor>> health_baseline{std::move(*created_health_baseline)};
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*publisher, *health_baseline));
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));
  auto first_visible_health = wait_for_catalog_tensor_operation(*health_baseline);
  ASSERT_TRUE(first_visible_health) << runtime_error_message(first_visible_health.error());
  // Key assertion: a health-first raw attachment accepts the established nonzero baseline.
  ASSERT_EQ(first_visible_health->kind,
            channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(first_visible_health->clock_health.publisher_operation_sequence, 5U);
  EXPECT_EQ(first_visible_health->clock_health.report_sequence, 3U);
}

// A replacement subscriber may briefly observe operations from the explicitly superseded and
// accepted producer generations on different catalog lanes. The known-retired generation must be
// removed before equal-sequence ambiguity is evaluated.
// REQ: CHAN-069
// REQ: RUNTIME-008
TEST(RuntimeExternalChannelTest,
     DescriptorTensorSchemaSupersededHealthDoesNotConflictWithReplacementData) {
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-superseded-tie"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  graph.add_clock_conversion(capture_clock, runtime_clock, AffineClockConversion{});
  TopicOptions options = external_image_import_options(endpoint_name);
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(options), external_image_contract());
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
  const auto* const endpoint_record = detail::ChannelEndpointCatalogAccess::record(*endpoint);
  const auto* const type_ops = detail::runtime_port_type_ops<schema::ImageT>();
  ASSERT_NE(endpoint_record, nullptr);
  ASSERT_NE(type_ops, nullptr);
  ASSERT_TRUE(endpoint_record->payload_identity.schema);

  auto graph_runtime = ChannelRuntime::create();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime) << runtime_error_message(graph_runtime.error());
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  const auto create_subscriber = [&]() {
    return detail::RuntimeChannelBindingAccess::create_catalog_tensor_schema_subscriber(
        *graph_runtime,
        *endpoint,
        *endpoint_record->payload_identity.schema,
        type_ops->wire_capacity,
        type_ops->descriptor_tensor_static_count);
  };
  const auto emit = [&](ChannelTensorSchemaPublisher<ExternalImagePayload>& publisher,
                        const std::uint8_t value,
                        const std::int64_t capture_timestamp_ns) -> expected<void, Error> {
    auto tensor = make_external_image_tensor(value);
    if (!tensor) {
      return make_unexpected(std::move(tensor).error());
    }
    auto metadata = schema::build_image_metadata(
        *tensor,
        schema::ImageMetadataOptions{
            .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
    if (!metadata) {
      return make_unexpected(std::move(metadata).error());
    }
    auto emitted = publisher.emit_captured(*metadata, std::move(*tensor), capture_timestamp_ns);
    return emitted ? expected<void, Error>{}
                   : expected<void, Error>{make_unexpected(std::move(emitted).error())};
  };

  auto first = create_subscriber();
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  auto created_retired = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(created_retired) << runtime_error_message(created_retired.error());
  std::optional<ChannelTensorSchemaPublisher<ExternalImagePayload>> retired{
      std::move(*created_retired)};
  ASSERT_TRUE(wait_for_catalog_schema_subscriber(*retired, **first));
  ASSERT_TRUE(emit(*retired, 0x11U, 1));
  auto retired_data = wait_for_catalog_tensor_schema_operation(**first);
  ASSERT_TRUE(retired_data) << runtime_error_message(retired_data.error());
  ASSERT_EQ(retired_data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  (*first)->release_payload_loan();
  auto retired_generation =
      detail::RuntimeChannelBindingAccess::latched_catalog_tensor_schema_producer_generation(
          **first);
  ASSERT_TRUE(retired_generation) << runtime_error_message(retired_generation.error());
  ASSERT_NE(*retired_generation, 0U);

  auto replacement = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  ASSERT_TRUE(wait_for_catalog_schema_subscriber(*replacement, **first));
  ASSERT_TRUE(replacement->report_clock_health(ClockHealthReport::healthy()));
  auto replacement_warmup = wait_for_catalog_tensor_schema_operation(**first);
  ASSERT_TRUE(replacement_warmup) << runtime_error_message(replacement_warmup.error());
  ASSERT_EQ(replacement_warmup->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  first->reset();
  ASSERT_TRUE(wait_for_no_subscribers(*retired));
  ASSERT_TRUE(wait_for_no_subscribers(*replacement));

  auto current = create_subscriber();
  ASSERT_TRUE(current) << runtime_error_message(current.error());
  ASSERT_TRUE(
      detail::RuntimeChannelBindingAccess::supersede_catalog_tensor_schema_producer_generation(
          **current, *retired_generation));
  ASSERT_TRUE(wait_for_catalog_schema_subscriber(*retired, **current));
  ASSERT_TRUE(wait_for_catalog_schema_subscriber(*replacement, **current));

  // Key setup: queue equal operation sequence two on separate lanes: retired clock health and
  // replacement descriptor-Tensor data.
  ASSERT_TRUE(retired->report_clock_health(ClockHealthReport::healthy()));
  ASSERT_TRUE(emit(*replacement, 0x42U, 2));

  auto pending =
      detail::RuntimeChannelBindingAccess::catalog_tensor_schema_import_quiescence(**current);
  ASSERT_TRUE(pending) << runtime_error_message(pending.error());
  EXPECT_TRUE(pending->pending_current);
  EXPECT_TRUE(pending->pending_foreign);

  // Key assertion: the stale lane reports a generation cut instead of a protocol conflict, then
  // the replacement data remains queued and establishes the accepted producer generation.
  auto fenced = wait_for_catalog_tensor_schema_operation(**current);
  ASSERT_TRUE(fenced) << runtime_error_message(fenced.error());
  ASSERT_EQ(fenced->kind, channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  auto accepted = wait_for_catalog_tensor_schema_operation(**current);
  ASSERT_TRUE(accepted) << runtime_error_message(accepted.error());
  ASSERT_EQ(accepted->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(accepted->metadata.sequence, 2U);
  (*current)->release_payload_loan();
  auto accepted_generation =
      detail::RuntimeChannelBindingAccess::latched_catalog_tensor_schema_producer_generation(
          **current);
  ASSERT_TRUE(accepted_generation) << runtime_error_message(accepted_generation.error());
  EXPECT_NE(*accepted_generation, *retired_generation);

  // The endpoint admits two publishers; the retired lane is fully fenced, so release that
  // publisher before reusing its slot for an unknown third incarnation.
  retired.reset();
  auto foreign = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(foreign) << runtime_error_message(foreign.error());
  ASSERT_TRUE(wait_for_catalog_schema_subscriber(*foreign, **current));
  ASSERT_TRUE(emit(*foreign, 0x11U, 1));
  auto foreign_first = wait_for_catalog_tensor_schema_operation(**current);
  ASSERT_TRUE(foreign_first) << runtime_error_message(foreign_first.error());
  EXPECT_EQ(foreign_first->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  ASSERT_TRUE(foreign->report_clock_health(ClockHealthReport::healthy()));
  auto foreign_second = wait_for_catalog_tensor_schema_operation(**current);
  ASSERT_TRUE(foreign_second) << runtime_error_message(foreign_second.error());
  EXPECT_EQ(foreign_second->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);

  // Key setup: after latching the replacement, queue its health and unknown-generation data at
  // sequence three on distinct catalog lanes.
  ASSERT_TRUE(emit(*foreign, 0x13U, 3));
  ASSERT_TRUE(replacement->report_clock_health(ClockHealthReport::healthy()));
  auto latched_fence = wait_for_catalog_tensor_schema_operation(**current);
  ASSERT_TRUE(latched_fence) << runtime_error_message(latched_fence.error());
  EXPECT_EQ(latched_fence->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  auto accepted_health = wait_for_catalog_tensor_schema_operation(**current);
  ASSERT_TRUE(accepted_health) << runtime_error_message(accepted_health.error());
  EXPECT_EQ(accepted_health->kind, channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(accepted_health->clock_health.publisher_operation_sequence, 3U);
}

// Raw Tensor catalog subscribers enforce the same superseded-generation-before-ambiguity rule as
// descriptor-backed schemas.
// REQ: CHAN-069
// REQ: RUNTIME-008
TEST(RuntimeExternalChannelTest, RawTensorSupersededHealthDoesNotConflictWithReplacementData) {
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-tensor-superseded-tie"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  graph.add_clock_conversion(capture_clock, runtime_clock, AffineClockConversion{});
  TopicOptions options = optional_tensor_export_options(endpoint_name);
  options.bounds.subscriber.reset();
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalTensorExportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
  auto graph_runtime = ChannelRuntime::create();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime) << runtime_error_message(graph_runtime.error());
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  const auto create_subscriber = [&]() {
    return detail::RuntimeChannelBindingAccess::create_catalog_tensor_subscriber(
        *graph_runtime, *endpoint, 2U);
  };
  const auto publish = [&](ChannelPublisher<Tensor>& publisher,
                           const std::int64_t capture_timestamp_ns) -> expected<void, Error> {
    auto capture_time = publisher.make_capture_time(capture_timestamp_ns);
    if (!capture_time) {
      return make_unexpected(std::move(capture_time).error());
    }
    return publish_external_host_tensor(publisher, *capture_time);
  };

  auto created_first = create_subscriber();
  ASSERT_TRUE(created_first) << runtime_error_message(created_first.error());
  std::optional<ChannelSubscriber<Tensor>> first{std::move(*created_first)};
  auto created_retired =
      external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(created_retired) << runtime_error_message(created_retired.error());
  std::optional<ChannelPublisher<Tensor>> retired{std::move(*created_retired)};
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*retired, *first));
  ASSERT_TRUE(publish(*retired, 1));
  auto retired_data = wait_for_catalog_tensor_operation(*first);
  ASSERT_TRUE(retired_data) << runtime_error_message(retired_data.error());
  ASSERT_EQ(retired_data->kind, channel_internal::CatalogReceivedOperationKind::kData);
  retired_data->tensor_sample.reset();
  const std::uint64_t retired_generation =
      detail::RuntimeChannelBindingAccess::latched_catalog_tensor_producer_generation(*first);
  ASSERT_NE(retired_generation, 0U);

  auto replacement =
      external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*replacement, *first));
  ASSERT_TRUE(replacement->report_clock_health(ClockHealthReport::healthy()));
  auto replacement_warmup = wait_for_catalog_tensor_operation(*first);
  ASSERT_TRUE(replacement_warmup) << runtime_error_message(replacement_warmup.error());
  ASSERT_EQ(replacement_warmup->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  first.reset();
  ASSERT_TRUE(wait_for_no_subscribers(*retired));
  ASSERT_TRUE(wait_for_no_subscribers(*replacement));

  auto created_current = create_subscriber();
  ASSERT_TRUE(created_current) << runtime_error_message(created_current.error());
  std::optional<ChannelSubscriber<Tensor>> current{std::move(*created_current)};
  ASSERT_TRUE(detail::RuntimeChannelBindingAccess::supersede_catalog_tensor_producer_generation(
      *current, retired_generation));
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*retired, *current));
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*replacement, *current));

  // Key setup: queue the retired publisher's health operation and replacement publisher's data
  // operation at the same sequence number on independent catalog lanes.
  ASSERT_TRUE(retired->report_clock_health(ClockHealthReport::healthy()));
  ASSERT_TRUE(publish(*replacement, 2));

  auto pending = detail::RuntimeChannelBindingAccess::catalog_tensor_import_quiescence(*current);
  ASSERT_TRUE(pending) << runtime_error_message(pending.error());
  EXPECT_TRUE(pending->pending_current);
  EXPECT_TRUE(pending->pending_foreign);

  // Key assertion: the raw subscriber fences the retired lane first, then accepts sequence two
  // from the replacement generation without reporting an ambiguous-incarnation protocol error.
  auto fenced = wait_for_catalog_tensor_operation(*current);
  ASSERT_TRUE(fenced) << runtime_error_message(fenced.error());
  ASSERT_EQ(fenced->kind, channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  auto accepted = wait_for_catalog_tensor_operation(*current);
  ASSERT_TRUE(accepted) << runtime_error_message(accepted.error());
  ASSERT_EQ(accepted->kind, channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(accepted->metadata.sequence, 2U);
  accepted->tensor_sample.reset();
  EXPECT_NE(
      detail::RuntimeChannelBindingAccess::latched_catalog_tensor_producer_generation(*current),
      retired_generation);

  // The endpoint admits two publishers; the retired lane is fully fenced, so release that
  // publisher before reusing its slot for an unknown third incarnation.
  retired.reset();
  auto foreign = external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(foreign) << runtime_error_message(foreign.error());
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*foreign, *current));
  ASSERT_TRUE(publish(*foreign, 1));
  auto foreign_first = wait_for_catalog_tensor_operation(*current);
  ASSERT_TRUE(foreign_first) << runtime_error_message(foreign_first.error());
  EXPECT_EQ(foreign_first->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  ASSERT_TRUE(foreign->report_clock_health(ClockHealthReport::healthy()));
  auto foreign_second = wait_for_catalog_tensor_operation(*current);
  ASSERT_TRUE(foreign_second) << runtime_error_message(foreign_second.error());
  EXPECT_EQ(foreign_second->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);

  // Key setup: after latching the replacement, queue unknown-generation data and current health
  // at sequence three so equal per-generation counters meet on distinct lanes.
  ASSERT_TRUE(publish(*foreign, 3));
  ASSERT_TRUE(replacement->report_clock_health(ClockHealthReport::healthy()));
  auto latched_fence = wait_for_catalog_tensor_operation(*current);
  ASSERT_TRUE(latched_fence) << runtime_error_message(latched_fence.error());
  EXPECT_EQ(latched_fence->kind,
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  auto accepted_health = wait_for_catalog_tensor_operation(*current);
  ASSERT_TRUE(accepted_health) << runtime_error_message(accepted_health.error());
  EXPECT_EQ(accepted_health->kind, channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(accepted_health->clock_health.publisher_operation_sequence, 3U);
}

// Closing one descriptor-schema producer incarnation establishes the generation fence needed for
// a lifecycle restart. The replacement attachment rejects the retired generation and accepts a
// fresh publisher without rebuilding the immutable ExecutionPlan.
// REQ: EXTERNAL-010
// REQ: RUNTIME-008
// REQ: SYSTEM-024
TEST(RuntimeExternalChannelTest, DescriptorTensorSchemaImportRebindsAcrossLifecycleRestart) {
  ExternalImageImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-restart"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  graph.set_default_clock(capture_clock);
  TopicOptions options = external_image_import_options(endpoint_name);
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(options), external_image_contract());
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.lifecycle().allow_external_restart(sink);

  CompileOptions compile_options;
  compile_options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 4U,
      .max_candidate_generations = 4U,
      .max_events = 64U,
      .max_audit_records = 8U,
      .max_quiesce_records = 16U,
      .max_rollback_records = 16U,
  };
  const ExecutionPlan compiled = compile(graph, compile_options);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_EQ(compiled.lifecycle_restart_targets().size(), 1U);
  ASSERT_EQ(compiled.imported_generation_capacities().size(), 1U);
  EXPECT_TRUE(compiled.imported_generation_capacities().front().restart_capable_minimum_two);
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  RunSession session = run_async(plan);
  const auto phase_count = [&](const detail::RuntimePhaseCounter phase) {
    return phase_counters->snapshot()[detail::RuntimePhaseCounters::index(phase)];
  };
  const std::uint64_t readiness_baseline =
      phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished);
  const auto publish = [&](ChannelTensorSchemaPublisher<ExternalImagePayload>& publisher,
                           const std::uint8_t value,
                           const std::int64_t capture_timestamp_ns) -> expected<void, Error> {
    auto tensor = make_external_image_tensor(value);
    if (!tensor) {
      return make_unexpected(std::move(tensor).error());
    }
    auto metadata = schema::build_image_metadata(
        *tensor,
        schema::ImageMetadataOptions{
            .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
    if (!metadata) {
      return make_unexpected(std::move(metadata).error());
    }
    auto emitted = publisher.emit_captured(*metadata, std::move(*tensor), capture_timestamp_ns);
    if (!emitted) {
      return make_unexpected(std::move(emitted).error());
    }
    return expected<void, Error>{};
  };

  // Key setup: close the first descriptor-schema publisher, publish its ordered EOS, restart the
  // owning sink, and attach a replacement publisher whose operation sequence restarts at one.
  {
    auto first = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
        plan.endpoint_catalog(), endpoint_name);
    ASSERT_TRUE(first) << runtime_error_message(first.error());
    ASSERT_TRUE(wait_for_schema_subscriber(*first, 5s, 20ms));
    const auto first_publish = publish(*first, 0x21U, 1);
    ASSERT_TRUE(first_publish) << runtime_error_message(first_publish.error());
    ASSERT_TRUE(ExternalImageImportProbe::wait_for_count(1U));
    ASSERT_TRUE(first->close());
    const auto eos_deadline = std::chrono::steady_clock::now() + 5s;
    while (phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished) <
               readiness_baseline + 2U &&
           session.is_running() && std::chrono::steady_clock::now() < eos_deadline) {
      std::this_thread::yield();
    }
    ASSERT_EQ(phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished),
              readiness_baseline + 2U);
  }

  // Prove the fully drained import remains in its bounded lifecycle-restart wait state instead of
  // racing natural completion before the host can submit the replacement transaction.
  const std::uint64_t idle_baseline = phase_count(detail::RuntimePhaseCounter::kNoProgressScans);
  const auto idle_deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(detail::RuntimePhaseCounter::kNoProgressScans) < idle_baseline + 2U &&
         session.is_running() && std::chrono::steady_clock::now() < idle_deadline) {
    std::this_thread::yield();
  }
  ASSERT_GE(phase_count(detail::RuntimePhaseCounter::kNoProgressScans), idle_baseline + 2U);
  ASSERT_TRUE(session.is_running());

  const auto run_target = session.operator_ref(compiled.lifecycle_restart_targets().front().target);
  ASSERT_TRUE(run_target) << runtime_error_message(run_target.error());
  const std::array targets{*run_target};
  constexpr std::array stages{LifecycleStage::kConfigure};
  auto control =
      session.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
  ASSERT_TRUE(control) << runtime_error_message(control.error());
  auto receipt = control->request_restart(*run_target);
  ASSERT_TRUE(receipt) << runtime_error_message(receipt.error());
  auto completed = control->await(*receipt, 5s);
  ASSERT_TRUE(completed) << runtime_error_message(completed.error());
  ASSERT_EQ(completed->state, LifecycleTransactionState::kPublished);

  auto second = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  // A replacement HoloMQ subscriber becomes visible before every publisher-side data segment is
  // necessarily mapped. Require a short stable-presence window so the first measured payload is
  // the post-restart sample rather than a transport-connectivity probe.
  ASSERT_TRUE(wait_for_schema_subscriber(*second, 5s, 20ms));
  const auto second_publish = publish(*second, 0x42U, 2);
  ASSERT_TRUE(second_publish) << runtime_error_message(second_publish.error());
  ASSERT_TRUE(ExternalImageImportProbe::wait_for_count(2U));
  ASSERT_TRUE(second->close());

  session.wait();
  // Key assertion: the replacement generation delivers its Tensor through the unchanged plan and
  // supersedes the first generation's observed byte value. Exhausting the two-generation budget
  // restores ordinary natural completion after the replacement EOS.
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(ExternalImageImportProbe::count.load(std::memory_order_acquire), 2U);
  EXPECT_EQ(ExternalImageImportProbe::first_byte.load(std::memory_order_relaxed), 0x42U);
}

// An imported generation's EOS must retire only the input side of a restart-capable forwarding
// operator.  Its output remains live until the replacement generation has published its payload;
// the final EOS then closes the forwarding edge and permits ordinary completion.
// REQ: EXTERNAL-010
// REQ: RUNTIME-008
// REQ: SYSTEM-024
TEST(RuntimeExternalChannelTest,
     DescriptorTensorSchemaImportForwarderKeepsOutputOpenAcrossRestart) {
  ExternalImageImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-image-forwarder-restart"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  graph.set_default_clock(capture_clock);
  TopicOptions options = external_image_import_options(endpoint_name);
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + endpoint_name, std::move(options), external_image_contract());
  const auto forwarder = graph.op<ExternalImageForwarder>("forwarder");
  const auto sink = graph.op<ExternalImageImportSink>("sink");
  graph.add_flow(imported.output(), forwarder->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(forwarder->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.lifecycle().allow_external_restart(forwarder);

  CompileOptions compile_options;
  compile_options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 4U,
      .max_candidate_generations = 4U,
      .max_events = 64U,
      .max_audit_records = 8U,
      .max_quiesce_records = 16U,
      .max_rollback_records = 16U,
  };
  const ExecutionPlan compiled = compile(graph, compile_options);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto restart_target =
      std::find_if(compiled.lifecycle_restart_targets().begin(),
                   compiled.lifecycle_restart_targets().end(),
                   [](const auto& target) { return target.external_control_allowed; });
  ASSERT_NE(restart_target, compiled.lifecycle_restart_targets().end()) << compiled.json();
  ASSERT_EQ(std::count_if(compiled.lifecycle_restart_targets().begin(),
                          compiled.lifecycle_restart_targets().end(),
                          [](const auto& target) { return target.external_control_allowed; }),
            1U)
      << compiled.json();
  ASSERT_EQ(compiled.imported_generation_capacities().size(), 1U);
  EXPECT_TRUE(compiled.imported_generation_capacities().front().restart_capable_minimum_two);
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  RunSession session = run_async(plan);
  const auto phase_count = [&](const detail::RuntimePhaseCounter phase) {
    return phase_counters->snapshot()[detail::RuntimePhaseCounters::index(phase)];
  };
  const std::uint64_t readiness_baseline =
      phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished);
  const auto publish = [&](ChannelTensorSchemaPublisher<ExternalImagePayload>& publisher,
                           const std::uint8_t value,
                           const std::int64_t capture_timestamp_ns) -> expected<void, Error> {
    auto tensor = make_external_image_tensor(value);
    if (!tensor) {
      return make_unexpected(std::move(tensor).error());
    }
    auto metadata = schema::build_image_metadata(
        *tensor,
        schema::ImageMetadataOptions{
            .height = 1U, .width = 8U, .encoding = schema::ImageEncoding_MONO8});
    if (!metadata) {
      return make_unexpected(std::move(metadata).error());
    }
    auto emitted = publisher.emit_captured(*metadata, std::move(*tensor), capture_timestamp_ns);
    if (!emitted) {
      return make_unexpected(std::move(emitted).error());
    }
    return expected<void, Error>{};
  };

  {
    auto first = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
        plan.endpoint_catalog(), endpoint_name);
    ASSERT_TRUE(first) << runtime_error_message(first.error());
    ASSERT_TRUE(wait_for_schema_subscriber(*first, 5s, 20ms));
    ASSERT_TRUE(publish(*first, 0x21U, 1));
    ASSERT_TRUE(ExternalImageImportProbe::wait_for_count(1U));
    ASSERT_TRUE(first->close());
    const auto eos_deadline = std::chrono::steady_clock::now() + 5s;
    while (phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished) <
               readiness_baseline + 2U &&
           session.is_running() && std::chrono::steady_clock::now() < eos_deadline) {
      std::this_thread::yield();
    }
    ASSERT_EQ(phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished),
              readiness_baseline + 2U);
  }

  // Wait until the first generation's terminal drain and the downstream callback have both
  // crossed the controller cut. This proves the run remains live specifically because a bounded
  // import replacement is admissible, and prevents the restart preflight from racing a retained
  // downstream Tensor claim.
  const std::uint64_t idle_baseline = phase_count(detail::RuntimePhaseCounter::kNoProgressScans);
  const auto idle_deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(detail::RuntimePhaseCounter::kNoProgressScans) < idle_baseline + 2U &&
         session.is_running() && std::chrono::steady_clock::now() < idle_deadline) {
    std::this_thread::yield();
  }
  ASSERT_GE(phase_count(detail::RuntimePhaseCounter::kNoProgressScans), idle_baseline + 2U);
  ASSERT_TRUE(session.is_running());

  const auto run_target = session.operator_ref(restart_target->target);
  ASSERT_TRUE(run_target) << runtime_error_message(run_target.error());
  const std::array targets{*run_target};
  constexpr std::array stages{LifecycleStage::kConfigure};
  auto control =
      session.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
  ASSERT_TRUE(control) << runtime_error_message(control.error());
  auto receipt = control->request_restart(*run_target);
  ASSERT_TRUE(receipt) << runtime_error_message(receipt.error());
  auto completed = control->await(*receipt, 5s);
  ASSERT_TRUE(completed) << runtime_error_message(completed.error());
  ASSERT_EQ(completed->state, LifecycleTransactionState::kPublished);

  auto second = external_runtime->create_tensor_schema_publisher<ExternalImagePayload>(
      plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  ASSERT_TRUE(wait_for_schema_subscriber(*second, 5s, 20ms));
  ASSERT_TRUE(publish(*second, 0x42U, 2));
  ASSERT_TRUE(ExternalImageImportProbe::wait_for_count(2U));
  ASSERT_TRUE(second->close());

  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(ExternalImageImportProbe::count.load(std::memory_order_acquire), 2U);
  EXPECT_EQ(ExternalImageImportProbe::first_byte.load(std::memory_order_relaxed), 0x42U);
}

// Raw Tensor imports use the same catalog generation fence as descriptor-backed schemas. The
// Runtime must replace the typed endpoint after restart and accept the replacement producer's
// reset publication sequence without rebuilding the immutable ExecutionPlan.
// REQ: EXTERNAL-010
// REQ: RUNTIME-008
// REQ: SYSTEM-024
TEST(RuntimeExternalChannelTest, RawTensorImportRebindsAcrossLifecycleRestart) {
  ExternalTensorExportSink::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-tensor-restart"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  graph.set_default_clock(capture_clock);
  TopicOptions options = optional_tensor_export_options(endpoint_name);
  options.bounds.subscriber.reset();
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalTensorExportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.lifecycle().allow_external_restart(sink);

  CompileOptions compile_options;
  compile_options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 4U,
      .max_candidate_generations = 4U,
      .max_events = 64U,
      .max_audit_records = 8U,
      .max_quiesce_records = 16U,
      .max_rollback_records = 16U,
  };
  const ExecutionPlan compiled = compile(graph, compile_options);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_EQ(compiled.lifecycle_restart_targets().size(), 1U);
  ASSERT_EQ(compiled.imported_generation_capacities().size(), 1U);
  EXPECT_TRUE(compiled.imported_generation_capacities().front().restart_capable_minimum_two);
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  RunSession session = run_async(plan);
  const auto phase_count = [&](const detail::RuntimePhaseCounter phase) {
    return phase_counters->snapshot()[detail::RuntimePhaseCounters::index(phase)];
  };
  const std::uint64_t readiness_baseline =
      phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished);

  // Key setup: close the first raw-Tensor publisher, wait for ordered EOS readiness, restart the
  // sink, then attach a replacement publisher with a reset sequence.
  {
    auto first = external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
    ASSERT_TRUE(first) << runtime_error_message(first.error());
    ASSERT_TRUE(wait_for_subscriber(*first, 5s, 20ms));
    auto first_capture = first->make_capture_time(1);
    ASSERT_TRUE(first_capture) << runtime_error_message(first_capture.error());
    const auto first_publish = publish_external_host_tensor(*first, *first_capture);
    ASSERT_TRUE(first_publish) << runtime_error_message(first_publish.error());
    ASSERT_TRUE(ExternalTensorExportSink::wait_for_count(1U));
    ASSERT_TRUE(first->close());
    const auto eos_deadline = std::chrono::steady_clock::now() + 5s;
    while (phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished) <
               readiness_baseline + 2U &&
           session.is_running() && std::chrono::steady_clock::now() < eos_deadline) {
      std::this_thread::yield();
    }
    ASSERT_EQ(phase_count(detail::RuntimePhaseCounter::kChannelReadinessPublished),
              readiness_baseline + 2U);
  }

  // Prove the fully drained import remains in its bounded lifecycle-restart wait state instead of
  // racing natural completion before the host can submit the replacement transaction.
  const std::uint64_t idle_baseline = phase_count(detail::RuntimePhaseCounter::kNoProgressScans);
  const auto idle_deadline = std::chrono::steady_clock::now() + 5s;
  while (phase_count(detail::RuntimePhaseCounter::kNoProgressScans) < idle_baseline + 2U &&
         session.is_running() && std::chrono::steady_clock::now() < idle_deadline) {
    std::this_thread::yield();
  }
  ASSERT_GE(phase_count(detail::RuntimePhaseCounter::kNoProgressScans), idle_baseline + 2U);
  ASSERT_TRUE(session.is_running());

  const auto run_target = session.operator_ref(compiled.lifecycle_restart_targets().front().target);
  ASSERT_TRUE(run_target) << runtime_error_message(run_target.error());
  const std::array targets{*run_target};
  constexpr std::array stages{LifecycleStage::kConfigure};
  auto control =
      session.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
  ASSERT_TRUE(control) << runtime_error_message(control.error());
  auto receipt = control->request_restart(*run_target);
  ASSERT_TRUE(receipt) << runtime_error_message(receipt.error());
  auto completed = control->await(*receipt, 5s);
  ASSERT_TRUE(completed) << runtime_error_message(completed.error());
  ASSERT_EQ(completed->state, LifecycleTransactionState::kPublished);

  auto second = external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  ASSERT_TRUE(wait_for_subscriber(*second, 5s, 20ms));
  auto second_capture = second->make_capture_time(2);
  ASSERT_TRUE(second_capture) << runtime_error_message(second_capture.error());
  const auto second_publish = publish_external_host_tensor(*second, *second_capture);
  ASSERT_TRUE(second_publish) << runtime_error_message(second_publish.error());
  ASSERT_TRUE(ExternalTensorExportSink::wait_for_count(2U));
  ASSERT_TRUE(second->close());

  session.wait();
  // Key assertion: both producer generations deliver exactly one Tensor through one immutable
  // execution plan. The replacement EOS exhausts the admitted two-generation budget, so the run
  // returns to ordinary natural completion without diagnostics.
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(ExternalTensorExportSink::received.load(std::memory_order_acquire), 2U);
}

// A raw Tensor subscriber that accepted EOS is generation-fenced until Runtime explicitly replaces
// it. Traffic from a newly created publisher must not keep the retired pump endpoint ready.
// REQ: RUNTIME-008
// REQ: SYSTEM-024
TEST(RuntimeExternalChannelTest, ClosedRawTensorCatalogSubscriberIgnoresReplacementTraffic) {
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-external-closed-tensor-readiness"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  TopicOptions options = optional_tensor_export_options(endpoint_name);
  options.bounds.subscriber.reset();
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  const auto imported = graph.import_topic<Tensor>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalTensorExportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 1U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << (plan.diagnostics().empty() ? "<no diagnostic>"
                                                        : plan.diagnostics().front().token);
  const auto endpoint = plan.endpoint_catalog().endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << runtime_error_message(endpoint.error());
  auto graph_runtime = ChannelRuntime::create();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime) << runtime_error_message(graph_runtime.error());
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto subscriber = detail::RuntimeChannelBindingAccess::create_catalog_tensor_subscriber(
      *graph_runtime, *endpoint, 2U);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  // Key setup: attach a replacement publisher before accepting the first publisher's ordered EOS,
  // preserving a live transport mapping that can receive traffic after the generation fence.
  auto first = external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*first, *subscriber));
  auto replacement =
      external_runtime->create_publisher<Tensor>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(replacement) << runtime_error_message(replacement.error());
  ASSERT_TRUE(wait_for_catalog_tensor_subscriber(*replacement, *subscriber));
  ASSERT_TRUE(first->close());
  auto terminal = wait_for_catalog_tensor_operation(*subscriber);
  ASSERT_TRUE(terminal) << runtime_error_message(terminal.error());
  ASSERT_EQ(terminal->kind, channel_internal::CatalogReceivedOperationKind::kEndOfStream);

  auto loan = replacement->loan_tensor(
      TensorLoanRequest{.shape = kExternalTensorShape, .dtype = kExternalTensorDtype});
  ASSERT_TRUE(loan) << runtime_error_message(loan.error());
  auto writer = loan->write_host();
  ASSERT_TRUE(writer) << runtime_error_message(writer.error());
  std::fill_n(static_cast<std::byte*>(writer->data()), writer->byte_size(), std::byte{0x5a});
  ASSERT_TRUE(std::move(*writer).commit());
  auto emitted = replacement->emit(std::move(*loan));
  ASSERT_TRUE(emitted) << runtime_error_message(emitted.error());
  ASSERT_EQ(*emitted, 1U);

  // Key assertion: EOS keeps the retired catalog subscriber non-ready even when replacement data
  // has reached its still-live HoloMQ recipient; explicit rebind owns the generation cutover.
  auto pending =
      detail::RuntimeChannelBindingAccess::has_pending_catalog_tensor_operation(*subscriber);
  ASSERT_TRUE(pending) << runtime_error_message(pending.error());
  EXPECT_FALSE(*pending);
}

// REQ: CHAN-069
// REQ: RUNTIME-016
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest,
     CanonicalScalarImportAppliesEveryPrecedingHealthOperationBeforeCompute) {
  ExternalImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  //   external_publisher --(holomq://endpoint, clock-health=kFreshHealthyReport)-->
  //       imported_topic --[depth=8]--> sink
  //   Health sequence (publisher total order): H1, S1, S2, H2(unhealthy), S3
  Graph graph{"runtime-external-import"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{0});
  const ClockKey age_clock = graph.add_clock<RealtimeClock>("health-age-clock");
  graph.set_default_clock(age_clock);
  graph.add_clock_conversion(capture_clock, age_clock, AffineClockConversion{});

  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 8U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  if (auto connected = wait_for_subscriber(*publisher); !connected) {
    session.request_stop();
    session.wait();
    FAIL() << "external subscriber did not connect: " << runtime_error_message(connected.error())
           << "; termination=" << static_cast<int>(session.termination()) << ", diagnostics="
           << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token)
           << ", plan=" << plan.json();
  }

  // Key setup: RequireCaptureTime + ClockHealthOptions{kFreshHealthyReport}; health ops
  // precede each sample in the publisher's total order and filter via Standard-profile policy.
  // H1 has no readiness authority. S1 and S2 carry the healthy state that precedes them in the
  // publisher's total operation order. H2 is then applied before S3, so Standard-profile skip
  // policy suppresses S3 without losing either earlier sample or fabricating a health callback.
  ASSERT_TRUE(publisher->report_clock_health(ClockHealthReport::healthy()));  // H1
  std::this_thread::sleep_for(20ms);
  ASSERT_TRUE(session.is_running())
      << "health-only operation failed the run; diagnostics="
      << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token);
  EXPECT_EQ(ExternalImportProbe::count.load(std::memory_order_acquire), 0U);

  constexpr std::uint32_t kFirst = 0x01020304U;
  ASSERT_TRUE(publisher->emit_captured(kFirst, 101));  // S1
  std::this_thread::sleep_for(100ms);
  ASSERT_TRUE(session.is_running())
      << "first data operation failed the run; diagnostics="
      << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token);
  if (!ExternalImportProbe::wait_for_count(1U)) {
    session.request_stop();
    session.wait();
    FAIL() << "first healthy sample did not reach compute; termination="
           << static_cast<int>(session.termination()) << ", diagnostics="
           << (session.diagnostics().empty() ? "<none>" : session.diagnostics().front().token)
           << ", vertex="
           << (session.diagnostics().empty() ? 0U
                                             : session.diagnostics().front().source.vertex.value);
  }
  ASSERT_TRUE(publisher->emit_captured(0x11223344U, 102));  // S2
  ASSERT_TRUE(ExternalImportProbe::wait_for_count(2U));

  ASSERT_TRUE(publisher->report_clock_health(
      ClockHealthReport::unhealthy(ClockHealthReason::kSynchronizationLockLost)));  // H2
  std::this_thread::sleep_for(20ms);
  EXPECT_EQ(ExternalImportProbe::count.load(std::memory_order_acquire), 2U);
  ASSERT_TRUE(publisher->emit_captured(0xaabbccddU, 103));  // S3
  std::this_thread::sleep_for(100ms);

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  // Key assertion: exactly 2 samples arrive; S3 is suppressed by H2 (unhealthy) without error.
  EXPECT_EQ(ExternalImportProbe::count.load(std::memory_order_acquire), 2U);
  EXPECT_EQ(ExternalImportProbe::values[0], kFirst);
  EXPECT_EQ(ExternalImportProbe::values[1], 0x11223344U);
  EXPECT_EQ(ExternalImportProbe::sequences[0], 2U);
  EXPECT_EQ(ExternalImportProbe::sequences[1], 3U);
}

}  // namespace
}  // namespace holoscan
