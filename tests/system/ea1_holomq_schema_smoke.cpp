/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Standalone smoke test for the HoloMQ Channel schema layer — no GTest harness.
// Exercises five end-to-end scenarios that together cover the public install-tree API:
// custom-POD catalogs, generated FlatBuffers schema packages, Tensor loan/emit/EOS,
// catalog attachment policies, and a full Graph-operator round-trip with metadata.
// Each scenario is mapped to a distinct non-zero exit code for CI failure triage.

#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <holoscan/channel/channel.hpp>
#include <holoscan/channel/endpoint_catalog_builder.hpp>
#include <holoscan/channel/pod_type.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/flatbuffer_schema_package.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

// Minimal two-field POD used as the application-defined payload in the custom-POD and
// graph-workflow scenarios.  Kept trivial so the focus is on the Channel transport
// contract rather than serialization complexity.
struct ControlPod {
  std::uint32_t frame{};
  std::int32_t value{};

  friend bool operator==(const ControlPod&, const ControlPod&) noexcept = default;
};

}  // namespace

namespace holoscan {

// Register ControlPod as a named, versioned channel type so the runtime can
// validate schema compatibility between publisher and subscriber endpoints.
template <>
struct ChannelTypeIdentity<::ControlPod> {
  static constexpr std::string_view name = "holoscan.tests.ea1.ControlPod";
  static constexpr std::uint32_t version = 1U;
};

// Describe ControlPod field offsets so the Channel layer can encode and decode the
// struct without reflection or generated code.
template <>
struct ChannelPodLayout<::ControlPod> {
  static constexpr auto fields =
      pod_fields(pod_field("frame", &::ControlPod::frame, offsetof(::ControlPod, frame)),
                 pod_field("value", &::ControlPod::value, offsetof(::ControlPod, value)));
};

}  // namespace holoscan

namespace {

// Concept that constrains the helpers below to types that expose report_clock_health().
// Satisfying this contract requires both custom-POD and generated-schema publishers
// to return expected<void, Error>, preventing accidental silent drops.
template <typename Publisher>
concept ClockHealthPublisher = requires(Publisher& publisher, holoscan::ClockHealthReport report) {
  {
    publisher.report_clock_health(report)
  } -> std::same_as<holoscan::expected<void, holoscan::Error>>;
};

// Q61-T12 requires custom POD and owning generated-schema facades to expose one exact result type.
static_assert(ClockHealthPublisher<holoscan::ChannelPublisher<ControlPod>>);
static_assert(ClockHealthPublisher<holoscan::ChannelPublisher<holoscan::schema::StringT>>);
static_assert(ClockHealthPublisher<holoscan::ChannelPublisher<holoscan::Tensor>>);

// Graph operator that bridges the external-publisher import topic to the external-subscriber
// export topic in the graph-workflow scenario.  Receives a ControlPod, formats its fields
// as "frame=N;value=M", and re-emits a StringT while forwarding capture-time metadata.
class CatalogWorkflowTransform final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "control").queue_depth(1U);
    spec.output(output, "description").max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto received = input.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }

    holoscan::schema::StringT transformed;
    transformed.data = "frame=" + std::to_string(received->data.frame) +
                       ";value=" + std::to_string(received->data.value);
    holoscan::SampleMetadata metadata = received->metadata;
    metadata.sequence = 0U;
    metadata.byte_size = 0U;
    metadata.payload_kind = holoscan::PayloadKind::kInlineTyped;
    return output.emit(holoscan::Sample<holoscan::schema::StringT>{
        std::move(transformed),
        metadata,
    });
  }

  holoscan::Input<ControlPod> input;
  holoscan::Output<holoscan::schema::StringT> output;
};

// Generate a process-scoped unique topic name to prevent collisions between concurrent
// test runs.  Combines PID, a monotonic ordinal, a steady-clock tick, and a caller suffix.
[[nodiscard]] std::string unique_name(std::string_view suffix) {
  static std::atomic<std::uint64_t> ordinal{};
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
#if defined(_WIN32)
  const auto process_id = static_cast<std::uint64_t>(::_getpid());
#else
  const auto process_id = static_cast<std::uint64_t>(::getpid());
#endif
  return "ea1_holomq_schema_smoke_" + std::to_string(process_id) + "_" +
         std::to_string(ordinal.fetch_add(1U, std::memory_order_relaxed)) + "_" +
         std::to_string(tick) + "_" + std::string(suffix);
}

// Convenience wrapper: constructs a single-endpoint ChannelEndpointCatalog for type T
// in the given direction.  Keeps each smoke scenario self-contained without repeating
// the builder boilerplate.
template <typename T>
[[nodiscard]] holoscan::expected<holoscan::ChannelEndpointCatalog, holoscan::Error> build_catalog(
    holoscan::TopicDirection direction, std::string_view endpoint_name, std::string_view uri,
    const holoscan::TopicOptions& options = {}) {
  auto builder = holoscan::ChannelEndpointCatalog::builder("ea1-holomq-schema-smoke");
  auto added = direction == holoscan::TopicDirection::kImport
                   ? builder.template import_topic<T>(endpoint_name, uri, options)
                   : builder.template export_topic<T>(endpoint_name, uri, options);
  if (!added) {
    return holoscan::make_unexpected(std::move(added).error());
  }
  return std::move(builder).build();
}

// Verify that ClockHealthOptions are validated at catalog-build time across four modes:
//   - no options       → capability=Unavailable, requirement=NotRequired, no max_report_age
//   - full monitoring  → options survive the catalog round-trip unchanged
//   - diagnostic-only  → capability set but requirement=NotRequired is rejected (kInvalidArgument)
//   - wrong role       → monitoring options on an export-role endpoint are rejected
template <typename T>
[[nodiscard]] bool validate_clock_health_contract_modes(std::string_view payload_name) {
  const std::string ordinary_name = unique_name(std::string(payload_name) + "_ordinary");
  auto ordinary = build_catalog<T>(
      holoscan::TopicDirection::kImport, ordinary_name, "holomq://" + ordinary_name);
  if (!ordinary) {
    return false;
  }
  auto ordinary_endpoint = ordinary->endpoint(ordinary_name);
  if (!ordinary_endpoint ||
      ordinary_endpoint->clock_health().capability() !=
          holoscan::ClockHealthCapability::kUnavailable ||
      ordinary_endpoint->clock_health().requirement() !=
          holoscan::ClockHealthRequirement::kNotRequired ||
      ordinary_endpoint->clock_health().max_report_age().has_value()) {
    return false;
  }

  holoscan::TopicOptions monitored_options;
  monitored_options.clock_health = holoscan::ClockHealthOptions{
      .capability = holoscan::ClockHealthCapability::kStatusReports,
      .requirement = holoscan::ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  const std::string monitored_name = unique_name(std::string(payload_name) + "_monitored");
  auto monitored = build_catalog<T>(holoscan::TopicDirection::kImport,
                                    monitored_name,
                                    "holomq://" + monitored_name,
                                    monitored_options);
  if (!monitored) {
    return false;
  }
  auto monitored_endpoint = monitored->endpoint(monitored_name);
  if (!monitored_endpoint ||
      monitored_endpoint->clock_health().capability() !=
          holoscan::ClockHealthCapability::kStatusReports ||
      monitored_endpoint->clock_health().requirement() !=
          holoscan::ClockHealthRequirement::kFreshHealthyReport ||
      monitored_endpoint->clock_health().max_report_age() != 5s) {
    return false;
  }

  holoscan::TopicOptions diagnostic_only_options;
  diagnostic_only_options.clock_health = holoscan::ClockHealthOptions{
      .capability = holoscan::ClockHealthCapability::kStatusReports,
      .requirement = holoscan::ClockHealthRequirement::kNotRequired,
  };
  const std::string diagnostic_name = unique_name(std::string(payload_name) + "_diagnostic");
  auto diagnostic_only = build_catalog<T>(holoscan::TopicDirection::kImport,
                                          diagnostic_name,
                                          "holomq://" + diagnostic_name,
                                          diagnostic_only_options);
  if (diagnostic_only || diagnostic_only.error().code != holoscan::ErrorCode::kInvalidArgument) {
    return false;
  }

  const std::string wrong_role_name = unique_name(std::string(payload_name) + "_wrong_role");
  auto wrong_role = build_catalog<T>(holoscan::TopicDirection::kExport,
                                     wrong_role_name,
                                     "holomq://" + wrong_role_name,
                                     monitored_options);
  return !wrong_role && wrong_role.error().code == holoscan::ErrorCode::kInvalidArgument;
}

// Confirm that a publisher without a ClockHealthOptions contract rejects report_clock_health()
// with kInvalidArgument rather than silently dropping the report.
template <ClockHealthPublisher Publisher>
[[nodiscard]] bool rejects_unconfigured_clock_health(Publisher& publisher) {
  auto result = publisher.report_clock_health(holoscan::ClockHealthReport::healthy());
  return !result && result.error().code == holoscan::ErrorCode::kInvalidArgument;
}

// Poll until both sides see each other as connected.  Two consecutive observations are
// required to avoid a race where one side briefly sees a peer before the other does.
// Times out after 3 s.
template <typename Publisher, typename Subscriber>
[[nodiscard]] bool wait_for_connection(Publisher& publisher, Subscriber& subscriber) {
  const auto deadline = std::chrono::steady_clock::now() + 3s;
  std::uint32_t consecutive_observations{};
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    auto readiness = subscriber.has_samples();
    if (!refreshed || !readiness) {
      return false;
    }
    if (publisher.has_subscribers() && subscriber.has_publishers()) {
      ++consecutive_observations;
      if (consecutive_observations == 2U) {
        return true;
      }
    } else {
      consecutive_observations = 0U;
    }
    std::this_thread::sleep_for(1ms);
  }
  return false;
}

// Retry subscriber.receive_sample() until a non-kNotReady result arrives or the 3 s
// deadline expires.  Returns kTimeout on deadline so callers get a uniform error code.
template <typename Subscriber>
[[nodiscard]] auto receive_sample_until(Subscriber& subscriber) {
  using Result = decltype(subscriber.receive_sample());
  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (std::chrono::steady_clock::now() < deadline) {
    Result received = subscriber.receive_sample();
    if (received || received.error().code != holoscan::ErrorCode::kNotReady) {
      return received;
    }
    std::this_thread::sleep_for(1ms);
  }
  return Result{holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout})};
}

// This is the readable public-only workflow that an installed application can copy: an external
// custom-POD publisher attaches by the immutable compiled catalog, feeds one Graph Operator, and
// an external generated-schema subscriber receives the transformed sample from a Graph export.
// The capture coordinate and semantic metadata survive both boundaries, while each boundary owns
// its transport sequence and payload representation.
[[nodiscard]] bool run_graph_catalog_workflow() {
  using Payload = holoscan::schema::StringT;
  const std::string import_name = unique_name("graph_import");
  const std::string export_name = unique_name("graph_export");

  holoscan::Graph graph{"ea1-external-catalog-workflow"};
  const auto clock = graph.add_clock<holoscan::RealtimeClock>("capture-clock");
  graph.set_default_clock(clock);

  holoscan::TopicOptions import_options;
  import_options.endpoint_name = import_name;
  import_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
  import_options.bounds.publisher = holoscan::PublisherEndpointBounds{.max_loaned_samples = 1U};
  const auto imported =
      graph.import_topic<ControlPod>("holomq://" + import_name, std::move(import_options));

  holoscan::TopicOptions export_options;
  export_options.endpoint_name = export_name;
  export_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
  export_options.bounds.subscriber = holoscan::SubscriberEndpointBounds{
      .queue_depth = 1U,
      .max_borrowed_samples = 1U,
  };
  const auto exported =
      graph.export_topic<Payload>("holomq://" + export_name, std::move(export_options));
  const auto transform = graph.op<CatalogWorkflowTransform>("transform");
  graph.add_flow(
      imported.output(), transform->input, holoscan::ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(
      transform->output, exported.input(), holoscan::ConnectionOptions{.queue_depth = 1U});

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << "graph workflow compile rejected: " << plan.json() << '\n';
    return false;
  }
  auto import_endpoint = plan.endpoint_catalog().endpoint(import_name);
  auto export_endpoint = plan.endpoint_catalog().endpoint(export_name);
  if (!import_endpoint || !export_endpoint || import_endpoint->name() != import_name ||
      export_endpoint->name() != export_name ||
      import_endpoint->direction() != holoscan::TopicDirection::kImport ||
      export_endpoint->direction() != holoscan::TopicDirection::kExport) {
    std::cerr << "graph workflow catalog identity mismatch\n";
    return false;
  }

  auto external_runtime = holoscan::ChannelRuntime::create();
  if (!external_runtime) {
    std::cerr << "graph workflow runtime: "
              << holoscan::runtime_error_message(external_runtime.error()) << '\n';
    return false;
  }
  auto publisher =
      external_runtime->create_publisher<ControlPod>(plan.endpoint_catalog(), import_name);
  auto subscriber = external_runtime->create_subscriber<Payload>(*export_endpoint);
  if (!publisher || !subscriber) {
    const auto& error = !publisher ? publisher.error() : subscriber.error();
    std::cerr << "graph workflow attach: " << holoscan::runtime_error_message(error) << '\n';
    return false;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  if (!wait_for_connection(*publisher, *subscriber)) {
    session.request_stop();
    session.wait();
    std::cerr << "graph workflow peers did not attach\n";
    return false;
  }

  constexpr std::int64_t kCaptureTimestamp = 1'234'567;
  constexpr holoscan::TraceContext kTrace{
      .trace_id = 301U,
      .span_id = 302U,
      .parent_span_id = 300U,
  };
  auto emitted = publisher->emit_captured(ControlPod{.frame = 17U, .value = -42},
                                          kCaptureTimestamp,
                                          holoscan::EmitOptions{
                                              .frame_id = 94U,
                                              .trace = kTrace,
                                              .flags = holoscan::SampleFlags::kDegraded,
                                          });
  if (!emitted || *emitted != 1U) {
    session.request_stop();
    session.wait();
    std::cerr << "graph workflow emit failed\n";
    return false;
  }

  auto received = receive_sample_until(*subscriber);
  if (!received) {
    session.request_stop();
    session.wait();
    std::cerr << "graph workflow receive: " << holoscan::runtime_error_message(received.error())
              << '\n';
    return false;
  }
  const bool sample_matches =
      received->data.data == "frame=17;value=-42" &&
      received->metadata.capture_timestamp_ns == kCaptureTimestamp &&
      received->metadata.source_clock_id.valid() && received->metadata.sequence != 0U &&
      received->metadata.trace_id == kTrace.trace_id && received->metadata.frame_id == 94U &&
      received->metadata.degraded() &&
      received->metadata.payload_kind == holoscan::PayloadKind::kSerializedBytes &&
      received->metadata.byte_size != 0U;

  auto closed = publisher->close();
  auto closed_twice = publisher->close();
  session.request_stop();
  session.wait();
  if (!sample_matches || !closed || closed_twice ||
      closed_twice.error().code != holoscan::ErrorCode::kClosed ||
      session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "graph workflow metadata, EOS, or teardown contract failed\n";
    return false;
  }
  return true;
}

// Custom-POD direct smoke: builds separate import/export catalogs for ControlPod,
// wires a publisher and subscriber, emits one sample with trace/frame metadata and
// the kDegraded flag, and validates the received payload and all metadata fields.
// Also covers clock-health-contract validation and the ClockHealthPublisher static_asserts.
[[nodiscard]] bool run_catalog_pod_smoke() {
  if (!validate_clock_health_contract_modes<ControlPod>("pod_health")) {
    return false;
  }
  const std::string endpoint_name = unique_name("pod");
  const std::string uri = "holomq://" + endpoint_name;
  auto import_catalog =
      build_catalog<ControlPod>(holoscan::TopicDirection::kImport, endpoint_name, uri);
  auto export_catalog =
      build_catalog<ControlPod>(holoscan::TopicDirection::kExport, endpoint_name, uri);
  if (!import_catalog || !export_catalog) {
    return false;
  }

  auto publisher_runtime = holoscan::ChannelRuntime::create();
  auto subscriber_runtime = holoscan::ChannelRuntime::create();
  if (!publisher_runtime || !subscriber_runtime) {
    return false;
  }
  auto subscriber =
      subscriber_runtime->create_subscriber<ControlPod>(*export_catalog, endpoint_name);
  auto publisher = publisher_runtime->create_publisher<ControlPod>(*import_catalog, endpoint_name);
  if (!subscriber || !publisher || !wait_for_connection(*publisher, *subscriber)) {
    return false;
  }
  if (!rejects_unconfigured_clock_health(*publisher)) {
    return false;
  }

  constexpr ControlPod expected{.frame = 17U, .value = -42};
  constexpr holoscan::TraceContext kTrace{
      .trace_id = 101U, .span_id = 102U, .parent_span_id = 100U};
  auto emitted = publisher->emit(expected,
                                 holoscan::EmitOptions{
                                     .frame_id = 91U,
                                     .trace = kTrace,
                                     .flags = holoscan::SampleFlags::kDegraded,
                                 });
  if (!emitted || emitted.value() != 1U) {
    return false;
  }
  auto received = receive_sample_until(*subscriber);
  return received && received->data == expected && received->metadata.frame_id == 91U &&
         received->metadata.trace_id == kTrace.trace_id && received->metadata.degraded() &&
         received->metadata.byte_size == sizeof(ControlPod);
}

// Attachment is always finite and catalog-authoritative. Optional peers return without waiting,
// while wrong-role, unsupported-storage, and absent-required-peer requests fail before exposing a
// facade that could violate the frozen endpoint contract.
[[nodiscard]] bool run_attach_policy_smoke() {
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    return false;
  }

  const std::string optional_name = unique_name("optional_peer");
  holoscan::TopicOptions optional_options;
  optional_options.attach = holoscan::TopicAttachPolicy{
      .peer = holoscan::PeerRequirement::kOptional,
      .max_wait = 1s,
  };
  optional_options.bounds.publisher = holoscan::PublisherEndpointBounds{.max_loaned_samples = 1U};
  auto optional_catalog = build_catalog<ControlPod>(holoscan::TopicDirection::kImport,
                                                    optional_name,
                                                    "holomq://" + optional_name,
                                                    optional_options);
  if (!optional_catalog) {
    return false;
  }
  const auto optional_started = std::chrono::steady_clock::now();
  auto optional = runtime->create_publisher<ControlPod>(
      *optional_catalog, optional_name, holoscan::ChannelAttachOptions{.timeout = 1ns});
  const auto optional_elapsed = std::chrono::steady_clock::now() - optional_started;
  if (!optional || optional_elapsed >= 250ms) {
    return false;
  }

  const std::string wrong_role_name = unique_name("wrong_role");
  auto wrong_role_catalog = build_catalog<ControlPod>(
      holoscan::TopicDirection::kExport, wrong_role_name, "holomq://" + wrong_role_name);
  if (!wrong_role_catalog) {
    return false;
  }
  auto wrong_role = runtime->create_publisher<ControlPod>(*wrong_role_catalog, wrong_role_name);
  if (wrong_role || wrong_role.error().code != holoscan::ErrorCode::kInvalidArgument) {
    return false;
  }

  const std::string required_name = unique_name("required_peer");
  holoscan::TopicOptions required_options;
  required_options.attach = holoscan::TopicAttachPolicy{
      .peer = holoscan::PeerRequirement::kRequired,
      .max_wait = 50ms,
  };
  required_options.bounds.publisher = holoscan::PublisherEndpointBounds{.max_loaned_samples = 1U};
  auto required_catalog = build_catalog<ControlPod>(holoscan::TopicDirection::kImport,
                                                    required_name,
                                                    "holomq://" + required_name,
                                                    required_options);
  if (!required_catalog) {
    return false;
  }
  const auto required_started = std::chrono::steady_clock::now();
  auto required = runtime->create_publisher<ControlPod>(
      *required_catalog, required_name, holoscan::ChannelAttachOptions{.timeout = 2ms});
  const auto required_elapsed = std::chrono::steady_clock::now() - required_started;
  if (required || required.error().code != holoscan::ErrorCode::kTimeout ||
      required_elapsed >= 250ms) {
    return false;
  }

  const std::string tensor_name = unique_name("unsupported_tensor_storage");
  constexpr std::array axes{holoscan::TensorAxisBounds{
      .max_extent = 8U,
      .max_abs_stride_bytes = 1U,
  }};
  holoscan::TopicOptions tensor_options;
  tensor_options.tensor = holoscan::LogicalTensorRequirements{
      .memory_kind = holoscan::MemoryKind::kCudaManaged,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
      .rank = 1U,
      .logical_bounds =
          holoscan::LogicalTensorBounds{
              .max_rank = 1U,
              .max_byte_span = 8U,
              .axes = axes,
          },
      .strides = holoscan::TensorStridePolicy::kContiguousOnly,
      .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
  };
  tensor_options.bounds.subscriber = holoscan::SubscriberEndpointBounds{
      .queue_depth = 1U,
      .max_borrowed_samples = 1U,
  };
  auto tensor_catalog = build_catalog<holoscan::Tensor>(
      holoscan::TopicDirection::kExport, tensor_name, "holomq://" + tensor_name, tensor_options);
  if (!tensor_catalog) {
    return false;
  }
  auto unsupported_storage =
      runtime->create_subscriber<holoscan::Tensor>(*tensor_catalog, tensor_name);
  return !unsupported_storage &&
         unsupported_storage.error().code == holoscan::ErrorCode::kNotSupported;
}

// Generated-schema smoke: validates the FlatBuffers schema package for StringT, then
// exercises a publisher/subscriber pair end-to-end.  Confirms the received payload data,
// frame_id, trace_id, degraded flag, and serialized byte_size match expectations.
// Also validates the package encode() path and clock-health rejection before the first emit.
[[nodiscard]] bool run_generated_schema_smoke() {
  using Payload = holoscan::schema::StringT;
  const auto& package = holoscan::schema_package<Payload>();
  auto package_valid = holoscan::validate_schema_package(package);
  if (!package_valid) {
    std::cerr << "generated schema package validation: "
              << holoscan::runtime_error_message(package_valid.error()) << '\n';
    return false;
  }
  if (!validate_clock_health_contract_modes<Payload>("schema_health")) {
    return false;
  }

  const std::string endpoint_name = unique_name("schema");
  const std::string uri = "holomq://" + endpoint_name;
  auto import_catalog =
      build_catalog<Payload>(holoscan::TopicDirection::kImport, endpoint_name, uri);
  auto export_catalog =
      build_catalog<Payload>(holoscan::TopicDirection::kExport, endpoint_name, uri);
  if (!import_catalog || !export_catalog) {
    const auto& error = !import_catalog ? import_catalog.error() : export_catalog.error();
    std::cerr << "generated schema catalog: " << holoscan::runtime_error_message(error) << '\n';
    return false;
  }

  auto publisher_runtime = holoscan::ChannelRuntime::create();
  auto subscriber_runtime = holoscan::ChannelRuntime::create();
  if (!publisher_runtime || !subscriber_runtime) {
    const auto& error = !publisher_runtime ? publisher_runtime.error() : subscriber_runtime.error();
    std::cerr << "generated schema runtime: " << holoscan::runtime_error_message(error) << '\n';
    return false;
  }
  auto subscriber = subscriber_runtime->create_subscriber<Payload>(*export_catalog, endpoint_name);
  auto publisher = publisher_runtime->create_publisher<Payload>(*import_catalog, endpoint_name);
  if (!subscriber || !publisher) {
    const auto& error = !subscriber ? subscriber.error() : publisher.error();
    std::cerr << "generated schema attach: " << holoscan::runtime_error_message(error) << '\n';
    return false;
  }
  if (!wait_for_connection(*publisher, *subscriber)) {
    std::cerr << "generated schema connection timed out\n";
    return false;
  }
  if (!rejects_unconfigured_clock_health(*publisher)) {
    std::cerr << "generated schema accepted health without an admitted contract\n";
    return false;
  }

  Payload expected;
  expected.data = "EA1 generated schema";
  alignas(std::max_align_t)
      std::array<std::byte, holoscan::generated_schema_max_serialized_size_v<Payload>>
          encoded_storage{};
  auto encoded = package.encode(expected, encoded_storage);
  if (!encoded) {
    std::cerr << "generated schema encode: " << holoscan::runtime_error_message(encoded.error())
              << '\n';
    return false;
  }
  constexpr holoscan::TraceContext kTrace{
      .trace_id = 201U, .span_id = 202U, .parent_span_id = 200U};
  auto emitted = publisher->emit(expected,
                                 holoscan::EmitOptions{
                                     .frame_id = 92U,
                                     .trace = kTrace,
                                     .flags = holoscan::SampleFlags::kDegraded,
                                 });
  if (!emitted || emitted.value() != 1U) {
    if (!emitted) {
      std::cerr << "generated schema emit: " << holoscan::runtime_error_message(emitted.error())
                << '\n';
    } else {
      std::cerr << "generated schema emit recipient count: " << emitted.value() << '\n';
    }
    return false;
  }
  auto received = receive_sample_until(*subscriber);
  if (!received) {
    std::cerr << "generated schema receive: " << holoscan::runtime_error_message(received.error())
              << '\n';
    return false;
  }
  return received && received->data.data == expected.data && received->metadata.frame_id == 92U &&
         received->metadata.trace_id == kTrace.trace_id && received->metadata.degraded() &&
         received->metadata.payload_kind == holoscan::PayloadKind::kSerializedBytes &&
         received->metadata.byte_size == *encoded;
}

// Diagnostic helpers: print a "catalog Tensor <stage>: <detail>" prefix and return false
// so run_catalog_tensor_transfer() can propagate any failure with a single return statement.
[[nodiscard]] bool fail_catalog_tensor_error(const std::string_view stage,
                                             const holoscan::Error& error) {
  std::cerr << "catalog Tensor " << stage << ": " << holoscan::runtime_error_message(error) << '\n';
  return false;
}

[[nodiscard]] bool fail_catalog_tensor_message(const std::string_view stage,
                                               const std::string_view detail) {
  std::cerr << "catalog Tensor " << stage << ": " << detail << '\n';
  return false;
}

// Core Tensor transfer sequence shared by the Tensor smoke scenario:
//   loan → host-write (fill 1..N) → commit → emit → receive → validate payload/layout/metadata
//   → close publisher → confirm EOS (kClosed) → confirm repeated close returns kClosed.
template <typename Publisher, typename Subscriber>
[[nodiscard]] bool run_catalog_tensor_transfer(Publisher& publisher, Subscriber& subscriber) {
  constexpr std::array<std::int64_t, 1U> shape{8};
  auto loan = publisher.loan_tensor(holoscan::TensorLoanRequest{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  });
  if (!loan) {
    return fail_catalog_tensor_error("tensor loan", loan.error());
  }
  auto writer = loan->write_host();
  if (!writer) {
    return fail_catalog_tensor_error("host writer", writer.error());
  }
  if (writer->byte_size() != 8U) {
    std::cerr << "catalog Tensor host writer span: expected 8 bytes, got " << writer->byte_size()
              << '\n';
    return false;
  }
  auto* bytes = static_cast<std::byte*>(writer->data());
  for (std::size_t index = 0U; index < writer->byte_size(); ++index) {
    bytes[index] = static_cast<std::byte>(index + 1U);
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return fail_catalog_tensor_error("host write commit", committed.error());
  }
  auto emitted = publisher.emit(std::move(*loan), holoscan::EmitOptions{.frame_id = 93U});
  if (!emitted) {
    return fail_catalog_tensor_error("emit", emitted.error());
  }
  if (*emitted != 1U) {
    std::cerr << "catalog Tensor emit recipient count: expected 1, got " << *emitted << '\n';
    return false;
  }
  auto received = receive_sample_until(subscriber);
  if (!received) {
    return fail_catalog_tensor_error("receive", received.error());
  }
  const auto received_shape = received->data.shape_span();
  const auto received_strides = received->data.strides();
  if (!received->data.is_read_only() || received->metadata.frame_id != 93U ||
      received->metadata.payload_kind != holoscan::PayloadKind::kDescriptorBacked ||
      received->data.dtype().code != kDLUInt || received->data.dtype().bits != 8U ||
      received->data.dtype().lanes != 1U || received_shape.size() != 1U ||
      received_shape.front() != 8 || !received->data.is_contiguous() ||
      received_strides.size() != 1U || received_strides.front() != 1 ||
      received->data.nbytes() != 8U || received->data.data() == nullptr) {
    return fail_catalog_tensor_message("receive contract",
                                       "read-only/layout/metadata contract mismatch");
  }
  const auto* received_bytes = static_cast<const std::byte*>(received->data.data());
  for (std::size_t index = 0U; index < received->data.nbytes(); ++index) {
    if (received_bytes[index] != static_cast<std::byte>(index + 1U)) {
      std::cerr << "catalog Tensor payload validation: byte mismatch at index " << index << '\n';
      return false;
    }
  }
  auto closed = publisher.close();
  if (!closed) {
    return fail_catalog_tensor_error("close", closed.error());
  }
  auto end_of_stream = receive_sample_until(subscriber);
  if (end_of_stream) {
    return fail_catalog_tensor_message("end of stream", "received a sample after publisher close");
  }
  if (end_of_stream.error().code != holoscan::ErrorCode::kClosed) {
    return fail_catalog_tensor_error("end of stream", end_of_stream.error());
  }
  auto repeated_close = publisher.close();
  if (repeated_close) {
    return fail_catalog_tensor_message("repeated close", "unexpectedly succeeded");
  }
  if (repeated_close.error().code != holoscan::ErrorCode::kClosed) {
    return fail_catalog_tensor_error("repeated close", repeated_close.error());
  }
  return true;
}

// Tensor smoke: builds host-memory Tensor catalogs with explicit layout bounds, attaches a
// publisher and subscriber, confirms clock-health rejection, then delegates the full
// loan/emit/receive/EOS sequence to run_catalog_tensor_transfer().
[[nodiscard]] bool run_catalog_tensor_smoke() {
  const std::string endpoint_name = unique_name("tensor");
  const std::string uri = "holomq://" + endpoint_name;
  constexpr std::array axes{holoscan::TensorAxisBounds{
      .max_extent = 8U,
      .max_abs_stride_bytes = 1U,
  }};
  holoscan::TopicOptions import_options;
  import_options.tensor = holoscan::LogicalTensorRequirements{
      .memory_kind = holoscan::MemoryKind::kHost,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
      .rank = 1U,
      .logical_bounds =
          holoscan::LogicalTensorBounds{
              .max_rank = 1U,
              .max_byte_span = 8U,
              .axes = axes,
          },
      .strides = holoscan::TensorStridePolicy::kContiguousOnly,
      .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
  };
  import_options.bounds.publisher = holoscan::PublisherEndpointBounds{.max_loaned_samples = 1U};
  holoscan::TopicOptions export_options = import_options;
  export_options.bounds.publisher.reset();
  export_options.bounds.subscriber = holoscan::SubscriberEndpointBounds{
      .queue_depth = 1U,
      .max_borrowed_samples = 1U,
  };
  auto import_catalog = build_catalog<holoscan::Tensor>(
      holoscan::TopicDirection::kImport, endpoint_name, uri, import_options);
  auto export_catalog = build_catalog<holoscan::Tensor>(
      holoscan::TopicDirection::kExport, endpoint_name, uri, export_options);
  if (!import_catalog) {
    return fail_catalog_tensor_error("import catalog", import_catalog.error());
  }
  if (!export_catalog) {
    return fail_catalog_tensor_error("export catalog", export_catalog.error());
  }

  auto publisher_runtime = holoscan::ChannelRuntime::create();
  auto subscriber_runtime = holoscan::ChannelRuntime::create();
  if (!publisher_runtime) {
    return fail_catalog_tensor_error("publisher runtime", publisher_runtime.error());
  }
  if (!subscriber_runtime) {
    return fail_catalog_tensor_error("subscriber runtime", subscriber_runtime.error());
  }
  auto export_endpoint = export_catalog->endpoint(endpoint_name);
  if (!export_endpoint) {
    return fail_catalog_tensor_error("export endpoint lookup", export_endpoint.error());
  }
  auto subscriber = subscriber_runtime->create_subscriber<holoscan::Tensor>(*export_endpoint);
  if (!subscriber) {
    return fail_catalog_tensor_error("subscriber attach", subscriber.error());
  }
  auto publisher =
      publisher_runtime->create_publisher<holoscan::Tensor>(*import_catalog, endpoint_name);
  if (!publisher) {
    return fail_catalog_tensor_error("publisher attach", publisher.error());
  }
  if (!wait_for_connection(*publisher, *subscriber)) {
    return fail_catalog_tensor_message("peer connection", "connection refresh failed or timed out");
  }
  if (!rejects_unconfigured_clock_health(*publisher)) {
    return fail_catalog_tensor_message(
        "clock health", "unconfigured report was accepted or returned the wrong error");
  }
  return run_catalog_tensor_transfer(*publisher, *subscriber);
}

}  // namespace

// REQ: CHAN-001
// REQ: CHAN-003
// REQ: CHAN-012
// REQ: CHAN-069
// REQ: CHAN-074
// REQ: CHAN-075
// REQ: CHAN-120
// REQ: EXTERNAL-001
// REQ: EXTERNAL-002
// REQ: EXTERNAL-003
// REQ: EXTERNAL-004
// REQ: EXTERNAL-005
//
// Data-flow overview
// ──────────────────
// Scenario 1 — graph catalog workflow:
//
//   [External Publisher<ControlPod>]
//           │  holomq://<import_name>
//           ▼
//   [CatalogWorkflowTransform]   (inside holoscan::Graph)
//           │  internal queue
//           ▼
//   [External Subscriber<StringT>]
//           holomq://<export_name>
//
// Scenarios 2–5 — direct catalog pairs:
//
//   [Publisher<T>] ──holomq://──▶ [Subscriber<T>]
//     T ∈ { ControlPod, StringT, Tensor }
//   plus attachment-policy rejection probes (no peer, wrong role, unsupported storage)
int main() {
  if (!run_graph_catalog_workflow()) {
    std::cerr << "external publisher -> Graph -> external subscriber workflow failed\n";
    return 1;  // exit 1: external-publisher → Graph operator → external-subscriber round-trip
  }
  if (!run_attach_policy_smoke()) {
    std::cerr << "catalog attachment policy smoke failed\n";
    return 2;  // exit 2: catalog attach-policy (optional/wrong-role/required-missing/unsupported)
  }
  if (!run_catalog_pod_smoke()) {
    std::cerr << "catalog-backed custom POD smoke failed\n";
    return 3;  // exit 3: custom-POD direct catalog round-trip with trace/frame/degraded metadata
  }
  if (!run_generated_schema_smoke()) {
    std::cerr << "generated-schema package smoke failed\n";
    return 4;  // exit 4: generated FlatBuffers schema package encode/emit/receive round-trip
  }
  if (!run_catalog_tensor_smoke()) {
    std::cerr << "catalog-backed Tensor smoke failed\n";
    return 5;  // exit 5: Tensor loan → host-write → emit → receive → EOS lifecycle
  }
  return 0;
}
