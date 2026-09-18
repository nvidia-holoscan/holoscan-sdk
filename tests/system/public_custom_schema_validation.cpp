/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <span>
#include <string>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/flatbuffer_schema_package.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/time/realtime_clock.hpp>

// Registers the generated CustomControl owning type with schema concepts.
#include "custom_control_schema_traits.hpp"  // NOLINT(misc-include-cleaner)

namespace {

using namespace std::chrono_literals;

constexpr std::size_t kControlCount = 5U;
constexpr std::chrono::nanoseconds kTickPeriod = 20ms;

static_assert(holoscan::PublishedPayload<app::CustomControlT>);
static_assert(holoscan::detail::HasSchemaIdentity<app::CustomControlT>);
static_assert(holoscan::detail::HasGeneratedSchemaNativeTraits<app::CustomControlT>);
static_assert(holoscan::detail::HasSchemaPackage<app::CustomControlT>);
static_assert(holoscan::detail::kRuntimeGeneratedSchemaPayload<app::CustomControlT>);
static_assert(holoscan::generated_schema_max_serialized_size_v<app::CustomControlT> == 512U);

struct Observation {
  static void reset() {
    const std::lock_guard lock(mutex);
    received = 0U;
    valid = true;
  }

  static void record(const app::CustomControlT& control) {
    {
      const std::lock_guard lock(mutex);
      ++received;
      const auto expected_sequence = static_cast<std::uint64_t>(received);
      valid = valid && control.sequence == expected_sequence &&
              control.gain == 0.25F * static_cast<float>(expected_sequence) &&
              control.label == "ctrl-" + std::to_string(expected_sequence);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [] { return received >= kControlCount; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::size_t received{};
  static inline bool valid{true};
};

// Emits a bounded sequence of generated custom-schema controls for the public-package smoke test.
// Representative test: CHANNEL_SCHEMA.public_custom_schema_detailed_validation.
class Source final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    if (emitted_ >= kControlCount) {
      return {};
    }
    ++emitted_;
    app::CustomControlT control;
    control.sequence = emitted_;
    control.gain = 0.25F * static_cast<float>(emitted_);
    control.label = "ctrl-" + std::to_string(emitted_);
    return output.emit(std::move(control));
  }

  holoscan::Output<app::CustomControlT> output;

 private:
  std::uint64_t emitted_{};
};

// Receives the generated controls and records their exact scalar and string fields.
// Representative test: CHANNEL_SCHEMA.public_custom_schema_detailed_validation.
class Sink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto control = input.receive_data();
    if (!control) {
      return holoscan::make_unexpected(std::move(control).error());
    }
    Observation::record(*control);
    return {};
  }

  holoscan::Input<app::CustomControlT> input;
};

[[nodiscard]] bool validates_generated_package_and_type_ops() {
  const auto package =
      holoscan::validate_schema_package(holoscan::schema_package<app::CustomControlT>());
  if (!package) {
    return false;
  }
  const holoscan::detail::RuntimePortTypeOps* ops =
      holoscan::detail::runtime_port_type_ops<app::CustomControlT>();
  return ops != nullptr &&
         ops->codec_family == holoscan::detail::RuntimePayloadCodecFamily::kGeneratedSchema &&
         ops->payload_kind == holoscan::PayloadKind::kSerializedBytes &&
         ops->wire_capacity == 512U && ops->encode_payload != nullptr &&
         ops->construct_sample != nullptr && ops->destroy_sample != nullptr;
}

[[nodiscard]] bool has_expected_plan_shape(const holoscan::ExecutionPlan& plan) {
  if (plan.ports().size() != 2U || plan.payload_edges().size() != 1U ||
      plan.payload_edges().front().payload_identity.kind !=
          holoscan::PayloadIdentityKind::kSchema) {
    return false;
  }
  for (const holoscan::PortPlanRecord& port : plan.ports()) {
    if (port.payload_identity.kind != holoscan::PayloadIdentityKind::kSchema ||
        port.payload_identity.payload_kind != holoscan::PayloadKind::kSerializedBytes ||
        port.serialized_payload_capacity_bytes != 512U) {
      return false;
    }
  }
  return true;
}

// Preserve the application-owned generated-schema rejection contract independently from the
// pedagogical example's normal Operator/Graph flow.
// REQ: CHAN-120
// REQ: SYSTEM-022
[[nodiscard]] bool rejects_invalid_peer_payloads() {
  constexpr std::uint64_t kEndpointGeneration = 41U;
  const auto& package = holoscan::schema_package<app::CustomControlT>();
  const holoscan::SchemaIdentity identity = holoscan::schema_identity<app::CustomControlT>();
  app::CustomControlT control;
  control.sequence = 7U;
  control.gain = 1.75F;
  control.label = "peer-validation";

  // Key setup: encode one valid application-owned value, then mutate independent payload and
  // envelope copies for the corrupt-bytes, foreign-ID, and version-drift cases.
  alignas(std::max_align_t)
      std::array<std::byte, holoscan::generated_schema_max_serialized_size_v<app::CustomControlT>>
          storage{};
  auto encoded = package.encode(control, storage);
  if (!encoded) {
    std::cerr << holoscan::runtime_error_message(encoded.error()) << '\n';
    return false;
  }
  const holoscan::SchemaEnvelopeHeader header{
      .schema_family_id = identity.family_id,
      .payload_size = static_cast<std::uint32_t>(*encoded),
      .transport_profile_tag = 0U,
      .endpoint_generation = kEndpointGeneration,
  };
  const auto payload = std::span<const std::byte>{storage}.first(*encoded);
  auto accepted = holoscan::verify_and_materialize_received_schema(
      package, header, identity, kEndpointGeneration, payload);
  if (!accepted) {
    std::cerr << holoscan::runtime_error_message(accepted.error()) << '\n';
    return false;
  }

  auto corrupted_storage = storage;
  std::fill_n(corrupted_storage.begin(), 4U, std::byte{0xFF});
  const auto corrupted_payload = std::span<const std::byte>{corrupted_storage}.first(*encoded);
  const auto corrupted = holoscan::verify_and_materialize_received_schema(
      package, header, identity, kEndpointGeneration, corrupted_payload);

  auto foreign_header = header;
  foreign_header.schema_family_id.value[0] ^= std::byte{0xFF};
  const auto foreign = holoscan::verify_and_materialize_received_schema(
      package, foreign_header, identity, kEndpointGeneration, payload);

  auto profiled_header = header;
  ++profiled_header.transport_profile_tag;
  const auto profiled = holoscan::verify_and_materialize_received_schema(
      package, profiled_header, identity, kEndpointGeneration, payload);

  // Key assertion: every peer mutation is rejected as a protocol error before an owning
  // CustomControlT can be exposed.
  return !corrupted && corrupted.error().code == holoscan::ErrorCode::kProtocolError && !foreign &&
         foreign.error().code == holoscan::ErrorCode::kProtocolError && !profiled &&
         profiled.error().code == holoscan::ErrorCode::kProtocolError;
}

}  // namespace

// REQ: CHAN-077
// REQ: CHAN-120
//   source(Source) --[CustomControlT, queue_depth=4]--> sink(Sink)
int main() {
  if (!validates_generated_package_and_type_ops() || !rejects_invalid_peer_payloads()) {
    return 1;
  }

  Observation::reset();
  holoscan::Graph graph{"public-custom-schema-validation"};
  const auto source = graph.op<Source>("source");
  const auto sink = graph.op<Sink>("sink");
  graph.add_flow(source->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok() || !has_expected_plan_shape(plan)) {
    std::cerr << plan.json() << '\n';
    return 2;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool observed = Observation::wait();
  session.request_stop();
  session.wait();

  const std::lock_guard lock(Observation::mutex);
  return observed && Observation::received == kControlCount && Observation::valid &&
                 session.termination() == holoscan::RunTermination::kStopped &&
                 session.diagnostics().empty()
             ? 0
             : 3;
}
