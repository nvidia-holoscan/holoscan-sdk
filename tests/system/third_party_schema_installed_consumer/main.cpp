/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/idl.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "schema_tooling_tensor_probe_generated.h"
#include "third_party_frame_traits.hpp"

namespace {

using namespace std::chrono_literals;

constexpr std::uint64_t kExpectedSequence = 42U;
constexpr std::string_view kExpectedLabel = "installed third-party schema";

static_assert(holoscan::PublishedPayload<app3p::FrameT>);
static_assert(sizeof(holoscan::schema::SchemaToolingTensorProbeT) > 0U);
static_assert(holoscan::detail::HasSchemaIdentity<app3p::FrameT>);
static_assert(holoscan::detail::HasGeneratedSchemaNativeTraits<app3p::FrameT>);
static_assert(holoscan::detail::HasSchemaPackage<app3p::FrameT>);
static_assert(holoscan::detail::kRuntimeGeneratedSchemaPayload<app3p::FrameT>);
static_assert(holoscan::detail::serialized_payload_capacity_bytes<app3p::FrameT>() ==
              app3p::package::kMaxSerializedSize);
static_assert(
    std::same_as<
        decltype(holoscan::schema::make_flatbuffer_package<app3p::Frame, app3p::FrameBinarySchema>(
            app3p::package::kIdentity, holoscan::SchemaPackageEvidence{})),
        holoscan::SchemaPackageDescriptor<app3p::Frame>>);

[[nodiscard]] bool flatbuffers_runtime_surface_works() {
  flatbuffers::FlatBufferBuilder builder;
  const auto label = builder.CreateString(kExpectedLabel);
  builder.Finish(app3p::CreateFrame(builder, kExpectedSequence, label));

  flatbuffers::Verifier verifier{builder.GetBufferPointer(), builder.GetSize()};
  if (!verifier.VerifyBuffer<app3p::Frame>(nullptr)) {
    return false;
  }

  app3p::FrameT unpacked;
  app3p::GetFrame(builder.GetBufferPointer())->UnPackTo(&unpacked);
  flatbuffers::FlatBufferBuilder repacked_builder;
  repacked_builder.Finish(app3p::Frame::Pack(repacked_builder, &unpacked));
  return unpacked.sequence == kExpectedSequence && unpacked.label == kExpectedLabel &&
         repacked_builder.GetSize() != 0U;
}

[[nodiscard]] bool bundled_flatbuffers_library_works() {
  flatbuffers::Parser parser;
  return parser.Parse("table Probe { value:uint; } root_type Probe;");
}

struct SinkObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    done = false;
    valid = false;
  }

  static void record(const app3p::FrameT& value) {
    {
      const std::lock_guard lock(mutex);
      valid = value.sequence == kExpectedSequence && value.label == kExpectedLabel;
      done = true;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [] { return done; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline bool done{};
  static inline bool valid{};
};

// Emits one object-API Frame through schema traits generated from the installed SDK tooling.
// Representative test: CHANNEL_SCHEMA.installed_third_party_native_consumer.
class FrameSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    app3p::FrameT frame;
    frame.sequence = kExpectedSequence;
    frame.label = kExpectedLabel;
    return output.emit(std::move(frame));
  }

  holoscan::Output<app3p::FrameT> output;

 private:
  bool emitted_{};
};

// Receives the generated Frame and validates its exact sequence and label.
// Representative test: CHANNEL_SCHEMA.installed_third_party_native_consumer.
class FrameSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto frame = input.receive_data();
    if (!frame) {
      return holoscan::make_unexpected(std::move(frame).error());
    }
    SinkObservation::record(*frame);
    return {};
  }

  holoscan::Input<app3p::FrameT> input;
};

void print_diagnostics(const holoscan::RunSession& session) {
  for (const auto& diagnostic : session.diagnostics()) {
    std::cerr << diagnostic.token << " required=" << diagnostic.required
              << " provided=" << diagnostic.provided << '\n';
  }
}

}  // namespace

// REQ: CHAN-120
//   source(FrameSource) --[FrameT, queue_depth=1]--> sink(FrameSink)
int main() {
  if (!bundled_flatbuffers_library_works()) {
    return 1;
  }
  if (!flatbuffers_runtime_surface_works()) {
    return 1;
  }

  const auto package_valid =
      holoscan::validate_schema_package(holoscan::schema_package<app3p::FrameT>());
  if (!package_valid) {
    std::cerr << "third-party package validation failed\n";
    return 2;
  }
  const holoscan::SchemaPackageEvidence& package_evidence =
      holoscan::schema_package<app3p::FrameT>().evidence();
  if (package_evidence.includes.size() != 1U ||
      package_evidence.includes.front().logical_path != "third_party_common.fbs" ||
      package_evidence.include_root_mapping != "schema-root-0=.;schema-root-1=." ||
      !package_evidence.host_paths_stripped || package_evidence.generator_format_version != 1U) {
    std::cerr << "generated package evidence is incomplete\n";
    return 3;
  }

  const holoscan::detail::RuntimePortTypeOps* type_ops =
      holoscan::detail::runtime_port_type_ops<app3p::FrameT>();
  if (type_ops == nullptr ||
      type_ops->codec_family != holoscan::detail::RuntimePayloadCodecFamily::kGeneratedSchema ||
      type_ops->payload_kind != holoscan::PayloadKind::kSerializedBytes ||
      type_ops->wire_capacity != app3p::package::kMaxSerializedSize ||
      type_ops->encode_payload == nullptr || type_ops->construct_sample == nullptr ||
      type_ops->destroy_sample == nullptr) {
    std::cerr << "generated Runtime type operations are incomplete\n";
    return 4;
  }

  SinkObservation::reset();
  holoscan::Graph graph{"third-party-schema-installed-consumer"};
  const auto source = graph.op<FrameSource>("source");
  const auto sink = graph.op<FrameSink>("sink");
  graph.add_flow(source->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 1U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok() || plan.ports().size() != 2U) {
    std::cerr << plan.json() << '\n';
    return 5;
  }
  for (const holoscan::PortPlanRecord& port : plan.ports()) {
    if (port.payload_identity.kind != holoscan::PayloadIdentityKind::kSchema ||
        port.payload_identity.payload_kind != holoscan::PayloadKind::kSerializedBytes ||
        port.serialized_payload_capacity_bytes != app3p::package::kMaxSerializedSize) {
      std::cerr << "compiled port does not retain the generated schema contract\n";
      return 6;
    }
  }
  if (plan.payload_edges().size() != 1U || plan.payload_edges().front().payload_identity.kind !=
                                               holoscan::PayloadIdentityKind::kSchema) {
    std::cerr << "compiled edge does not retain the generated schema identity\n";
    return 7;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool observed = SinkObservation::wait();
  session.request_stop();
  session.wait();
  if (!observed || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    print_diagnostics(session);
    return 8;
  }
  const std::lock_guard lock(SinkObservation::mutex);
  if (!SinkObservation::valid) {
    std::cerr << "sink did not observe the expected generated owning value\n";
  }
  return SinkObservation::valid ? 0 : 9;
}
