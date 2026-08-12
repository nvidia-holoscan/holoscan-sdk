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
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "descriptor_frame_traits.hpp"

namespace {

using namespace std::chrono_literals;

struct Observation {
  static void reset() {
    const std::lock_guard lock(mutex);
    source_pointer = nullptr;
    done = false;
    valid = false;
    error.reset();
  }

  static void record_source(const void* pointer) {
    const std::lock_guard lock(mutex);
    source_pointer = pointer;
  }

  static void record(const holoscan::Sample<app3p::DescriptorFrameT>& sample) {
    const std::lock_guard lock(mutex);
    const holoscan::Tensor* tensor = sample.data.tensor.get();
    const std::uint8_t* bytes = tensor == nullptr ? nullptr : tensor->data_as<std::uint8_t>();
    valid = tensor != nullptr && tensor->data() == source_pointer && tensor->nbytes() == 8 &&
            std::ranges::equal(tensor->shape_span(), kShape) && bytes != nullptr &&
            std::ranges::equal(std::span<const std::uint8_t>{bytes, 8U}, kBytes) &&
            sample.data.sequence == 17U && sample.data.label == "installed descriptor frame" &&
            sample.data.gain == 0.25F && sample.data.samples == kSamples &&
            sample.data.aliases == kAliases && sample.data.details != nullptr &&
            sample.data.details->code == 29U && sample.data.details->note == "generated nested" &&
            sample.data.point != nullptr && sample.data.point->x() == 1.5F &&
            sample.data.point->y() == -2.0F && sample.metadata.frame_id == 23U;
    done = true;
    ready.notify_all();
  }

  static void fail(holoscan::ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
      done = true;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [] { return done; });
  }

  static constexpr std::array<std::int64_t, 1U> kShape{8};
  static constexpr std::array<std::uint8_t, 8U> kBytes{1U, 3U, 5U, 7U, 9U, 11U, 13U, 15U};
  static inline const std::vector<std::int16_t> kSamples{2, 4, 8, 16};
  static inline const std::vector<std::string> kAliases{"alpha", "beta"};
  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline const void* source_pointer{};
  static inline bool done{};
  static inline bool valid{};
  static inline std::optional<holoscan::ErrorCode> error;
};

// Publishes one generated third-party descriptor schema using a host Tensor output loan.
// Representative test: CHANNEL_SCHEMA.public_descriptor_schema_detailed_validation.
class Source final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::TensorPortLayout{
            .memory_kind = holoscan::MemoryKind::kHost,
            .dtype = DLDataType{kDLUInt, 8U, 1U},
            .rank = 1U,
        })
        .tensor_allocation(holoscan::TensorAllocationBounds{.max_rank = 1U, .max_byte_span = 8U})
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<app3p::DescriptorFrameT>(),
            .max_tensor_byte_span = 8U,
            .interface_tensor_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
        });
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
    auto loan = output.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = Observation::kShape,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
    });
    if (!loan) {
      return fail(loan.error().code);
    }
    auto written = loan->write_host_and_commit([](const holoscan::TensorOutputWriteGuard& writer) {
      Observation::record_source(writer.data());
      std::ranges::copy(Observation::kBytes, writer.data_as<std::uint8_t>());
    });
    if (!written) {
      return fail(written.error().code);
    }
    app3p::DescriptorFrameT frame{};
    frame.sequence = 17U;
    frame.label = "installed descriptor frame";
    frame.gain = 0.25F;
    frame.samples = Observation::kSamples;
    frame.aliases = Observation::kAliases;
    frame.details = std::make_shared<app3p::FrameDetailsT>();
    frame.details->code = 29U;
    frame.details->note = "generated nested";
    frame.point = std::make_shared<app3p::FramePoint>(1.5F, -2.0F);
    auto emitted =
        output.emit_tensor(std::move(*loan), frame, holoscan::EmitOptions{.frame_id = 23U});
    if (!emitted) {
      return fail(emitted.error().code);
    }
    return {};
  }

  holoscan::Output<app3p::DescriptorFrameT> output;

 private:
  [[nodiscard]] static holoscan::expected<void, holoscan::Error> fail(holoscan::ErrorCode code) {
    Observation::fail(code);
    return holoscan::make_unexpected(holoscan::Error{code});
  }

  bool emitted_{};
};

// Receives and validates the generated metadata, Tensor bytes, and retained source-pool identity.
// Representative test: CHANNEL_SCHEMA.public_descriptor_schema_detailed_validation.
class Sink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(holoscan::TensorPortLayout{
            .memory_kind = holoscan::MemoryKind::kHost,
            .dtype = DLDataType{kDLUInt, 8U, 1U},
            .rank = 1U,
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      Observation::fail(sample.error().code);
      return holoscan::make_unexpected(std::move(sample).error());
    }
    Observation::record(*sample);
    return {};
  }

  holoscan::Input<app3p::DescriptorFrameT> input;
};

[[nodiscard]] bool validate_generated_adapter() {
  static_assert(holoscan::detail::HasDescriptorTensorSchemaNativeTraits<app3p::DescriptorFrameT>);
  static_assert(!holoscan::detail::HasGeneratedSchemaNativeTraits<app3p::DescriptorFrameT>);
  static_assert(holoscan::detail::kRuntimeDescriptorTensorSchemaPayload<app3p::DescriptorFrameT>);

  const auto package =
      holoscan::validate_schema_package(holoscan::schema_package<app3p::DescriptorFrame>());
  if (!package) {
    return false;
  }
  const holoscan::detail::RuntimePortTypeOps* ops =
      holoscan::detail::runtime_port_type_ops<app3p::DescriptorFrameT>();
  if (ops == nullptr ||
      ops->codec_family != holoscan::detail::RuntimePayloadCodecFamily::kDescriptorTensorSchema ||
      ops->payload_kind != holoscan::PayloadKind::kDescriptorBacked ||
      ops->descriptor_tensor == nullptr || ops->descriptor_tensor_count == nullptr ||
      ops->descriptor_tensor_at == nullptr || ops->encode_descriptor_tensor_sample == nullptr ||
      ops->descriptor_tensor_view_metadata_at == nullptr ||
      ops->assign_descriptor_tensor_sample == nullptr ||
      ops->assign_descriptor_tensors_sample == nullptr) {
    return false;
  }
  alignas(holoscan::Sample<app3p::DescriptorFrameT>)
      std::array<std::byte, sizeof(holoscan::Sample<app3p::DescriptorFrameT>)>
          sample_storage{};
  auto prepared = ops->prepare_sample_storage(sample_storage.data());
  if (!prepared) {
    return false;
  }
  auto* prepared_sample = std::launder(
      reinterpret_cast<holoscan::Sample<app3p::DescriptorFrameT>*>(sample_storage.data()));
  if (prepared_sample->data.tensor == nullptr || prepared_sample->data.details != nullptr ||
      prepared_sample->data.point != nullptr || !prepared_sample->data.label.empty() ||
      !prepared_sample->data.samples.empty() || !prepared_sample->data.aliases.empty() ||
      ops->descriptor_tensor_count(&prepared_sample->data) != 1U ||
      ops->descriptor_tensor(&prepared_sample->data) !=
          ops->descriptor_tensor_at(&prepared_sample->data, 0U) ||
      ops->descriptor_tensor_at(&prepared_sample->data, 1U) != nullptr) {
    ops->destroy_sample(sample_storage.data());
    return false;
  }
  prepared_sample->data.details = std::make_shared<app3p::FrameDetailsT>();
  prepared_sample->data.point = std::make_shared<app3p::FramePoint>();
  const auto prepared_details = prepared_sample->data.details;
  const auto prepared_point = prepared_sample->data.point;
  auto nested_prepared =
      app3p::package::detail::GeneratedDescriptorCodec{}.prepare(prepared_sample->data);
  if (!nested_prepared || prepared_sample->data.details != prepared_details ||
      prepared_sample->data.point != prepared_point || !prepared_details->note.empty()) {
    ops->destroy_sample(sample_storage.data());
    return false;
  }
  prepared_sample->data.sequence = 99U;
  prepared_sample->data.label = "stale";
  prepared_sample->data.samples = {7};
  prepared_sample->data.aliases = {"stale"};
  prepared_details->code = 99U;
  prepared_details->note = "stale";
  *prepared_point = app3p::FramePoint{9.0F, 9.0F};
  ops->reset_sample_storage(sample_storage.data());
  const bool reset =
      prepared_sample->data.sequence == 0U && prepared_sample->data.label.empty() &&
      prepared_sample->data.samples.empty() && prepared_sample->data.aliases.empty() &&
      prepared_sample->data.details == prepared_details && prepared_details->code == 0U &&
      prepared_details->note.empty() && prepared_sample->data.point == prepared_point &&
      prepared_point->x() == 0.0F && prepared_point->y() == 0.0F;
  ops->destroy_sample(sample_storage.data());
  return reset;
}

[[nodiscard]] bool has_expected_plan_shape(const holoscan::ExecutionPlan& plan) {
  return std::ranges::any_of(plan.payload_edges(), [](const auto& edge) {
    return edge.payload_identity.payload_kind == holoscan::PayloadKind::kDescriptorBacked;
  });
}

}  // namespace

// REQ: CHAN-120
// REQ: OPERATOR-035
//   source(Source) --[descriptor-backed DescriptorFrameT]--> sink(Sink)
int main() {
  if (!validate_generated_adapter()) {
    return 1;
  }

  Observation::reset();
  holoscan::Graph graph{"public-descriptor-schema-validation"};
  const auto source = graph.op<Source>("source");
  const auto sink = graph.op<Sink>("sink");
  graph.add_flow(source->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 1U});
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
  return observed && Observation::valid && !Observation::error.has_value() &&
                 session.termination() == holoscan::RunTermination::kStopped &&
                 session.diagnostics().empty()
             ? 0
             : 3;
}
