/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/predicate.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/sdk_schemas.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/run.hpp>
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/domain/tensor_access.hpp"
#include "runtime/run_session_internal.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_inline_execution_test_support.hpp"

namespace holoscan {

// REQ: RUNTIME-040
// REQ: RUNTIME-041

struct InlinePartitionParams {
  std::uint32_t value{1U};
};

HOLOSCAN_PARAM_SCHEMA(InlinePartitionParams, param_schema("holoscan.test.InlinePartitionParams", 1),
                      field<&InlinePartitionParams::value>("value"));

struct InlinePartitionRuntimeParams {
  std::uint32_t value{1U};
};

HOLOSCAN_PARAM_SCHEMA(InlinePartitionRuntimeParams,
                      param_schema("holoscan.test.InlinePartitionRuntimeParams", 1),
                      field<&InlinePartitionRuntimeParams::value>("value").runtime_mutable());

struct InlineTemporalParams {
  std::uint32_t value{200U};
  std::chrono::nanoseconds max_age{std::chrono::seconds{1}};
  std::chrono::nanoseconds deadline{std::chrono::milliseconds{1}};
};

HOLOSCAN_PARAM_SCHEMA(InlineTemporalParams, param_schema("holoscan.test.InlineTemporalParams", 1),
                      field<&InlineTemporalParams::value>("value").runtime_mutable(),
                      field<&InlineTemporalParams::max_age>("max_age").runtime_mutable().range(
                          std::chrono::milliseconds{1}, std::chrono::seconds{10}),
                      field<&InlineTemporalParams::deadline>("deadline")
                          .runtime_mutable()
                          .range(std::chrono::milliseconds{1}, std::chrono::seconds{10}));

namespace {

using namespace std::chrono_literals;

using namespace test::inline_execution;

// Emits a POD value read from immutable typed parameters and records parameter pinning.
// Representative test: FrozenTypedParamsRemainCoherentAcrossRepeatedInlineRunsWithoutHandoffs.
class InlineParamSource final : public Operator<InlinePartitionParams> {
 public:
  explicit InlineParamSource(std::uint32_t operator_id) : operator_id_(operator_id) {}

  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kStart, &InlineParamSource::start_stage)
        .stage(LifecycleStage::kStop, &InlineParamSource::stop_stage);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    InlinePartitionProbeState::record_param_compute<InlinePartitionParams>(operator_id_, context);
    const auto& params = context.params<InlinePartitionParams>();
    InlinePartitionProbeState::record_compute(operator_id_, params.value);
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    return output.emit(InlinePartitionPod{.value = params.value});
  }

  Output<InlinePartitionPod> output;

 private:
  LifecycleStatus start_stage(LifecycleContext& context) noexcept {
    InlinePartitionProbeState::record_param_lifecycle<InlinePartitionParams>(
        operator_id_, LifecycleStage::kStart, context);
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStart);
    return LifecycleStatus::kOk;
  }

  LifecycleStatus stop_stage(LifecycleContext& context) noexcept {
    InlinePartitionProbeState::record_param_lifecycle<InlinePartitionParams>(
        operator_id_, LifecycleStage::kStop, context);
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStop);
    return LifecycleStatus::kOk;
  }

  std::uint32_t operator_id_{};
  bool emitted_{};
};

// Adds an immutable typed parameter to each POD sample and records parameter pinning.
// Representative test: FrozenTypedParamsRemainCoherentAcrossRepeatedInlineRunsWithoutHandoffs.
class InlineParamRelay final : public Operator<InlinePartitionParams> {
 public:
  explicit InlineParamRelay(std::uint32_t operator_id) : operator_id_(operator_id) {}

  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kStart, &InlineParamRelay::start_stage)
        .stage(LifecycleStage::kStop, &InlineParamRelay::stop_stage);
    spec.input(input, "input").queue_depth(2U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    InlinePartitionProbeState::record_param_compute<InlinePartitionParams>(operator_id_, context);
    const std::uint32_t value = sample->data.value + context.params<InlinePartitionParams>().value;
    InlinePartitionProbeState::record_compute(operator_id_, value);
    return output.emit(InlinePartitionPod{.value = value});
  }

  Input<InlinePartitionPod> input;
  Output<InlinePartitionPod> output;

 private:
  LifecycleStatus start_stage(LifecycleContext& context) noexcept {
    InlinePartitionProbeState::record_param_lifecycle<InlinePartitionParams>(
        operator_id_, LifecycleStage::kStart, context);
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStart);
    return LifecycleStatus::kOk;
  }

  LifecycleStatus stop_stage(LifecycleContext& context) noexcept {
    InlinePartitionProbeState::record_param_lifecycle<InlinePartitionParams>(
        operator_id_, LifecycleStage::kStop, context);
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStop);
    return LifecycleStatus::kOk;
  }

  std::uint32_t operator_id_{};
};

// Emits values from two parameter generations while allowing a NextCompute update between them.
// Representative test: RuntimeMutableNextComputePublishesPinsAndObservesOnTheController.
class RuntimeMutableInlineParamSource final : public Operator<InlinePartitionRuntimeParams> {
 public:
  explicit RuntimeMutableInlineParamSource(std::uint32_t operator_id) : operator_id_(operator_id) {}

  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kStart, &RuntimeMutableInlineParamSource::start_stage)
        .stage(LifecycleStage::kStop, &RuntimeMutableInlineParamSource::stop_stage);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 50ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (compute_count_ >= 2U) {
      return {};
    }
    const bool first = compute_count_ == 0U;
    if (!InlinePartitionProbeState::record_runtime_param_compute<InlinePartitionRuntimeParams>(
            operator_id_, context, first)) {
      return make_unexpected(Error{ErrorCode::kFailure, "INLINE_RUNTIME_PARAM_RELEASE_TIMEOUT"});
    }
    const std::uint32_t value = context.params<InlinePartitionRuntimeParams>().value;
    InlinePartitionProbeState::record_compute(operator_id_, value);
    ++compute_count_;
    return output.emit(InlinePartitionPod{
        .value = value,
    });
  }

  Output<InlinePartitionPod> output;

 private:
  LifecycleStatus start_stage(LifecycleContext&) noexcept {
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStart);
    return LifecycleStatus::kOk;
  }

  LifecycleStatus stop_stage(LifecycleContext&) noexcept {
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStop);
    return LifecycleStatus::kOk;
  }

  std::uint32_t operator_id_{};
  std::uint32_t compute_count_{};
};

// Emits driver and deliberately older side samples for temporal-selection parameter tests.
// Representative test: UpdatedSelectorDeadlineAndCallbackShareOnePinnedGeneration.
class InlineTemporalForkSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kStart, &InlineTemporalForkSource::start_stage)
        .stage(LifecycleStage::kStop, &InlineTemporalForkSource::stop_stage);
    spec.output(driver, "driver").max_emits_per_compute(1U);
    spec.output(side, "side").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    InlinePartitionProbeState::record_compute(1U, 100U);
    if (emitted_) {
      return {};
    }
    if (!InlinePartitionProbeState::block_first_runtime_compute()) {
      return make_unexpected(Error{ErrorCode::kFailure, "INLINE_TEMPORAL_RELEASE_TIMEOUT"});
    }
    const TimePoint activation = context.activation_time();
    if (!activation.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const SampleMetadata driver_metadata{
        .capture_timestamp_ns = activation.timestamp_ns,
        .source_clock_id = activation.clock_id,
    };
    const SampleMetadata side_metadata{
        .capture_timestamp_ns = activation.timestamp_ns - 2'000'000'000LL,
        .source_clock_id = activation.clock_id,
    };
    emitted_ = true;
    auto side_result =
        side.emit(Sample<InlinePartitionPod>{InlinePartitionPod{.value = 102U}, side_metadata});
    if (!side_result) {
      return side_result;
    }
    return driver.emit(
        Sample<InlinePartitionPod>{InlinePartitionPod{.value = 101U}, driver_metadata});
  }

  Output<InlinePartitionPod> driver;
  Output<InlinePartitionPod> side;

 private:
  LifecycleStatus start_stage(LifecycleContext&) noexcept {
    InlinePartitionProbeState::record_lifecycle(1U, LifecycleStage::kStart);
    return LifecycleStatus::kOk;
  }

  LifecycleStatus stop_stage(LifecycleContext&) noexcept {
    InlinePartitionProbeState::record_lifecycle(1U, LifecycleStage::kStop);
    return LifecycleStatus::kOk;
  }

  bool emitted_{};
};

// Evaluates a parameterized temporal predicate and joins its driver and side inputs.
// Representative test: UpdatedSelectorDeadlineAndCallbackShareOnePinnedGeneration.
class InlineTemporalRelay final : public Operator<InlineTemporalParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kStart, &InlineTemporalRelay::start_stage)
        .stage(LifecycleStage::kStop, &InlineTemporalRelay::stop_stage);
    spec.input(driver, "driver").queue_depth(2U);
    spec.input(side, "side").queue_depth(2U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    auto evaluator = predicate("inline-temporal-predicate",
                               predicate_inputs(side),
                               bind_predicate(*this, &InlineTemporalRelay::evaluate));
    Contract result;
    result
        .trigger(OnPredicate{
            .evaluate_on = OnInputUpdate{driver},
            .predicate = std::move(evaluator),
            .firing = PredicateFiring::kOnEveryTrueEvaluation,
        })
        .input(driver, Exact{})
        .input(side, LatestWithin{.max_age = param_ref<&InlineTemporalParams::max_age>()})
        .on_missing(side, Skip{})
        .max_start_delay(param_ref<&InlineTemporalParams::deadline>())
        .on_violation(OnViolation::kEmitWitness);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto driver_sample = driver.receive();
    if (!driver_sample) {
      return make_unexpected(std::move(driver_sample.error()));
    }
    auto side_sample = side.receive();
    if (!side_sample) {
      return make_unexpected(std::move(side_sample.error()));
    }
    InlinePartitionProbeState::record_param_compute<InlineTemporalParams>(2U, context);
    const std::uint32_t value = driver_sample->data.value + side_sample->data.value;
    InlinePartitionProbeState::record_compute(2U, value);
    return output.emit(InlinePartitionPod{.value = value});
  }

  [[nodiscard]] bool evaluate(
      const PredicateEvaluationContext&, const PredicateInputView<InlinePartitionPod>& driver_view,
      const PredicateInputView<InlinePartitionPod>& side_view) const noexcept {
    constexpr std::uint32_t threshold = 200U;
    const std::uint32_t driver_value = driver_view.sample().data.value;
    const std::uint32_t side_value = side_view.sample().data.value;
    InlinePartitionProbeState::record_temporal_predicate(threshold, driver_value, side_value);
    return driver_value + side_value >= threshold;
  }

  Input<InlinePartitionPod> driver;
  Input<InlinePartitionPod> side;
  Output<InlinePartitionPod> output;

 private:
  LifecycleStatus start_stage(LifecycleContext&) noexcept {
    InlinePartitionProbeState::record_lifecycle(2U, LifecycleStage::kStart);
    return LifecycleStatus::kOk;
  }

  LifecycleStatus stop_stage(LifecycleContext&) noexcept {
    InlinePartitionProbeState::record_lifecycle(2U, LifecycleStage::kStop);
    return LifecycleStatus::kOk;
  }
};

// Selects an anchor plus a recent side sample to exercise LatestWithin partition planning.
// Representative test: LatestWithinSelectionUsesTheInlinePartitionPlan.
class LatestWithinInlineRelay final : public Operator<>,
                                      public InlinePartitionLifecycle<LatestWithinInlineRelay> {
 public:
  LatestWithinInlineRelay() : InlinePartitionLifecycle<LatestWithinInlineRelay>(2U) {}

  void setup(OperatorSpec& spec) override {
    setup_lifecycle(spec);
    spec.input(anchor, "anchor").queue_depth(2U);
    spec.input(side, "side").queue_depth(2U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{anchor})
        .input(side, LatestWithin{.max_age = 30s})
        .on_missing(side, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = anchor.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    return output.emit(std::move(*sample));
  }

  Input<InlinePartitionPod> anchor;
  Input<InlinePartitionPod> side;
  Output<InlinePartitionPod> output;
};

// Publishes one POD sample through a single output connected to two consumers.
// Representative test: AdmitsOneRootWithTwoConsumersInStableSiblingOrder.
class InlineFanoutSource final : public Operator<>,
                                 public InlinePartitionLifecycle<InlineFanoutSource> {
 public:
  InlineFanoutSource() : InlinePartitionLifecycle<InlineFanoutSource>(1U) {}

  void setup(OperatorSpec& spec) override {
    setup_lifecycle(spec);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    InlinePartitionProbeState::record_compute(operator_id_, 100U);
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    return output.emit(InlinePartitionPod{.value = 101U});
  }

  Output<InlinePartitionPod> output;

 private:
  bool emitted_{};
};

// Publishes numbered owning String schema samples from a configurable clock.
// Representative test: RunsAnOwningSchemaChainOnTheControllerWithoutHandoffs.
class InlineSchemaSource final : public Operator<>,
                                 public InlinePartitionLifecycle<InlineSchemaSource> {
 public:
  explicit InlineSchemaSource(std::size_t emission_limit = 1U,
                              std::chrono::nanoseconds period = 30s)
      : InlinePartitionLifecycle<InlineSchemaSource>(1U),
        emission_limit_(emission_limit),
        period_(period) {}

  void setup(OperatorSpec& spec) override {
    setup_lifecycle(spec);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = period_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (next_frame_ >= emission_limit_) {
      return {};
    }
    const std::uint32_t frame = static_cast<std::uint32_t>(next_frame_);
    schema::StringT sample;
    sample.data = "root";
    auto emitted = output.emit(Sample<schema::StringT>{
        std::move(sample),
        SampleMetadata{.frame_id = frame},
    });
    if (!emitted) {
      return emitted;
    }
    InlinePartitionProbeState::record_compute(operator_id_, frame, context.tag_time());
    ++next_frame_;
    return {};
  }

  Output<schema::StringT> output;

 private:
  std::size_t emission_limit_{};
  std::chrono::nanoseconds period_{};
  std::size_t next_frame_{};
};

// Appends its operator ID to an owning String schema while preserving frame metadata.
// Representative test: RunsAnOwningSchemaChainOnTheControllerWithoutHandoffs.
class InlineSchemaRelay final : public Operator<>,
                                public InlinePartitionLifecycle<InlineSchemaRelay> {
 public:
  explicit InlineSchemaRelay(std::uint32_t operator_id)
      : InlinePartitionLifecycle<InlineSchemaRelay>(operator_id) {}

  void setup(OperatorSpec& spec) override {
    setup_lifecycle(spec);
    spec.input(input, "input").queue_depth(2U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    sample->data.data += ":" + std::to_string(operator_id_);
    const std::uint32_t frame = static_cast<std::uint32_t>(sample->metadata.frame_id);
    auto emitted = output.emit(std::move(*sample));
    if (!emitted) {
      return emitted;
    }
    InlinePartitionProbeState::record_compute(operator_id_, frame, context.tag_time());
    return {};
  }

  Input<schema::StringT> input;
  Output<schema::StringT> output;
};

// Records the final owning String schema, frame order, tag time, and lifecycle activity.
// Representative test: RunsAnOwningSchemaChainOnTheControllerWithoutHandoffs.
class InlineSchemaSink final : public Operator<>,
                               public InlinePartitionLifecycle<InlineSchemaSink> {
 public:
  InlineSchemaSink() : InlinePartitionLifecycle<InlineSchemaSink>(4U) {}

  void setup(OperatorSpec& spec) override {
    setup_lifecycle(spec);
    spec.input(input, "input").queue_depth(2U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    const std::uint32_t frame = static_cast<std::uint32_t>(sample->metadata.frame_id);
    InlinePartitionProbeState::record_compute(operator_id_, frame, context.tag_time());
    InlinePartitionProbeState::record_sink(frame);
    InlinePartitionProbeState::record_schema(sample->data.data);
    return {};
  }

  Input<schema::StringT> input;
};

[[nodiscard]] ExecutionPlan make_restart_chain_plan() {
  // [restarting source] -> [relay] -> [sink]
  Graph graph{"inline-restart-chain"};
  auto source = graph.op<InlinePodSource>("source", 1U, 41U, true);
  auto relay = graph.op<InlinePodRelay>("relay", 2U, 1U);
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  graph.add_flow(source->output, relay->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(relay->output, sink->input, queue_options(QueuePolicy::kBlocking));
  graph.lifecycle().allow_self_restart(source);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  CompileOptions options;
  options.lifecycle_transactions = inline_lifecycle_capacity();
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_failing_chain_plan() {
  // [source] -> [failing relay] -> [sink]
  Graph graph{"inline-failing-chain"};
  auto source = graph.op<InlinePodSource>("source", 1U, 41U);
  auto relay = graph.op<InlinePodRelay>("failing-relay", 2U, 1U, true);
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  graph.add_flow(source->output, relay->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(relay->output, sink->input, queue_options(QueuePolicy::kBlocking));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_param_chain_plan(bool explicit_placement = false) {
  // [parameter source] -> [parameter relay] -> [sink]
  Graph graph{explicit_placement ? "inline-param-chain-fallback" : "inline-param-chain"};
  auto source = graph.op<InlineParamSource>("source", 1U);
  auto relay = graph.op<InlineParamRelay>("relay", 2U);
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  source.set_params(InlinePartitionParams{.value = 41U});
  relay.set_params(InlinePartitionParams{.value = 3U});
  graph.add_flow(source->output, relay->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(relay->output, sink->input, queue_options(QueuePolicy::kBlocking));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  CompileOptions options;
  if (explicit_placement) {
    auto partition = graph.partition("inline-partition-fallback-worker");
    partition.add(source).add(relay).add(sink);
    options.deployment.bind_thread_placement("inline-partition-fallback-worker", ThreadPlacement{});
  }
  return compile(graph, std::move(options));
}

enum class RuntimeMutablePlanRoute : std::uint8_t {
  kInline,
  kExplicitPlacement,
  kHoloMQ,
};

[[nodiscard]] ExecutionPlan make_runtime_mutable_param_chain_plan(
    RuntimeMutablePlanRoute route = RuntimeMutablePlanRoute::kInline) {
  // [runtime-parameter source] -> [relay] -> [sink]
  Graph graph{"runtime-mutable-param-chain"};
  const bool explicit_placement = route == RuntimeMutablePlanRoute::kExplicitPlacement;
  auto source = graph.op<RuntimeMutableInlineParamSource>("source", 1U);
  auto relay = graph.op<InlinePodRelay>("relay", 2U, 3U);
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  source.set_params(InlinePartitionRuntimeParams{.value = 41U});
  graph.add_flow(source->output, relay->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(relay->output, sink->input, queue_options(QueuePolicy::kBlocking));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  if (explicit_placement) {
    auto partition = graph.partition("inline-partition-fallback-worker");
    partition.add(source).add(relay).add(sink);
    options.deployment.bind_thread_placement("inline-partition-fallback-worker", ThreadPlacement{});
  }
  return route == RuntimeMutablePlanRoute::kHoloMQ
             ? detail::compile_with_process_local_holomq(graph, std::move(options))
             : compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_temporal_param_chain_plan() {
  // [source] -- driver + side --> [temporal relay] -> [sink]
  Graph graph{"inline-temporal-param-chain"};
  auto source = graph.op<InlineTemporalForkSource>("source");
  auto relay = graph.op<InlineTemporalRelay>("relay");
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  relay.set_params(InlineTemporalParams{});
  graph.add_flow(source->driver, relay->driver, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(source->side, relay->side, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(relay->output, sink->input, queue_options(QueuePolicy::kBlocking));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_latest_within_chain_plan() {
  // [source] -- anchor + side --> [LatestWithin relay] -> [sink]
  Graph graph{"latest-within-inline-fallback"};
  auto source = graph.op<InlineForkSource>("source");
  auto relay = graph.op<LatestWithinInlineRelay>("relay");
  auto sink = graph.op<InlinePodSink>("sink", 3U);
  graph.add_flow(source->first, relay->anchor, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(source->second, relay->side, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(relay->output, sink->input, queue_options(QueuePolicy::kBlocking));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_fork_plan() {
  // [source] -> [first sink]
  //     \----> [second sink]
  Graph graph{"inline-pod-fork"};
  auto source = graph.op<InlineFanoutSource>("source");
  auto first = graph.op<InlinePodSink>("first-sink", 2U);
  auto second = graph.op<InlinePodSink>("second-sink", 3U);
  graph.add_flow(source->output, first->input, queue_options(QueuePolicy::kBlocking));
  graph.add_flow(source->output, second->input, queue_options(QueuePolicy::kBlocking));
  graph.partition("inline-pod-fork-complete-cover").include_all();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_schema_chain_plan(std::size_t emission_limit = 1U,
                                                   std::chrono::nanoseconds period = 30s) {
  // [source] -> [first relay] -> [second relay] -> [sink]
  Graph graph{"inline-schema-chain"};
  auto source = graph.op<InlineSchemaSource>("source", emission_limit, period);
  auto first = graph.op<InlineSchemaRelay>("first-relay", 2U);
  auto second = graph.op<InlineSchemaRelay>("second-relay", 3U);
  auto sink = graph.op<InlineSchemaSink>("sink");
  graph.add_flow(source->output, first->input, queue_options(QueuePolicy::kDropOldest));
  graph.add_flow(first->output, second->input, queue_options(QueuePolicy::kDropOldest));
  graph.add_flow(second->output, sink->input, queue_options(QueuePolicy::kDropOldest));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  return compile(graph);
}

void expect_fast_chain_controller_evidence(const InlinePartitionRunEvidence& evidence) {
  ASSERT_TRUE(evidence.controller_thread.has_value());
  EXPECT_TRUE(std::ranges::all_of(evidence.compute_records, [&](const auto& record) {
    return record.thread == *evidence.controller_thread;
  }));
  EXPECT_TRUE(std::ranges::all_of(evidence.lifecycle_records, [&](const auto& record) {
    return record.thread == *evidence.controller_thread;
  }));
  expect_no_legacy_lane_handoffs(evidence.counters);
  EXPECT_GE(phase_count(evidence, detail::RuntimePhaseCounter::kLaneInlineOwnerInvocations),
            evidence.compute_records.size() + evidence.lifecycle_records.size());
}

void expect_param_coherence(const InlinePartitionRunEvidence& evidence, std::uint32_t operator_id,
                            std::uint32_t expected_value, bool inline_owner = true) {
  ASSERT_TRUE(evidence.controller_thread.has_value());
  const auto compute = std::ranges::find(
      evidence.param_compute_records, operator_id, &InlinePartitionParamComputeRecord::operator_id);
  ASSERT_NE(compute, evidence.param_compute_records.end());
  EXPECT_EQ(compute->value, expected_value);
  EXPECT_EQ(compute->generation, 0U);
  EXPECT_TRUE(compute->repeated_address_stable);
  EXPECT_TRUE(compute->repeated_generation_stable);
  // The fatal assertion above proves the optional is engaged for these test-only dereferences.
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  if (inline_owner) {
    EXPECT_EQ(compute->thread, *evidence.controller_thread);
  } else {
    EXPECT_NE(compute->thread, *evidence.controller_thread);
  }

  std::size_t lifecycle_count = 0U;
  for (const auto& lifecycle : evidence.param_lifecycle_records) {
    if (lifecycle.operator_id != operator_id) {
      continue;
    }
    ++lifecycle_count;
    EXPECT_EQ(lifecycle.value, expected_value);
    EXPECT_EQ(lifecycle.generation, compute->generation);
    EXPECT_EQ(lifecycle.address, compute->address);
    EXPECT_TRUE(lifecycle.repeated_address_stable);
    EXPECT_TRUE(lifecycle.repeated_generation_stable);
    if (inline_owner) {
      EXPECT_EQ(lifecycle.thread, *evidence.controller_thread);
    } else {
      EXPECT_NE(lifecycle.thread, *evidence.controller_thread);
    }
  }
  // NOLINTEND(bugprone-unchecked-optional-access)
  EXPECT_EQ(lifecycle_count, 2U);
}

class RuntimeMutableComputeRelease {
 public:
  RuntimeMutableComputeRelease() = default;
  RuntimeMutableComputeRelease(const RuntimeMutableComputeRelease&) = delete;
  RuntimeMutableComputeRelease& operator=(const RuntimeMutableComputeRelease&) = delete;

  ~RuntimeMutableComputeRelease() { release(); }

  void release() noexcept {
    if (!released_) {
      InlinePartitionProbeState::release_first_runtime_compute();
      released_ = true;
    }
  }

 private:
  bool released_{};
};

void expect_runtime_mutable_update_route(RuntimeMutablePlanRoute route, bool expect_inline_owner) {
  InlinePartitionProbeState::reset();
  ExecutionPlan compiled = make_runtime_mutable_param_chain_plan(route);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_EQ(compiled.parameter_fields().size(), 1U);
  const ParameterFieldPlanRecord field = compiled.parameter_fields().front();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  if (route != RuntimeMutablePlanRoute::kHoloMQ) {
    EXPECT_EQ(state->inline_partition_execution.has_value(),
              route == RuntimeMutablePlanRoute::kInline)
        << compiled.json();
  }
  ASSERT_EQ(state->partition_runners.size(), 1U);
  ASSERT_TRUE(std::ranges::any_of(state->operator_parameters, [](const auto& parameters) {
    return parameters.runtime_updates_enabled;
  }));

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<InlinePartitionControllerProbe>();
  compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  compiled = detail::RuntimeTestPlanAccess::with_support_thread_control(compiled, control);

  RunSession session = run_async(compiled);
  RuntimeMutableComputeRelease release;
  ASSERT_TRUE(InlinePartitionProbeState::wait_for_first_runtime_compute());
  const auto run_owner = session.operator_ref(field.owner);
  ASSERT_TRUE(run_owner);
  const ParamAccessPolicy policy{
      .origin = ParamControlOrigin::kLocalApplication,
      .permissions = ParamPermission::kRead | ParamPermission::kUpdate | ParamPermission::kAwait,
  };
  auto handle_result = session.parameter_control(policy);
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = std::move(*handle_result);
  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = *run_owner, .field = field.field},
          .value = ParamValue::from(std::uint32_t{43U}),
      },
  };
  const ParamBatchUpdateResult accepted =
      handle.submit(patch, ParamUpdateOptions{.effective = NextCompute{}});
  ASSERT_TRUE(accepted.committed);
  EXPECT_EQ(accepted.status, ParamUpdateStatus::kAccepted);

  release.release();
  const bool observed = InlinePartitionProbeState::wait_for_sinks(2U);
  session.request_stop();
  session.wait();

  const InlinePartitionRunEvidence evidence{
      .observed = observed,
      .termination = session.termination(),
      .diagnostic_count = session.diagnostics().size(),
      .controller_thread = control->controller_thread(),
      .compute_records = InlinePartitionProbeState::compute_records(),
      .lifecycle_records = InlinePartitionProbeState::lifecycle_records(),
      .param_compute_records = InlinePartitionProbeState::param_compute_records(),
      .param_lifecycle_records = InlinePartitionProbeState::param_lifecycle_records(),
      .temporal_predicate_records = InlinePartitionProbeState::temporal_predicate_records(),
      .sink_values = InlinePartitionProbeState::sink_values(),
      .schema_value = InlinePartitionProbeState::schema_value(),
      .counters = counters->snapshot(),
  };
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{44U, 46U}));
  ASSERT_EQ(evidence.param_compute_records.size(), 2U);
  const auto& initial = evidence.param_compute_records[0];
  EXPECT_EQ(initial.operator_id, 1U);
  EXPECT_EQ(initial.value, 41U);
  EXPECT_EQ(initial.generation, 0U);
  EXPECT_TRUE(initial.repeated_address_stable);
  EXPECT_TRUE(initial.repeated_generation_stable);
  const auto& updated = evidence.param_compute_records[1];
  EXPECT_EQ(updated.operator_id, 1U);
  EXPECT_EQ(updated.value, 43U);
  EXPECT_EQ(updated.generation, accepted.target_generation);
  EXPECT_TRUE(updated.repeated_address_stable);
  EXPECT_TRUE(updated.repeated_generation_stable);
  const auto observed_update = handle.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(observed_update);
  EXPECT_EQ(observed_update->state, ParamUpdateState::kObserved);
  EXPECT_EQ(observed_update->target_generation, accepted.target_generation);

  if (expect_inline_owner) {
    ASSERT_TRUE(evidence.controller_thread.has_value());
    // The fatal assertion above proves the optional is engaged for these test-only dereferences.
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    expect_inline_thread_evidence(evidence, 6U, 3U);
    EXPECT_EQ(initial.thread, *evidence.controller_thread);
    EXPECT_EQ(updated.thread, *evidence.controller_thread);
    // NOLINTEND(bugprone-unchecked-optional-access)
  }
}

class InlinePodChainTest : public testing::TestWithParam<QueuePolicy> {};

// REQ: RUNTIME-041
TEST_P(InlinePodChainTest, RunsComputeAndLifecycleCallbacksOnTheControllerWithoutHandoffs) {
  constexpr std::size_t kFrameCount = 3U;
  InlinePartitionProbeState::reset();
  // Key setup: run three frames through a four-Operator direct POD chain whose 1 ns source remains
  // ready while every edge is eligible for causal continuation.
  const ExecutionPlan compiled = make_pod_chain_plan(GetParam(), false, kFrameCount, 1ns);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, kFrameCount);
  // Key assertion: each frame drains in topological order at one logical time and successive
  // microsteps, with one accepted causal frontier per edge and no fallback.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  ASSERT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{44U, 44U, 44U}));
  ASSERT_EQ(evidence.compute_records.size(), kFrameCount * 4U);
  constexpr std::array<std::uint32_t, 4U> kStageValues{41U, 42U, 44U, 44U};
  for (std::size_t frame = 0U; frame < kFrameCount; ++frame) {
    const std::size_t source_index = frame * kStageValues.size();
    const TagTime source_tag = evidence.compute_records[source_index].tag_time;
    ASSERT_TRUE(source_tag.valid());
    ASSERT_LE(source_tag.microstep, std::numeric_limits<std::uint32_t>::max() - 3U);
    for (std::size_t stage = 0U; stage < kStageValues.size(); ++stage) {
      const auto& record = evidence.compute_records[source_index + stage];
      SCOPED_TRACE(frame);
      SCOPED_TRACE(stage);
      EXPECT_EQ(record.operator_id, stage + 1U);
      EXPECT_EQ(record.value, kStageValues[stage]);
      EXPECT_EQ(record.tag_time.logical_ns, source_tag.logical_ns);
      EXPECT_EQ(record.tag_time.microstep,
                source_tag.microstep + static_cast<std::uint32_t>(stage));
    }
  }
  expect_fast_chain_controller_evidence(evidence);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
            kFrameCount * 3U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
            kFrameCount * 3U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks),
            0U);
}

// Once the causal path destructively accepts a staged receipt, freeze failure is terminal. The
// controller must retire the sole pending coordinator credit instead of attempting an ordinary
// readiness poll that could neither reconstruct nor safely replay the consumed receipt.
// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution,
     CausalFrontierFreezeFailureIsTerminalAndRetiresCoordinatorCredit) {
  // Key setup: force causal-frontier freezing to fail after the source receipt is consumed.
  InlinePartitionProbeState::reset();
  ExecutionPlan plan = make_pod_chain_plan(QueuePolicy::kBlocking);
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << plan.json();

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<detail::RuntimeDirectContinuationTestControl>();
  control->force_causal_freeze_failure();
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  plan = detail::RuntimeTestPlanAccess::with_direct_continuation_test_control(plan, control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool cleanup_recorded = control->wait_until_causal_freeze_failure_cleanup();
  if (!cleanup_recorded) {
    session.request_stop();
  }
  session.wait();

  // Key assertion: the run fails without polling fallback and retires its sole coordinator credit.
  ASSERT_TRUE(cleanup_recorded);
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
  EXPECT_EQ(session.diagnostics().front().service, "partition_causal_frontier_freeze");
  const Diagnostic& diagnostic = session.diagnostics().front();
  const auto vertices = plan.graph().vertices();
  const auto successor = std::ranges::find(vertices, "first-relay", &VertexRecord::name);
  ASSERT_NE(successor, vertices.end());
  EXPECT_EQ(diagnostic.source.vertex, successor->id);
  if (!diagnostic.detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  const auto& detail = *diagnostic.detail;
  EXPECT_EQ(detail.error.code, ErrorCode::kProtocolError);
  EXPECT_TRUE(detail.error.message.empty());
  EXPECT_TRUE(detail.causes.empty());
  EXPECT_FALSE(detail.truncated);
  EXPECT_TRUE(InlinePartitionProbeState::sink_values().empty());
  const auto records = InlinePartitionProbeState::compute_records();
  ASSERT_EQ(records.size(), 1U);
  EXPECT_EQ(records.front().operator_id, 1U);

  const auto cleanup = control->causal_freeze_failure_observation();
  ASSERT_TRUE(cleanup.recorded);
  EXPECT_EQ(cleanup.pending_before_cancel, 1U);
  EXPECT_EQ(cleanup.outstanding_before_cancel, 1U);
  EXPECT_EQ(cleanup.pending_after_cancel, 0U);
  EXPECT_EQ(cleanup.outstanding_after_cancel, 0U);
  EXPECT_FALSE(cleanup.receipt_present_after_cancel);
  EXPECT_TRUE(cleanup.retired_all);

  const auto snapshot = counters->snapshot();
  const auto count = [&](const detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelEmitAccepted), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelSamplesReceived), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectContinuationPollElisions), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kActivationsCompleted), 1U);
}

INSTANTIATE_TEST_SUITE_P(QueuePolicies, InlinePodChainTest,
                         testing::Values(QueuePolicy::kBlocking, QueuePolicy::kDropOldest),
                         [](const testing::TestParamInfo<QueuePolicy>& info) {
                           return info.param == QueuePolicy::kBlocking ? "Blocking" : "DropOldest";
                         });

TEST(RuntimeInlinePartitionExecution,
     CachedClockRootCoordinateRefreshesAcrossConsecutiveCausalChains) {
  // Key setup: run three clock activations through the same cached four-Operator POD chain.
  InlinePartitionProbeState::reset();
  constexpr std::size_t kActivationCount = 3U;
  constexpr std::size_t kOperatorsPerActivation = 4U;
  const ExecutionPlan compiled =
      make_pod_chain_plan(QueuePolicy::kBlocking, false, kActivationCount, 1ms);
  ASSERT_TRUE(compiled.ok()) << compiled.json();

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, kActivationCount);
  // Key assertion: each chain has a newer root coordinate and internally consecutive microsteps.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values,
            (std::vector<std::uint32_t>(kActivationCount, std::uint32_t{44U})));
  ASSERT_EQ(evidence.compute_records.size(), kActivationCount * kOperatorsPerActivation);

  std::optional<std::int64_t> previous_root_logical_ns;
  for (std::size_t activation = 0U; activation < kActivationCount; ++activation) {
    const std::size_t begin = activation * kOperatorsPerActivation;
    const TagTime root_tag_time = evidence.compute_records[begin].tag_time;
    ASSERT_TRUE(root_tag_time.valid());
    if (previous_root_logical_ns) {
      EXPECT_GT(root_tag_time.logical_ns, *previous_root_logical_ns);
    }
    previous_root_logical_ns = root_tag_time.logical_ns;
    for (std::size_t member = 0U; member < kOperatorsPerActivation; ++member) {
      SCOPED_TRACE(activation);
      SCOPED_TRACE(member);
      const auto& record = evidence.compute_records[begin + member];
      EXPECT_EQ(record.operator_id, static_cast<std::uint32_t>(member + 1U));
      EXPECT_EQ(record.tag_time.logical_ns, root_tag_time.logical_ns);
      EXPECT_EQ(record.tag_time.microstep,
                root_tag_time.microstep + static_cast<std::uint32_t>(member));
    }
  }
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
            9U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
            9U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks),
            0U);
}

// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution, RunsAnOwningSchemaChainOnTheControllerWithoutHandoffs) {
  constexpr std::size_t kFrameCount = 3U;
  InlinePartitionProbeState::reset();
  // Key setup: drive three frames through a four-Operator generated-schema chain from a 1 ns
  // source, providing the established non-Tensor continuation behavior for comparison.
  const ExecutionPlan compiled = make_schema_chain_plan(kFrameCount, 1ns);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, kFrameCount);
  // Key assertion: every generated-schema frame drains at successive microsteps and all three
  // direct edges accept their causal continuations without fallback.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{0U, 1U, 2U}));
  ASSERT_TRUE(evidence.schema_value.has_value());
  // The fatal assertion above proves the optional is engaged.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(*evidence.schema_value, "root:2:3");
  ASSERT_EQ(evidence.compute_records.size(), kFrameCount * 4U);
  for (std::size_t frame = 0U; frame < kFrameCount; ++frame) {
    const std::size_t source_index = frame * 4U;
    const TagTime source_tag = evidence.compute_records[source_index].tag_time;
    ASSERT_TRUE(source_tag.valid());
    ASSERT_LE(source_tag.microstep, std::numeric_limits<std::uint32_t>::max() - 3U);
    for (std::size_t stage = 0U; stage < 4U; ++stage) {
      const auto& record = evidence.compute_records[source_index + stage];
      SCOPED_TRACE(frame);
      SCOPED_TRACE(stage);
      EXPECT_EQ(record.operator_id, stage + 1U);
      EXPECT_EQ(record.value, frame);
      EXPECT_EQ(record.tag_time.logical_ns, source_tag.logical_ns);
      EXPECT_EQ(record.tag_time.microstep,
                source_tag.microstep + static_cast<std::uint32_t>(stage));
    }
  }
  expect_fast_chain_controller_evidence(evidence);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted),
            kFrameCount * 3U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
            kFrameCount * 3U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
            kFrameCount * 3U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks),
            0U);
}

TEST(RuntimeInlinePartitionExecution, AdmitsOneRootWithTwoConsumersInStableSiblingOrder) {
  // Key setup: connect one direct POD output to two sinks in a single authored partition.
  InlinePartitionProbeState::reset();
  const ExecutionPlan compiled = make_fork_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), 1U) << compiled.json();
  EXPECT_FALSE(state->inline_partition_execution.has_value());
  ASSERT_EQ(state->partition_runners.front().operators.size(), 3U);
  EXPECT_EQ(state->partition_runners.front().internal_edges.size(), 2U);

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, 2U);
  // Key assertion: the ordinary partition runner executes both siblings once in stable order.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{101U, 101U}));
  ASSERT_EQ(evidence.compute_records.size(), 3U);
  EXPECT_EQ(evidence.compute_records[0].operator_id, 1U);
  EXPECT_EQ(evidence.compute_records[1].operator_id, 2U);
  EXPECT_EQ(evidence.compute_records[2].operator_id, 3U);
  EXPECT_TRUE(std::ranges::all_of(evidence.compute_records, [&](const auto& record) {
    return record.thread == evidence.compute_records.front().thread;
  }));
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
  EXPECT_GE(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionRunnerSweeps), 1U);
  EXPECT_GE(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionRunnerWakeSignals), 1U);
  const std::uint64_t tail_attempts =
      phase_count(evidence, detail::RuntimePhaseCounter::kPartitionTailDispatchAttempts);
  EXPECT_GT(tail_attempts, 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionTailDispatchAccepted), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionTailDispatchFallbacks),
            tail_attempts);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
            0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
            0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks),
            0U);
  EXPECT_GE(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionControlBarriers), 1U);
}

TEST(RuntimeInlinePartitionExecution,
     FrozenTypedParamsRemainCoherentAcrossRepeatedInlineRunsWithoutHandoffs) {
  // Key setup: reuse immutable typed-parameter source and relay chains across two inline runs.
  for (std::size_t iteration = 0U; iteration < 2U; ++iteration) {
    SCOPED_TRACE(iteration);
    InlinePartitionProbeState::reset();
    const ExecutionPlan compiled = make_param_chain_plan();
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    ASSERT_TRUE(compiled.parameter_runtime_capacity().has_value());
    // The fatal assertion above proves the optional is engaged.
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    EXPECT_FALSE(compiled.parameter_runtime_capacity()->runtime_updates_enabled);
    EXPECT_FALSE(compiled.parameter_runtime_capacity()->lifecycle_updates_enabled);
    // NOLINTEND(bugprone-unchecked-optional-access)
    EXPECT_EQ(compiled.parameter_fields().size(), 2U);
    EXPECT_EQ(std::ranges::count(
                  compiled.operator_parameters(), false, &OperatorParameterPlanRecord::no_params),
              2);
    const auto state = detail::ExecutionPlanAccess::retain(compiled);
    ASSERT_TRUE(state);
    ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

    const InlinePartitionRunEvidence evidence = run_and_observe(compiled, 1U);
    // Key assertion: callback and lifecycle views share frozen values on the inline owner.
    ASSERT_TRUE(evidence.observed);
    EXPECT_EQ(evidence.termination, RunTermination::kStopped);
    EXPECT_EQ(evidence.diagnostic_count, 0U);
    EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{44U}));
    ASSERT_EQ(evidence.param_compute_records.size(), 2U);
    ASSERT_EQ(evidence.param_lifecycle_records.size(), 4U);
    expect_param_coherence(evidence, 1U, 41U);
    expect_param_coherence(evidence, 2U, 3U);
    expect_inline_thread_evidence(evidence, 3U, 3U);
  }
}

TEST(RuntimeInlinePartitionExecution,
     ExplicitSinglePartitionKeepsFrozenParamsCoherentOnItsInlineOwner) {
  // Key setup: explicitly place the immutable parameter chain in one authored partition.
  InlinePartitionProbeState::reset();
  const ExecutionPlan compiled = make_param_chain_plan(true);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_FALSE(state->inline_partition_execution.has_value());
  ASSERT_EQ(state->partition_runners.size(), 1U);

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, 1U);
  // Key assertion: frozen parameter views remain coherent on the authored partition owner.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{44U}));
  ASSERT_TRUE(evidence.controller_thread.has_value());
  ASSERT_EQ(evidence.param_compute_records.size(), 2U);
  ASSERT_EQ(evidence.param_lifecycle_records.size(), 4U);
  expect_param_coherence(evidence, 1U, 41U);
  expect_param_coherence(evidence, 2U, 3U);
  expect_inline_thread_evidence(evidence, 3U, 3U);
}

TEST(RuntimeInlinePartitionExecution,
     RuntimeMutableNextComputePublishesPinsAndObservesOnTheController) {
  // Key setup: submit a NextCompute parameter update while the implicit inline source is blocked.
  // Key assertion: the shared route helper verifies pin publication and observation on controller.
  expect_runtime_mutable_update_route(RuntimeMutablePlanRoute::kInline, true);
}

TEST(RuntimeInlinePartitionExecution,
     RuntimeMutableNextComputeMatchesExplicitPartitionAndHoloMQControls) {
  // Key setup: submit equivalent NextCompute updates through explicit-partition and HoloMQ routes.
  // Key assertion: both control routes publish and observe the same pinned parameter generation.
  expect_runtime_mutable_update_route(RuntimeMutablePlanRoute::kExplicitPlacement, true);
  expect_runtime_mutable_update_route(RuntimeMutablePlanRoute::kHoloMQ, false);
}

TEST(RuntimeInlinePartitionExecution, UpdatedSelectorDeadlineAndCallbackShareOnePinnedGeneration) {
  // Key setup: update value, LatestWithin age, and deadline while temporal source compute is held.
  InlinePartitionProbeState::reset();
  ExecutionPlan compiled = make_temporal_param_chain_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_EQ(state->partition_runners.size(), 1U) << compiled.json();
  ASSERT_EQ(compiled.parameter_fields().size(), 3U);
  const auto value_field = std::ranges::find(
      compiled.parameter_fields(), "value", &ParameterFieldPlanRecord::canonical_name);
  const auto max_age_field = std::ranges::find(
      compiled.parameter_fields(), "max_age", &ParameterFieldPlanRecord::canonical_name);
  const auto deadline_field = std::ranges::find(
      compiled.parameter_fields(), "deadline", &ParameterFieldPlanRecord::canonical_name);
  ASSERT_NE(value_field, compiled.parameter_fields().end());
  ASSERT_NE(max_age_field, compiled.parameter_fields().end());
  ASSERT_NE(deadline_field, compiled.parameter_fields().end());
  ASSERT_EQ(compiled.parameter_derived_terms().size(), 2U);
  EXPECT_TRUE(std::ranges::any_of(compiled.parameter_derived_terms(), [](const auto& term) {
    return term.kind == ParameterDerivedTermKind::kLatestWithinMaxAge;
  }));
  EXPECT_TRUE(std::ranges::any_of(compiled.parameter_derived_terms(), [](const auto& term) {
    return term.kind == ParameterDerivedTermKind::kContractMaxStartDelay;
  }));

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<InlinePartitionControllerProbe>();
  compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  compiled = detail::RuntimeTestPlanAccess::with_support_thread_control(compiled, control);
  RunSession session = run_async(compiled);
  RuntimeMutableComputeRelease release;
  ASSERT_TRUE(InlinePartitionProbeState::wait_for_first_runtime_compute());
  const auto run_owner = session.operator_ref(value_field->owner);
  ASSERT_TRUE(run_owner);
  const ParamAccessPolicy policy{
      .origin = ParamControlOrigin::kLocalApplication,
      .permissions = ParamPermission::kRead | ParamPermission::kUpdate | ParamPermission::kAwait,
  };
  auto handle_result = session.parameter_control(policy);
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = std::move(*handle_result);
  const ParamAddress value_address{
      .operator_instance = *run_owner,
      .field = value_field->field,
  };
  const std::array patch{
      ParamPatch{
          .target = value_address,
          .value = ParamValue::from(std::uint32_t{100U}),
      },
      ParamPatch{
          .target = ParamAddress{.operator_instance = *run_owner, .field = max_age_field->field},
          .value = ParamValue::from(3s),
      },
      ParamPatch{
          .target = ParamAddress{.operator_instance = *run_owner, .field = deadline_field->field},
          .value = ParamValue::from(2s),
      },
  };
  const ParamBatchUpdateResult accepted =
      handle.submit(patch, ParamUpdateOptions{.effective = NextCompute{}});
  ASSERT_TRUE(accepted.committed);
  const auto pending = handle.read(value_address, ParamReadView::kLatestAccepted);
  ASSERT_TRUE(pending);
  EXPECT_EQ(pending->generation, accepted.target_generation);
  EXPECT_EQ(pending->update_sequence, accepted.update_sequence);
  const auto pending_value = pending->value.as<std::uint32_t>();
  ASSERT_TRUE(pending_value);
  EXPECT_EQ(*pending_value, 100U);
  release.release();
  const bool observed = InlinePartitionProbeState::wait_for_sinks(1U);
  session.request_stop();
  session.wait();

  const InlinePartitionRunEvidence evidence{
      .observed = observed,
      .termination = session.termination(),
      .diagnostic_count = session.diagnostics().size(),
      .controller_thread = control->controller_thread(),
      .compute_records = InlinePartitionProbeState::compute_records(),
      .lifecycle_records = InlinePartitionProbeState::lifecycle_records(),
      .param_compute_records = InlinePartitionProbeState::param_compute_records(),
      .param_lifecycle_records = InlinePartitionProbeState::param_lifecycle_records(),
      .temporal_predicate_records = InlinePartitionProbeState::temporal_predicate_records(),
      .sink_values = InlinePartitionProbeState::sink_values(),
      .schema_value = InlinePartitionProbeState::schema_value(),
      .counters = counters->snapshot(),
  };
  // Key assertion: predicate, deadline, and callback observe one newly pinned generation.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{203U}));
  ASSERT_EQ(evidence.temporal_predicate_records.size(), 1U);
  EXPECT_EQ(evidence.temporal_predicate_records[0].threshold, 200U);
  EXPECT_EQ(evidence.temporal_predicate_records[0].driver, 101U);
  EXPECT_EQ(evidence.temporal_predicate_records[0].side, 102U);
  ASSERT_EQ(evidence.param_compute_records.size(), 1U);
  EXPECT_EQ(evidence.param_compute_records[0].value, 100U);
  EXPECT_EQ(evidence.param_compute_records[0].generation, accepted.target_generation);
  EXPECT_EQ(evidence.param_compute_records[0].activation_reason, ActivationReason::kNominal);
  EXPECT_EQ(evidence.param_compute_records[0].activation_policy, OnViolation::kEmitWitness);
  ASSERT_TRUE(evidence.controller_thread.has_value());
  // The fatal assertion above proves the optional is engaged.
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  EXPECT_EQ(evidence.temporal_predicate_records[0].thread, *evidence.controller_thread);
  EXPECT_EQ(evidence.param_compute_records[0].thread, *evidence.controller_thread);
  // NOLINTEND(bugprone-unchecked-optional-access)
  expect_inline_thread_evidence(evidence, 3U, 3U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
            0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
            0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks),
            0U);
  const auto observed_update = handle.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(observed_update);
  EXPECT_EQ(observed_update->state, ParamUpdateState::kObserved);
}

TEST(RuntimeInlinePartitionExecution, SelfRestartRunsAtTheCallbackBoundaryWithoutWorkerHandoffs) {
  // Key setup: request a source self-restart before its first POD emission in an inline chain.
  InlinePartitionProbeState::reset();
  const ExecutionPlan compiled = make_restart_chain_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();
  ASSERT_FALSE(compiled.lifecycle_restart_targets().empty());

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, 1U);
  // Key assertion: restart lifecycle stages complete at the boundary before the chain drains.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_EQ(evidence.sink_values, (std::vector<std::uint32_t>{42U}));
  ASSERT_EQ(evidence.compute_records.size(), 4U);
  EXPECT_EQ(evidence.compute_records[0].operator_id, 1U);
  EXPECT_EQ(evidence.compute_records[1].operator_id, 1U);
  EXPECT_EQ(evidence.compute_records[2].operator_id, 2U);
  EXPECT_EQ(evidence.compute_records[3].operator_id, 3U);

  std::vector<LifecycleStage> source_stages;
  for (const auto& record : evidence.lifecycle_records) {
    if (record.operator_id == 1U) {
      source_stages.push_back(record.stage);
    }
  }
  EXPECT_EQ(source_stages,
            (std::vector{LifecycleStage::kStart,
                         LifecycleStage::kStop,
                         LifecycleStage::kStart,
                         LifecycleStage::kStop}));
  expect_inline_thread_evidence(evidence, 4U, 3U, 8U);
}

TEST(RuntimeInlinePartitionExecution,
     StopDuringCallbackCancelsPendingGenerationAndFencesSuccessors) {
  // Key setup: queue a NextCompute parameter generation while the current source callback is held.
  InlinePartitionProbeState::reset();
  ExecutionPlan compiled = make_runtime_mutable_param_chain_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();
  ASSERT_EQ(compiled.parameter_fields().size(), 1U);
  const ParameterFieldPlanRecord field = compiled.parameter_fields().front();

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<InlinePartitionControllerProbe>();
  compiled = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  compiled = detail::RuntimeTestPlanAccess::with_support_thread_control(compiled, control);

  RunSession session = run_async(compiled);
  RuntimeMutableComputeRelease release;
  ASSERT_TRUE(InlinePartitionProbeState::wait_for_first_runtime_compute());
  const auto run_owner = session.operator_ref(field.owner);
  ASSERT_TRUE(run_owner);
  const ParamAccessPolicy policy{
      .origin = ParamControlOrigin::kLocalApplication,
      .permissions = ParamPermission::kRead | ParamPermission::kUpdate | ParamPermission::kAwait,
  };
  auto handle_result = session.parameter_control(policy);
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = std::move(*handle_result);
  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = *run_owner, .field = field.field},
          .value = ParamValue::from(std::uint32_t{43U}),
      },
  };
  const ParamBatchUpdateResult accepted =
      handle.submit(patch, ParamUpdateOptions{.effective = NextCompute{}});
  ASSERT_TRUE(accepted.committed);

  session.request_stop();
  release.release();
  session.wait();
  const InlinePartitionRunEvidence evidence =
      capture_run_evidence(session, true, counters, control);

  // Key assertion: stop cancels the pending generation and prevents relay and sink activation.
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  EXPECT_TRUE(evidence.sink_values.empty());
  ASSERT_EQ(evidence.compute_records.size(), 1U);
  EXPECT_EQ(evidence.compute_records.front().operator_id, 1U);
  ASSERT_EQ(evidence.param_compute_records.size(), 1U);
  EXPECT_EQ(evidence.param_compute_records.front().value, 41U);
  EXPECT_EQ(evidence.param_compute_records.front().generation, 0U);
  const auto terminal = handle.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(terminal);
  EXPECT_EQ(terminal->status, ParamUpdateStatus::kClosed);
  EXPECT_EQ(terminal->state, ParamUpdateState::kCanceled);
  EXPECT_EQ(terminal->target_generation, accepted.target_generation);
  expect_inline_thread_evidence(evidence, 1U, 3U);
}

TEST(RuntimeInlinePartitionExecution, CallbackFailureReleasesInlineStateAndReplaysTheTypedFailure) {
  // Key setup: reuse an inline POD plan whose relay callback fails for two consecutive runs.
  const ExecutionPlan compiled = make_failing_chain_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->inline_partition_execution.has_value()) << compiled.json();

  for (std::size_t iteration = 0U; iteration < 2U; ++iteration) {
    SCOPED_TRACE(iteration);
    InlinePartitionProbeState::reset();
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    const auto control = std::make_shared<InlinePartitionControllerProbe>();
    ExecutionPlan run_plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
    run_plan = detail::RuntimeTestPlanAccess::with_support_thread_control(run_plan, control);
    RunSession session = run_async(run_plan);
    session.wait();
    const InlinePartitionRunEvidence evidence =
        capture_run_evidence(session, true, counters, control);

    // Key assertion: each run reports the same typed failure and releases inline callback state.
    EXPECT_EQ(evidence.termination, RunTermination::kFailed);
    ASSERT_EQ(session.diagnostics().size(), 1U);
    EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
    EXPECT_TRUE(evidence.sink_values.empty());
    ASSERT_EQ(evidence.compute_records.size(), 2U);
    EXPECT_EQ(evidence.compute_records[0].operator_id, 1U);
    EXPECT_EQ(evidence.compute_records[1].operator_id, 2U);
    expect_inline_thread_evidence(evidence, 2U, 3U);
  }
}

TEST(RuntimeInlinePartitionExecution, LatestWithinSelectionUsesTheInlinePartitionPlan) {
  // Key setup: compile a two-input POD graph with a LatestWithin side selection.
  const ExecutionPlan compiled = make_latest_within_chain_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  // Key assertion: the selection remains represented within a single partition-runner plan.
  ASSERT_TRUE(std::ranges::any_of(state->input_selections, [](const auto& selection) {
    return selection.selector == InputSelectorKind::kLatestWithin;
  }));
  EXPECT_EQ(state->partition_runners.size(), 1U);
}

TEST(RuntimeInlinePartitionExecution, ExplicitSinglePartitionPlacementUsesTheControllerOwner) {
  // Key setup: explicitly place the four-Operator POD chain into one authored partition.
  InlinePartitionProbeState::reset();
  const ExecutionPlan compiled = make_pod_chain_plan(QueuePolicy::kBlocking, true);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_TRUE(state);
  ASSERT_FALSE(state->inline_partition_execution.has_value());
  ASSERT_EQ(state->partition_runners.size(), 1U);

  const InlinePartitionRunEvidence evidence = run_and_observe(compiled, 1U);
  // Key assertion: all callbacks execute on the controller-owned partition without lane dispatch.
  ASSERT_TRUE(evidence.observed);
  EXPECT_EQ(evidence.termination, RunTermination::kStopped);
  EXPECT_EQ(evidence.diagnostic_count, 0U);
  ASSERT_TRUE(evidence.controller_thread.has_value());
  ASSERT_EQ(evidence.compute_records.size(), 4U);
  EXPECT_TRUE(std::ranges::all_of(evidence.compute_records, [&](const auto& record) {
    return record.thread == *evidence.controller_thread;
  }));
  EXPECT_GT(phase_count(evidence, detail::RuntimePhaseCounter::kLaneInlineOwnerInvocations), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneCompletionsObserved), 0U);
}

}  // namespace
}  // namespace holoscan
