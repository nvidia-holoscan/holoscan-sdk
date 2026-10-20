/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/run.hpp>

#include "compiler/execution_plan_state.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_support_thread_control.hpp"

// Inline partition execution fixtures shared by the POD/schema and Tensor payload tests.
namespace holoscan {

struct InlinePartitionPod {
  std::uint32_t value{};
};

template <>
struct ChannelTypeIdentity<InlinePartitionPod> {
  static constexpr std::string_view name = "holoscan.test.InlinePartitionPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<InlinePartitionPod> {
  static constexpr auto fields = pod_fields(
      pod_field("value", &InlinePartitionPod::value, offsetof(InlinePartitionPod, value)));
};

namespace test::inline_execution {

struct InlinePartitionComputeRecord {
  std::uint32_t operator_id{};
  std::uint32_t value{};
  TagTime tag_time{};
  std::thread::id thread;
};

struct InlinePartitionLifecycleRecord {
  std::uint32_t operator_id{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  std::thread::id thread;
};

struct InlinePartitionParamComputeRecord {
  std::uint32_t operator_id{};
  std::uint32_t value{};
  ParamGeneration generation{};
  const void* address{};
  bool repeated_address_stable{};
  bool repeated_generation_stable{};
  ActivationReason activation_reason{ActivationReason::kNominal};
  OnViolation activation_policy{OnViolation::kEmitWitness};
  std::thread::id thread;
};

struct InlinePartitionParamLifecycleRecord {
  std::uint32_t operator_id{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  std::uint32_t value{};
  ParamGeneration generation{};
  const void* address{};
  bool repeated_address_stable{};
  bool repeated_generation_stable{};
  std::thread::id thread;
};

struct InlineTemporalPredicateRecord {
  std::uint32_t threshold{};
  std::uint32_t driver{};
  std::uint32_t side{};
  std::thread::id thread;
};

struct InlineTensorObservation {
  MemoryKind memory_kind{MemoryKind::kHost};
  const void* source_pointer{};
  const void* relay_pointer{};
  cudaStream_t source_stream{};
  cudaStream_t relay_stream{};
  cudaStream_t readiness_stream{};
  std::int32_t source_device{-1};
  std::int32_t relay_device{-1};
  std::int32_t stop_device{-1};
  std::uint8_t value{};
  bool source_recorded{};
  bool relay_recorded{};
  bool stop_device_recorded{};
};

class InlineTensorProbeState {
 public:
  static void reset(MemoryKind memory_kind) {
    const std::lock_guard lock(mutex_);
    observation_ = InlineTensorObservation{.memory_kind = memory_kind};
    retained_tensor_.reset();
  }

  static void record_source(const void* pointer, cudaStream_t stream, std::int32_t device) {
    const std::lock_guard lock(mutex_);
    observation_.source_pointer = pointer;
    observation_.source_stream = stream;
    observation_.source_device = device;
    observation_.source_recorded = true;
  }

  static void record_relay(const void* pointer, cudaStream_t stream, cudaStream_t readiness_stream,
                           std::int32_t device, std::uint8_t value) {
    const std::lock_guard lock(mutex_);
    observation_.relay_pointer = pointer;
    observation_.relay_stream = stream;
    observation_.readiness_stream = readiness_stream;
    observation_.relay_device = device;
    observation_.value = value;
    observation_.relay_recorded = true;
  }

  static void retain_tensor(const Tensor& tensor) {
    const std::lock_guard lock(mutex_);
    retained_tensor_ = tensor;
  }

  [[nodiscard]] static bool has_retained_tensor() {
    const std::lock_guard lock(mutex_);
    return retained_tensor_.has_value();
  }

  static void release_retained_tensor() {
    const std::lock_guard lock(mutex_);
    retained_tensor_.reset();
  }

  static void record_stop_device(std::int32_t device) {
    const std::lock_guard lock(mutex_);
    observation_.stop_device = device;
    observation_.stop_device_recorded = true;
  }

  [[nodiscard]] static InlineTensorObservation observation() {
    const std::lock_guard lock(mutex_);
    return observation_;
  }

 private:
  static inline std::mutex mutex_;
  static inline InlineTensorObservation observation_;
  static inline std::optional<Tensor> retained_tensor_;
};

class InlinePartitionProbeState {
 public:
  static void reset() {
    const std::lock_guard lock(mutex_);
    compute_records_.clear();
    lifecycle_records_.clear();
    param_compute_records_.clear();
    param_lifecycle_records_.clear();
    temporal_predicate_records_.clear();
    sink_values_.clear();
    schema_value_.reset();
    first_runtime_compute_entered_ = false;
    release_first_runtime_compute_ = false;
  }

  static void record_compute(std::uint32_t operator_id, std::uint32_t value,
                             TagTime tag_time = {}) {
    const std::lock_guard lock(mutex_);
    compute_records_.push_back(InlinePartitionComputeRecord{
        .operator_id = operator_id,
        .value = value,
        .tag_time = tag_time,
        .thread = std::this_thread::get_id(),
    });
  }

  static void record_lifecycle(std::uint32_t operator_id, LifecycleStage stage) {
    const std::lock_guard lock(mutex_);
    lifecycle_records_.push_back(InlinePartitionLifecycleRecord{
        .operator_id = operator_id,
        .stage = stage,
        .thread = std::this_thread::get_id(),
    });
  }

  template <typename Params>
  static void record_param_compute(std::uint32_t operator_id, ExecutionContext& context) {
    const auto* const first = &context.params<Params>();
    const ParamGeneration first_generation = context.params_generation();
    const auto* const second = &context.params<Params>();
    const std::lock_guard lock(mutex_);
    param_compute_records_.push_back(InlinePartitionParamComputeRecord{
        .operator_id = operator_id,
        .value = first->value,
        .generation = first_generation,
        .address = first,
        .repeated_address_stable = first == second,
        .repeated_generation_stable = first_generation == context.params_generation(),
        .activation_reason = context.activation().reason,
        .activation_policy = context.activation().policy,
        .thread = std::this_thread::get_id(),
    });
    cv_.notify_all();
  }

  template <typename Params>
  [[nodiscard]] static bool record_runtime_param_compute(std::uint32_t operator_id,
                                                         ExecutionContext& context, bool hold) {
    const auto* const first = &context.params<Params>();
    const ParamGeneration first_generation = context.params_generation();
    if (hold) {
      std::unique_lock lock(mutex_);
      first_runtime_compute_entered_ = true;
      cv_.notify_all();
      if (!cv_.wait_for(
              lock, std::chrono::seconds{5}, [] { return release_first_runtime_compute_; })) {
        return false;
      }
    }
    const auto* const second = &context.params<Params>();
    const std::lock_guard lock(mutex_);
    param_compute_records_.push_back(InlinePartitionParamComputeRecord{
        .operator_id = operator_id,
        .value = first->value,
        .generation = first_generation,
        .address = first,
        .repeated_address_stable = first == second,
        .repeated_generation_stable = first_generation == context.params_generation(),
        .activation_reason = context.activation().reason,
        .activation_policy = context.activation().policy,
        .thread = std::this_thread::get_id(),
    });
    cv_.notify_all();
    return true;
  }

  template <typename Params>
  static void record_param_lifecycle(std::uint32_t operator_id, LifecycleStage stage,
                                     LifecycleContext& context) {
    const auto* const first = &context.params<Params>();
    const ParamGeneration first_generation = context.params_generation();
    const auto* const second = &context.params<Params>();
    const std::lock_guard lock(mutex_);
    param_lifecycle_records_.push_back(InlinePartitionParamLifecycleRecord{
        .operator_id = operator_id,
        .stage = stage,
        .value = first->value,
        .generation = first_generation,
        .address = first,
        .repeated_address_stable = first == second,
        .repeated_generation_stable = first_generation == context.params_generation(),
        .thread = std::this_thread::get_id(),
    });
  }

  static void record_sink(std::uint32_t value) {
    {
      const std::lock_guard lock(mutex_);
      sink_values_.push_back(value);
    }
    cv_.notify_all();
  }

  static void record_schema(std::string value) {
    {
      const std::lock_guard lock(mutex_);
      schema_value_ = std::move(value);
    }
    cv_.notify_all();
  }

  static void record_temporal_predicate(std::uint32_t threshold, std::uint32_t driver,
                                        std::uint32_t side) {
    const std::lock_guard lock(mutex_);
    temporal_predicate_records_.push_back(InlineTemporalPredicateRecord{
        .threshold = threshold,
        .driver = driver,
        .side = side,
        .thread = std::this_thread::get_id(),
    });
  }

  [[nodiscard]] static bool wait_for_first_runtime_compute() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(
        lock, std::chrono::seconds{5}, [] { return first_runtime_compute_entered_; });
  }

  static void release_first_runtime_compute() {
    {
      const std::lock_guard lock(mutex_);
      release_first_runtime_compute_ = true;
    }
    cv_.notify_all();
  }

  [[nodiscard]] static bool block_first_runtime_compute() {
    std::unique_lock lock(mutex_);
    first_runtime_compute_entered_ = true;
    cv_.notify_all();
    return cv_.wait_for(
        lock, std::chrono::seconds{5}, [] { return release_first_runtime_compute_; });
  }

  [[nodiscard]] static bool wait_for_sinks(std::size_t count) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(
        lock, std::chrono::seconds{5}, [count] { return sink_values_.size() >= count; });
  }

  [[nodiscard]] static bool wait_for_schema() {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds{5}, [] { return schema_value_.has_value(); });
  }

  [[nodiscard]] static std::vector<InlinePartitionComputeRecord> compute_records() {
    const std::lock_guard lock(mutex_);
    return compute_records_;
  }

  [[nodiscard]] static std::vector<InlinePartitionLifecycleRecord> lifecycle_records() {
    const std::lock_guard lock(mutex_);
    return lifecycle_records_;
  }

  [[nodiscard]] static std::vector<InlinePartitionParamComputeRecord> param_compute_records() {
    const std::lock_guard lock(mutex_);
    return param_compute_records_;
  }

  [[nodiscard]] static std::vector<InlinePartitionParamLifecycleRecord> param_lifecycle_records() {
    const std::lock_guard lock(mutex_);
    return param_lifecycle_records_;
  }

  [[nodiscard]] static std::vector<InlineTemporalPredicateRecord> temporal_predicate_records() {
    const std::lock_guard lock(mutex_);
    return temporal_predicate_records_;
  }

  [[nodiscard]] static std::vector<std::uint32_t> sink_values() {
    const std::lock_guard lock(mutex_);
    return sink_values_;
  }

  [[nodiscard]] static std::optional<std::string> schema_value() {
    const std::lock_guard lock(mutex_);
    return schema_value_;
  }

 private:
  static inline std::mutex mutex_;
  static inline std::condition_variable cv_;
  static inline std::vector<InlinePartitionComputeRecord> compute_records_;
  static inline std::vector<InlinePartitionLifecycleRecord> lifecycle_records_;
  static inline std::vector<InlinePartitionParamComputeRecord> param_compute_records_;
  static inline std::vector<InlinePartitionParamLifecycleRecord> param_lifecycle_records_;
  static inline std::vector<InlineTemporalPredicateRecord> temporal_predicate_records_;
  static inline std::vector<std::uint32_t> sink_values_;
  static inline std::optional<std::string> schema_value_;
  static inline bool first_runtime_compute_entered_{};
  static inline bool release_first_runtime_compute_{};
};

class InlinePartitionControllerProbe final : public detail::RuntimeSupportThreadControl {
 public:
  [[nodiscard]] expected<void, Error> enter(SupportThreadRole role) override {
    if (role == SupportThreadRole::kController) {
      const std::lock_guard lock(mutex_);
      controller_thread_ = std::this_thread::get_id();
    }
    return {};
  }

  [[nodiscard]] std::optional<std::thread::id> controller_thread() const {
    const std::lock_guard lock(mutex_);
    return controller_thread_;
  }

 private:
  mutable std::mutex mutex_;
  std::optional<std::thread::id> controller_thread_;
};

template <typename Derived>
class InlinePartitionLifecycle {
 protected:
  explicit InlinePartitionLifecycle(std::uint32_t operator_id, bool record_stop_device = false)
      : operator_id_(operator_id), record_stop_device_(record_stop_device) {}

  void setup_lifecycle(OperatorSpec& spec) {
    using Callback = LifecycleStatus (Derived::*)(LifecycleContext&) noexcept;
    spec.lifecycle()
        .stage(LifecycleStage::kStart,
               static_cast<Callback>(&InlinePartitionLifecycle::start_stage))
        .stage(LifecycleStage::kStop, static_cast<Callback>(&InlinePartitionLifecycle::stop_stage));
  }

  LifecycleStatus start_stage(LifecycleContext&) noexcept {
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStart);
    return LifecycleStatus::kOk;
  }

  LifecycleStatus stop_stage(LifecycleContext&) noexcept {
    if (record_stop_device_) {
      std::int32_t device = -1;
      if (cudaGetDevice(&device) == cudaSuccess) {
        InlineTensorProbeState::record_stop_device(device);
      }
    }
    InlinePartitionProbeState::record_lifecycle(operator_id_, LifecycleStage::kStop);
    return LifecycleStatus::kOk;
  }

  std::uint32_t operator_id_{};
  bool record_stop_device_{};
};

// Emits bounded POD samples and optionally requests a self-restart before its first emission.
// Representative test: RunsComputeAndLifecycleCallbacksOnTheControllerWithoutHandoffs.
class InlinePodSource final : public Operator<>, public InlinePartitionLifecycle<InlinePodSource> {
 public:
  InlinePodSource(std::uint32_t operator_id, std::uint32_t value, bool restart_before_emit = false,
                  std::size_t emission_limit = 1U,
                  std::chrono::nanoseconds period = std::chrono::seconds{30})
      : InlinePartitionLifecycle<InlinePodSource>(operator_id),
        value_(value),
        emission_limit_(emission_limit),
        period_(period),
        restart_before_emit_(restart_before_emit) {}

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
    if (emitted_count_ >= emission_limit_) {
      return {};
    }
    InlinePartitionProbeState::record_compute(operator_id_, value_, context.tag_time());
    if (restart_before_emit_ && !restart_requested_) {
      restart_requested_ = true;
      auto receipt =
          context.request_restart(LifecycleRestartOptions{.from_stage = LifecycleStage::kStart});
      if (!receipt) {
        return make_unexpected(std::move(receipt.error()));
      }
      return {};
    }
    ++emitted_count_;
    return output.emit(InlinePartitionPod{.value = value_});
  }

  Output<InlinePartitionPod> output;

 private:
  std::uint32_t value_{};
  std::size_t emission_limit_{};
  std::chrono::nanoseconds period_{};
  bool restart_before_emit_{};
  bool restart_requested_{};
  std::size_t emitted_count_{};
};

// Transforms POD samples while recording lifecycle and optional callback-failure behavior.
// Representative test: RunsComputeAndLifecycleCallbacksOnTheControllerWithoutHandoffs.
class InlinePodRelay final : public Operator<>, public InlinePartitionLifecycle<InlinePodRelay> {
 public:
  InlinePodRelay(std::uint32_t operator_id, std::uint32_t increment, bool fail_compute = false)
      : InlinePartitionLifecycle<InlinePodRelay>(operator_id),
        increment_(increment),
        fail_compute_(fail_compute) {}

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
    const std::uint32_t value = sample->data.value + increment_;
    InlinePartitionProbeState::record_compute(operator_id_, value, context.tag_time());
    if (fail_compute_) {
      return make_unexpected(Error{ErrorCode::kFailure, "INLINE_CALLBACK_FAILURE"});
    }
    return output.emit(InlinePartitionPod{.value = value});
  }

  Input<InlinePartitionPod> input;
  Output<InlinePartitionPod> output;

 private:
  std::uint32_t increment_{};
  bool fail_compute_{};
};

// Records terminal POD values, tag times, lifecycle calls, and callback ownership.
// Representative test: RunsComputeAndLifecycleCallbacksOnTheControllerWithoutHandoffs.
class InlinePodSink final : public Operator<>, public InlinePartitionLifecycle<InlinePodSink> {
 public:
  explicit InlinePodSink(std::uint32_t operator_id)
      : InlinePartitionLifecycle<InlinePodSink>(operator_id) {}

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
    InlinePartitionProbeState::record_compute(operator_id_, sample->data.value, context.tag_time());
    InlinePartitionProbeState::record_sink(sample->data.value);
    return {};
  }

  Input<InlinePartitionPod> input;
};

// Emits distinct POD values on two output ports for multi-input selection tests.
// Representative test: LatestWithinSelectionUsesTheInlinePartitionPlan.
class InlineForkSource final : public Operator<>,
                               public InlinePartitionLifecycle<InlineForkSource> {
 public:
  InlineForkSource() : InlinePartitionLifecycle<InlineForkSource>(1U) {}

  void setup(OperatorSpec& spec) override {
    setup_lifecycle(spec);
    spec.output(first, "first").max_emits_per_compute(1U);
    spec.output(second, "second").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::seconds{30}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    InlinePartitionProbeState::record_compute(operator_id_, 100U);
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    auto first_result = first.emit(InlinePartitionPod{.value = 101U});
    if (!first_result) {
      return first_result;
    }
    return second.emit(InlinePartitionPod{.value = 102U});
  }

  Output<InlinePartitionPod> first;
  Output<InlinePartitionPod> second;

 private:
  bool emitted_{};
};

struct InlinePartitionRunEvidence {
  bool observed{};
  RunTermination termination{RunTermination::kFailed};
  std::size_t diagnostic_count{};
  std::optional<std::thread::id> controller_thread;
  std::vector<InlinePartitionComputeRecord> compute_records;
  std::vector<InlinePartitionLifecycleRecord> lifecycle_records;
  std::vector<InlinePartitionParamComputeRecord> param_compute_records;
  std::vector<InlinePartitionParamLifecycleRecord> param_lifecycle_records;
  std::vector<InlineTemporalPredicateRecord> temporal_predicate_records;
  std::vector<std::uint32_t> sink_values;
  std::optional<std::string> schema_value;
  detail::RuntimePhaseCounters::Snapshot counters{};
};

[[nodiscard]] std::uint64_t phase_count(const InlinePartitionRunEvidence& evidence,
                                        detail::RuntimePhaseCounter counter);

void expect_no_legacy_lane_handoffs(const detail::RuntimePhaseCounters::Snapshot& counters);

[[nodiscard]] InlinePartitionRunEvidence capture_run_evidence(
    const RunSession& session, bool observed,
    const std::shared_ptr<detail::RuntimePhaseCounters>& counters,
    const std::shared_ptr<InlinePartitionControllerProbe>& control);

[[nodiscard]] InlinePartitionRunEvidence run_and_observe(ExecutionPlan plan, std::size_t sink_count,
                                                         bool schema_sink = false);

[[nodiscard]] bool wait_for_terminal(const RunSession& session);

[[nodiscard]] ConnectionOptions queue_options(QueuePolicy policy);

[[nodiscard]] ExecutionPlan make_pod_chain_plan(
    QueuePolicy policy, bool explicit_placement = false, std::size_t emission_limit = 1U,
    std::chrono::nanoseconds period = std::chrono::seconds{30});

[[nodiscard]] LifecycleTransactionOptions inline_lifecycle_capacity();

void expect_inline_thread_evidence(
    const InlinePartitionRunEvidence& evidence, std::size_t expected_compute_count,
    std::size_t expected_operator_count,
    std::optional<std::size_t> expected_lifecycle_count = std::nullopt);

}  // namespace test::inline_execution
}  // namespace holoscan
