/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace holoscan::python::test {

struct CounterSourceParams {
  std::uint32_t limit{10U};
};

HOLOSCAN_PARAM_SCHEMA(CounterSourceParams, param_schema("holoscan.tests.CounterSourceParams", 1),
                      field<&CounterSourceParams::limit>("limit").range(1U, 1'000'000U));

// Captures sink progress for test_first_pipeline_runs_until_explicit_stop_without_signal_changes
// and RunSessionReaperTest.StopsAndJoinsAnAbandonedSession.
class PipelineObservation final {
 public:
  static void reset() {
    const std::lock_guard lock(mutex_);
    count_ = 0U;
    last_value_ = 0U;
  }

  static void record(std::uint32_t value) {
    {
      const std::lock_guard lock(mutex_);
      ++count_;
      last_value_ = value;
    }
    ready_.notify_all();
  }

  [[nodiscard]] static bool wait_for(std::size_t expected, std::chrono::nanoseconds timeout) {
    std::unique_lock lock(mutex_);
    return ready_.wait_for(lock, timeout, [expected] { return count_ >= expected; });
  }

  [[nodiscard]] static std::size_t count() {
    const std::lock_guard lock(mutex_);
    return count_;
  }

  [[nodiscard]] static std::uint32_t last_value() {
    const std::lock_guard lock(mutex_);
    return last_value_;
  }

 private:
  static inline std::mutex mutex_;
  static inline std::condition_variable ready_;
  static inline std::size_t count_{};
  static inline std::uint32_t last_value_{};
};

// Supplies clock-driven scalar values for
// test_plan_matches_independently_authored_cpp_and_outlives_graph.
class CounterSource final : public Operator<CounterSourceParams> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(out, "out"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{10}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (next_ >= context.params<CounterSourceParams>().limit) {
      return {};
    }
    const std::uint32_t value = next_ + 1U;
    auto emitted = out.emit(value);
    if (emitted) {
      next_ = value;
    }
    return emitted;
  }

  Output<std::uint32_t> out;

 private:
  std::uint32_t next_{};
};

// Doubles scalar samples for test_first_pipeline_runs_until_explicit_stop_without_signal_changes.
class DoubleIt final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{in});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(const std::uint32_t value, in.receive_data());
    return out.emit(value * 2U);
  }

  Input<std::uint32_t> in;
  Output<std::uint32_t> out;
};

// Records doubled samples for test_first_pipeline_runs_until_explicit_stop_without_signal_changes.
class Sink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(in, "in"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{in});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(const std::uint32_t value, in.receive_data());
    PipelineObservation::record(value);
    return {};
  }

  Input<std::uint32_t> in;
};

struct ScalePairGraphParams {
  std::uint32_t factor{2U};
};

HOLOSCAN_PARAM_SCHEMA(ScalePairGraphParams, param_schema("holoscan.tests.ScalePairGraphParams", 1),
                      field<&ScalePairGraphParams::factor>("factor").range(1U, 1'000'000U));

struct WrappedScaleGraphParams {
  std::uint32_t factor{3U};
};

HOLOSCAN_PARAM_SCHEMA(WrappedScaleGraphParams,
                      param_schema("holoscan.tests.WrappedScaleGraphParams", 1),
                      field<&WrappedScaleGraphParams::factor>("factor").range(1U, 1'000'000U));

struct ConditionalInterfaceGraphParams {
  bool include_debug{false};
};

HOLOSCAN_PARAM_SCHEMA(ConditionalInterfaceGraphParams,
                      param_schema("holoscan.tests.ConditionalInterfaceGraphParams", 1),
                      field<&ConditionalInterfaceGraphParams::include_debug>("include_debug"));

// Multiplies scalar samples inside ScalePairGraph for registered-Graph embedding tests.
// Representative test: test_registered_cpp_graph_embedding_matches_independent_cpp_authoring.
class ScaleOp final : public Operator<> {
 public:
  explicit ScaleOp(std::uint32_t factor) noexcept : factor_(factor) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(const std::uint32_t value, input.receive_data());
    return output.emit(value * factor_);
  }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;

 private:
  std::uint32_t factor_{};
};

// Exports a parameter-dependent interface for
// test_registered_graph_ports_follow_real_parameterized_instance.
class ConditionalInterfaceGraph final : public Graph {
 public:
  using Params = ConditionalInterfaceGraphParams;

  explicit ConditionalInterfaceGraph(const Params& params) noexcept : params_(params) {}

  Input<std::uint32_t> input{"input"};
  Output<std::uint32_t> output{"output"};
  Output<std::uint32_t> debug{"debug"};

 protected:
  void compose() override {
    const auto primary = op<ScaleOp>("primary", 1U);
    export_input(input, primary->input);
    export_output(output, primary->output);
    if (params_.include_debug) {
      const auto debug_stage = op<ScaleOp>("debug", 1U);
      add_flow(primary->output, debug_stage->input);
      export_output(debug, debug_stage->output);
    }
  }

 private:
  Params params_;
};

// Exports one configurable scale stage for registered-Graph discovery and embedding tests.
// Representative test: test_registered_cpp_graph_discovery_and_interface_metadata.
class ScalePairGraph final : public Graph {
 public:
  using Params = ScalePairGraphParams;

  explicit ScalePairGraph(const Params& params) noexcept : factor_(params.factor) {}

  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

  Input<std::uint32_t> input{"input"};
  Output<std::uint32_t> output{"output"};

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    const auto scale = op<ScaleOp>("scale", factor_);
    export_input(input, scale->input);
    export_output(output, scale->output);
  }

 private:
  std::uint32_t factor_{};
  ClockKey clock_import_;
};

// Re-exports a nested ScalePairGraph for registered nested-Graph and clock-binding tests.
// Representative test: test_registered_cpp_graph_embedding_matches_independent_cpp_authoring.
class WrappedScaleGraph final : public Graph {
 public:
  using Params = WrappedScaleGraphParams;

  explicit WrappedScaleGraph(const Params& params) noexcept : factor_(params.factor) {}

  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

  Input<std::uint32_t> input{"input"};
  Output<std::uint32_t> output{"output"};

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    const auto inner = embed<ScalePairGraph>("inner", ScalePairGraphParams{.factor = factor_});
    bind_clock(inner, inner->clock_import(), clock_import_);
    export_input(input, inner->input);
    export_output(output, inner->output);
  }

 private:
  std::uint32_t factor_{};
  ClockKey clock_import_;
};

// Fails composition so test_registered_graph_discovery_isolates_failing_rows can verify that
// discovery does not hide healthy providers.
class FailingDiscoveryGraph final : public Graph {
 protected:
  void compose() override {
    throw RuntimeError{Error{ErrorCode::kFailure, "TEST_GRAPH_DISCOVERY_FAILED"}};
  }
};

// Produces an unbounded clock-driven stream for test_blocking_run_returns_terminal_session.
class TickSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(out, "out"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{5}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    return out.emit(++next_);
  }

  Output<std::uint32_t> out;

 private:
  std::uint32_t next_{};
};

// Publishes one timestamped scalar while remaining clock-driven so
// test_depth_one_reuse_last_preserves_the_initial_native_python_sample can verify Python-authored
// temporal selection after the ordinary queue record has retired.
class SparseTimestampedSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(out, "out"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (published_) {
      return {};
    }
    const TimePoint capture_time = context.activation_time();
    auto emitted = out.emit(Sample<std::uint32_t>{
        17U,
        SampleMetadata{.capture_timestamp_ns = capture_time.timestamp_ns,
                       .source_clock_id = capture_time.clock_id},
    });
    if (emitted) {
      published_ = true;
    }
    return emitted;
  }

  Output<std::uint32_t> out;

 private:
  bool published_{};
};

// Requests successful run completion for test_blocking_run_returns_terminal_session.
class CompletingSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(in, "in"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{in});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    HOLOSCAN_TRY(const std::uint32_t value, in.receive_data());
    if (value == 5U) {
      context.request_completion();
    }
    return {};
  }

  Input<std::uint32_t> in;
};

// Produces a deterministic callback failure for
// test_failed_session_exposes_structured_diagnostics_and_bounded_summary.
class StructuredFailureSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(out, "out"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    try {
      throw RuntimeError{Error{ErrorCode::kNotFound, "nested runtime cause"}};
    } catch (...) {
      std::throw_with_nested(
          RuntimeError{Error{ErrorCode::kInvalidArgument, "primary runtime failure"}});
    }
  }

  Output<std::uint32_t> out;
};

inline constexpr DLDataType kTopicFrameDtype{kDLUInt, 8U, 1U};
inline constexpr TensorRepresentation kTopicFrameRepresentation{
    .memory_kind = MemoryKind::kHost,
    .dtype = kTopicFrameDtype,
    .rank = 1U,
};
inline constexpr std::array<std::int64_t, 1U> kTopicFrameShape{64};

// Supplies Tensor frames for test_camera_imu_topic_plan_matches_independent_cpp_authoring.
class TopicCameraSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(frame, "frame")
        .produces_tensor(TensorOutputSpec{.representation = kTopicFrameRepresentation,
                                          .bounds = tensor_bounds(64U),
                                          .storage = TensorOutputStorage::kRuntimePool});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{20}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto loan = frame.allocate_tensor(
        TensorLoanRequest{.shape = kTopicFrameShape, .dtype = kTopicFrameDtype});
    if (!loan) {
      return make_unexpected(std::move(loan).error());
    }
    auto committed = loan->write_host_and_commit(
        [this](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
          auto* bytes = writer.data_as<std::uint8_t>();
          if (bytes == nullptr || writer.byte_size() != kTopicFrameShape.front()) {
            return make_unexpected(Error{ErrorCode::kInvalidArgument});
          }
          std::fill_n(bytes, writer.byte_size(), static_cast<std::uint8_t>(sequence_++));
          return {};
        });
    if (!committed) {
      return make_unexpected(std::move(committed).error());
    }
    return frame.emit(std::move(*loan), EmitOptions{.capture_time = context.activation_time()});
  }

  Output<Tensor> frame;

 private:
  std::uint32_t sequence_{};
};

// Fuses camera and topic samples for test_camera_imu_topic_plan_matches_independent_cpp_authoring.
class TopicWindowFusion final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(camera, "camera")
        .queue_depth(4U)
        .expects_tensor(TensorInputSpec{.representation = kTopicFrameRepresentation});
    spec.input(imu, "imu").queue_depth(32U);
    spec.output(control, "control");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result
        .trigger(OnWindow{
            .anchor = camera, .max_wait = std::chrono::milliseconds{20}, .max_pending_windows = 2U})
        .input(camera, Exact{})
        .input(imu, BracketedWindow{.lookbehind = 1U, .lookahead = 1U, .max_samples = 16U})
        .on_missing(imu, SafeDefault{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto received_frame = camera.receive();
    if (!received_frame) {
      return make_unexpected(std::move(received_frame).error());
    }
    std::uint32_t integrated{};
    if (imu.selection().state == InputState::kSelected) {
      auto batch = imu.selected_batch();
      if (!batch) {
        return make_unexpected(std::move(batch).error());
      }
      for (const auto& sample : *batch) {
        integrated += sample.data;
      }
    }
    auto published =
        control.emit(static_cast<std::uint32_t>(received_frame->data.nbytes()) + integrated);
    return !published && published.error().code == ErrorCode::kNotReady ? expected<void, Error>{}
                                                                        : std::move(published);
  }

  Input<Tensor> camera;
  Input<std::uint32_t> imu;
  Output<std::uint32_t> control;
};

}  // namespace holoscan::python::test
