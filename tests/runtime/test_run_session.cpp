/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/managed_shutdown_signals.hpp>

#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_placement_internal.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/parameter_update_coordinator.hpp"
#include "runtime/runtime_callback_context.hpp"
#include "runtime/runtime_partition_test_plan_access.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_work_inventory.hpp"
#include "runtime/runtime_worker.hpp"
#include "runtime/source_wait.hpp"
#include "runtime_channel_binding_test_support.hpp"
#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"

namespace holoscan {

struct RuntimeCompletionFencePayload {
  std::uint32_t value{};
};

template <>
struct ChannelTypeIdentity<RuntimeCompletionFencePayload> {
  static constexpr std::string_view name = "nvidia.holoscan.test.RuntimeCompletionFencePayload";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<RuntimeCompletionFencePayload> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &RuntimeCompletionFencePayload::value,
                           offsetof(RuntimeCompletionFencePayload, value)));
};

namespace {

using namespace std::chrono_literals;

enum class DiagnosticReplayMode { kNone, kConstruct, kDeclaration, kContract };

class DiagnosticReplayProbe final : public Operator<> {
 public:
  explicit DiagnosticReplayProbe(bool direct) : direct_(direct) {
    if (mode.load() == DiagnosticReplayMode::kConstruct) {
      try {
        throw RuntimeError{ErrorCode::kInvalidArgument, "nested factory cause"};
      } catch (...) {
        std::throw_with_nested(RuntimeError{ErrorCode::kNotSupported, "factory cause"});
      }
    }
  }
  void setup(OperatorSpec& spec) override {
    spec.output(output_, "output");
    if (mode.load() == DiagnosticReplayMode::kDeclaration) {
      spec.output(output_, "output");
    }
    if (direct_) {
      spec.spin_source(spin_, "activation").capacity(1U);
    }
  }
  Contract contract() const override {
    Contract result;
    if (direct_) {
      result.trigger(OnSpin{.event = spin_});
    } else {
      result.trigger(OnClock{.period = 1ms});
    }
    if (mode.load() == DiagnosticReplayMode::kContract) {
      result.deadline(0ns).on_violation(OnViolation::kFailRun);
    }
    return result;
  }
  void start() override { callbacks.fetch_add(1U); }
  expected<void, Error> compute(ExecutionContext&) override {
    callbacks.fetch_add(1U);
    return {};
  }
  static inline std::atomic<DiagnosticReplayMode> mode{DiagnosticReplayMode::kNone};
  static inline std::atomic<std::size_t> callbacks{};

 private:
  bool direct_{};
  SpinSource spin_;
  Output<std::uint32_t> output_;
};

TEST(RunSessionTest, FactoryAndReturnedReplayErrorsRetainStructuredEvidence) {
  struct Reset {
    ~Reset() { DiagnosticReplayProbe::mode.store(DiagnosticReplayMode::kNone); }
  };
  const Reset reset;
  for (const bool direct : {false, true}) {
    DiagnosticReplayProbe::mode.store(DiagnosticReplayMode::kNone);
    Graph graph{"diagnostic-replay"};
    const auto probe = graph.op<DiagnosticReplayProbe>("probe", direct);
    graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
    CompileOptions options;
    if (direct) {
      graph.partition("worker").add(probe);
      options.deployment.bind_thread_placement("worker",
                                               ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
    }
    const auto compiled = compile(graph, options);
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    const auto plan = direct ? test::enable_direct_runtime(compiled) : compiled;
    for (const auto mode : {DiagnosticReplayMode::kDeclaration,
                            DiagnosticReplayMode::kContract,
                            DiagnosticReplayMode::kConstruct}) {
      SCOPED_TRACE(static_cast<int>(mode));
      SCOPED_TRACE(direct);
      DiagnosticReplayProbe::mode.store(mode);
      DiagnosticReplayProbe::callbacks.store(0U);
      auto session = run_async(plan);
      session.wait();
      ASSERT_EQ(session.termination(), RunTermination::kFailed);
      EXPECT_EQ(DiagnosticReplayProbe::callbacks.load(), 0U);
      ASSERT_EQ(session.diagnostics().size(), 1U);
      const auto& diagnostic = session.diagnostics().front();
      const auto& retained_detail = diagnostic.detail;
      if (!retained_detail.has_value()) {
        FAIL() << "Expected diagnostic detail";
      }
      const auto& evidence = *retained_detail;
      EXPECT_EQ(evidence.operator_path, "probe");
      if (mode == DiagnosticReplayMode::kConstruct) {
        EXPECT_EQ(diagnostic.token, "RUNTIME_OPERATOR_RECONSTRUCTION_FAILED");
        EXPECT_EQ(evidence.operation, DiagnosticOperation::kConstruct);
        EXPECT_EQ(evidence.error.code, ErrorCode::kNotSupported);
        EXPECT_EQ(evidence.error.message, "factory cause");
        ASSERT_EQ(evidence.causes.size(), 1U);
        EXPECT_EQ(evidence.causes.front().code, ErrorCode::kInvalidArgument);
        EXPECT_EQ(evidence.causes.front().message, "nested factory cause");
      } else {
        EXPECT_EQ(diagnostic.token, "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH");
        EXPECT_TRUE(diagnostic.required.starts_with("sha256:"));
        EXPECT_TRUE(diagnostic.provided.starts_with("sha256:"));
        EXPECT_NE(diagnostic.required, diagnostic.provided);
        EXPECT_EQ(evidence.operation,
                  mode == DiagnosticReplayMode::kDeclaration ? DiagnosticOperation::kSetup
                                                             : DiagnosticOperation::kContract);
        EXPECT_EQ(evidence.error.code,
                  mode == DiagnosticReplayMode::kDeclaration ? ErrorCode::kDuplicateName
                                                             : ErrorCode::kInvalidArgument);
        EXPECT_NE(evidence.error.message.find(mode == DiagnosticReplayMode::kDeclaration
                                                  ? "duplicated"
                                                  : "deadline budget must be positive"),
                  std::string::npos);
      }
    }
  }
}

class FactoryCaptureOperator final : public Operator<> {
 public:
  explicit FactoryCaptureOperator(std::exception_ptr failure) {
    std::rethrow_exception(std::move(failure));
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

class FactoryCaptureClock final : public Clock {
 public:
  explicit FactoryCaptureClock(std::exception_ptr failure) {
    std::rethrow_exception(std::move(failure));
  }
  double time() const override { return 0; }
  std::int64_t timestamp() const override { return 0; }
  expected<void, Error> sleep_for(std::int64_t) override { return {}; }
  expected<void, Error> sleep_until(std::int64_t) override { return {}; }
};

TEST(RunSessionTest, OperatorAndClockFactoryCaptureSurvivesAllocationFailure) {
  std::exception_ptr failure;
  try {
    throw RuntimeError{ErrorCode::kInvalidArgument,
                       "original factory exception survives allocation failure"};
  } catch (...) {
    failure = std::current_exception();
  }
  const detail::OperatorFactoryModel<FactoryCaptureOperator, std::exception_ptr> op_factory{
      failure};
  const detail::ClockFactoryModel<FactoryCaptureClock, std::exception_ptr> clock_factory{failure};
  const auto verify = [](const auto& factory) {
    detail::DiagnosticCapture capture;
    const auto created = [&] {
      const test::ScopedAllocationFailure fail_after_object_allocation{1U};
      return factory.create(&capture);
    }();
    ASSERT_FALSE(created);
    EXPECT_EQ(created.error().code, ErrorCode::kInvalidArgument);
    ASSERT_EQ(capture.count, 1U);
    const auto evidence = capture.detail(DiagnosticOperation::kConstruct);
    EXPECT_EQ(evidence.error.message, "original factory exception survives allocation failure");
    EXPECT_FALSE(evidence.truncated);
  };
  verify(op_factory);
  verify(clock_factory);
}

TEST(RunSessionTest, ConcurrentFactoryCapturesRetainIndependentBoundedCauses) {
  std::exception_ptr failure;
  try {
    throw RuntimeError{ErrorCode::kInvalidArgument, std::string(8192U, 'x')};
  } catch (...) {
    failure = std::current_exception();
  }
  for (std::size_t depth = 0U; depth < 5U; ++depth) {
    try {
      try {
        std::rethrow_exception(failure);
      } catch (...) {
        std::throw_with_nested(RuntimeError{ErrorCode::kFailure, "outer cause"});
      }
    } catch (...) {
      failure = std::current_exception();
    }
  }
  const detail::OperatorFactoryModel<FactoryCaptureOperator, std::exception_ptr> factory{failure};
  std::array<detail::DiagnosticCapture, 2U> captures;
  std::thread first([&] { static_cast<void>(factory.create(&captures[0])); });
  std::thread second([&] { static_cast<void>(factory.create(&captures[1])); });
  first.join();
  second.join();
  for (const auto& capture : captures) {
    EXPECT_EQ(capture.count, detail::DiagnosticCapture::kCauseCapacity);
    EXPECT_TRUE(capture.truncated);
    const auto evidence = capture.detail(DiagnosticOperation::kConstruct);
    EXPECT_EQ(evidence.error.message, "outer cause");
    EXPECT_EQ(evidence.causes.size(), 3U);
  }
}

TEST(RunSessionTest, ClockReconstructionRetainsNestedFactoryEvidence) {
  DiagnosticReplayProbe::mode.store(DiagnosticReplayMode::kNone);
  Graph graph{"clock-failure"};
  graph.op<DiagnosticReplayProbe>("probe", false);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  const auto plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  std::exception_ptr failure;
  try {
    try {
      throw RuntimeError{ErrorCode::kNotFound, "clock device missing"};
    } catch (...) {
      std::throw_with_nested(RuntimeError{ErrorCode::kNotSupported, "clock construction failed"});
    }
  } catch (...) {
    failure = std::current_exception();
  }
  auto state =
      std::make_shared<detail::ExecutionPlanState>(*detail::ExecutionPlanAccess::retain(plan));
  ASSERT_EQ(state->executable_factories.clocks.size(), 1U);
  state->executable_factories.clocks.front().factory =
      std::make_shared<detail::ClockFactoryModel<FactoryCaptureClock, std::exception_ptr>>(failure);
  detail::RuntimeDiagnosticDetail evidence;
  const auto reconstructed = detail::GraphExec::reconstruct(state, 1U, {}, nullptr, &evidence);
  ASSERT_FALSE(reconstructed);
  ASSERT_TRUE(evidence.capture);
  EXPECT_EQ(evidence.operation, DiagnosticOperation::kConstruct);
  EXPECT_EQ(evidence.component_path, "clock/clock");
  EXPECT_FALSE(evidence.vertex);
  const auto captured = evidence.capture->detail(evidence.operation);
  EXPECT_EQ(captured.error.message, "clock construction failed");
  ASSERT_EQ(captured.causes.size(), 1U);
  EXPECT_EQ(captured.causes.front().code, ErrorCode::kNotFound);
  EXPECT_EQ(captured.causes.front().message, "clock device missing");
}

class CleanupOverflowProbe final : public Operator<> {
 public:
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    context.request_completion();
    return {};
  }
  void stop() override { throw RuntimeError{ErrorCode::kFailure, "cleanup overflow cause"}; }
};

TEST(RunSessionTest, LifecycleFailureOverflowIsExplicitAndKeepsThePrimary) {
  Graph graph{"lifecycle-failure-overflow"};
  for (std::size_t index = 0U; index < 6U; ++index) {
    graph.op<CleanupOverflowProbe>("probe" + std::to_string(index));
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  const auto plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = run_async(plan);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 4U);
  for (std::size_t index = 0U; index < session.diagnostics().size(); ++index) {
    const auto& diagnostic = session.diagnostics()[index];
    const auto& retained_detail = diagnostic.detail;
    if (!retained_detail.has_value()) {
      FAIL() << "Expected diagnostic detail";
    }
    EXPECT_EQ(retained_detail->role, DiagnosticFailureRole::kTeardown);
    EXPECT_EQ(retained_detail->error.message, "cleanup overflow cause");
    EXPECT_EQ(retained_detail->truncated, index == 3U);
    EXPECT_EQ(retained_detail->primary_diagnostic_index,
              index == 0U ? std::nullopt : std::optional<std::uint32_t>{0U});
  }
}

class ThrowingTerminalClosePublisher final : public channel_internal::InlinePayloadPublisher {
 public:
  explicit ThrowingTerminalClosePublisher(
      std::shared_ptr<std::atomic<std::size_t>> close_count) noexcept
      : close_count_(std::move(close_count)) {}

  [[nodiscard]] channel_internal::InlinePublicationResult emit_phase(
      std::span<const std::byte>, const SampleMetadata&) override {
    return channel_internal::InlinePublicationResult{
        .phase = channel_internal::InlinePublicationPhase::kAccepted,
        .recipient_count = 1U,
        .error = Error{ErrorCode::kSuccess},
    };
  }

  [[nodiscard]] expected<void, Error> close_stream() override {
    close_count_->fetch_add(1U, std::memory_order_relaxed);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    throw std::runtime_error{"injected terminal close failure"};
#else
    return make_unexpected(Error{ErrorCode::kNotSupported});
#endif
  }

  [[nodiscard]] expected<void, Error> update_connections() noexcept override { return {}; }
  [[nodiscard]] bool has_subscribers() const noexcept override { return true; }

 private:
  std::shared_ptr<std::atomic<std::size_t>> close_count_;
};

class ThrowingTerminalCloseChannelBindControl final
    : public detail::RuntimeChannelSessionBindTestControl {
 public:
  explicit ThrowingTerminalCloseChannelBindControl(const PortId output) noexcept
      : output_(output), close_count_(std::make_shared<std::atomic<std::size_t>>()) {}

  [[nodiscard]] expected<void, Error> bind(
      detail::RuntimeChannelSession& session) noexcept override {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
      auto first = std::make_unique<ThrowingTerminalClosePublisher>(close_count_);
      std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers;
      publishers.push_back(std::move(first));
      return detail::RuntimeChannelSessionTestAccess::replace_output_publishers(
          session, output_, std::move(publishers));
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
#else
    static_cast<void>(session);
    return make_unexpected(Error{ErrorCode::kNotSupported});
#endif
  }

  [[nodiscard]] std::size_t first_close_count() const noexcept {
    return close_count_->load(std::memory_order_relaxed);
  }

 private:
  PortId output_{};
  std::shared_ptr<std::atomic<std::size_t>> close_count_;
};

// REQ: RUNTIME-046
static_assert(static_cast<std::uint8_t>(RunTermination::kRunning) == 0U);
static_assert(static_cast<std::uint8_t>(RunTermination::kCompleted) == 1U);
static_assert(static_cast<std::uint8_t>(RunTermination::kRequestedCompleted) == 2U);
static_assert(static_cast<std::uint8_t>(RunTermination::kStopped) == 3U);
static_assert(static_cast<std::uint8_t>(RunTermination::kFailed) == 4U);

// REQ: RUNTIME-047
// REQ: RUNTIME-049
static_assert(static_cast<std::uint8_t>(RunSignalPolicy::kManaged) == 0U);
static_assert(static_cast<std::uint8_t>(RunSignalPolicy::kCallerManaged) == 1U);
static_assert(std::move_constructible<ManagedShutdownSignals>);
static_assert(std::is_nothrow_move_constructible_v<ManagedShutdownSignals>);
static_assert(std::is_nothrow_destructible_v<ManagedShutdownSignals>);
static_assert(!std::copyable<ManagedShutdownSignals>);
static_assert(!std::is_move_assignable_v<ManagedShutdownSignals>);

TEST(RunSessionApi, ExposesManagedAndCallerManagedExecutionOverloads) {
  const auto blocking = static_cast<int (*)(const ExecutionPlan&, RunSignalPolicy)>(&run);
  const auto asynchronous =
      static_cast<RunSession (*)(const ExecutionPlan&, ManagedShutdownSignals&)>(&run_async);
  EXPECT_NE(blocking, nullptr);
  EXPECT_NE(asynchronous, nullptr);
}

// REQ: RUNTIME-050
class RuntimeCompletionFenceSource final : public Operator<> {
 public:
  explicit RuntimeCompletionFenceSource(bool with_deadline = false) noexcept
      : with_deadline_(with_deadline) {}

  void setup(OperatorSpec& spec) override {
    spec.spin_source(spin_, "activation").capacity(1U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = spin_});
    if (with_deadline_) {
      result.deadline(1h).on_violation(OnViolation::kEmitWitness);
    }
    result.complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    compute_count.fetch_add(1U, std::memory_order_relaxed);
    return output.emit(RuntimeCompletionFencePayload{.value = 41U});
  }

  static inline std::atomic<std::size_t> compute_count{};
  Output<RuntimeCompletionFencePayload> output;

 private:
  SpinSource spin_;
  bool with_deadline_{};
};

class RuntimeCompletionFenceSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    observed_value.store(sample->data.value, std::memory_order_relaxed);
    compute_count.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }

  static inline std::atomic<std::size_t> compute_count{};
  static inline std::atomic<std::uint32_t> observed_value{};
  Input<RuntimeCompletionFencePayload> input;
};

[[nodiscard]] ExecutionPlan make_runtime_completion_fence_plan(bool with_deadline = false) {
  Graph graph{"runtime-completion-final-entry-fence"};
  const auto source = graph.op<RuntimeCompletionFenceSource>("source", with_deadline);
  const auto sink = graph.op<RuntimeCompletionFenceSink>("sink");
  graph.add_flow(source->output,
                 sink->input,
                 ConnectionOptions{
                     .queue_policy = QueuePolicy::kBlocking,
                     .queue_depth = 1U,
                 });
  graph.partition("runtime-completion-final-entry-fence").add(source).add(sink);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-completion-fence-clock"));
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-completion-final-entry-fence",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

void reset_runtime_completion_fence_probe() noexcept {
  RuntimeCompletionFenceSource::compute_count.store(0U, std::memory_order_relaxed);
  RuntimeCompletionFenceSink::compute_count.store(0U, std::memory_order_relaxed);
  RuntimeCompletionFenceSink::observed_value.store(0U, std::memory_order_relaxed);
}

class OptionalLifecycleReplayProbe final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    if (!omit_callbacks.load(std::memory_order_relaxed)) {
      spec.lifecycle()
          .optional(LifecycleStage::kStart, &OptionalLifecycleReplayProbe::start_stage)
          .optional(LifecycleStage::kStop, &OptionalLifecycleReplayProbe::stop_stage);
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }
  LifecycleStatus start_stage(LifecycleContext&) noexcept { return LifecycleStatus::kOk; }
  LifecycleStatus stop_stage(LifecycleContext&) noexcept { return LifecycleStatus::kOk; }

  static inline std::atomic_bool omit_callbacks{};
};

class CudaGraphEligibilityReplayProbe final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.cuda_graph_capture_eligible(eligible.load(std::memory_order_relaxed));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  static inline std::atomic_bool eligible{true};
};

struct RuntimeRateProbeState {
  static void reset() noexcept {
    bind_error = ErrorCode::kSuccess;
    same_stage_error = ErrorCode::kSuccess;
    allocate_error = ErrorCode::kSuccess;
    compute_error = ErrorCode::kSuccess;
    same_stage = {};
    allocate = {};
    compute = {};
  }

  static inline ErrorCode bind_error{ErrorCode::kSuccess};
  static inline ErrorCode same_stage_error{ErrorCode::kSuccess};
  static inline ErrorCode allocate_error{ErrorCode::kSuccess};
  static inline ErrorCode compute_error{ErrorCode::kSuccess};
  static inline PortRateView same_stage;
  static inline PortRateView allocate;
  static inline PortRateView compute;
};

class RuntimeFixedRateQueryProbe final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.spin_source(spin_, "activation").capacity(1U);
    spec.output(output, "output").nominal_rate(Rate::hz(60U));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = spin_}).complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto view = context.rate(output);
    if (!view) {
      RuntimeRateProbeState::compute_error = view.error().code;
      return make_unexpected(std::move(view.error()));
    }
    RuntimeRateProbeState::compute = *view;
    return output.emit(RuntimeCompletionFencePayload{.value = 17U});
  }

  Output<RuntimeCompletionFencePayload> output;

 private:
  SpinSource spin_;
};

class RuntimeFixedRateQuerySink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    return {};
  }

  Input<RuntimeCompletionFencePayload> input;
};

class RuntimeLifecycleRateQueryProbe final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.spin_source(spin_, "activation").capacity(1U);
    spec.output(output, "output")
        .lifecycle_nominal_rate(LifecycleStage::kDiscover)
        .max_nominal_rate(Rate::hz(120U));
    spec.lifecycle()
        .stage(LifecycleStage::kDiscover, &RuntimeLifecycleRateQueryProbe::discover)
        .stage(LifecycleStage::kAllocate, &RuntimeLifecycleRateQueryProbe::allocate);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = spin_}).complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto view = context.rate(output);
    if (!view) {
      RuntimeRateProbeState::compute_error = view.error().code;
      return make_unexpected(std::move(view.error()));
    }
    RuntimeRateProbeState::compute = *view;
    return {};
  }

  LifecycleStatus discover(LifecycleContext& context) noexcept {
    auto bound = context.bind_nominal_rate(output, Rate::hz(60U));
    if (!bound) {
      RuntimeRateProbeState::bind_error = bound.error().code;
      return LifecycleStatus::kFatalFailure;
    }
    auto view = context.rate(output);
    if (!view) {
      RuntimeRateProbeState::same_stage_error = view.error().code;
      return LifecycleStatus::kFatalFailure;
    }
    RuntimeRateProbeState::same_stage = *view;
    return LifecycleStatus::kOk;
  }

  LifecycleStatus allocate(LifecycleContext& context) noexcept {
    auto view = context.rate(output);
    if (!view) {
      RuntimeRateProbeState::allocate_error = view.error().code;
      return LifecycleStatus::kFatalFailure;
    }
    RuntimeRateProbeState::allocate = *view;
    return LifecycleStatus::kOk;
  }

  Output<std::uint32_t> output;

 private:
  SpinSource spin_;
};

struct RuntimeLifecycleRateQueueProbeState {
  static void reset() noexcept {
    bind_error.store(0U, std::memory_order_release);
    allocate_error.store(0U, std::memory_order_release);
    compute_error.store(0U, std::memory_order_release);
    allocate_saw_resolved.store(false, std::memory_order_release);
    sink_received.store(false, std::memory_order_release);
    observed_value.store(0U, std::memory_order_release);
  }

  static void fail(std::atomic<std::uint32_t>& target, ErrorCode code) noexcept {
    target.store(static_cast<std::uint32_t>(code), std::memory_order_release);
  }

  static inline std::atomic<std::uint32_t> bind_error{};
  static inline std::atomic<std::uint32_t> allocate_error{};
  static inline std::atomic<std::uint32_t> compute_error{};
  static inline std::atomic_bool allocate_saw_resolved{};
  static inline std::atomic_bool sink_received{};
  static inline std::atomic<std::uint32_t> observed_value{};
};

class RuntimeLifecycleRateQueueSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").lifecycle_nominal_rate(LifecycleStage::kDiscover);
    spec.lifecycle()
        .stage(LifecycleStage::kDiscover, &RuntimeLifecycleRateQueueSource::discover)
        .stage(LifecycleStage::kAllocate, &RuntimeLifecycleRateQueueSource::allocate);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto view = context.rate(output);
    if (!view) {
      RuntimeLifecycleRateQueueProbeState::fail(RuntimeLifecycleRateQueueProbeState::compute_error,
                                                view.error().code);
      return make_unexpected(std::move(view.error()));
    }
    auto emitted = output.emit(kValue);
    if (!emitted) {
      RuntimeLifecycleRateQueueProbeState::fail(RuntimeLifecycleRateQueueProbeState::compute_error,
                                                emitted.error().code);
    }
    return emitted;
  }

  LifecycleStatus discover(LifecycleContext& context) noexcept {
    auto bound = context.bind_nominal_rate(output, Rate::hz(60U));
    if (!bound) {
      RuntimeLifecycleRateQueueProbeState::fail(RuntimeLifecycleRateQueueProbeState::bind_error,
                                                bound.error().code);
      return LifecycleStatus::kFatalFailure;
    }
    return LifecycleStatus::kOk;
  }

  LifecycleStatus allocate(LifecycleContext& context) noexcept {
    auto view = context.rate(output);
    if (!view) {
      RuntimeLifecycleRateQueueProbeState::fail(RuntimeLifecycleRateQueueProbeState::allocate_error,
                                                view.error().code);
      return LifecycleStatus::kFatalFailure;
    }
    RuntimeLifecycleRateQueueProbeState::allocate_saw_resolved.store(
        view->state == PortRateViewState::kResolved && view->resolved_nominal_rate == Rate::hz(60U),
        std::memory_order_release);
    return LifecycleStatus::kOk;
  }

  Output<std::uint32_t> output;
  static constexpr std::uint32_t kValue = 0x4155544FU;
};

class RuntimeLifecycleRateQueueSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      RuntimeLifecycleRateQueueProbeState::fail(RuntimeLifecycleRateQueueProbeState::compute_error,
                                                sample.error().code);
      return make_unexpected(std::move(sample.error()));
    }
    RuntimeLifecycleRateQueueProbeState::observed_value.store(sample->data,
                                                              std::memory_order_release);
    RuntimeLifecycleRateQueueProbeState::sink_received.store(true, std::memory_order_release);
    return {};
  }

  Input<std::uint32_t> input;
};

struct RuntimeLifecycleDrainAllScratchProbeState {
  static void reset() noexcept {
    error.store(0U, std::memory_order_release);
    driver_value.store(0U, std::memory_order_release);
    batch_size.store(0U, std::memory_order_release);
  }

  static void fail(ErrorCode code) noexcept {
    error.store(static_cast<std::uint32_t>(code), std::memory_order_release);
  }

  static inline std::atomic<std::uint32_t> error{};
  static inline std::atomic<std::uint32_t> driver_value{};
  static inline std::atomic<std::size_t> batch_size{};
};

// Build the secondary backlog over twelve 1 ms callbacks, then publish the OnEach driver from a
// separate callback. The extra callback makes the full retained prefix independent of cross-channel
// ingress scan order.
class RuntimeLifecycleDrainAllScratchSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(secondary, "secondary").lifecycle_nominal_rate(LifecycleStage::kDiscover);
    spec.output(driver, "driver").nominal_rate(Rate::hz(100U));
    spec.lifecycle().stage(LifecycleStage::kDiscover,
                           &RuntimeLifecycleDrainAllScratchSource::discover);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = kBacklogSize + 1U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    ++next_value_;
    if (next_value_ > kBacklogSize) {
      auto driver_emitted = driver.emit(kDriverValue);
      if (!driver_emitted) {
        RuntimeLifecycleDrainAllScratchProbeState::fail(driver_emitted.error().code);
      }
      return driver_emitted;
    }
    auto secondary_emitted = secondary.emit(next_value_);
    if (!secondary_emitted) {
      RuntimeLifecycleDrainAllScratchProbeState::fail(secondary_emitted.error().code);
    }
    return secondary_emitted;
  }

  LifecycleStatus discover(LifecycleContext& context) noexcept {
    auto bound = context.bind_nominal_rate(secondary, Rate::hz(1000U));
    if (!bound) {
      RuntimeLifecycleDrainAllScratchProbeState::fail(bound.error().code);
      return LifecycleStatus::kFatalFailure;
    }
    return LifecycleStatus::kOk;
  }

  static constexpr std::uint32_t kBacklogSize = 12U;
  static constexpr std::uint32_t kDriverValue = 0xD41U;
  Output<std::uint32_t> secondary;
  Output<std::uint32_t> driver;

 private:
  std::uint32_t next_value_{};
};

class RuntimeLifecycleDrainAllScratchSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(secondary, "secondary");
    spec.input(driver, "driver");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{driver}).input(secondary, DrainAll{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto driver_sample = driver.receive();
    if (!driver_sample) {
      RuntimeLifecycleDrainAllScratchProbeState::fail(driver_sample.error().code);
      return make_unexpected(std::move(driver_sample.error()));
    }
    auto batch = secondary.selected_batch();
    if (!batch) {
      RuntimeLifecycleDrainAllScratchProbeState::fail(batch.error().code);
      return make_unexpected(std::move(batch.error()));
    }
    RuntimeLifecycleDrainAllScratchProbeState::driver_value.store(driver_sample->data,
                                                                  std::memory_order_release);
    RuntimeLifecycleDrainAllScratchProbeState::batch_size.store(batch->size(),
                                                                std::memory_order_release);
    return {};
  }

  Input<std::uint32_t> secondary;
  Input<std::uint32_t> driver;
};

template <typename OperatorT>
[[nodiscard]] ExecutionPlan make_runtime_rate_probe_plan(std::string_view name) {
  Graph graph{std::string(name)};
  const auto probe = graph.op<OperatorT>("probe");
  graph.partition(std::string{name}).add(probe);
  graph.set_default_clock(graph.add_clock<RealtimeClock>(std::string(name) + "-clock"));
  CompileOptions options;
  options.deployment.bind_thread_placement(std::string{name},
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return test::enable_direct_runtime(compile(graph, std::move(options)));
}

[[nodiscard]] ExecutionPlan make_runtime_fixed_rate_query_plan(std::string_view name) {
  Graph graph{std::string(name)};
  const auto source = graph.op<RuntimeFixedRateQueryProbe>("source");
  const auto sink = graph.op<RuntimeFixedRateQuerySink>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.partition(std::string{name}).add(source).add(sink);
  graph.set_default_clock(graph.add_clock<RealtimeClock>(std::string(name) + "-clock"));
  CompileOptions options;
  options.deployment.bind_thread_placement(std::string{name},
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return test::enable_direct_runtime(compile(graph, std::move(options)));
}

// Starts the connected index fixture from a clock and exposes its first output. Used by
// GraphExecIndexTest.ResolvesDenseVertexIndexWithAbsentNonOperatorSlots.
class GraphExecIndexProbeOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

// Extends the index fixture with one input-triggered callback and output. Used by
// GraphExecIndexTest.ResolvesDenseVertexIndexWithAbsentNonOperatorSlots.
class GraphExecIndexRelayOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{}).input(input, DrainAll{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
};

// Captures one round-trip observation: the data value and SampleMetadata seen
// at either the source (expected) or the sink (observed) side.
// REQ: RUNTIME-012
TEST(RunSessionTime, BoundsFullRangeSourceWaitWithoutSignedOverflow) {
  constexpr std::chrono::nanoseconds kMaximum{1ms};
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();

  EXPECT_EQ(detail::bounded_source_wait_duration(kMin, kMax, kMaximum), kMaximum);
  EXPECT_EQ(detail::bounded_source_wait_duration(kMin, kMin + 999, kMaximum), 999ns);
  EXPECT_EQ(detail::bounded_source_wait_duration(kMax - 999, kMax, kMaximum), 999ns);
  EXPECT_EQ(detail::bounded_source_wait_duration(kMax - kMaximum.count(), kMax, kMaximum),
            kMaximum);
}

// The producer can signal after the controller's last drain but before it begins waiting. The
// durable token must make the subsequent timed wait return without relying on a fallback poll.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, RetainsSignalAcrossTheReadinessScanToWaitBoundary) {
  detail::RuntimeWakeState wake;

  // Key setup: signal after the controller-equivalent drain and before its nonblocking wait.
  EXPECT_EQ(wake.drain(), 0U);
  wake.signal();
  // Key assertion: exactly one durable token crosses the scan-to-wait boundary.
  EXPECT_TRUE(wake.wait_for(0ns));
  EXPECT_FALSE(wake.wait_for(0ns));
}

// A controller discards old wake hints before scanning because exact witness cardinality lives in
// the bounded physical source, not in this coalescing mechanism.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, DrainsEveryCoalescedSignalBeforeTheNextReadinessScan) {
  detail::RuntimeWakeState wake;

  // Key setup: coalesce three producer hints before the next controller scan.
  wake.signal();
  wake.signal();
  wake.signal();
  // Key assertion: drain accounts for every hint and leaves no stale wake token.
  EXPECT_EQ(wake.drain(), 3U);
  EXPECT_FALSE(wake.wait_for(0ns));
}

// Direct-retention progress is a rescan hint rather than a counted witness. One parked partition
// therefore retains at most one durable token while producers continue to advance the bounded
// typed store.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, BoundsCoalescedRescanHintsToOneDurableToken) {
  detail::RuntimeWakeState wake;

  EXPECT_TRUE(wake.signal_coalesced());
  for (std::size_t duplicate = 0U; duplicate < 1024U; ++duplicate) {
    EXPECT_FALSE(wake.signal_coalesced());
  }
  EXPECT_EQ(wake.drain(), 1U);
  EXPECT_FALSE(wake.wait_for(0ns));

  // A new hint after consumption must arm the next scan-to-wait boundary.
  EXPECT_TRUE(wake.signal_coalesced());
  EXPECT_TRUE(wake.wait_for(0ns));
  EXPECT_FALSE(wake.wait_for(0ns));
}

// A producer can be preempted while publishing the semaphore permit as the controller drains its
// previous scan. The durable pending count must keep that race bounded without losing a
// successfully accepted rescan hint.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, ConcurrentCoalescedSignalsAndDrainsRemainBoundedWithoutLosingWakeTokens) {
  detail::RuntimeWakeState wake;
  constexpr std::size_t kProducerCount = 4U;
  constexpr std::size_t kSignalsPerProducer = 4096U;
  std::atomic_bool start{};
  std::atomic_size_t active_producers{kProducerCount};
  std::atomic_size_t accepted{};
  std::array<std::thread, kProducerCount> producers;

  for (std::thread& producer : producers) {
    producer = std::thread([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (std::size_t signal = 0U; signal < kSignalsPerProducer; ++signal) {
        if (wake.signal_coalesced()) {
          accepted.fetch_add(1U, std::memory_order_relaxed);
        }
        std::this_thread::yield();
      }
      active_producers.fetch_sub(1U, std::memory_order_release);
    });
  }

  start.store(true, std::memory_order_release);
  std::size_t consumed = 0U;
  while (active_producers.load(std::memory_order_acquire) != 0U) {
    const std::size_t drained = wake.drain();
    EXPECT_LE(drained, 1U);
    consumed += drained;
    if (wake.wait_for(0ns)) {
      ++consumed;
    }
    std::this_thread::yield();
  }
  for (std::thread& producer : producers) {
    producer.join();
  }

  while (true) {
    const std::size_t drained = wake.drain();
    EXPECT_LE(drained, 1U);
    consumed += drained;
    if (drained == 0U) {
      break;
    }
  }
  EXPECT_FALSE(wake.wait_for(0ns));
  EXPECT_EQ(consumed, accepted.load(std::memory_order_relaxed));
}

// Counted readiness evidence and a coalesced direct-retention rescan can arrive concurrently.
// Every accepted counted token remains distinct while the rescan hint stays bounded to one.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, MixedCountedAndCoalescedSignalsRemainDurableAcrossConcurrentConsumers) {
  detail::RuntimeWakeState wake;
  constexpr std::size_t kCountedSignals = 4096U;
  std::atomic_bool start{};
  std::atomic_size_t accepted_coalesced{};
  std::atomic_size_t consumed{};

  std::thread counted_producer([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (std::size_t signal = 0U; signal < kCountedSignals; ++signal) {
      wake.signal();
      std::this_thread::yield();
    }
  });
  std::thread coalesced_producer([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (std::size_t signal = 0U; signal < kCountedSignals; ++signal) {
      if (wake.signal_coalesced()) {
        accepted_coalesced.fetch_add(1U, std::memory_order_relaxed);
      }
      std::this_thread::yield();
    }
  });
  std::thread waiting_consumer([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    while (consumed.load(std::memory_order_relaxed) <
           kCountedSignals + accepted_coalesced.load(std::memory_order_relaxed)) {
      if (wake.wait_for(0ns)) {
        consumed.fetch_add(1U, std::memory_order_relaxed);
      } else {
        std::this_thread::yield();
      }
    }
  });

  start.store(true, std::memory_order_release);
  counted_producer.join();
  coalesced_producer.join();
  const std::size_t expected = kCountedSignals + accepted_coalesced.load(std::memory_order_relaxed);
  while (consumed.load(std::memory_order_relaxed) < expected) {
    consumed.fetch_add(wake.drain(), std::memory_order_relaxed);
    std::this_thread::yield();
  }
  waiting_consumer.join();

  EXPECT_EQ(consumed.load(std::memory_order_relaxed), expected);
  EXPECT_FALSE(wake.wait_for(0ns));
}

// Cold partition routing maps one compiler-frozen owner to exactly one wake state. Together with
// the Channel callback test that verifies destination/source owner identities, this proves direct
// progress cannot broadcast to an unrelated partition.
// REQ: RUNTIME-041
TEST(RuntimePartitionWakeRouter, TargetsExactlyOneCompilerFrozenOwnerWithoutBroadcast) {
  detail::RuntimeWakeState source;
  detail::RuntimeWakeState sink;
  detail::RuntimeWakeState parked;
  detail::RuntimePartitionWakeRouter router;
  router.reset(4U);

  ASSERT_TRUE(router.bind(VertexId{1U}, &source));
  ASSERT_TRUE(router.bind(VertexId{2U}, &sink));
  ASSERT_TRUE(router.bind(VertexId{3U}, &parked));
  EXPECT_FALSE(router.bind(VertexId{2U}, &parked));
  EXPECT_EQ(router.find(VertexId{}), nullptr);
  EXPECT_EQ(router.find(VertexId{4U}), nullptr);

  ASSERT_NE(router.find(VertexId{2U}), nullptr);
  EXPECT_TRUE(router.find(VertexId{2U})->signal_coalesced());
  EXPECT_FALSE(source.wait_for(0ns));
  EXPECT_TRUE(sink.wait_for(0ns));
  EXPECT_FALSE(parked.wait_for(0ns));

  ASSERT_NE(router.find(VertexId{1U}), nullptr);
  EXPECT_TRUE(router.find(VertexId{1U})->signal_coalesced());
  EXPECT_TRUE(source.wait_for(0ns));
  EXPECT_FALSE(sink.wait_for(0ns));
  EXPECT_FALSE(parked.wait_for(0ns));
}

// The controller drains before every readiness scan, including the common case where no producer
// posted a wake. Empty drains must stay on the atomic fast path rather than probing the semaphore.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, RepeatedEmptyDrainsRemainEmptyWithoutInventingWakeTokens) {
  detail::RuntimeWakeState wake;

  // Key setup: model consecutive controller scans with no intervening physical-source signal.
  for (std::size_t scan = 0U; scan < 1024U; ++scan) {
    // Key assertion: every drain observes the empty pending count and leaves the wait unarmed.
    EXPECT_EQ(wake.drain(), 0U);
  }
  EXPECT_FALSE(wake.wait_for(0ns));
}

// Signaling is valid whether the controller has already entered the kernel wait or is still
// between announcing that wait and blocking; both interleavings consume the same durable token.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, ReleasesAWaitingController) {
  detail::RuntimeWakeState wake;
  std::atomic_bool waiter_started{};
  std::atomic_bool signaled{};
  // Key setup: place one controller-equivalent thread in a bounded kernel wait.
  std::thread waiter([&] {
    waiter_started.store(true, std::memory_order_release);
    signaled.store(wake.wait_for(1s), std::memory_order_release);
  });
  while (!waiter_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  wake.signal();
  waiter.join();

  // Key assertion: a foreign-thread signal releases the waiting controller.
  EXPECT_TRUE(signaled.load(std::memory_order_acquire));
}

// Clock-driven sources still require bounded timed waits when no physical source signals.
// REQ: RUNTIME-025
TEST(RuntimeWakeState, TimesOutWithoutAReadinessSignal) {
  detail::RuntimeWakeState wake;

  // Key assertion: no physical signal produces no synthetic readiness wake.
  EXPECT_FALSE(wake.wait_for(0ns));
}

// REQ: RUNTIME-003
static_assert(!std::is_default_constructible_v<RunSession>);
static_assert(std::is_nothrow_move_constructible_v<RunSession>);
static_assert(!std::is_copy_constructible_v<RunSession>);
static_assert(!std::is_copy_assignable_v<RunSession>);
static_assert(!std::is_move_assignable_v<RunSession>);
static_assert(std::is_nothrow_destructible_v<RunSession>);

// The analyzer intentionally follows the moved-from source into this helper;
// the public contract guarantees that source is inert and remains callable.
// NOLINTBEGIN(clang-analyzer-cplusplus.Move)
void expect_inert_moved_from_session(RunSession& session, OperatorRef plan_ref) {
  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  session.request_stop();
  session.wait();
  EXPECT_EQ(session.operator_ref(plan_ref).error().code, ErrorCode::kStaleOperatorRef);
}
// NOLINTEND(clang-analyzer-cplusplus.Move)

void expect_diagnostic_sequences_equal(std::span<const Diagnostic> expected,
                                       std::span<const Diagnostic> actual) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(actual[index].id, expected[index].id);
    EXPECT_EQ(actual[index].severity, expected[index].severity);
    EXPECT_EQ(actual[index].token, expected[index].token);
    EXPECT_EQ(actual[index].source.graph_generation, expected[index].source.graph_generation);
    EXPECT_EQ(actual[index].source.graph, expected[index].source.graph);
    EXPECT_EQ(actual[index].source.vertex, expected[index].source.vertex);
    EXPECT_EQ(actual[index].source.port, expected[index].source.port);
    EXPECT_EQ(actual[index].service, expected[index].service);
    EXPECT_EQ(actual[index].endpoint, expected[index].endpoint);
    EXPECT_EQ(actual[index].json_pointer, expected[index].json_pointer);
    EXPECT_EQ(actual[index].required, expected[index].required);
    EXPECT_EQ(actual[index].provided, expected[index].provided);
    EXPECT_EQ(actual[index].cause_id, expected[index].cause_id);
    const auto& actual_detail = actual[index].detail;
    const auto& expected_detail = expected[index].detail;
    ASSERT_EQ(actual_detail.has_value(), expected_detail.has_value());
    if (!actual_detail.has_value() || !expected_detail.has_value()) {
      continue;
    }
    const auto& observed = *actual_detail;
    const auto& wanted = *expected_detail;
    EXPECT_EQ(observed.operation, wanted.operation);
    EXPECT_EQ(observed.error.code, wanted.error.code);
    EXPECT_EQ(observed.error.message, wanted.error.message);
    EXPECT_EQ(observed.stage, wanted.stage);
    EXPECT_EQ(observed.phase, wanted.phase);
    EXPECT_EQ(observed.role, wanted.role);
    EXPECT_EQ(observed.transaction_id, wanted.transaction_id);
    EXPECT_EQ(observed.operator_path, wanted.operator_path);
    EXPECT_EQ(observed.port_name, wanted.port_name);
    EXPECT_EQ(observed.primary_diagnostic_index, wanted.primary_diagnostic_index);
    EXPECT_EQ(observed.truncated, wanted.truncated);
    ASSERT_EQ(observed.causes.size(), wanted.causes.size());
    for (std::size_t cause = 0U; cause < wanted.causes.size(); ++cause) {
      EXPECT_EQ(observed.causes[cause].code, wanted.causes[cause].code);
      EXPECT_EQ(observed.causes[cause].message, wanted.causes[cause].message);
    }
  }
}

// Graph topology:
//   a-first --[lifecycle: before(kStart)]--> z-second
// Independent operators with an authored start-dependency used to probe
// parameter-update startup failure with selective stop rollback.
[[nodiscard]] ExecutionPlan make_two_legacy_parameter_startup_plan() {
  Graph graph{"runtime-startup-pre-callback-failure"};
  const auto first = graph.op<test::RuntimeParamProbeOperator>("a-first");
  const auto second = graph.op<test::RuntimeParamProbeOperator>("z-second");
  graph.partition("runtime-first-spin").add(first);
  graph.partition("runtime-second-spin").add(second);
  graph.lifecycle().before(first, second, LifecycleStage::kStart);
  const ClockKey clock = graph.add_clock<test::RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(clock);
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  options.deployment
      .bind_thread_placement("runtime-first-spin",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("runtime-second-spin",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

// Graph topology:
//   z-second --[lifecycle: before(kConfigure)]--> a-first
// Reversed configure dependency verifies the compiler produces canonical topological
// checkpoint order regardless of authored sequence.
[[nodiscard]] ExecutionPlan make_reverse_topological_lifecycle_plan() {
  Graph graph{"runtime-reverse-topological-lifecycle"};
  const auto first = graph.op<test::RuntimeParamProbeOperator>("a-first");
  const auto second = graph.op<test::RuntimeParamProbeOperator>("z-second");
  graph.partition("runtime-first-spin").add(first);
  graph.partition("runtime-second-spin").add(second);
  graph.lifecycle().before(second, first, LifecycleStage::kConfigure);
  const ClockKey clock = graph.add_clock<test::RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(clock);
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  options.deployment
      .bind_thread_placement("runtime-first-spin",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("runtime-second-spin",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

// Captures every published Diagnostic sequence, runtime detail sequence, and lane
// startup evidence in memory for post-run assertion. When fail_=true it throws on
// publish to verify that sink exceptions cannot alter run outcome.
// Used by: BlockingAndAsyncRunsPublishTheExactSameTerminalDiagnostics, and others.
class RecordingDiagnosticSink final : public detail::RuntimeStructuralDiagnosticSink {
 public:
  explicit RecordingDiagnosticSink(bool fail = false) : fail_(fail) {}

  void publish(std::span<const Diagnostic> diagnostics) override {
    if (discard_for_injected_failure()) {
      return;
    }
    const std::lock_guard lock(mutex_);
    sequences_.emplace_back(diagnostics.begin(), diagnostics.end());
  }

  void publish_runtime_details(std::span<const detail::RuntimeDiagnosticDetail> details) override {
    if (discard_for_injected_failure()) {
      return;
    }
    const std::lock_guard lock(mutex_);
    detail_sequences_.emplace_back(details.begin(), details.end());
  }

  void publish_lifecycle_receipts(
      std::span<const detail::RuntimeLifecycleReceiptDiagnostic> receipts) override {
    if (discard_for_injected_failure()) {
      return;
    }
    const std::lock_guard lock(mutex_);
    lifecycle_receipt_sequences_.emplace_back(receipts.begin(), receipts.end());
  }

  void publish_lane_startup_evidence(
      std::span<const detail::ExecutionLaneStartupEvidence> evidence) override {
    if (discard_for_injected_failure()) {
      return;
    }
    const std::lock_guard lock(mutex_);
    lane_sequences_.emplace_back(evidence.begin(), evidence.end());
  }

  void publish_operator_completion_progress(
      std::span<const OperatorCompletionProgressRecord> progress) override {
    if (discard_for_injected_failure()) {
      return;
    }
    const std::lock_guard lock(mutex_);
    completion_progress_sequences_.emplace_back(progress.begin(), progress.end());
  }

  [[nodiscard]] std::vector<std::vector<Diagnostic>> snapshot() const {
    const std::lock_guard lock(mutex_);
    return sequences_;
  }

  [[nodiscard]] std::vector<std::vector<detail::RuntimeDiagnosticDetail>> snapshot_details() const {
    const std::lock_guard lock(mutex_);
    return detail_sequences_;
  }

  [[nodiscard]] std::vector<std::vector<detail::ExecutionLaneStartupEvidence>>
  snapshot_lane_evidence() const {
    const std::lock_guard lock(mutex_);
    return lane_sequences_;
  }

  [[nodiscard]] std::vector<std::vector<detail::RuntimeLifecycleReceiptDiagnostic>>
  snapshot_lifecycle_receipts() const {
    const std::lock_guard lock(mutex_);
    return lifecycle_receipt_sequences_;
  }

  [[nodiscard]] std::vector<std::vector<OperatorCompletionProgressRecord>>
  snapshot_operator_completion_progress() const {
    const std::lock_guard lock(mutex_);
    return completion_progress_sequences_;
  }

 private:
  [[nodiscard]] bool discard_for_injected_failure() const {
    if (!fail_) {
      return false;
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    throw std::runtime_error{"injected diagnostic sink failure"};
#else
    return true;
#endif
  }

  bool fail_{};
  mutable std::mutex mutex_;
  std::vector<std::vector<Diagnostic>> sequences_;
  std::vector<std::vector<detail::RuntimeDiagnosticDetail>> detail_sequences_;
  std::vector<std::vector<detail::RuntimeLifecycleReceiptDiagnostic>> lifecycle_receipt_sequences_;
  std::vector<std::vector<detail::ExecutionLaneStartupEvidence>> lane_sequences_;
  std::vector<std::vector<OperatorCompletionProgressRecord>> completion_progress_sequences_;
};

struct TerminalSummarySnapshot {
  RunGenerationId run_generation{};
  detail::RuntimeTerminalCategory category{detail::RuntimeTerminalCategory::kFailed};
  std::size_t diagnostic_count{};
  DiagnosticId first_diagnostic{};
  DiagnosticSource first_source{};
  std::string first_token;
  std::size_t structured_publication_count{};
};

class RecordingTerminalSummarySink final : public detail::RuntimeTerminalSummarySink {
 public:
  explicit RecordingTerminalSummarySink(
      std::shared_ptr<RecordingDiagnosticSink> structured_sink = {}, bool fail = false)
      : structured_sink_(std::move(structured_sink)), fail_(fail) {}

  void publish(const detail::RuntimeTerminalSummary& summary) override {
    const std::size_t structured_publication_count =
        structured_sink_ != nullptr ? structured_sink_->snapshot().size() : 0U;
    {
      const std::lock_guard lock(mutex_);
      snapshots_.push_back(TerminalSummarySnapshot{
          .run_generation = summary.run_generation,
          .category = summary.category,
          .diagnostic_count = summary.diagnostic_count,
          .first_diagnostic = summary.first_diagnostic,
          .first_source = summary.first_source,
          .first_token = std::string{summary.first_token},
          .structured_publication_count = structured_publication_count,
      });
    }
    if (fail_) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      throw std::runtime_error{"injected terminal summary sink failure"};
#endif
    }
  }

  [[nodiscard]] std::vector<TerminalSummarySnapshot> snapshot() const {
    const std::lock_guard lock(mutex_);
    return snapshots_;
  }

 private:
  std::shared_ptr<RecordingDiagnosticSink> structured_sink_;
  bool fail_{};
  mutable std::mutex mutex_;
  std::vector<TerminalSummarySnapshot> snapshots_;
};

[[nodiscard]] ExecutionPlan make_unselected_process_group_test_plan() {
  Graph graph{"runtime-unselected-process-group"};
  const auto producer = graph.op<test::RuntimeProbeOperator>("producer", 1);
  const auto consumer = graph.op<test::RuntimeProbeOperator>("consumer", 2);
  graph.partition("runtime-spin-producer").add(producer);
  graph.partition("runtime-spin-consumer").add(consumer);
  const ClockKey default_clock =
      graph.add_clock<test::RuntimeProbeClock>("runtime-unselected-process-group-clock");
  graph.set_default_clock(default_clock);

  constexpr std::array producer_selector{detail::GraphObjectSelector{.graph_path = "producer"}};
  constexpr std::array consumer_selector{detail::GraphObjectSelector{.graph_path = "consumer"}};
  CompileOptions options;
  options.deployment
      .bind_thread_placement("runtime-spin-producer",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("runtime-spin-consumer",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("producer",
                         detail::ProcessPlacement{
                             .members = producer_selector,
                             .launch_group = "producer-process",
                         })
      .bind_process_role("consumer",
                         detail::ProcessPlacement{
                             .members = consumer_selector,
                             .launch_group = "consumer-process",
                         });
  return detail::compile_with_process_placement(graph, std::move(options), std::move(placement));
}

[[nodiscard]] ExecutionPlan make_process_group_completion_progress_test_plan() {
  Graph graph{"runtime-process-group-completion-progress"};
  const auto first_source = graph.op<test::RuntimeComputeCallSourceOperator>(
      "a-source", 1U, test::RuntimeComputeCallAction::kNone);
  const auto first_sink = graph.op<test::RuntimeComputeCallSinkOperator>("a-sink");
  const auto second_source = graph.op<test::RuntimeComputeCallSourceOperator>(
      "z-source", 2U, test::RuntimeComputeCallAction::kNone);
  const auto second_sink = graph.op<test::RuntimeComputeCallSinkOperator>("z-sink");
  graph.partition("runtime-completion-first-source").add(first_source);
  graph.partition("runtime-completion-first-sink").add(first_sink);
  graph.partition("runtime-completion-second-source").add(second_source);
  graph.partition("runtime-completion-second-sink").add(second_sink);
  graph.add_flow(first_source->output, first_sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(second_source->output, second_sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock =
      graph.add_clock<RealtimeClock>("runtime-process-group-completion-progress-clock");
  graph.set_default_clock(default_clock);

  constexpr std::array first_selector{
      detail::GraphObjectSelector{.graph_path = "a-source"},
      detail::GraphObjectSelector{.graph_path = "a-sink"},
  };
  constexpr std::array second_selector{
      detail::GraphObjectSelector{.graph_path = "z-source"},
      detail::GraphObjectSelector{.graph_path = "z-sink"},
  };
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("first",
                         detail::ProcessPlacement{
                             .members = first_selector,
                             .launch_group = "first-process",
                         })
      .bind_process_role("second",
                         detail::ProcessPlacement{
                             .members = second_selector,
                             .launch_group = "second-process",
                         });
  return detail::compile_with_process_placement(graph, CompileOptions{}, std::move(placement));
}

[[nodiscard]] ExecutionPlan make_unequal_compute_call_sources_test_plan() {
  Graph graph{"runtime-unequal-compute-call-sources"};
  const auto first_source = graph.op<test::RuntimeComputeCallSourceOperator>(
      "a-source", 1U, test::RuntimeComputeCallAction::kNone);
  const auto first_sink = graph.op<test::RuntimeComputeCallSinkOperator>("a-sink");
  const auto second_source = graph.op<test::RuntimeComputeCallSourceOperator>(
      "z-source", 3U, test::RuntimeComputeCallAction::kNone);
  const auto second_sink = graph.op<test::RuntimeComputeCallSinkOperator>("z-sink");
  graph.partition("runtime-unequal-first-source").add(first_source);
  graph.partition("runtime-unequal-first-sink").add(first_sink);
  graph.partition("runtime-unequal-second-source").add(second_source);
  graph.partition("runtime-unequal-second-sink").add(second_sink);
  graph.add_flow(first_source->output, first_sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(second_source->output, second_sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock =
      graph.add_clock<RealtimeClock>("runtime-unequal-compute-call-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_finite_and_unbounded_source_test_plan() {
  Graph graph{"runtime-finite-and-unbounded-sources"};
  const auto finite = graph.op<test::RuntimeComputeCallSourceOperator>(
      "finite", 1U, test::RuntimeComputeCallAction::kNone);
  const auto sink = graph.op<test::RuntimeComputeCallSinkOperator>("finite-sink");
  const auto unbounded = graph.op<test::RuntimeClockProbeOperator>("unbounded");
  graph.partition("runtime-finite-source").add(finite);
  graph.partition("runtime-finite-sink").add(sink);
  graph.partition("runtime-unbounded-source").add(unbounded);
  graph.add_flow(finite->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock =
      graph.add_clock<RealtimeClock>("runtime-finite-and-unbounded-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

#if GTEST_HAS_DEATH_TEST
// Triggers a self-join death scenario: publish() spins until the bound session
// pointer is set, then calls session->wait() from the runtime's own thread,
// which must terminate under external supervision.
// Used by: RunOwnedThreadSelfJoinTerminatesUnderExternalSupervision
class SelfJoinDiagnosticSink final : public detail::RuntimeStructuralDiagnosticSink {
 public:
  void publish(std::span<const Diagnostic>) override {
    RunSession* session = nullptr;
    while ((session = session_.load(std::memory_order_acquire)) == nullptr) {
      std::this_thread::yield();
    }
    session->wait();
  }

  void bind(RunSession& session) noexcept { session_.store(&session, std::memory_order_release); }

 private:
  std::atomic<RunSession*> session_{};
};
#endif

struct OrdinaryPollStatusProbe {
  std::atomic<std::size_t> poll_calls{};
  std::atomic<std::size_t> incomplete_status_calls{};
};

// One ordinary physical poll target with a single already-admitted witness. The status hook is
// instrumented because it exists only for a freshness recovery after a contended Channel poll;
// consulting it here adds a virtual call to every unrelated physical poll.
class OrdinaryPollStatusBinding final : public detail::PortBinding {
 public:
  OrdinaryPollStatusBinding(const detail::RuntimePortBindingRequest& request,
                            TimePoint start_coordinate,
                            std::shared_ptr<OrdinaryPollStatusProbe> probe)
      : request_(request), probe_(std::move(probe)) {
    event_ = detail::RuntimeEvent{
        .kind = request.kind,
        .source = request.source,
        .port = request.port,
        .source_decision_time = start_coordinate,
        .reference_time = start_coordinate,
        .source_generation = request.source_generation,
        .source_sequence = request.initial_source_sequence + 1U,
    };
  }

  [[nodiscard]] expected<void, Error> poll(TimePoint observed_at) noexcept override {
    if (!observed_at.valid() || observed_at.clock_id != request_.reference_clock) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    probe_->poll_calls.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }

  [[nodiscard]] std::optional<detail::PortBindingPollTarget> poll_target() const noexcept override {
    return detail::PortBindingPollTarget{
        .execution_lane = request_.execution_lane,
        .reference_clock = request_.reference_clock,
    };
  }

  [[nodiscard]] bool last_poll_incomplete() const noexcept override {
    probe_->incomplete_status_calls.fetch_add(1U, std::memory_order_relaxed);
    return false;
  }

  [[nodiscard]] SourceId source() const noexcept override { return request_.source; }

  [[nodiscard]] expected<std::size_t, Error> drain(
      std::span<detail::RuntimeEvent> destination) noexcept override {
    if (!pending_ || destination.empty()) {
      return 0U;
    }
    destination.front() = event_;
    pending_ = false;
    return 1U;
  }

  [[nodiscard]] std::size_t capacity() const noexcept override { return request_.capacity; }
  [[nodiscard]] bool closed() const noexcept override { return true; }

 private:
  detail::RuntimePortBindingRequest request_;
  std::shared_ptr<OrdinaryPollStatusProbe> probe_;
  detail::RuntimeEvent event_;
  bool pending_{true};
};

class OrdinaryPollStatusFactory final : public detail::RuntimePortBindingFactory {
 public:
  explicit OrdinaryPollStatusFactory(std::shared_ptr<OrdinaryPollStatusProbe> probe)
      : probe_(std::move(probe)) {}

  [[nodiscard]] expected<std::unique_ptr<detail::PortBinding>, Error> bind(
      const detail::RuntimePortBindingRequest& request,
      TimePoint start_coordinate) noexcept override {
    if (!request.source || request.capacity == 0U || !request.reference_clock.valid() ||
        request.initial_source_sequence == std::numeric_limits<std::uint64_t>::max() ||
        !start_coordinate.valid() || start_coordinate.clock_id != request.reference_clock) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return std::unique_ptr<detail::PortBinding>{
        std::make_unique<OrdinaryPollStatusBinding>(request, start_coordinate, probe_)};
  }

 private:
  std::shared_ptr<OrdinaryPollStatusProbe> probe_;
};

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// A cold allocation failure cannot retain session-local progress storage, so the shared startup
// terminal explicitly reports the empty sequence even when the selected plan has a completion rule.
// REQ: RUNTIME-050
TEST(RunSessionTest, CompletionPlanPreStorageAllocationFailurePublishesEmptyProgress) {
  test::RuntimeComputeCallChannelProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan baseline = test::make_runtime_compute_call_channel_plan(1U);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);

  const RunSession session = [&] {
    const test::ScopedAllocationFailure fail_next_allocation{0U};
    return run_async(plan);
  }();

  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_STARTUP_FAILED");
  EXPECT_TRUE(session.operator_completion_progress().empty());
  const auto published = diagnostic_sink->snapshot_operator_completion_progress();
  ASSERT_EQ(published.size(), 1U);
  EXPECT_TRUE(published.front().empty());
}

// A failure after progress materialization must retain the selected plan's terminal zero record.
// REQ: RUNTIME-050
TEST(RunSessionTest, CompletionPlanPostProgressStartupFailurePublishesZeroProgress) {
  test::RuntimeComputeCallChannelProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan baseline = test::make_runtime_compute_call_channel_plan(1U);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const ExecutionPlan with_diagnostics =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::
      fail_after_operator_completion_progress_materialization_for_test(with_diagnostics);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_STARTUP_FAILED");
  ASSERT_EQ(session.operator_completion_progress().size(), 1U);
  const auto& rule = plan.operator_completion_rules().front();
  const auto& progress = session.operator_completion_progress().front();
  EXPECT_EQ(progress.owner.id, rule.owner.id);
  EXPECT_EQ(progress.owner.plan_generation, rule.owner.plan_generation);
  EXPECT_TRUE(progress.owner.run_bound());
  EXPECT_EQ(progress.kind, OperatorCompletionRuleKind::kComputeCalls);
  EXPECT_EQ(progress.configured_limit, 1U);
  EXPECT_EQ(progress.successful_compute_calls, 0U);
  EXPECT_FALSE(progress.limit_reached);
  const auto published = diagnostic_sink->snapshot_operator_completion_progress();
  ASSERT_EQ(published.size(), 1U);
  const std::vector<OperatorCompletionProgressRecord> public_progress{
      session.operator_completion_progress().begin(), session.operator_completion_progress().end()};
  EXPECT_EQ(published.front(), public_progress);
}
#endif

// REQ: RUNTIME-001
TEST(RunSessionTest, RejectsEmptyPlanBeforeCreatingSession) {
  const ExecutionPlan plan;
  EXPECT_THROW((void)run_async(plan), CompileError);
  EXPECT_THROW((void)run(plan), CompileError);
}

// Freshness recovery state belongs only to a round that evicted a retained temporal claim. An
// ordinary physical predicate poll must not pay an extra virtual call to inspect that state.
// REQ: RUNTIME-017
TEST(RunSessionTest, OrdinaryPollDoesNotQueryFreshnessRecoveryStatus) {
  test::RuntimeProbeState::reset();
  const auto probe = std::make_shared<OrdinaryPollStatusProbe>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_port_binding_factory(
      test::make_runtime_test_plan(), std::make_shared<OrdinaryPollStatusFactory>(probe));
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  ASSERT_EQ(session.termination(), RunTermination::kCompleted);
  ASSERT_TRUE(session.diagnostics().empty());
  EXPECT_GT(probe->poll_calls.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(probe->incomplete_status_calls.load(std::memory_order_relaxed), 0U);
}

// REQ: RUNTIME-007
// REQ: RUNTIME-013
TEST(RunSessionTest, RejectsStrayDeadlineInventoryInOperatorFreePlan) {
  Graph graph;
  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const ExecutionPlan malformed = detail::RuntimeTestPlanAccess::with_stray_deadline(baseline);
  ASSERT_TRUE(malformed.ok()) << malformed.json();

  RunSession session = run_async(malformed);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_REPLAY_MISMATCH");
  EXPECT_TRUE(session.diagnostics().front().required.empty());
  EXPECT_TRUE(session.diagnostics().front().provided.empty());
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// REQ: RUNTIME-001
// REQ: RUNTIME-004
// REQ: RUNTIME-054
TEST(RunSessionTest, AcceptedPlanAllocationFailureReturnsTerminalSessionInsteadOfThrowing) {
  test::RuntimeProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>(diagnostic_sink);
  const ExecutionPlan baseline = test::enable_direct_runtime(test::make_runtime_test_plan());
  const ExecutionPlan with_diagnostics =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();

  // Key setup: fail the first startup allocation after an otherwise accepted plan is supplied.
  RunSession session = [&] {
    const test::ScopedAllocationFailure fail_next_allocation{0U};
    return run_async(plan);
  }();
  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_STARTUP_FAILED");
  ASSERT_EQ(diagnostic_sink->snapshot().size(), 1U);
  auto summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), 1U);
  EXPECT_NE(summaries.front().run_generation, 0U);
  EXPECT_EQ(summaries.front().first_source.graph_generation, plan.graph().generation());
  EXPECT_EQ(summaries.front().first_token, "RUNTIME_STARTUP_FAILED");
  EXPECT_EQ(summaries.front().structured_publication_count, 1U);

  const auto retained_span = session.diagnostics();
  const std::vector<Diagnostic> retained_diagnostics{retained_span.begin(), retained_span.end()};
  const std::string retained_report = format_diagnostics(retained_span, {});

  // Key assertion: the shared startup-failure state rejects parameter capture without a plan.
  auto captured = session.capture_params();
  ASSERT_FALSE(captured);
  EXPECT_EQ(captured.error().code, ErrorCode::kClosed);

  const ExecutionPlan other_baseline =
      test::enable_direct_runtime(test::make_runtime_test_plan(true));
  const ExecutionPlan other_with_diagnostics =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(other_baseline, diagnostic_sink);
  const ExecutionPlan other_plan = detail::RuntimeTestPlanAccess::with_terminal_summary_sink(
      other_with_diagnostics, summary_sink);
  ASSERT_TRUE(other_plan.ok()) << other_plan.json();
  ASSERT_NE(other_plan.graph().generation(), plan.graph().generation());

  int blocking_result{};
  {
    const test::ScopedAllocationFailure fail_next_allocation{0U};
    blocking_result = run(other_plan, RunSignalPolicy::kCallerManaged);
  }
  EXPECT_NE(blocking_result, 0);
  ASSERT_EQ(diagnostic_sink->snapshot().size(), 2U);
  summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), 2U);
  EXPECT_NE(summaries.back().run_generation, 0U);
  EXPECT_NE(summaries.back().run_generation, summaries.front().run_generation);
  EXPECT_EQ(summaries.front().first_source.graph_generation, plan.graph().generation());
  EXPECT_EQ(summaries.back().first_source.graph_generation, other_plan.graph().generation());
  EXPECT_EQ(summaries.back().first_token, "RUNTIME_STARTUP_FAILED");
  EXPECT_EQ(summaries.back().structured_publication_count, 2U);

  // A later request may report its own identity without rewriting an observed fallback session.
  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(session.diagnostics().data(), retained_span.data());
  expect_diagnostic_sequences_equal(retained_diagnostics, session.diagnostics());
  EXPECT_EQ(format_diagnostics(session.diagnostics(), {}), retained_report);
}
#endif

// REQ: RUNTIME-001
// REQ: RUNTIME-004
// REQ: RUNTIME-035
// REQ: RUNTIME-054
TEST(RunSessionTest, BlockingAndAsyncRunsPublishTheExactSameTerminalDiagnostics) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan baseline = test::make_runtime_test_plan();
  const auto sink = std::make_shared<RecordingDiagnosticSink>();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>(sink);
  const ExecutionPlan with_diagnostics =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);
  ASSERT_EQ(plan.json(), baseline.json());
  ASSERT_EQ(plan.digest(), baseline.digest());

  test::RuntimeProbeState::reset();
  RunSession session = run_async(plan);
  session.wait();
  const std::vector<Diagnostic> async_diagnostics{session.diagnostics().begin(),
                                                  session.diagnostics().end()};
  ASSERT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(async_diagnostics.empty());
  EXPECT_EQ(async_diagnostics.front().token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
  EXPECT_TRUE(async_diagnostics.front().id);

  test::RuntimeProbeState::reset();
  EXPECT_NE(run(plan, RunSignalPolicy::kCallerManaged), 0);
  const auto published = sink->snapshot();
  ASSERT_EQ(published.size(), 2U);
  expect_diagnostic_sequences_equal(async_diagnostics, published[0]);
  expect_diagnostic_sequences_equal(async_diagnostics, published[1]);

  const auto summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), 2U);
  for (std::size_t index = 0U; index < summaries.size(); ++index) {
    const auto& summary = summaries[index];
    EXPECT_EQ(summary.category, detail::RuntimeTerminalCategory::kFailed);
    EXPECT_EQ(summary.diagnostic_count, async_diagnostics.size());
    EXPECT_EQ(summary.first_diagnostic, async_diagnostics.front().id);
    EXPECT_EQ(summary.first_token, async_diagnostics.front().token);
    EXPECT_EQ(summary.structured_publication_count, index + 1U);
  }
}

// REQ: RUNTIME-036
TEST(RunSessionTest, UnselectedMultiProcessPlanFailsBeforeRuntimeMaterialization) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan baseline = make_unselected_process_group_test_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto roles = detail::ProcessPlacementPlanAccess::process_roles(baseline);
  const auto process_placements =
      detail::ProcessPlacementPlanAccess::vertex_process_placements(baseline);
  ASSERT_EQ(roles.size(), 2U);
  ASSERT_EQ(process_placements.size(), 2U);
  ASSERT_NE(process_placements[0].process_group, process_placements[1].process_group);

  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>(diagnostic_sink);
  const ExecutionPlan with_diagnostics =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);

  // Compilation constructs the authored instances. Reset after admission so any runtime replay,
  // lifecycle callback, or compute callback becomes observable.
  test::RuntimeProbeState::reset();
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_TRUE(session.diagnostics().front().id);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_PROCESS_GROUP_SELECTION_REQUIRED");
  EXPECT_EQ(test::RuntimeProbeState::constructed.load(std::memory_order_acquire), 0U);
  EXPECT_FALSE(test::RuntimeProbeState::setup_entered.load(std::memory_order_acquire));
  EXPECT_EQ(test::RuntimeProbeState::active.load(std::memory_order_acquire), 0U);
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());

  const auto published = diagnostic_sink->snapshot();
  ASSERT_EQ(published.size(), 1U);
  expect_diagnostic_sequences_equal(session.diagnostics(), published.front());

  const auto summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), 1U);
  EXPECT_NE(summaries.front().run_generation, 0U);
  EXPECT_EQ(summaries.front().diagnostic_count, 1U);
  EXPECT_EQ(summaries.front().first_diagnostic, session.diagnostics().front().id);
  EXPECT_EQ(summaries.front().first_token, "RUNTIME_PROCESS_GROUP_SELECTION_REQUIRED");
  EXPECT_EQ(summaries.front().structured_publication_count, 1U);
}

// REQ: RUNTIME-047
// REQ: RUNTIME-049
TEST(RunSessionTest, ManagedAsyncAdmissionAbortRetiresBindingBeforeClose) {
  const ExecutionPlan plan = make_unselected_process_group_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  for (std::size_t iteration = 0U; iteration < 20U; ++iteration) {
    auto created = ManagedShutdownSignals::create();
    ASSERT_TRUE(created) << static_cast<unsigned>(created.error().code);
    auto signals = std::move(created).value();

    RunSession session = run_async(plan, signals);
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    ASSERT_TRUE(signals.close()) << "iteration " << iteration;
  }
}

// REQ: RUNTIME-036
TEST(RunSessionTest, InvalidExplicitProcessGroupUsesGenericStartupBoundary) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan baseline = make_unselected_process_group_test_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>(diagnostic_sink);
  const ExecutionPlan with_diagnostics =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);
  constexpr ProcessGroupId kInvalidProcessGroup{std::numeric_limits<std::uint32_t>::max()};

  test::RuntimeProbeState::reset();
  RunSession session = detail::run_async_worker(plan, kInvalidProcessGroup);
  session.wait();

  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_STARTUP_FAILED");
  EXPECT_EQ(test::RuntimeProbeState::constructed.load(std::memory_order_acquire), 0U);
  EXPECT_FALSE(test::RuntimeProbeState::setup_entered.load(std::memory_order_acquire));
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  ASSERT_EQ(diagnostic_sink->snapshot().size(), 1U);
  const auto summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), 1U);
  EXPECT_EQ(summaries.front().run_generation, 0U);
  EXPECT_EQ(summaries.front().first_source.graph_generation, plan.graph().generation());
  EXPECT_EQ(summaries.front().first_token, "RUNTIME_STARTUP_FAILED");
  EXPECT_EQ(summaries.front().structured_publication_count, 1U);
}

// A process-group RunSession owns progress only for Operators reconstructed in that worker. A
// remote rule is absent, not a local zero, and the private terminal sink must publish the same
// canonical local subsequence as the public session accessor.
// REQ: RUNTIME-005
// REQ: RUNTIME-041
// REQ: RUNTIME-050
TEST(RunSessionTest, ProcessGroupWorkersPublishOnlyTheirLocalCompletionProgress) {
  const ExecutionPlan baseline = make_process_group_completion_progress_test_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  ASSERT_EQ(baseline.operator_completion_rules().size(), 2U);
  const auto accepted = detail::ExecutionPlanAccess::retain(baseline);
  ASSERT_NE(accepted, nullptr);

  std::array<ProcessGroupId, 2U> groups;
  for (std::size_t index = 0U; index < baseline.operator_completion_rules().size(); ++index) {
    const auto& rule = baseline.operator_completion_rules()[index];
    const auto factory =
        std::find_if(accepted->executable_factories.operators.begin(),
                     accepted->executable_factories.operators.end(),
                     [&](const auto& record) { return record.operator_id == rule.owner.id; });
    ASSERT_NE(factory, accepted->executable_factories.operators.end());
    groups[index] = detail::process_group_for(*accepted, factory->vertex);
    ASSERT_TRUE(groups[index]);
  }
  ASSERT_NE(groups[0], groups[1]);

  for (std::size_t selected_index = 0U; selected_index < groups.size(); ++selected_index) {
    SCOPED_TRACE(selected_index);
    const std::size_t remote_index = 1U - selected_index;
    const auto& local_rule = baseline.operator_completion_rules()[selected_index];
    const auto& remote_rule = baseline.operator_completion_rules()[remote_index];
    const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);

    test::RuntimeComputeCallChannelProbeState::reset(true);
    RunSession session = detail::run_async_worker(plan, groups[selected_index]);
    const bool compute_entered =
        test::RuntimeComputeCallChannelProbeState::wait_until_source_compute();
    EXPECT_TRUE(compute_entered);
    EXPECT_TRUE(session.operator_completion_progress().empty());
    const auto local_ref = session.operator_ref(local_rule.owner);
    EXPECT_TRUE(local_ref);
    const auto remote_ref = session.operator_ref(remote_rule.owner);
    EXPECT_FALSE(remote_ref);
    if (!remote_ref) {
      EXPECT_EQ(remote_ref.error().code, ErrorCode::kStaleOperatorRef);
    }
    test::RuntimeComputeCallChannelProbeState::release_source_compute();
    session.wait();

    ASSERT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_TRUE(session.diagnostics().empty());
    ASSERT_EQ(session.operator_completion_progress().size(), 1U);
    const auto& progress = session.operator_completion_progress().front();
    EXPECT_EQ(progress.owner.id, local_rule.owner.id);
    EXPECT_EQ(progress.owner.plan_generation, local_rule.owner.plan_generation);
    if (local_ref) {
      EXPECT_EQ(progress.owner, *local_ref);
    }
    EXPECT_EQ(progress.kind, local_rule.kind);
    EXPECT_EQ(progress.configured_limit, local_rule.limit);
    EXPECT_EQ(progress.successful_compute_calls, local_rule.limit);
    EXPECT_TRUE(progress.limit_reached);
    EXPECT_EQ(
        test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
        local_rule.limit);
    EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
              local_rule.limit);

    const auto published = diagnostic_sink->snapshot_operator_completion_progress();
    ASSERT_EQ(published.size(), 1U);
    const std::vector<OperatorCompletionProgressRecord> public_progress{
        session.operator_completion_progress().begin(),
        session.operator_completion_progress().end()};
    EXPECT_EQ(published.front(), public_progress);
  }
}

// A failed multi-process launch without a selected group reconstructs no Operator and therefore
// owns no completion-progress cell, while still invoking the terminal sink exactly once.
// REQ: RUNTIME-036
// REQ: RUNTIME-050
TEST(RunSessionTest, UnselectedMultiProcessPlanPublishesEmptyCompletionProgress) {
  const ExecutionPlan baseline = make_process_group_completion_progress_test_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  ASSERT_EQ(baseline.operator_completion_rules().size(), 2U);
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();

  test::RuntimeComputeCallChannelProbeState::reset();
  RunSession session = run_async(plan);
  session.wait();

  ASSERT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_PROCESS_GROUP_SELECTION_REQUIRED");
  EXPECT_TRUE(session.operator_completion_progress().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            0U);
  const auto published = diagnostic_sink->snapshot_operator_completion_progress();
  ASSERT_EQ(published.size(), 1U);
  EXPECT_TRUE(published.front().empty());
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// REQ: RUNTIME-004
TEST(RunSessionTest, StructuralDiagnosticSinkFailureCannotChangeRunOutcome) {
  test::RuntimeProbeState::reset();
  const auto sink = std::make_shared<RecordingDiagnosticSink>(true);
  const ExecutionPlan baseline = test::make_runtime_test_plan();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, sink);

  test::RuntimeProbeState::reset();
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
  test::RuntimeProbeState::reset();
  EXPECT_NE(run(plan, RunSignalPolicy::kCallerManaged), 0);
  EXPECT_TRUE(sink->snapshot().empty());
}

// The test sink's failure injection must cover every callback so a future Runtime publication
// surface cannot silently bypass the containment path exercised above.
// REQ: RUNTIME-004
TEST(RunSessionTest, StructuralDiagnosticSinkFailureInjectionCoversEveryCallback) {
  RecordingDiagnosticSink sink{true};
  EXPECT_THROW(sink.publish(std::span<const Diagnostic>{}), std::runtime_error);
  EXPECT_THROW(sink.publish_runtime_details(std::span<const detail::RuntimeDiagnosticDetail>{}),
               std::runtime_error);
  EXPECT_THROW(
      sink.publish_lifecycle_receipts(std::span<const detail::RuntimeLifecycleReceiptDiagnostic>{}),
      std::runtime_error);
  EXPECT_THROW(
      sink.publish_lane_startup_evidence(std::span<const detail::ExecutionLaneStartupEvidence>{}),
      std::runtime_error);
  EXPECT_THROW(sink.publish_operator_completion_progress(
                   std::span<const OperatorCompletionProgressRecord>{}),
               std::runtime_error);
}
#endif

// REQ: RUNTIME-035
// REQ: RUNTIME-054
TEST(RunSessionTest, TerminalFailureSummaryFollowsStructuredPublicationExactlyOnce) {
  test::RuntimeProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>(diagnostic_sink);
  const ExecutionPlan baseline = test::make_runtime_test_plan();
  const ExecutionPlan with_diagnostics =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);

  RunSession session = run_async(plan);
  session.wait();

  ASSERT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  const auto summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), 1U);
  const TerminalSummarySnapshot& summary = summaries.front();
  EXPECT_NE(summary.run_generation, 0U);
  EXPECT_EQ(summary.category, detail::RuntimeTerminalCategory::kFailed);
  EXPECT_EQ(summary.diagnostic_count, session.diagnostics().size());
  EXPECT_EQ(summary.first_diagnostic, session.diagnostics().front().id);
  EXPECT_EQ(summary.first_source.graph_generation,
            session.diagnostics().front().source.graph_generation);
  EXPECT_EQ(summary.first_source.graph, session.diagnostics().front().source.graph);
  EXPECT_EQ(summary.first_source.vertex, session.diagnostics().front().source.vertex);
  EXPECT_EQ(summary.first_source.port, session.diagnostics().front().source.port);
  EXPECT_EQ(summary.first_token, session.diagnostics().front().token);
  EXPECT_EQ(summary.structured_publication_count, 1U);
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// REQ: RUNTIME-035
// REQ: RUNTIME-054
TEST(RunSessionTest, TerminalSummarySinkFailureCannotChangeCleanupOrDiagnostics) {
  test::RuntimeProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>(diagnostic_sink, true);
  const ExecutionPlan baseline = test::make_runtime_test_plan();
  const ExecutionPlan with_diagnostics =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
  const auto summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), 1U);
  EXPECT_EQ(summaries.front().structured_publication_count, 1U);
  EXPECT_EQ(test::RuntimeProbeState::active.load(std::memory_order_acquire), 0U);
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
}
#endif

// REQ: RUNTIME-035
// REQ: RUNTIME-054
TEST(RunSessionTest, CompletedRecurringActivationsAttemptNoTerminalFailureSummary) {
  constexpr std::uint32_t kActivationCount = 3U;
  test::RuntimeProbeState::reset();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>();
  const ExecutionPlan direct =
      test::enable_direct_runtime(test::make_runtime_test_plan(), kActivationCount);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(direct, summary_sink);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_TRUE(summary_sink->snapshot().empty());
  EXPECT_EQ(std::ranges::count(test::RuntimeProbeState::snapshot(), 12), kActivationCount);
}

void expect_terminal_compute_call_progress(const RunSession& session, const ExecutionPlan& plan,
                                           std::uint64_t configured_limit,
                                           std::uint64_t successful_compute_calls,
                                           bool limit_reached) {
  ASSERT_EQ(plan.operator_completion_rules().size(), 1U);
  ASSERT_EQ(session.operator_completion_progress().size(), 1U);
  const auto& rule = plan.operator_completion_rules().front();
  const auto& progress = session.operator_completion_progress().front();
  EXPECT_TRUE(rule.owner.plan_bound());
  EXPECT_TRUE(progress.owner.run_bound());
  EXPECT_EQ(progress.owner.id, rule.owner.id);
  EXPECT_EQ(progress.owner.plan_generation, rule.owner.plan_generation);
  EXPECT_EQ(progress.kind, OperatorCompletionRuleKind::kComputeCalls);
  EXPECT_EQ(progress.configured_limit, configured_limit);
  EXPECT_EQ(progress.successful_compute_calls, successful_compute_calls);
  EXPECT_EQ(progress.limit_reached, limit_reached);
}

void expect_operator_completion_progress_sequences_equal(
    std::span<const OperatorCompletionProgressRecord> expected,
    std::span<const OperatorCompletionProgressRecord> actual) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(actual[index].owner, expected[index].owner);
    EXPECT_EQ(actual[index].kind, expected[index].kind);
    EXPECT_EQ(actual[index].configured_limit, expected[index].configured_limit);
    EXPECT_EQ(actual[index].successful_compute_calls, expected[index].successful_compute_calls);
    EXPECT_EQ(actual[index].limit_reached, expected[index].limit_reached);
  }
}

// Three physical Spin witnesses are available, but the source owns a one-call completion rule.
// The test control retains the finite receiver after the limit-reaching retirement so the next
// two stale selections reach the ordinary lane-side final-entry fence without entering compute.
// REQ: RUNTIME-050
TEST(RunSessionTest, OrdinaryDispatchRejectsQueuedWorkBeyondTheFinalComputeEntry) {
  reset_runtime_completion_fence_probe();
  const ExecutionPlan compiled = make_runtime_completion_fence_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto compiled_state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_NE(compiled_state, nullptr);
  ASSERT_EQ(compiled_state->closed_island_activations.size(), 1U);
  const ExecutionPlan finite = detail::RuntimeTestPlanAccess::enable_direct_count(compiled, 3U);
  const ExecutionPlan ordinary =
      detail::RuntimeTestPlanAccess::disable_direct_continuation_elision_for_test(finite);
  const auto final_entry_control =
      std::make_shared<detail::RuntimeOperatorCompletionFinalEntryTestControl>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_operator_completion_final_entry_test_control(
          ordinary, final_entry_control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(RuntimeCompletionFenceSource::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::observed_value.load(std::memory_order_relaxed), 41U);
  // Only the ruled source's three witnesses reach the final-entry control. Its unruled sibling
  // shares the partition wrapper but falls through after the owner lookup without being counted.
  EXPECT_EQ(final_entry_control->checked_count(), 3U);
  EXPECT_EQ(final_entry_control->rejected_count(), 2U);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// A stale retained witness reaches the same lane entry as ordinary admitted work, but the
// Operator-local completion fence precedes deadline sampling. Only the limit-reaching call may
// start deadline evaluation; the two post-limit selections are retired without that side effect.
// REQ: RUNTIME-050
TEST(RunSessionTest, FinalEntryFenceRejectsStaleWorkBeforeDeadlineEvaluation) {
  reset_runtime_completion_fence_probe();
  const ExecutionPlan compiled = make_runtime_completion_fence_plan(true);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_EQ(compiled.deadlines().size(), 1U);
  const auto compiled_state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_NE(compiled_state, nullptr);
  ASSERT_TRUE(compiled_state->closed_island_activations.empty());
  const ExecutionPlan finite = detail::RuntimeTestPlanAccess::enable_direct_count(compiled, 3U);
  const ExecutionPlan ordinary =
      detail::RuntimeTestPlanAccess::disable_direct_continuation_elision_for_test(finite);
  const auto final_entry_control =
      std::make_shared<detail::RuntimeOperatorCompletionFinalEntryTestControl>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_operator_completion_final_entry_test_control(
          ordinary, final_entry_control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(RuntimeCompletionFenceSource::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(final_entry_control->rejected_count(), 2U);
  EXPECT_EQ(final_entry_control->deadline_evaluation_count(), 1U);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// Disabling the two-callback composite retains the direct-continuation receipt path. Hold the
// receipt after the source gate has closed; releasing it may drain the final successor, but the
// final-entry fence must reject both retained source witnesses without entering compute.
// REQ: RUNTIME-050
TEST(RunSessionTest, DirectContinuationDrainsTheFinalSuccessorWithoutReenteringTheSource) {
  reset_runtime_completion_fence_probe();
  const ExecutionPlan compiled = make_runtime_completion_fence_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto compiled_state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_NE(compiled_state, nullptr);
  ASSERT_EQ(compiled_state->closed_island_activations.size(), 1U);
  const ExecutionPlan finite = detail::RuntimeTestPlanAccess::enable_direct_count(compiled, 3U);
  const ExecutionPlan ordinary =
      detail::RuntimeTestPlanAccess::disable_closed_island_composite_for_test(finite);
  const auto final_entry_control =
      std::make_shared<detail::RuntimeOperatorCompletionFinalEntryTestControl>();
  const ExecutionPlan with_final_entry =
      detail::RuntimeTestPlanAccess::with_operator_completion_final_entry_test_control(
          ordinary, final_entry_control);
  const auto control = std::make_shared<detail::RuntimeDirectContinuationTestControl>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_direct_continuation_test_control(
      with_final_entry, control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool receipt_staged = control->wait_until_receipt_staged();
  if (!receipt_staged) {
    session.request_stop();
    control->release();
    session.wait();
  }
  ASSERT_TRUE(receipt_staged);
  EXPECT_TRUE(session.is_running());
  EXPECT_EQ(RuntimeCompletionFenceSource::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::compute_count.load(std::memory_order_acquire), 0U);
  control->release();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(RuntimeCompletionFenceSource::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::observed_value.load(std::memory_order_relaxed), 41U);
  EXPECT_EQ(final_entry_control->checked_count(), 3U);
  EXPECT_EQ(final_entry_control->rejected_count(), 2U);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// The fused closed-island path uses the same final-entry fence. Pause between the source and
// successor transactions after the source's retirement closes its completion gate, then verify
// that the successor alone drains the accepted final publication while two stale source
// selections are rejected at final entry.
// REQ: RUNTIME-050
TEST(RunSessionTest, ClosedIslandDrainsTheFinalSuccessorWithoutReenteringTheSource) {
  reset_runtime_completion_fence_probe();
  const ExecutionPlan compiled = make_runtime_completion_fence_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto compiled_state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_NE(compiled_state, nullptr);
  ASSERT_EQ(compiled_state->closed_island_activations.size(), 1U);
  const ExecutionPlan finite = detail::RuntimeTestPlanAccess::enable_direct_count(compiled, 3U);
  const auto final_entry_control =
      std::make_shared<detail::RuntimeOperatorCompletionFinalEntryTestControl>();
  const ExecutionPlan with_final_entry =
      detail::RuntimeTestPlanAccess::with_operator_completion_final_entry_test_control(
          finite, final_entry_control);
  const auto control = std::make_shared<detail::RuntimeClosedIslandTestControl>();
  control->set_pause_before_successor(true);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_closed_island_test_control(with_final_entry, control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool source_retired = control->wait_until_before_successor();
  if (!source_retired) {
    session.request_stop();
    control->release();
    session.wait();
  }
  ASSERT_TRUE(source_retired);
  EXPECT_TRUE(session.is_running());
  EXPECT_EQ(RuntimeCompletionFenceSource::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::compute_count.load(std::memory_order_acquire), 0U);
  control->release();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(RuntimeCompletionFenceSource::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::compute_count.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(RuntimeCompletionFenceSink::observed_value.load(std::memory_order_relaxed), 41U);
  EXPECT_EQ(final_entry_control->checked_count(), 3U);
  EXPECT_EQ(final_entry_control->rejected_count(), 2U);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// REQ: RUNTIME-050
TEST(RunSessionTest, ComputeCallCompletionStopsTheSourceAtItsLimitWithoutAnExtraCallback) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(kLimit);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            kLimit);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            2U * kLimit);
  expect_terminal_compute_call_progress(session, plan, kLimit, kLimit, true);
}

// REQ: RUNTIME-050
TEST(RunSessionTest, OperatorCompletionProgressRemainsEmptyUntilTerminalOutcomeFreeze) {
  test::RuntimeComputeCallChannelProbeState::reset(true);
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::wait_until_source_compute());
  EXPECT_TRUE(session.operator_completion_progress().empty());
  test::RuntimeComputeCallChannelProbeState::release_source_compute();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// REQ: RUNTIME-050
TEST(RunSessionTest, FreshRunSessionStartsComputeCallProgressAtZero) {
  constexpr std::uint64_t kLimit = 2U;
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(kLimit);
  ASSERT_TRUE(plan.ok()) << plan.json();
  RunGenerationId first_run_generation{};

  for (std::size_t run = 0U; run < 2U; ++run) {
    test::RuntimeComputeCallChannelProbeState::reset();
    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_EQ(
        test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
        kLimit);
    expect_terminal_compute_call_progress(session, plan, kLimit, kLimit, true);
    ASSERT_EQ(session.operator_completion_progress().size(), 1U);
    const RunGenerationId run_generation =
        session.operator_completion_progress().front().owner.run_generation;
    if (run == 0U) {
      first_run_generation = run_generation;
    } else {
      EXPECT_NE(run_generation, first_run_generation);
    }
  }
}

// Completion records are immutable replay identity. Removing or changing the compiled rule fails
// before lifecycle callbacks or compute entry rather than silently changing the session budget.
// REQ: CONTRACT-020
// REQ: RUNTIME-050
TEST(RunSessionTest, CompletionRuleReplayRejectsMissingOrChangedRecordBeforeCompute) {
  const ExecutionPlan baseline = test::make_runtime_compute_call_channel_plan(2U);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  ASSERT_EQ(baseline.operator_completion_rules().size(), 1U);
  const OperatorRef owner = baseline.operator_completion_rules().front().owner;
  const std::array cases{
      std::pair{"missing",
                detail::RuntimeTestPlanAccess::without_operator_completion_rules(baseline)},
      std::pair{"changed",
                detail::RuntimeTestPlanAccess::with_operator_completion_rule(baseline, owner, 3U)},
  };

  for (const auto& [name, plan] : cases) {
    SCOPED_TRACE(name);
    test::RuntimeComputeCallChannelProbeState::reset();
    ASSERT_TRUE(plan.ok()) << plan.json();

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    ASSERT_FALSE(session.diagnostics().empty());
    EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH");
    EXPECT_EQ(
        test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
        0U);
  }
}

// REQ: RUNTIME-050
TEST(RunSessionTest, AcceptedRestartPreservesLogicalOperatorComputeCallProgress) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan baseline = test::make_runtime_compute_call_channel_plan(
      kLimit, test::RuntimeComputeCallAction::kRestartAfterFirst);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_TRUE(
      test::RuntimeComputeCallChannelProbeState::restart_accepted.load(std::memory_order_acquire));
  EXPECT_EQ(
      test::RuntimeComputeCallChannelProbeState::restart_error.load(std::memory_order_relaxed),
      ErrorCode::kSuccess);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            kLimit);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            2U);
  EXPECT_EQ(
      test::RuntimeComputeCallChannelProbeState::sink_compute_count.load(std::memory_order_acquire),
      2U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::first_value.load(std::memory_order_relaxed),
            78U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::second_value.load(std::memory_order_relaxed),
            78U);
  // Task 036 discards queued old-generation ingress when a restart publishes. Task 039 preserves
  // the logical compute-call count and protects the final limit-reaching call's publications; it
  // does not carry the first call's queued value across the generation boundary.
  const auto lifecycle_receipts = diagnostic_sink->snapshot_lifecycle_receipts();
  ASSERT_EQ(lifecycle_receipts.size(), 1U);
  ASSERT_EQ(lifecycle_receipts.front().size(), 1U);
  const auto& receipt = lifecycle_receipts.front().front();
  EXPECT_EQ(receipt.origin, detail::RuntimeLifecycleRequestOrigin::kComputeSelf);
  EXPECT_EQ(receipt.state, LifecycleTransactionState::kPublished);
  EXPECT_EQ(receipt.recovery, detail::RuntimeLifecycleRecoveryDisposition::kPublished);
  EXPECT_EQ(receipt.previous_epoch, 1U);
  EXPECT_EQ(receipt.candidate_epoch, 2U);
  EXPECT_EQ(receipt.effective_epoch, 2U);
  ASSERT_EQ(plan.operator_completion_rules().size(), 1U);
  EXPECT_EQ(receipt.receipt.target.id, plan.operator_completion_rules().front().owner.id);
  EXPECT_EQ(receipt.receipt.target.plan_generation,
            plan.operator_completion_rules().front().owner.plan_generation);
  expect_terminal_compute_call_progress(session, plan, kLimit, kLimit, true);
}

// An accepted self-restart preserves the logical Operator's cumulative progress while Task 036
// retires only the restart-boundary call's queued old-generation publications.
// REQ: RUNTIME-050
TEST(RunSessionTest, AcceptedRestartPreservesThreeHundredComputeCallsAcrossTwoGenerations) {
  constexpr std::uint64_t kSuccessfulComputeCallsBeforeRestart = 120U;
  constexpr std::uint64_t kSuccessfulComputeCallsAfterRestart = 180U;
  constexpr std::uint64_t kLimit =
      kSuccessfulComputeCallsBeforeRestart + kSuccessfulComputeCallsAfterRestart;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan baseline = test::make_runtime_compute_call_channel_plan(
      kLimit,
      test::RuntimeComputeCallAction::kRestartAfterComputeCalls,
      kSuccessfulComputeCallsBeforeRestart);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_TRUE(
      test::RuntimeComputeCallChannelProbeState::restart_accepted.load(std::memory_order_acquire));
  EXPECT_EQ(
      test::RuntimeComputeCallChannelProbeState::restart_error.load(std::memory_order_relaxed),
      ErrorCode::kSuccess);
  const std::uint64_t successful_compute_calls =
      test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire);
  const std::uint64_t successful_compute_calls_before_restart =
      test::RuntimeComputeCallChannelProbeState::successful_compute_calls_before_restart.load(
          std::memory_order_acquire);
  EXPECT_EQ(successful_compute_calls_before_restart, kSuccessfulComputeCallsBeforeRestart);
  EXPECT_EQ(successful_compute_calls - successful_compute_calls_before_restart,
            kSuccessfulComputeCallsAfterRestart);
  EXPECT_EQ(successful_compute_calls, kLimit);
  // Task 036 discards only restart-boundary publications that are still queued when the restart
  // publishes. Either fanout arm may already have been consumed, so zero, one, or both boundary
  // samples may survive; every other sample, including both final-call samples, must drain.
  const std::size_t sink_count =
      test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire);
  EXPECT_GE(sink_count, 2U * (kLimit - 1U));
  EXPECT_LE(sink_count, 2U * kLimit);
  const auto lifecycle_receipts = diagnostic_sink->snapshot_lifecycle_receipts();
  ASSERT_EQ(lifecycle_receipts.size(), 1U);
  ASSERT_EQ(lifecycle_receipts.front().size(), 1U);
  const auto& receipt = lifecycle_receipts.front().front();
  EXPECT_EQ(receipt.origin, detail::RuntimeLifecycleRequestOrigin::kComputeSelf);
  EXPECT_EQ(receipt.state, LifecycleTransactionState::kPublished);
  EXPECT_EQ(receipt.recovery, detail::RuntimeLifecycleRecoveryDisposition::kPublished);
  EXPECT_EQ(receipt.previous_epoch, 1U);
  EXPECT_EQ(receipt.candidate_epoch, 2U);
  EXPECT_EQ(receipt.effective_epoch, 2U);
  ASSERT_EQ(plan.operator_completion_rules().size(), 1U);
  EXPECT_EQ(receipt.receipt.target.id, plan.operator_completion_rules().front().owner.id);
  EXPECT_EQ(receipt.receipt.target.plan_generation,
            plan.operator_completion_rules().front().owner.plan_generation);
  expect_terminal_compute_call_progress(session, plan, kLimit, kLimit, true);
}

// A failed forward restart that restores the old generation does not create a fresh completion
// budget. The first accepted publication survives Task 036 rollback, then the second successful
// call reaches the same logical Operator's original limit.
// REQ: RUNTIME-050
TEST(RunSessionTest, RolledBackRestartPreservesLogicalOperatorComputeCallProgress) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan baseline = test::make_runtime_compute_call_channel_plan(
      kLimit, test::RuntimeComputeCallAction::kRestartRollbackAfterFirst);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, diagnostic_sink);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().severity, DiagnosticSeverity::kWarning);
  const auto& failure_detail = session.diagnostics().front().detail;
  if (!failure_detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  EXPECT_EQ(failure_detail->stage, LifecycleStage::kStart);
  EXPECT_NE(failure_detail->transaction_id, 0U);
  EXPECT_TRUE(
      test::RuntimeComputeCallChannelProbeState::restart_accepted.load(std::memory_order_acquire));
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::restart_forward_start_count.load(
                std::memory_order_acquire),
            1U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::restore_old_start_count.load(
                std::memory_order_acquire),
            1U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            kLimit);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            2U * kLimit);
  const auto lifecycle_receipts = diagnostic_sink->snapshot_lifecycle_receipts();
  ASSERT_EQ(lifecycle_receipts.size(), 1U);
  ASSERT_EQ(lifecycle_receipts.front().size(), 1U);
  const auto& receipt = lifecycle_receipts.front().front();
  EXPECT_EQ(receipt.state, LifecycleTransactionState::kRolledBack);
  EXPECT_EQ(receipt.recovery, detail::RuntimeLifecycleRecoveryDisposition::kRestoredOldEpoch);
  EXPECT_EQ(receipt.previous_epoch, 1U);
  EXPECT_EQ(receipt.candidate_epoch, 2U);
  EXPECT_EQ(receipt.effective_epoch, 1U);
  expect_terminal_compute_call_progress(session, plan, kLimit, kLimit, true);
}

// REQ: RUNTIME-050
TEST(RunSessionTest, ComputeCallCompletionDrainsFinalOutputAndEosBeforeNaturalCompletion) {
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            1U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            2U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::first_value.load(std::memory_order_relaxed),
            77U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::second_value.load(std::memory_order_relaxed),
            77U);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// An ordinary inputful transform must propagate natural closure only after forwarding the source's
// final accepted data. EOS is control, not an extra compute-call witness at either downstream hop.
// REQ: RUNTIME-050
// REQ: CHAN-136
TEST(RunSessionTest, ComputeCallCompletionPropagatesEosThroughOrdinaryThreeHopChain) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_chain_plan(kLimit);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.operator_completion_rules().size(), 1U);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            kLimit);
  EXPECT_EQ(
      test::RuntimeComputeCallChannelProbeState::transform_count.load(std::memory_order_acquire),
      kLimit);
  EXPECT_EQ(
      test::RuntimeComputeCallChannelProbeState::sink_compute_count.load(std::memory_order_acquire),
      kLimit);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            kLimit);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::first_value.load(std::memory_order_relaxed),
            77U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::second_value.load(std::memory_order_relaxed),
            78U);
  expect_terminal_compute_call_progress(session, plan, kLimit, kLimit, true);
}

// Each finite producer closes only its own gate. Natural completion waits for both unequal limits
// and their independent downstream EOS paths to drain.
// REQ: RUNTIME-050
// REQ: CHAN-136
TEST(RunSessionTest, UnequalFiniteSourcesCompleteOnlyAfterEveryOperatorLocalLimitDrains) {
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = make_unequal_compute_call_sources_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.operator_completion_rules().size(), 2U);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            4U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            4U);
  ASSERT_EQ(session.operator_completion_progress().size(), 2U);
  EXPECT_EQ(session.operator_completion_progress()[0].configured_limit, 1U);
  EXPECT_EQ(session.operator_completion_progress()[0].successful_compute_calls, 1U);
  EXPECT_TRUE(session.operator_completion_progress()[0].limit_reached);
  EXPECT_EQ(session.operator_completion_progress()[1].configured_limit, 3U);
  EXPECT_EQ(session.operator_completion_progress()[1].successful_compute_calls, 3U);
  EXPECT_TRUE(session.operator_completion_progress()[1].limit_reached);
}

// A finite producer reaching its limit does not make an unrelated ordinary source finite. The
// session remains live until an explicit Task 036 stop wins terminal arbitration.
// REQ: RUNTIME-050
// REQ: CHAN-136
TEST(RunSessionTest, FiniteProducerDrainDoesNotCompleteAnUnboundedSource) {
  test::RuntimeComputeCallChannelProbeState::reset();
  test::RuntimeClockProbeState::reset();
  const ExecutionPlan plan = make_finite_and_unbounded_source_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.operator_completion_rules().size(), 1U);

  RunSession session = run_async(plan);
  const auto finite_deadline = std::chrono::steady_clock::now() + 5s;
  while (test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire) ==
             0U &&
         std::chrono::steady_clock::now() < finite_deadline) {
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            1U);
  EXPECT_TRUE(test::RuntimeClockProbeState::wait_until_compute_count(1U));
  EXPECT_TRUE(session.is_running());
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// REQ: RUNTIME-050
TEST(RunSessionTest, ZeroComputeCallLimitClosesOutputsWithoutCallingTheSource) {
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(0U);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            0U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            0U);
  expect_terminal_compute_call_progress(session, plan, 0U, 0U, true);
}

// A terminal close that throws at the backend boundary must fail the RunSession without escaping
// the noexcept close path. The successful compute-call budget stays recorded, while terminal
// closure stops at the throwing arm.
// REQ: RUNTIME-050
// REQ: CHAN-136
TEST(RunSessionTest, ThrowingTerminalCloseFailsTheRunSessionAfterCompletionProgressCommits) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan baseline = test::make_runtime_compute_call_channel_plan(1U);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto accepted = detail::ExecutionPlanAccess::retain(baseline);
  ASSERT_TRUE(accepted);
  ASSERT_FALSE(accepted->process_local_channel_publishers.empty());
  const PortId output = accepted->process_local_channel_publishers.front().source;
  EXPECT_TRUE(
      std::ranges::all_of(accepted->process_local_channel_publishers,
                          [output](const auto& publisher) { return publisher.source == output; }));
  const auto control = std::make_shared<ThrowingTerminalCloseChannelBindControl>(output);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_channel_session_bind_test_control(baseline, control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_READINESS_BINDING_UNAVAILABLE"));
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            1U);
  // The test hook intentionally replaces the real transport publishers with scripted endpoints,
  // so this seam verifies completion and failure containment rather than downstream delivery.
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            0U);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
  EXPECT_EQ(control->first_close_count(), 1U);
#else
  GTEST_SKIP() << "exception containment requires language exception support";
#endif
}

// REQ: RUNTIME-050
TEST(RunSessionTest, FailedComputeCallRetainsOnlyPreviouslySuccessfulProgressAndPublication) {
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      3U, test::RuntimeComputeCallAction::kFailOnSecond);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            2U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            2U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::first_value.load(std::memory_order_relaxed),
            77U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::second_value.load(std::memory_order_relaxed),
            77U);
  expect_terminal_compute_call_progress(session, plan, 3U, 1U, false);
}

// An accepted first fanout does not consume completion authority when a later publication in the
// same callback fails and makes the callback return an Error.
// REQ: RUNTIME-050
TEST(RunSessionTest, FailedPublicationConsumesNoComputeCallCompletionUnit) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      kLimit, test::RuntimeComputeCallAction::kFailPublicationOnFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            1U);
  expect_terminal_compute_call_progress(session, plan, kLimit, 0U, false);
}

// The callback returns success and its first publication is accepted, but Channel activation
// retirement fails before the activation cut and completion-cell commit. This is neither a user
// callback failure nor a publication failure, and therefore consumes no completion unit.
// REQ: RUNTIME-050
// REQ: CHAN-136
TEST(RunSessionTest, FailedPostCallbackChannelRetirementConsumesNoCompletionUnit) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan baseline = test::make_runtime_compute_call_channel_plan(kLimit);
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::fail_next_channel_activation_retirement_for_test(baseline);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            1U);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_LIFECYCLE_CALLBACK_FAILED");
  EXPECT_TRUE(std::ranges::none_of(session.diagnostics(), [](const Diagnostic& diagnostic) {
    return diagnostic.token == "RUNTIME_OPERATOR_CALLBACK_FAILED";
  }));
  expect_terminal_compute_call_progress(session, plan, kLimit, 0U, false);
}

// The structural sink receives one terminal completion-progress sequence for every terminal
// outcome, including successful outcomes whose ordinary Diagnostic sequence is empty. Each
// sequence must be nonempty and byte-for-field identical to the public frozen accessor.
// REQ: RUNTIME-050
TEST(RunSessionTest, TerminalSinkPublishesOneNonemptyCompletionProgressSequenceForEveryOutcome) {
  struct TestCase {
    std::string_view name;
    std::uint64_t limit{};
    test::RuntimeComputeCallAction action{test::RuntimeComputeCallAction::kNone};
    RunTermination expected_termination{RunTermination::kCompleted};
    std::uint64_t expected_progress{};
    bool expected_limit_reached{};
  };
  constexpr std::array test_cases{
      TestCase{
          .name = "natural",
          .limit = 1U,
          .action = test::RuntimeComputeCallAction::kNone,
          .expected_termination = RunTermination::kCompleted,
          .expected_progress = 1U,
          .expected_limit_reached = true,
      },
      TestCase{
          .name = "stop",
          .limit = 3U,
          .action = test::RuntimeComputeCallAction::kRequestStopAfterFirst,
          .expected_termination = RunTermination::kStopped,
          .expected_progress = 1U,
          .expected_limit_reached = false,
      },
      TestCase{
          .name = "requested-completion",
          .limit = 3U,
          .action = test::RuntimeComputeCallAction::kRequestCompletionAfterFirst,
          .expected_termination = RunTermination::kRequestedCompleted,
          .expected_progress = 1U,
          .expected_limit_reached = false,
      },
      TestCase{
          .name = "failure",
          .limit = 3U,
          .action = test::RuntimeComputeCallAction::kFailOnSecond,
          .expected_termination = RunTermination::kFailed,
          .expected_progress = 1U,
          .expected_limit_reached = false,
      },
  };

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    test::RuntimeComputeCallChannelProbeState::reset();
    const ExecutionPlan baseline =
        test::make_runtime_compute_call_channel_plan(test_case.limit, test_case.action);
    ASSERT_TRUE(baseline.ok()) << baseline.json();
    const auto sink = std::make_shared<RecordingDiagnosticSink>();
    const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, sink);
    ASSERT_TRUE(plan.ok()) << plan.json();

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), test_case.expected_termination);
    expect_terminal_compute_call_progress(session,
                                          plan,
                                          test_case.limit,
                                          test_case.expected_progress,
                                          test_case.expected_limit_reached);
    const auto published = sink->snapshot_operator_completion_progress();
    ASSERT_EQ(published.size(), 1U);
    ASSERT_FALSE(published.front().empty());
    expect_operator_completion_progress_sequences_equal(session.operator_completion_progress(),
                                                        published.front());
  }
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// A thrown callback after an accepted publication is still a failed compute call and consumes no
// completion authority.
// REQ: RUNTIME-050
TEST(RunSessionTest, ThrownComputeCallConsumesNoCompletionUnit) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      kLimit, test::RuntimeComputeCallAction::kThrowOnFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            1U);
  expect_terminal_compute_call_progress(session, plan, kLimit, 0U, false);
}
#endif

// REQ: RUNTIME-050
TEST(RunSessionTest, StopPublishesPartialComputeCallProgressWithoutClaimingTheLimit) {
  test::RuntimeComputeCallChannelProbeState::reset(true);
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      3U, test::RuntimeComputeCallAction::kRequestStopAfterFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::wait_until_source_compute());
  EXPECT_TRUE(session.operator_completion_progress().empty());
  test::RuntimeComputeCallChannelProbeState::release_source_compute();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  expect_terminal_compute_call_progress(session, plan, 3U, 1U, false);
}

// UINT64_MAX is an exact limit rather than an absent/unbounded sentinel. A stopped session reports
// that unchanged configured value and its bounded partial progress.
// REQ: RUNTIME-050
TEST(RunSessionTest, MaximumComputeCallLimitIsAnOrdinaryExactValue) {
  constexpr std::uint64_t kLimit = std::numeric_limits<std::uint64_t>::max();
  test::RuntimeComputeCallChannelProbeState::reset(true);
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      kLimit, test::RuntimeComputeCallAction::kRequestStopAfterFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::wait_until_source_compute());
  test::RuntimeComputeCallChannelProbeState::release_source_compute();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  expect_terminal_compute_call_progress(session, plan, kLimit, 1U, false);
}

// Seed the executable progress immediately below its exact maximum, then require one and only one
// successful compute call to reach natural completion without increment wrap.
// REQ: RUNTIME-050
TEST(RunSessionTest, MaximumComputeCallBoundaryClosesWithoutWrapOrAnExtraCallback) {
  constexpr std::uint64_t kLimit = std::numeric_limits<std::uint64_t>::max();
  constexpr std::uint64_t kSeed = kLimit - 1U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan source = test::make_runtime_compute_call_channel_plan(kLimit);
  ASSERT_TRUE(source.ok()) << source.json();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_operator_compute_call_progress_seed_for_test(source,
                                                                                       kSeed);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            1U);
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::sink_count.load(std::memory_order_acquire),
            2U);
  expect_terminal_compute_call_progress(session, plan, kLimit, kLimit, true);
}

// A valid readiness source that publishes no event performs no callback entry and consumes no
// completion unit. Explicit stop remains distinct from natural Operator completion.
// REQ: RUNTIME-050
TEST(RunSessionTest, UnfiredNotificationConsumesNoComputeCallCompletionUnit) {
  constexpr std::uint64_t kLimit = 2U;
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_dormant_compute_call_plan(kLimit);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.operator_completion_rules().size(), 1U);

  RunSession session = run_async(plan);
  std::this_thread::sleep_for(25ms);
  EXPECT_TRUE(session.is_running());
  EXPECT_TRUE(session.operator_completion_progress().empty());
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            0U);
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  expect_terminal_compute_call_progress(session, plan, kLimit, 0U, false);
}

// Task 036 stop precedence wins even when the same successful return reaches the natural limit.
// REQ: RUNTIME-046
// REQ: RUNTIME-050
TEST(RunSessionTest, StopWinsTheFinalSuccessfulComputeCallRace) {
  test::RuntimeComputeCallChannelProbeState::reset(true);
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      1U, test::RuntimeComputeCallAction::kRequestStopAfterFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::wait_until_source_compute());
  test::RuntimeComputeCallChannelProbeState::release_source_compute();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// Requested completion similarly outranks natural completion without weakening the final
// successful-call accounting boundary.
// REQ: RUNTIME-046
// REQ: RUNTIME-050
TEST(RunSessionTest, RequestedCompletionWinsTheFinalSuccessfulComputeCallRace) {
  test::RuntimeComputeCallChannelProbeState::reset(true);
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      1U, test::RuntimeComputeCallAction::kRequestCompletionAfterFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::wait_until_source_compute());
  test::RuntimeComputeCallChannelProbeState::release_source_compute();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// A self-restart queued by the limit-reaching call cannot publish a new producer generation after
// the Operator-local completion gate closes. The lifecycle fence fails closed and Runtime failure
// outranks the otherwise-natural completion.
// REQ: RUNTIME-046
// REQ: RUNTIME-050
TEST(RunSessionTest, RestartQueuedAtTheLimitCannotReopenTerminalCompletion) {
  test::RuntimeComputeCallChannelProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      1U, test::RuntimeComputeCallAction::kRestartAfterFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(
      test::RuntimeComputeCallChannelProbeState::restart_accepted.load(std::memory_order_acquire));
  EXPECT_EQ(test::RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
            1U);
  expect_terminal_compute_call_progress(session, plan, 1U, 1U, true);
}

// REQ: RUNTIME-050
TEST(RunSessionTest, RequestedCompletionPublishesPartialProgressWithoutClaimingNaturalDrain) {
  test::RuntimeComputeCallChannelProbeState::reset(true);
  const ExecutionPlan plan = test::make_runtime_compute_call_channel_plan(
      3U, test::RuntimeComputeCallAction::kRequestCompletionAfterFirst);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  EXPECT_TRUE(test::RuntimeComputeCallChannelProbeState::wait_until_source_compute());
  EXPECT_TRUE(session.operator_completion_progress().empty());
  test::RuntimeComputeCallChannelProbeState::release_source_compute();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  expect_terminal_compute_call_progress(session, plan, 3U, 1U, false);
}

// REQ: RUNTIME-035
// REQ: RUNTIME-054
TEST(RunSessionTest, CooperativelyStoppedSessionAttemptsNoTerminalFailureSummary) {
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>();
  const ExecutionPlan direct = test::enable_direct_runtime(test::make_runtime_test_plan());
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(direct, summary_sink);

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_TRUE(summary_sink->snapshot().empty());
}

// REQ: RUNTIME-035
// REQ: RUNTIME-054
TEST(RunSessionTest, RepeatedFailedRunsEachAttemptOneTerminalSummary) {
  constexpr std::size_t kRunCount = 4U;
  test::RuntimeProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingDiagnosticSink>();
  const auto summary_sink = std::make_shared<RecordingTerminalSummarySink>(diagnostic_sink);
  const ExecutionPlan with_diagnostics = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      test::make_runtime_test_plan(), diagnostic_sink);
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_terminal_summary_sink(with_diagnostics, summary_sink);

  for (std::size_t run = 0U; run < kRunCount; ++run) {
    RunSession session = run_async(plan);
    session.wait();
    ASSERT_EQ(session.termination(), RunTermination::kFailed);
    ASSERT_EQ(session.diagnostics().size(), 1U);
  }

  const auto summaries = summary_sink->snapshot();
  ASSERT_EQ(summaries.size(), kRunCount);
  for (std::size_t run = 0U; run < summaries.size(); ++run) {
    EXPECT_EQ(summaries[run].category, detail::RuntimeTerminalCategory::kFailed);
    EXPECT_EQ(summaries[run].diagnostic_count, 1U);
    EXPECT_EQ(summaries[run].structured_publication_count, run + 1U);
  }
}

#if GTEST_HAS_DEATH_TEST
// REQ: RUNTIME-003
TEST(RunSessionDeathTest, RunOwnedThreadSelfJoinTerminatesUnderExternalSupervision) {
  EXPECT_DEATH(
      {
        test::RuntimeProbeState::reset();
        const auto sink = std::make_shared<SelfJoinDiagnosticSink>();
        const ExecutionPlan direct = test::enable_direct_runtime(test::make_runtime_test_plan());
        const ExecutionPlan plan =
            detail::RuntimeTestPlanAccess::with_diagnostic_sink(direct, sink);
        RunSession session = run_async(plan);
        sink->bind(session);
        session.wait();
      },
      "RUNTIME_SELF_JOIN_FORBIDDEN");
}

void expect_runtime_callback_session_action_terminates(test::RuntimeProbeMode mode,
                                                       bool destroy_session) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(mode);
  if (destroy_session) {
    auto* session = new RunSession{run_async(plan)};
    test::RuntimeProbeState::bind_callback_session(session);
    while (!test::RuntimeProbeState::callback_action_returned()) {
      std::this_thread::yield();
    }
  } else {
    RunSession session = run_async(plan);
    test::RuntimeProbeState::bind_callback_session(&session);
    session.wait();
  }
}

// REQ: RUNTIME-003
TEST(RunSessionDeathTest, WaitFromComputeCallbackTerminatesUnderExternalSupervision) {
  EXPECT_DEATH(
      {
        expect_runtime_callback_session_action_terminates(test::RuntimeProbeMode::kWaitFromCompute,
                                                          false);
      },
      "RUNTIME_SELF_JOIN_FORBIDDEN");
}

// REQ: RUNTIME-003
TEST(RunSessionDeathTest, WaitFromDispatcherContextTerminatesUnderExternalSupervision) {
  EXPECT_DEATH(
      {
        test::RuntimeProbeState::reset();
        const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
        RunSession session = run_async(plan);
        const detail::RuntimeCallbackContextScope dispatcher_context{
            detail::RuntimeCallbackContext::kDispatcher};
        session.wait();
      },
      "RUNTIME_SELF_JOIN_FORBIDDEN");
}

// REQ: RUNTIME-003
TEST(RunSessionDeathTest, DestructorFromComputeCallbackTerminatesUnderExternalSupervision) {
  EXPECT_DEATH(
      {
        expect_runtime_callback_session_action_terminates(
            test::RuntimeProbeMode::kDestroyFromCompute, true);
      },
      ".*");
}

// REQ: RUNTIME-003
TEST(RunSessionDeathTest, WaitFromLifecycleCallbackTerminatesUnderExternalSupervision) {
  EXPECT_DEATH(
      {
        expect_runtime_callback_session_action_terminates(
            test::RuntimeProbeMode::kWaitFromLifecycle, false);
      },
      ".*");
}

// REQ: RUNTIME-003
TEST(RunSessionDeathTest, DestructorFromLifecycleCallbackTerminatesUnderExternalSupervision) {
  EXPECT_DEATH(
      {
        expect_runtime_callback_session_action_terminates(
            test::RuntimeProbeMode::kDestroyFromLifecycle, true);
      },
      ".*");
}
#endif

// REQ: RUNTIME-001
// REQ: RUNTIME-011
TEST(RunSessionTest, RunsCompilerAcceptedEmptyGraphAsNoOp) {
  // Graph topology:
  //   (empty)
  Graph graph{"empty-runtime-plan"};
  // Key setup: compile an empty graph with a diagnostic sink attached to its accepted plan.
  const ExecutionPlan compiled = compile(graph);
  const auto sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(compiled, sink);
  ASSERT_TRUE(plan.ok());
  EXPECT_TRUE(plan.graph().vertices().empty());
  EXPECT_TRUE(plan.execution_partitions().empty());
  EXPECT_TRUE(plan.partition_decisions().empty());
  EXPECT_TRUE(plan.execution_lanes().empty());
  EXPECT_TRUE(plan.partition_placements().empty());
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 0U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 0U);
  EXPECT_TRUE(plan.profile_manifest().materialization_obligations.empty());

  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);
  EXPECT_FALSE(detail::has_structural_runtime_topology(*accepted));
  EXPECT_FALSE(detail::has_plan_callback_owners(*accepted));
  auto reconstructed = detail::GraphExec::reconstruct(accepted, RunGenerationId{17U});
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  EXPECT_FALSE((*reconstructed)->has_process_local_callback_owners());
  EXPECT_TRUE((*reconstructed)->parameter_updates().catalog().operators().empty());
  EXPECT_EQ((*reconstructed)->parameter_updates().max_event_cursors(), 0U);

  RunSession session = run_async(plan);
  session.wait();

  // Key assertion: empty topology authenticates and completes without inventing callback owners.
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  const auto lane_evidence = sink->snapshot_lane_evidence();
  ASSERT_EQ(lane_evidence.size(), 1U);
  EXPECT_TRUE(lane_evidence.front().empty());
  EXPECT_EQ(run(plan, RunSignalPolicy::kCallerManaged), 0);
}

TEST(GraphExecIndexTest, ResolvesDenseVertexIndexWithAbsentNonOperatorSlots) {
  constexpr std::size_t kOperatorCount = 32U;
  // Graph:
  //   operator-000 -> operator-001 -> ... -> operator-031 -> exported topic
  Graph graph{"graph-exec-dense-index"};
  const auto first = graph.op<GraphExecIndexProbeOperator>("operator-000");
  auto exported = graph.export_topic<std::uint32_t>("holomq://graph-exec-index-topic");
  auto previous = graph.op<GraphExecIndexRelayOperator>("operator-001");
  graph.add_flow(first->output, previous->input);
  for (std::size_t index = 2U; index < kOperatorCount; ++index) {
    const auto next = graph.op<GraphExecIndexRelayOperator>("operator-" + std::to_string(index));
    graph.add_flow(previous->output, next->input);
    previous = next;
  }
  graph.add_flow(previous->output, exported.input());
  graph.set_default_clock(graph.add_clock<test::RuntimeProbeClock>("graph-exec-index-clock"));
  const ExecutionPlan plan = compile(graph);
  std::string diagnostic_tokens;
  for (const Diagnostic& diagnostic : plan.diagnostics()) {
    diagnostic_tokens += diagnostic.token + "\n";
  }
  ASSERT_TRUE(plan.ok()) << diagnostic_tokens;
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);

  // Key setup: one non-Operator exported-topic vertex creates an absent slot within the otherwise
  // contiguous accepted VertexId range.
  ASSERT_EQ(accepted->graph.vertices().size(), kOperatorCount + 1U);
  EXPECT_TRUE(detail::has_structural_runtime_topology(*accepted));
  EXPECT_TRUE(detail::has_plan_callback_owners(*accepted));
  const auto non_operator = std::ranges::find_if(
      accepted->graph.vertices(),
      [](const VertexRecord& vertex) { return vertex.kind != VertexKind::kOperator; });
  ASSERT_NE(non_operator, accepted->graph.vertices().end());
  auto reconstructed = detail::GraphExec::reconstruct(accepted, RunGenerationId{17U});
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  detail::GraphExec& exec = **reconstructed;
  EXPECT_TRUE(exec.has_process_local_callback_owners());
  ASSERT_EQ(exec.operators().size(), kOperatorCount);

  // Key assertion: first/last identities resolve in O(1), the non-Operator and invalid slots remain
  // absent, and both const and mutable accessors identify the exact reconstructed object.
  const VertexId first_vertex = exec.operators().front().vertex;
  const VertexId last_vertex = exec.operators().back().vertex;
  EXPECT_EQ(exec.receiver_index(first_vertex), 0U);
  EXPECT_EQ(exec.receiver_index(last_vertex), kOperatorCount - 1U);
  EXPECT_EQ(exec.receiver_index(exec.operators().front().operator_id), 0U);
  EXPECT_EQ(exec.receiver_index(exec.operators().back().operator_id), kOperatorCount - 1U);
  EXPECT_FALSE(exec.receiver_index(non_operator->id));
  EXPECT_FALSE(exec.receiver_index(VertexId{}));
  EXPECT_FALSE(exec.receiver_index(VertexId{accepted->graph.vertices().size() + 1U}));
  EXPECT_FALSE(exec.receiver_index(OperatorInstanceId{}));
  EXPECT_FALSE(exec.receiver_index(static_cast<OperatorInstanceId>(kOperatorCount + 1U)));
  EXPECT_EQ(exec.operator_by_vertex(first_vertex), &exec.operators().front());
  EXPECT_EQ(exec.operator_by_id(exec.operators().back().operator_id), &exec.operators().back());
  const detail::GraphExec& const_exec = exec;
  EXPECT_EQ(const_exec.operator_by_vertex(last_vertex), &const_exec.operators().back());
  EXPECT_EQ(const_exec.operator_by_id(const_exec.operators().front().operator_id),
            &const_exec.operators().front());
}

TEST(GraphExecIndexTest, RejectsDuplicateReconstructedVertexIdentityBeforePublication) {
  const ExecutionPlan baseline = test::make_runtime_multi_lane_test_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto accepted = detail::ExecutionPlanAccess::retain(baseline);
  ASSERT_NE(accepted, nullptr);
  ASSERT_GE(accepted->executable_factories.operators.size(), 2U);
  auto malformed = std::make_shared<detail::ExecutionPlanState>(*accepted);

  // Key setup: two executable factories claim the exact same accepted graph vertex.
  malformed->executable_factories.operators[1].vertex =
      malformed->executable_factories.operators.front().vertex;
  const auto reconstructed =
      detail::GraphExec::reconstruct(std::move(malformed), RunGenerationId{17U});

  // Key assertion: the shared accepted-cover authority now rejects the factory-derived partition
  // lookup mismatch before a GraphExec with an overwritten index can escape.
  ASSERT_FALSE(reconstructed);
  EXPECT_EQ(reconstructed.error().code, ErrorCode::kNotSupported);
  EXPECT_EQ(reconstructed.error().message, "RUNTIME_PLACEMENT_BINDING_UNAVAILABLE");
}

// REQ: OPERATOR-050
TEST(RunSessionTest, RejectsEmptyOperatorPathBeforeReconstruction) {
  const ExecutionPlan baseline = test::make_runtime_test_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  const auto accepted = detail::ExecutionPlanAccess::retain(baseline);
  ASSERT_NE(accepted, nullptr);
  auto malformed = std::make_shared<detail::ExecutionPlanState>(*accepted);
  ASSERT_FALSE(malformed->operator_parameters.empty());

  malformed->operator_parameters.front().operator_path.clear();
  test::RuntimeProbeState::reset();
  const auto reconstructed =
      detail::GraphExec::reconstruct(std::move(malformed), RunGenerationId{17U});

  ASSERT_FALSE(reconstructed);
  EXPECT_EQ(reconstructed.error().code, ErrorCode::kFailure);
  EXPECT_EQ(reconstructed.error().message, "RUNTIME_PARAMETER_INVENTORY_MISMATCH");
  EXPECT_EQ(test::RuntimeProbeState::constructed.load(std::memory_order_acquire), 0U);
  EXPECT_FALSE(test::RuntimeProbeState::setup_entered.load(std::memory_order_acquire));
}

// REQ: PARAM-017
TEST(RunSessionTest, RejectsNoncontiguousParameterOwnerInventoriesBeforeReconstruction) {
  using Mutation = detail::RuntimeTestPlanAccess::ParameterInventoryMutation;
  const std::array cases{
      std::pair{test::make_runtime_two_param_test_plan(), Mutation::kInterleavedFields},
      std::pair{test::make_runtime_two_deadline_param_test_plan(),
                Mutation::kInterleavedDerivedTerms},
  };
  for (const auto& [baseline, mutation] : cases) {
    ASSERT_TRUE(baseline.ok()) << baseline.json();
    const ExecutionPlan malformed =
        detail::RuntimeTestPlanAccess::mutate_parameter_inventory(baseline, mutation);
    ASSERT_TRUE(malformed.ok()) << malformed.json();
    const auto accepted = detail::ExecutionPlanAccess::retain(malformed);
    ASSERT_NE(accepted, nullptr);
    const auto reconstructed = detail::GraphExec::reconstruct(accepted, RunGenerationId{17U});
    ASSERT_FALSE(reconstructed);
    EXPECT_EQ(reconstructed.error().code, ErrorCode::kFailure);
    EXPECT_EQ(reconstructed.error().message, "RUNTIME_PARAMETER_INVENTORY_MISMATCH");
  }
}

// REQ: PARAM-017
TEST(RunSessionTest, RejectsZeroParameterOwnersBeforeReconstruction) {
  using Mutation = detail::RuntimeTestPlanAccess::ParameterInventoryMutation;
  const std::array cases{
      std::pair{test::make_runtime_two_param_test_plan(), Mutation::kZeroFieldOwner},
      std::pair{test::make_runtime_two_deadline_param_test_plan(), Mutation::kZeroDerivedTermOwner},
  };
  for (const auto& [baseline, mutation] : cases) {
    ASSERT_TRUE(baseline.ok()) << baseline.json();
    // Key setup: corrupt one otherwise valid owner reference with the reserved zero identity.
    const ExecutionPlan malformed =
        detail::RuntimeTestPlanAccess::mutate_parameter_inventory(baseline, mutation);
    ASSERT_TRUE(malformed.ok()) << malformed.json();
    const auto accepted = detail::ExecutionPlanAccess::retain(malformed);
    ASSERT_NE(accepted, nullptr);

    const auto reconstructed = detail::GraphExec::reconstruct(accepted, RunGenerationId{17U});

    // Key assertion: startup rejects zero before using it to index owner-group state.
    ASSERT_FALSE(reconstructed);
    EXPECT_EQ(reconstructed.error().code, ErrorCode::kFailure);
    EXPECT_EQ(reconstructed.error().message, "RUNTIME_PARAMETER_INVENTORY_MISMATCH");
  }
}

// REQ: RUNTIME-010
// REQ: RUNTIME-011
TEST(RunSessionTest, EmptyGraphTreatsAuthoredClockCatalogAsInertAuthoringEvidence) {
  Graph graph{"empty-runtime-plan-with-clocks"};
  const ClockKey default_clock = graph.add_clock<test::RuntimeProbeClock>("default-clock");
  const ClockKey comparison_clock = graph.add_clock<test::RuntimeProbeClock>("comparison-clock");
  graph.set_default_clock(default_clock);
  graph.add_clock_conversion(default_clock, comparison_clock, AffineClockConversion{});
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.clock_catalog().clocks.size(), 2U);
  ASSERT_EQ(plan.clock_conversions().size(), 1U);
  EXPECT_TRUE(plan.profile_manifest().materialization_obligations.empty());

  test::RuntimeProbeState::reset();
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeProbeState::clocks_constructed.load(std::memory_order_relaxed), 0U);
}

// REQ: RUNTIME-010
TEST(RunSessionTest, EmptyGraphRejectsTamperedInertClockInventoryWithoutConstruction) {
  using InertClockMutation = detail::RuntimeTestPlanAccess::InertClockMutation;
  constexpr std::array mutations{
      InertClockMutation::kPlanClockKey,
      InertClockMutation::kMissingFactory,
      InertClockMutation::kFactoryMode,
      InertClockMutation::kRootDefault,
      InertClockMutation::kConversionGeneration,
      InertClockMutation::kConversionEndpoint,
  };
  Graph graph{"empty-runtime-plan-with-corrupted-clocks"};
  const ClockKey default_clock = graph.add_clock<test::RuntimeProbeClock>("default-clock");
  const ClockKey comparison_clock = graph.add_clock<test::RuntimeProbeClock>("comparison-clock");
  graph.set_default_clock(default_clock);
  graph.add_clock_conversion(default_clock, comparison_clock, AffineClockConversion{});
  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok()) << baseline.json();

  for (const InertClockMutation mutation : mutations) {
    SCOPED_TRACE(static_cast<unsigned>(mutation));
    test::RuntimeProbeState::reset();
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::mutate_inert_clock(baseline, mutation);
    ASSERT_TRUE(plan.ok());

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    EXPECT_EQ(test::RuntimeProbeState::clocks_constructed.load(std::memory_order_relaxed), 0U);
  }
}

// REQ: RUNTIME-011
TEST(RunSessionTest, RejectsMalformedZeroWorkSchedulerAndPartitionFacts) {
  using ZeroWorkMutation = detail::RuntimePartitionTestPlanAccess::ZeroWorkMutation;
  constexpr std::array mutations{
      ZeroWorkMutation::kWorkerCount,
      ZeroWorkMutation::kExecutionLaneCount,
      ZeroWorkMutation::kMaxConcurrentActivations,
      ZeroWorkMutation::kIntraPathParallelism,
      ZeroWorkMutation::kExecutionLane,
      ZeroWorkMutation::kOrphanPartition,
      ZeroWorkMutation::kPartitionDecision,
      ZeroWorkMutation::kScheduleEdge,
      ZeroWorkMutation::kPlacementProbe,
      ZeroWorkMutation::kPartitionPlacement,
  };
  Graph graph{"malformed-empty-runtime-plan"};
  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok());

  for (const ZeroWorkMutation mutation : mutations) {
    SCOPED_TRACE(static_cast<unsigned>(mutation));
    const ExecutionPlan plan =
        detail::RuntimePartitionTestPlanAccess::mutate_zero_work(baseline, mutation);
    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_PLACEMENT_BINDING_UNAVAILABLE"));
  }
}

// REQ: RUNTIME-010
TEST(RunSessionTest, RejectsMalformedZeroWorkRuntimeOwnedInventoriesBeforeReconstruction) {
  using ZeroWorkMutation = detail::RuntimePartitionTestPlanAccess::ZeroWorkMutation;
  struct Case {
    ZeroWorkMutation mutation;
    std::string_view token;
  };
  constexpr std::array cases{
      Case{ZeroWorkMutation::kOperatorFactory, "RUNTIME_OPERATOR_INVENTORY_MISMATCH"},
      Case{ZeroWorkMutation::kMaterializationObligation, "PROFILE_MATERIALIZATION_FAILED"},
  };
  Graph graph{"malformed-empty-runtime-inventory"};
  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok());

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.token);
    const ExecutionPlan plan =
        detail::RuntimePartitionTestPlanAccess::mutate_zero_work(baseline, test_case.mutation);
    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, test_case.token));
  }
}

// REQ: RUNTIME-010
TEST(RunSessionTest, EmptyGraphStillRejectsEveryPackageMismatch) {
  using ProfileMutation = detail::RuntimeTestPlanAccess::ProfileMutation;
  struct Case {
    ProfileMutation mutation;
    std::string_view token;
  };
  constexpr std::array cases{
      Case{ProfileMutation::kPlanSchema, "PROFILE_PLAN_SCHEMA_UNSUPPORTED"},
      Case{ProfileMutation::kProfilePolicy, "PROFILE_POLICY_UNSUPPORTED"},
      Case{ProfileMutation::kRuntimeAbi, "PROFILE_RUNTIME_ABI_INCOMPATIBLE"},
      Case{ProfileMutation::kTarget, "PROFILE_TARGET_INCOMPATIBLE"},
      Case{ProfileMutation::kCapability, "PROFILE_CAPABILITY_UNSATISFIED"},
      Case{ProfileMutation::kEvidence, "PROFILE_EVIDENCE_UNSATISFIED"},
  };
  Graph graph{"empty-runtime-profile-plan"};
  const ExecutionPlan baseline = compile(graph);
  ASSERT_TRUE(baseline.ok());

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.token);
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::mutate_profile(baseline, test_case.mutation);
    ASSERT_TRUE(plan.ok());

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, test_case.token));
  }
}

// REQ: RUNTIME-001
// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-041
TEST(RunSessionTest, RunsAfterSourceGraphIsDestroyedAndPreservesPartitionLocalOrder) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan(true));
  ASSERT_TRUE(plan.ok()) << plan.json();
  test::RuntimeProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  const auto events = test::RuntimeProbeState::snapshot();
  auto sorted_events = events;
  std::sort(sorted_events.begin(), sorted_events.end());
  EXPECT_EQ(sorted_events, (std::vector<std::int32_t>{11, 12, 13, 21, 22, 23}));
  for (const std::int32_t identity : {1, 2}) {
    const auto start = std::find(events.begin(), events.end(), identity * 10 + 1);
    const auto compute = std::find(events.begin(), events.end(), identity * 10 + 2);
    const auto stop = std::find(events.begin(), events.end(), identity * 10 + 3);
    ASSERT_NE(start, events.end());
    ASSERT_NE(compute, events.end());
    ASSERT_NE(stop, events.end());
    EXPECT_LT(start, compute);
    EXPECT_LT(compute, stop);
  }
  const auto max_active = test::RuntimeProbeState::max_active.load(std::memory_order_relaxed);
  EXPECT_GE(max_active, 1U);
  EXPECT_LE(max_active, 2U);
}

// REQ: RUNTIME-004
// REQ: LIFECYCLE-004
TEST(RunSessionTest, PreservesAuthoredStopDependenciesWhileReversingIndependentOperators) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan =
      test::enable_direct_runtime(test::make_runtime_stop_dependency_test_plan());
  ASSERT_TRUE(plan.ok()) << plan.json();
  test::RuntimeProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  const auto events = test::RuntimeProbeState::snapshot();
  auto sorted_events = events;
  std::sort(sorted_events.begin(), sorted_events.end());
  EXPECT_EQ(sorted_events, (std::vector<std::int32_t>{11, 12, 13, 21, 22, 23}));
  for (const std::int32_t identity : {1, 2}) {
    const auto start = std::find(events.begin(), events.end(), identity * 10 + 1);
    const auto compute = std::find(events.begin(), events.end(), identity * 10 + 2);
    const auto stop = std::find(events.begin(), events.end(), identity * 10 + 3);
    ASSERT_NE(start, events.end());
    ASSERT_NE(compute, events.end());
    ASSERT_NE(stop, events.end());
    EXPECT_LT(start, compute);
    EXPECT_LT(compute, stop);
  }
  const auto first_stop = std::find(events.begin(), events.end(), 13);
  const auto dependent_stop = std::find(events.begin(), events.end(), 23);
  ASSERT_NE(first_stop, events.end());
  ASSERT_NE(dependent_stop, events.end());
  EXPECT_LT(first_stop, dependent_stop);
}

// REQ: RUNTIME-001
// REQ: RUNTIME-002
TEST(RunSessionTest, BlockingRunCompletesThroughExplicitBackendWithBoundClockContext) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset();

  EXPECT_EQ(run(plan, RunSignalPolicy::kCallerManaged), 0);
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
  EXPECT_EQ(test::RuntimeProbeState::constructed.load(std::memory_order_relaxed), 1U);
  EXPECT_TRUE(
      test::RuntimeProbeState::observed_clock_read_succeeded.load(std::memory_order_relaxed));
  const auto clock_id = test::RuntimeProbeState::observed_clock_id.load(std::memory_order_relaxed);
  EXPECT_NE(clock_id, 0U);
  EXPECT_EQ(test::RuntimeProbeState::observed_activation_clock_id.load(std::memory_order_relaxed),
            clock_id);
  EXPECT_EQ(test::RuntimeProbeState::observed_activation_time_ns.load(std::memory_order_relaxed),
            0);
  EXPECT_EQ(test::RuntimeProbeState::observed_clock_time_ns.load(std::memory_order_relaxed), 0);
}

// Source-tree performance counters are absent from ordinary compiled plans and, when explicitly
// attached, count the same accepted candidate through freeze, selection, compute, and retirement.
// REQ: RUNTIME-011
TEST(RunSessionTest, PrivatePhaseCountersTrackOneCompleteActivationPerDirectWitness) {
  constexpr std::uint32_t kActivationCount = 3U;
  constexpr std::array expected_points{
      detail::RuntimePhaseTracePoint::kCandidateFrozen,
      detail::RuntimePhaseTracePoint::kCandidateSelected,
      detail::RuntimePhaseTracePoint::kInputSelectionPrepared,
      detail::RuntimePhaseTracePoint::kActivationPrepared,
      detail::RuntimePhaseTracePoint::kComputeDispatchStarted,
      detail::RuntimePhaseTracePoint::kComputeCallbackEntered,
      detail::RuntimePhaseTracePoint::kComputeCallbackReturned,
      detail::RuntimePhaseTracePoint::kComputeDispatchReturned,
      detail::RuntimePhaseTracePoint::kActivationRetirementCompleted,
      detail::RuntimePhaseTracePoint::kActivationCutCompleted,
  };
  test::RuntimeProbeState::reset();
  const ExecutionPlan baseline = test::make_runtime_test_plan();
  const auto baseline_state = detail::ExecutionPlanAccess::retain(baseline);
  ASSERT_TRUE(baseline_state);
  ASSERT_EQ(baseline_state->executable_factories.phase_counters, nullptr);

  std::array<detail::RuntimePhaseTraceRecord, kActivationCount * expected_points.size()>
      trace_storage;
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>(std::span{trace_storage});
  const ExecutionPlan direct = test::enable_direct_runtime(baseline, kActivationCount);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(direct, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  // Key setup: run three finite direct witnesses through the production coordinator and lane.
  RunSession session = run_async(plan);
  session.wait();
  ASSERT_EQ(session.termination(), RunTermination::kCompleted);
  const auto snapshot = counters->snapshot();
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };

  // Key assertion: every accepted witness has exactly one frozen candidate, selected compute,
  // and completed activation; the direct backend needs neither Channel polling nor a parked wait.
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kCandidatesFrozen), kActivationCount);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kCandidatesSelected), kActivationCount);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kComputeDispatches), kActivationCount);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kActivationsCompleted), kActivationCount);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kReadinessPollDispatches), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kWaitCalls), 0U);

  // Key assertion: opt-in trace storage preserves exact controller checkpoint order without
  // inventing a readiness poll for this direct source.
  const auto trace = counters->trace_records();
  ASSERT_EQ(trace.size(), trace_storage.size());
  EXPECT_EQ(counters->trace_dropped(), 0U);
  for (std::size_t activation = 0U; activation < kActivationCount; ++activation) {
    const auto group = trace.subspan(activation * expected_points.size(), expected_points.size());
    const SourceId source = group.front().source;
    const std::uint64_t source_sequence = group.front().source_sequence;
    EXPECT_TRUE(source);
    EXPECT_NE(source_sequence, 0U);
    for (std::size_t point = 0U; point < group.size(); ++point) {
      EXPECT_EQ(group[point].point, expected_points[point]);
      EXPECT_EQ(group[point].receiver_index, 0U);
      EXPECT_EQ(group[point].source, source);
      EXPECT_EQ(group[point].source_sequence, source_sequence);
      if (point != 0U) {
        EXPECT_LE(group[point - 1U].timestamp_ns, group[point].timestamp_ns);
      }
    }
  }

  const auto usage = counters->thread_usage_snapshot();
  const auto& controller_usage = usage[detail::RuntimePhaseCounters::thread_usage_index(
      detail::RuntimeThreadUsageRole::kController)];
  const auto& lane_usage = usage[detail::RuntimePhaseCounters::thread_usage_index(
      detail::RuntimeThreadUsageRole::kExecutionLane)];
  EXPECT_EQ(controller_usage.supported, detail::runtime_thread_usage_detail::kSupported);
  EXPECT_EQ(controller_usage.samples, detail::runtime_thread_usage_detail::kSupported ? 1U : 0U);
  // The singleton partition is controller-owned, so opt-in accounting must not fabricate a
  // separate execution-lane worker sample for its inline callbacks.
  EXPECT_EQ(lane_usage.samples, 0U);
}

// A source-tree-only seed places the run-owned allocator at its final valid sequence. The first
// witness executes with that identity; the next accepted witness must stop the session with the
// exact bounded diagnostic rather than wrap, reuse a Tag, or collapse the cause into generic
// readiness failure.
// REQ: RUNTIME-007
TEST(RunSessionTest, TrueTagSequenceExhaustionPublishesExactTerminalDiagnostic) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan direct = test::enable_direct_runtime(test::make_runtime_test_plan(), 2U);
  const ExecutionPlan seeded = detail::RuntimeTestPlanAccess::with_initial_tag_sequence_for_test(
      direct, std::numeric_limits<std::uint64_t>::max());
  const auto sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(seeded, sink);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_TAG_CAPACITY_EXHAUSTED");
  EXPECT_EQ(session.diagnostics().front().service, "candidate_freeze");
  const Diagnostic& diagnostic = session.diagnostics().front();
  EXPECT_FALSE(diagnostic.source.vertex);
  if (!diagnostic.detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  const auto& detail = *diagnostic.detail;
  EXPECT_EQ(detail.error.code, ErrorCode::kResourceExhausted);
  EXPECT_TRUE(detail.error.message.empty());
  EXPECT_TRUE(detail.causes.empty());
  EXPECT_FALSE(detail.truncated);
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));

  const auto published = sink->snapshot_details();
  ASSERT_EQ(published.size(), 1U);
  ASSERT_EQ(published.front().size(), 1U);
  EXPECT_TRUE(published.front().front().present);
  EXPECT_EQ(published.front().front().error.code, ErrorCode::kResourceExhausted);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-011
TEST(RunSessionTest, RoutesLifecycleAndComputeThroughAcceptedPartitionLane) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan compiled = test::make_runtime_role_test_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_EQ(compiled.execution_lanes().size(), 1U);
  ASSERT_EQ(compiled.scheduler_plan().execution_lane_count, 1U);
  const ExecutionPlan plan = test::enable_direct_runtime(compiled);
  test::RuntimeProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted)
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
  const auto callback_threads = test::RuntimeProbeState::snapshot_threads();
  ASSERT_EQ(callback_threads.size(), 3U);
  EXPECT_NE(callback_threads.front(), std::thread::id{});
  EXPECT_TRUE(std::all_of(callback_threads.begin(), callback_threads.end(), [&](const auto& id) {
    return id == callback_threads.front();
  }));
}

// REQ: RUNTIME-002
TEST(RunSessionTest, ReconstructsEveryPlannedClockCapability) {
  const auto expect_capabilities = [](const ExecutionPlan& source) {
    ASSERT_EQ(source.clock_catalog().clocks.size(), 1U);
    const ClockPlanRecord planned = source.clock_catalog().clocks.front();
    const ExecutionPlan plan = test::enable_direct_runtime(source);
    test::RuntimeProbeState::reset();

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_EQ(test::RuntimeProbeState::observed_clock_can_read_now.load(std::memory_order_relaxed),
              planned.can_read_now);
    EXPECT_EQ(
        test::RuntimeProbeState::observed_clock_can_wait_until.load(std::memory_order_relaxed),
        planned.can_wait_until);
    EXPECT_EQ(test::RuntimeProbeState::observed_clock_steady.load(std::memory_order_relaxed),
              planned.steady);
  };

  expect_capabilities(test::make_runtime_test_plan_with_clock<RealtimeClock>(false));
  expect_capabilities(test::make_runtime_test_plan_with_clock<ManualClock>(false, std::int64_t{0}));
  expect_capabilities(
      test::make_runtime_test_plan_with_clock<SyntheticClock>(false, std::int64_t{0}));
  expect_capabilities(test::make_runtime_test_plan());
  EXPECT_FALSE(test::RuntimeProbeState::observed_clock_steady.load(std::memory_order_relaxed));
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-027
TEST(RunSessionTest, RejectsReplayedContractMismatchBeforeLifecycleCallbacks) {
  struct CallbackProbeReset {
    CallbackProbeReset() {
      test::RuntimeParamProbeState::reset();
      test::RuntimeDeadlineParamProbeState::reset();
    }
    ~CallbackProbeReset() {
      test::RuntimeParamProbeState::reset();
      test::RuntimeDeadlineParamProbeState::reset();
    }
  };
  const CallbackProbeReset callback_probe_reset{};
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kMismatchedContract);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH");
  {
    const std::lock_guard lock(test::RuntimeParamProbeState::mutex);
    EXPECT_FALSE(test::RuntimeParamProbeState::start_entered);
    EXPECT_FALSE(test::RuntimeParamProbeState::stop_entered);
  }
  EXPECT_EQ(test::RuntimeDeadlineParamProbeState::compute_count.load(std::memory_order_acquire),
            0U);
}

// REQ: RUNTIME-004
// REQ: LIFECYCLE-002
TEST(RunSessionTest, RejectsMissingAdvancedOptionalLifecycleCallbacksDuringReplay) {
  struct ReplayReset {
    ~ReplayReset() {
      OptionalLifecycleReplayProbe::omit_callbacks.store(false, std::memory_order_relaxed);
    }
  };
  const ReplayReset reset{};

  Graph graph{"optional-lifecycle-replay"};
  static_cast<void>(graph.op<OptionalLifecycleReplayProbe>("probe"));
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  OptionalLifecycleReplayProbe::omit_callbacks.store(true, std::memory_order_relaxed);
  auto reconstructed = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                      RunGenerationId{29U});

  ASSERT_FALSE(reconstructed);
  EXPECT_EQ(reconstructed.error().message, "RUNTIME_OPERATOR_REPLAY_MISMATCH");
}

// REQ: OPERATOR-051
TEST(RunSessionTest, RejectsCudaGraphEligibilityReplayMismatchBeforeCallbacks) {
  struct EligibilityReset {
    ~EligibilityReset() {
      CudaGraphEligibilityReplayProbe::eligible.store(true, std::memory_order_relaxed);
    }
  };
  const EligibilityReset reset{};

  CudaGraphEligibilityReplayProbe::eligible.store(true, std::memory_order_relaxed);
  Graph graph{"cuda-graph-eligibility-runtime-replay"};
  (void)graph.op<CudaGraphEligibilityReplayProbe>("probe");
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-test-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  CudaGraphEligibilityReplayProbe::eligible.store(false, std::memory_order_relaxed);
  detail::RuntimeReplayMismatchDetail mismatch;
  auto reconstructed = detail::GraphExec::reconstruct(
      detail::ExecutionPlanAccess::retain(plan), RunGenerationId{30U}, {}, &mismatch);

  ASSERT_FALSE(reconstructed);
  EXPECT_EQ(reconstructed.error().message, "RUNTIME_OPERATOR_REPLAY_MISMATCH");
  ASSERT_TRUE(mismatch.present);
  EXPECT_EQ(mismatch.category, detail::RuntimeReplayMismatchCategory::kDeclaration);
}

// A runtime-mutable deadline retains its independently frozen violation action. Replaying the
// same Parameter dependency must not allow a different action to bypass cold reconstruction.
// REQ: RUNTIME-004
// REQ: RUNTIME-027
// REQ: PARAM-017
TEST(RunSessionTest, RejectsParameterDerivedDeadlineViolationReplayMismatchBeforeCallbacks) {
  struct ReplayViolationReset {
    ~ReplayViolationReset() { test::RuntimeDeadlineParamProbeState::reset(); }
  };
  const ReplayViolationReset reset{};

  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();
  test::RuntimeDeadlineParamProbeState::reset();
  const ExecutionPlan compiled = test::make_runtime_deadline_param_test_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(compiled);

  test::RuntimeDeadlineParamProbeState::replay_violation.store(OnViolation::kSafeDefault,
                                                               std::memory_order_relaxed);
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH");
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
}

// REQ: RUNTIME-004
// REQ: RUNTIME-016
TEST(RunSessionTest, ReconstructsPhaseSeparatedInputPredicateWithoutFakeReadinessSource) {
  const ExecutionPlan plan = test::make_runtime_input_predicate_replay_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.predicate_triggers().size(), 1U);
  ASSERT_EQ(plan.temporal_triggers().size(), 2U);
  const auto predicate_trigger = std::find_if(
      plan.temporal_triggers().begin(), plan.temporal_triggers().end(), [](const auto& trigger) {
        return trigger.kind == TemporalRecordTriggerKind::kPredicate;
      });
  ASSERT_NE(predicate_trigger, plan.temporal_triggers().end());
  ASSERT_TRUE(static_cast<bool>(predicate_trigger->readiness_source));
  EXPECT_TRUE(std::none_of(plan.readiness().begin(), plan.readiness().end(), [&](const auto& row) {
    return row.source_id == predicate_trigger->readiness_source;
  }));

  auto reconstructed = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                      RunGenerationId{17U});
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  EXPECT_EQ((*reconstructed)->operators().size(), 2U);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-015
TEST(RunSessionTest, ReconstructsEveryInputBackedTriggerWithoutFakeReadinessRecords) {
  constexpr std::array trigger_kinds{
      test::RuntimeInputTriggerReplayKind::kAny,
      test::RuntimeInputTriggerReplayKind::kEach,
      test::RuntimeInputTriggerReplayKind::kAligned,
      test::RuntimeInputTriggerReplayKind::kWindow,
  };
  for (const auto kind : trigger_kinds) {
    SCOPED_TRACE(static_cast<unsigned>(kind));
    const ExecutionPlan plan = test::make_runtime_input_trigger_replay_plan(kind);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.temporal_triggers().size(), 2U);
    const auto trigger = std::find_if(
        plan.temporal_triggers().begin(), plan.temporal_triggers().end(), [](const auto& row) {
          return row.kind != TemporalRecordTriggerKind::kClock;
        });
    ASSERT_NE(trigger, plan.temporal_triggers().end());
    ASSERT_TRUE(static_cast<bool>(trigger->readiness_source));
    EXPECT_TRUE(
        std::none_of(plan.readiness().begin(), plan.readiness().end(), [&](const auto& row) {
          return row.source_id == trigger->readiness_source;
        }));

    auto reconstructed = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                        RunGenerationId{18U});
    ASSERT_TRUE(reconstructed) << reconstructed.error().message;
    EXPECT_EQ((*reconstructed)->operators().size(), 2U);
  }
}

// Clock scheduling is independent of input arrival, but its callback still consumes the complete
// frozen selector, missing-policy, and deadline plan. Replay must preserve both dimensions.
// REQ: RUNTIME-004
// REQ: RUNTIME-015
// REQ: RUNTIME-016
TEST(RunSessionTest, ReconstructsClockDrivenOperatorsWithConnectedInputs) {
  constexpr std::array kinds{
      test::RuntimeClockInputReplayKind::kClock,
      test::RuntimeClockInputReplayKind::kPredicate,
  };
  constexpr std::array profiles{Profile::kStandard, Profile::kResearch};
  for (const auto profile : profiles) {
    for (const auto kind : kinds) {
      SCOPED_TRACE(static_cast<unsigned>(profile));
      SCOPED_TRACE(static_cast<unsigned>(kind));
      const ExecutionPlan plan = test::make_runtime_clock_input_replay_plan(kind, profile);
      ASSERT_TRUE(plan.ok()) << plan.json();
      ASSERT_EQ(plan.input_selections().size(), 2U);
      ASSERT_EQ(plan.deadlines().size(), 1U);
      ASSERT_EQ(plan.clock_triggers().size(), 2U);

      auto reconstructed = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                          RunGenerationId{19U});
      ASSERT_TRUE(reconstructed) << reconstructed.error().message;
      EXPECT_EQ((*reconstructed)->operators().size(), 2U);
    }
  }
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-016
// REQ: RUNTIME-027
TEST(RunSessionTest, ClockReplayMismatchCarriesStableOperatorAndFingerprintContext) {
  struct ReplayModeReset {
    ~ReplayModeReset() {
      test::RuntimeClockInputReplayOperator::mismatch_ordinal.store(0U, std::memory_order_relaxed);
    }
  };
  const ReplayModeReset reset{};

  const ExecutionPlan baseline = test::make_runtime_dual_clock_input_replay_plan();
  ASSERT_TRUE(baseline.ok()) << baseline.json();

  const auto reconstruct_mismatch = [&](std::uint32_t ordinal) {
    test::RuntimeClockInputReplayOperator::mismatch_ordinal.store(ordinal,
                                                                  std::memory_order_relaxed);
    detail::RuntimeReplayMismatchDetail mismatch;
    auto reconstructed = detail::GraphExec::reconstruct(
        detail::ExecutionPlanAccess::retain(baseline), RunGenerationId{19U}, {}, &mismatch);
    EXPECT_FALSE(reconstructed);
    if (!reconstructed) {
      EXPECT_EQ(reconstructed.error().message, "RUNTIME_OPERATOR_REPLAY_MISMATCH");
    }
    return mismatch;
  };

  const detail::RuntimeReplayMismatchDetail ordinary = reconstruct_mismatch(1U);
  const detail::RuntimeReplayMismatchDetail predicate = reconstruct_mismatch(2U);
  ASSERT_TRUE(ordinary.present);
  ASSERT_TRUE(predicate.present);
  EXPECT_EQ(ordinary.category, detail::RuntimeReplayMismatchCategory::kTemporalContract);
  EXPECT_EQ(predicate.category, detail::RuntimeReplayMismatchCategory::kTemporalContract);
  EXPECT_EQ(ordinary.factory_slot, 1U);
  EXPECT_EQ(predicate.factory_slot, 2U);
  EXPECT_EQ(ordinary.operator_id, 1U);
  EXPECT_EQ(predicate.operator_id, 2U);
  EXPECT_NE(ordinary.vertex, predicate.vertex);
  EXPECT_NE(ordinary.expected, predicate.expected);
  EXPECT_NE(ordinary.observed, predicate.observed);

  const auto sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(baseline, sink);
  test::RuntimeClockInputReplayOperator::mismatch_ordinal.store(1U, std::memory_order_relaxed);
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  const Diagnostic& diagnostic = session.diagnostics().front();
  EXPECT_EQ(diagnostic.token, "RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH");
  EXPECT_EQ(diagnostic.source.vertex, ordinary.vertex);
  EXPECT_EQ(diagnostic.required.size(), 71U);
  EXPECT_EQ(diagnostic.provided.size(), 71U);
  EXPECT_TRUE(diagnostic.required.starts_with("sha256:"));
  EXPECT_TRUE(diagnostic.provided.starts_with("sha256:"));
  EXPECT_NE(diagnostic.required, diagnostic.provided);

  const auto published = sink->snapshot_details();
  ASSERT_EQ(published.size(), 1U);
  ASSERT_EQ(published.front().size(), 1U);
  const detail::RuntimeDiagnosticDetail& retained = published.front().front();
  EXPECT_TRUE(retained.present);
  EXPECT_EQ(retained.error.code, ErrorCode::kFailure);
  EXPECT_EQ(retained.operator_id, ordinary.operator_id);
  EXPECT_EQ(retained.replay_mismatch.vertex, ordinary.vertex);
  EXPECT_EQ(retained.replay_mismatch.operator_id, ordinary.operator_id);
  EXPECT_EQ(retained.replay_mismatch.factory_slot, ordinary.factory_slot);
  EXPECT_EQ(retained.replay_mismatch.category, ordinary.category);
  EXPECT_EQ(retained.replay_mismatch.expected, ordinary.expected);
  EXPECT_EQ(retained.replay_mismatch.observed, ordinary.observed);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: PARAM-017
TEST(RunSessionTest, BindsTypedParameterGenerationZeroBeforeFirstCallback) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_param_test_plan());
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  const auto observations = test::RuntimeParamProbeState::snapshot();
  ASSERT_EQ(observations.size(), 1U);
  EXPECT_EQ(observations[0].before_value, 7);
  EXPECT_EQ(observations[0].after_value, 7);
  EXPECT_EQ(observations[0].before_generation, 0U);
  EXPECT_EQ(observations[0].after_generation, 0U);
  EXPECT_EQ(observations[0].before_address, observations[0].after_address);
}

// REQ: LIFECYCLE-012
TEST(RunSessionTest, ComputeReadsThePublishedFixedRateGenerationInConstantTime) {
  const ExecutionPlan compiled = make_runtime_fixed_rate_query_plan("runtime-fixed-rate-query");
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const auto compiled_state = detail::ExecutionPlanAccess::retain(compiled);
  ASSERT_NE(compiled_state, nullptr);
  ASSERT_EQ(compiled_state->closed_island_activations.size(), 1U);

  for (const bool closed_island : {true, false}) {
    SCOPED_TRACE(closed_island ? "closed-island" : "ordinary-dispatch");
    RuntimeRateProbeState::reset();
    const ExecutionPlan plan =
        closed_island
            ? compiled
            : detail::RuntimeTestPlanAccess::disable_closed_island_composite_for_test(compiled);

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_TRUE(session.diagnostics().empty());
    EXPECT_EQ(RuntimeRateProbeState::compute_error, ErrorCode::kSuccess);
    EXPECT_EQ(RuntimeRateProbeState::compute.resolved_nominal_rate, Rate::hz(60U));
    EXPECT_EQ(RuntimeRateProbeState::compute.sizing_rate, Rate::hz(60U));
    EXPECT_EQ(RuntimeRateProbeState::compute.origin, ResolvedRateOrigin::kOutputNominal);
    EXPECT_EQ(RuntimeRateProbeState::compute.state, PortRateViewState::kResolved);
    EXPECT_NE(RuntimeRateProbeState::compute.run_generation, 0U);
    EXPECT_NE(RuntimeRateProbeState::compute.rate_generation, 0U);
  }
}

// REQ: LIFECYCLE-012
TEST(RunSessionTest, DiscoverBindingResolvesAtTheGlobalBarrierBeforeAllocateAndCompute) {
  RuntimeRateProbeState::reset();
  const ExecutionPlan plan =
      make_runtime_rate_probe_plan<RuntimeLifecycleRateQueryProbe>("runtime-lifecycle-rate-query");
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(RuntimeRateProbeState::bind_error, ErrorCode::kSuccess);
  EXPECT_EQ(RuntimeRateProbeState::same_stage_error, ErrorCode::kSuccess);
  EXPECT_EQ(RuntimeRateProbeState::allocate_error, ErrorCode::kSuccess);
  EXPECT_EQ(RuntimeRateProbeState::compute_error, ErrorCode::kSuccess);
  EXPECT_EQ(RuntimeRateProbeState::same_stage.state, PortRateViewState::kPendingBarrier);
  EXPECT_FALSE(RuntimeRateProbeState::same_stage.resolved_nominal_rate.has_value());
  EXPECT_EQ(RuntimeRateProbeState::allocate.resolved_nominal_rate, Rate::hz(60U));
  EXPECT_EQ(RuntimeRateProbeState::allocate.max_nominal_rate, Rate::hz(120U));
  EXPECT_EQ(RuntimeRateProbeState::allocate.origin, ResolvedRateOrigin::kLifecycleBinding);
  EXPECT_EQ(RuntimeRateProbeState::allocate.state, PortRateViewState::kResolved);
  EXPECT_EQ(RuntimeRateProbeState::compute, RuntimeRateProbeState::allocate);
}

// REQ: LIFECYCLE-012
// REQ: CHAN-141
TEST(RunSessionTest, DiscoverRateMaterializesGenericQueueBeforeUserAllocate) {
  RuntimeLifecycleRateQueueProbeState::reset();
  Graph graph{"runtime-lifecycle-rate-generic-queue"};
  auto source = graph.op<RuntimeLifecycleRateQueueSource>("source");
  auto sink = graph.op<RuntimeLifecycleRateQueueSink>("sink");
  graph.add_flow(source->output, sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-generic-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.queue_sizing_recipes().size(), 1U);
  EXPECT_EQ(plan.queue_sizing_recipes().front().geometry_mode,
            AllocationGeometryMode::kLifecycleCandidate);
  ASSERT_EQ(plan.generic_output_loans().size(), 1U);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(RuntimeLifecycleRateQueueProbeState::bind_error.load(std::memory_order_acquire), 0U);
  EXPECT_EQ(RuntimeLifecycleRateQueueProbeState::allocate_error.load(std::memory_order_acquire),
            0U);
  EXPECT_EQ(RuntimeLifecycleRateQueueProbeState::compute_error.load(std::memory_order_acquire), 0U);
  EXPECT_TRUE(
      RuntimeLifecycleRateQueueProbeState::allocate_saw_resolved.load(std::memory_order_acquire));
  EXPECT_TRUE(RuntimeLifecycleRateQueueProbeState::sink_received.load(std::memory_order_acquire));
  EXPECT_EQ(RuntimeLifecycleRateQueueProbeState::observed_value.load(std::memory_order_acquire),
            RuntimeLifecycleRateQueueSource::kValue);
}

// REQ: LIFECYCLE-012
// REQ: RUNTIME-007
// An OnEach driver without an authored selector is pinned separately from a lifecycle-expanded
// DrainAll batch. The batch exceeds the cold placeholder, exercising both backing-ceiling scratch
// reservation and stable trigger-pin storage through the complete RunSession/Channel path.
TEST(RunSessionTest, LifecycleDrainAllExpansionPreservesSeparateOnEachTriggerPin) {
  RuntimeLifecycleDrainAllScratchProbeState::reset();
  Graph graph{"runtime-lifecycle-drain-all-selection-scratch"};
  auto source = graph.op<RuntimeLifecycleDrainAllScratchSource>("source");
  auto sink = graph.op<RuntimeLifecycleDrainAllScratchSink>("sink");
  graph.add_flow(source->secondary, sink->secondary);
  graph.add_flow(source->driver, sink->driver);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-drain-all-scratch-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto secondary_port = std::ranges::find_if(plan.ports(), [](const auto& port) {
    return port.local_name == "secondary" && port.direction == PortDirection::kInput;
  });
  ASSERT_NE(secondary_port, plan.ports().end());
  const auto selection = std::ranges::find(
      plan.input_selections(), secondary_port->id, &InputSelectionPlanRecord::input);
  ASSERT_NE(selection, plan.input_selections().end());
  EXPECT_EQ(selection->selector, InputSelectorKind::kDrainAll);
  EXPECT_EQ(selection->max_selected_samples_origin,
            SelectedSampleBoundOrigin::kResolvedRetainedCapacity);
  ASSERT_LT(selection->max_selected_samples, RuntimeLifecycleDrainAllScratchSource::kBacklogSize)
      << plan.json();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(RuntimeLifecycleDrainAllScratchProbeState::error.load(std::memory_order_acquire), 0U);
  EXPECT_EQ(RuntimeLifecycleDrainAllScratchProbeState::driver_value.load(std::memory_order_acquire),
            RuntimeLifecycleDrainAllScratchSource::kDriverValue);
  EXPECT_EQ(RuntimeLifecycleDrainAllScratchProbeState::batch_size.load(std::memory_order_acquire),
            RuntimeLifecycleDrainAllScratchSource::kBacklogSize);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-007
TEST(RunSessionTest, FreezesActivationTimeAfterEveryStartCallbackCompletes) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kAdvanceClockInStart);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_EQ(test::RuntimeProbeState::observed_activation_time_ns.load(std::memory_order_relaxed),
            37);
  EXPECT_EQ(test::RuntimeProbeState::observed_clock_time_ns.load(std::memory_order_relaxed), 37);
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
}

// REQ: RUNTIME-004
// REQ: RUNTIME-012
TEST(RunSessionTest, OnClockRepeatsDeterministicallyUntilCooperativeStop) {
  constexpr std::size_t kObservedActivations = 3U;
  constexpr std::size_t kRuns = 2U;
  std::array<std::array<Tag, kObservedActivations>, kRuns> tags{};
  std::array<std::array<TagTime, kObservedActivations>, kRuns> tag_times{};
  std::array<std::array<TimePoint, kObservedActivations>, kRuns> activation_times{};

  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_clock_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  for (std::size_t run_index = 0U; run_index < kRuns; ++run_index) {
    test::RuntimeProbeState::reset();
    test::RuntimeClockProbeState::reset();
    RunSession session = run_async(plan);
    ASSERT_TRUE(test::RuntimeClockProbeState::wait_until_compute_count(kObservedActivations));

    session.request_stop();
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty());
    ASSERT_GE(test::RuntimeClockProbeState::compute_count.load(std::memory_order_acquire),
              kObservedActivations);
    std::copy_n(
        test::RuntimeClockProbeState::tags.begin(), kObservedActivations, tags[run_index].begin());
    std::copy_n(test::RuntimeClockProbeState::tag_times.begin(),
                kObservedActivations,
                tag_times[run_index].begin());
    std::copy_n(test::RuntimeClockProbeState::activation_times.begin(),
                kObservedActivations,
                activation_times[run_index].begin());
  }

  EXPECT_EQ(tags[0], tags[1]);
  EXPECT_EQ(tag_times[0], tag_times[1]);
  EXPECT_EQ(activation_times[0], activation_times[1]);
  for (std::size_t index = 0U; index < kObservedActivations; ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(tags[0][index], (Tag{index + 1U, 1U, 0U}));
    EXPECT_EQ(
        tag_times[0][index],
        (TagTime{
            static_cast<std::int64_t>(index) * test::RuntimeClockProbeState::kPeriodNs, 0U, 0U}));
    EXPECT_EQ(activation_times[0][index].timestamp_ns, tag_times[0][index].logical_ns);
    EXPECT_EQ(activation_times[0][index].clock_id, activation_times[0].front().clock_id);
  }
  EXPECT_TRUE(activation_times[0].front().valid());
}

// REQ: RUNTIME-046
TEST(RunSessionTest, OperatorTerminationRequestsUseTheirStrongestSuccessfulReason) {
  const ExecutionPlan plan = test::make_runtime_clock_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto run = [&](test::RuntimeClockProbeState::TerminationRequestMode mode,
                       RunTermination expected,
                       bool callback_completed_normally) {
    test::RuntimeClockProbeState::reset();
    test::RuntimeClockProbeState::termination_request_mode.store(mode, std::memory_order_release);
    RunSession session = run_async(plan);
    session.wait();
    EXPECT_EQ(session.termination(), expected);
    if (callback_completed_normally) {
      EXPECT_GE(test::RuntimeClockProbeState::compute_count.load(std::memory_order_acquire), 1U);
    }
  };

  run(test::RuntimeClockProbeState::TerminationRequestMode::kCompletion,
      RunTermination::kRequestedCompleted,
      true);
  run(test::RuntimeClockProbeState::TerminationRequestMode::kStop, RunTermination::kStopped, true);
  run(test::RuntimeClockProbeState::TerminationRequestMode::kCompletionThenStop,
      RunTermination::kStopped,
      true);
  run(test::RuntimeClockProbeState::TerminationRequestMode::kStopThenCompletion,
      RunTermination::kStopped,
      true);
  run(test::RuntimeClockProbeState::TerminationRequestMode::kCompletionThenFailure,
      RunTermination::kFailed,
      false);
}

// REQ: RUNTIME-046
TEST(RunSessionTest, OperatorStopRequestBeatsNaturalCompletionAndIsIdempotent) {
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  ASSERT_TRUE(plan.ok()) << plan.json();

  for (const test::RuntimeProbeMode mode :
       {test::RuntimeProbeMode::kRequestStop, test::RuntimeProbeMode::kRequestStopTwice}) {
    test::RuntimeProbeState::reset(mode);
    RunSession session = run_async(plan);
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
  }
}

// REQ: RUNTIME-046
TEST(RunSessionTest, BlockingRunReturnsZeroForRequestedCompletion) {
  const ExecutionPlan plan = test::make_runtime_clock_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  test::RuntimeClockProbeState::reset();
  test::RuntimeClockProbeState::termination_request_mode.store(
      test::RuntimeClockProbeState::TerminationRequestMode::kCompletion, std::memory_order_release);

  EXPECT_EQ(run(plan, RunSignalPolicy::kCallerManaged), 0);
}

// REQ: OPERATOR-038
// REQ: RUNTIME-046
TEST(RunSessionTest, OperatorCompletionRequestDoesNotWaitForParameterControlMutex) {
  test::RuntimeProbeState::reset();
  test::RuntimeClockProbeState::reset();
  auto control = std::make_shared<detail::RuntimeTerminationAdmissionTestControl>(
      detail::RuntimeTerminationAdmissionTestControl::Point::kParameterMutexHeld, 1U);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_termination_admission_test_control(
      test::make_runtime_clock_test_plan(), control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool first_callback_returned = test::RuntimeClockProbeState::wait_until_compute_count(1U);
  if (!first_callback_returned) {
    session.request_stop();
    session.wait();
  }
  ASSERT_TRUE(first_callback_returned);
  const std::size_t completed_before_request =
      test::RuntimeClockProbeState::compute_count.load(std::memory_order_acquire);
  std::thread capture{[&] { static_cast<void>(session.capture_params()); }};
  const bool parameter_mutex_held = control->wait_until_arrived();
  test::RuntimeClockProbeState::termination_request_mode.store(
      test::RuntimeClockProbeState::TerminationRequestMode::kCompletion, std::memory_order_release);

  // The callback must publish completion and return while capture still owns parameter_mutex_.
  // Releasing the test barrier first would hide a synchronous lock acquisition in the request.
  std::unique_lock probe_lock(test::RuntimeClockProbeState::mutex);
  const bool callback_returned =
      test::RuntimeClockProbeState::cv.wait_for(probe_lock, 5s, [completed_before_request] {
        return test::RuntimeClockProbeState::termination_request_mode.load(
                   std::memory_order_acquire) ==
                   test::RuntimeClockProbeState::TerminationRequestMode::kNone &&
               test::RuntimeClockProbeState::compute_count.load(std::memory_order_acquire) >
                   completed_before_request;
      });
  probe_lock.unlock();
  control->release();
  capture.join();
  session.wait();

  ASSERT_TRUE(parameter_mutex_held);
  EXPECT_TRUE(callback_returned);
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
}

// The ordinary run path, without an injected binding factory, must expose the exact bounded
// sender only during Arm and preserve capacity through complete callback retirement.
// REQ: LIFECYCLE-005
// REQ: RUNTIME-025
TEST(RunSessionTest, NotificationSenderIsLifecycleBoundCapacityLimitedAndClosedAtStop) {
  // Key setup: compile and run an OnNotified graph without an injected PortBinding factory.
  test::RuntimeNotificationProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_notification_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeNotificationProbeState::wait_until_sender());
  // Key assertion: Arm can acquire only the admitted sender, but cannot post before global Start.
  EXPECT_EQ(test::RuntimeNotificationProbeState::arm_post_error, ErrorCode::kNotReady);
  EXPECT_EQ(test::RuntimeNotificationProbeState::second_acquire_error,
            ErrorCode::kResourceExhausted);

  const NotificationSender sender = test::RuntimeNotificationProbeState::sender_copy();
  auto first_post = sender.post();
  const auto start_deadline = std::chrono::steady_clock::now() + 5s;
  while (!first_post && first_post.error().code == ErrorCode::kNotReady &&
         std::chrono::steady_clock::now() < start_deadline) {
    std::this_thread::yield();
    first_post = sender.post();
  }
  ASSERT_TRUE(first_post);
  ASSERT_TRUE(test::RuntimeNotificationProbeState::wait_until_compute_enters());

  // Key assertion: capacity=1 remains charged while the first callback is still active.
  const auto backpressured = sender.post();
  ASSERT_FALSE(backpressured);
  EXPECT_EQ(backpressured.error().code, ErrorCode::kBackpressured);

  test::RuntimeNotificationProbeState::release_blocked_compute();
  ASSERT_TRUE(test::RuntimeNotificationProbeState::wait_until_compute_count(1U));

  auto second_post = sender.post();
  const auto retirement_deadline = std::chrono::steady_clock::now() + 5s;
  while (!second_post && second_post.error().code == ErrorCode::kBackpressured &&
         std::chrono::steady_clock::now() < retirement_deadline) {
    std::this_thread::yield();
    second_post = sender.post();
  }
  ASSERT_TRUE(second_post);
  ASSERT_TRUE(test::RuntimeNotificationProbeState::wait_until_compute_count(2U));

  session.request_stop();
  session.wait();

  // Key assertion: the escaped sender retains only a closed tombstone after session teardown.
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const auto after_stop = sender.post();
  ASSERT_FALSE(after_stop);
  EXPECT_EQ(after_stop.error().code, ErrorCode::kClosed);
}

// REQ: RUNTIME-001, RUNTIME-007, RUNTIME-012
TEST(RunSessionTest, MaterializesMixedPhysicalClockAndClockPredicateSourcesTogether) {
  test::RuntimeProbeState::reset();
  test::RuntimeClockProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();
  const ExecutionPlan plan =
      test::enable_direct_runtime(test::make_runtime_mixed_source_test_plan());
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.readiness().size(), 1U);
  ASSERT_EQ(plan.temporal_triggers().size(), 3U);

  test::RuntimeProbeState::reset();
  test::RuntimeClockProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();
  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());
  ASSERT_TRUE(test::RuntimeClockProbeState::wait_until_compute_count(1U));
  ASSERT_TRUE(test::RuntimeClockPredicateProbeState::wait_until_compute_count(1U));

  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
}

void expect_clock_predicate_snapshot(
    const test::RuntimeClockPredicateEvaluationEvidence& evaluation,
    const test::RuntimeClockPredicateComputeEvidence& compute) {
  EXPECT_EQ(evaluation.selection_count, 0U);
  EXPECT_EQ(compute.tag, evaluation.tag);
  EXPECT_EQ(compute.tag_time, evaluation.tag_time);
  EXPECT_EQ(compute.activation_time, evaluation.reference_time);
  EXPECT_EQ(evaluation.reference_time.timestamp_ns, evaluation.tag_time.logical_ns);
}

// REQ: RUNTIME-007
// REQ: RUNTIME-012
TEST(RunSessionTest, ClockPredicateCommitsFalseAndRetiresNoFireBeforeNextTrue) {
  test::RuntimeProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_clock_predicate_plan(
      test::RuntimeClockPredicatePattern::kTrueFalseTrue, PredicateFiring::kOnFalseToTrue);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.predicate_triggers().size(), 1U);
  EXPECT_EQ(plan.predicate_triggers().front().firing, PredicateFiring::kOnFalseToTrue);
  EXPECT_TRUE(plan.predicate_inputs().empty());
  EXPECT_TRUE(plan.input_selections().empty());
  test::RuntimeProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeClockPredicateProbeState::wait_until_compute_count(2U));
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  ASSERT_EQ(test::RuntimeClockPredicateProbeState::evaluation_count.load(std::memory_order_acquire),
            3U);
  ASSERT_EQ(test::RuntimeClockPredicateProbeState::compute_count.load(std::memory_order_acquire),
            2U);
  const auto& evaluations = test::RuntimeClockPredicateProbeState::evaluations;
  EXPECT_EQ(evaluations[0].state_count, 1U);
  EXPECT_TRUE(evaluations[0].result);
  EXPECT_EQ(evaluations[1].state_count, 2U);
  EXPECT_FALSE(evaluations[1].result);
  EXPECT_EQ(evaluations[2].state_count, 3U);
  EXPECT_TRUE(evaluations[2].result);
  EXPECT_EQ(evaluations[0].tag.sequence + 1U, evaluations[1].tag.sequence);
  EXPECT_EQ(evaluations[1].tag.sequence + 1U, evaluations[2].tag.sequence);
  EXPECT_EQ(evaluations[0].tag.generation, evaluations[2].tag.generation);
  expect_clock_predicate_snapshot(evaluations[0],
                                  test::RuntimeClockPredicateProbeState::computes[0]);
  expect_clock_predicate_snapshot(evaluations[2],
                                  test::RuntimeClockPredicateProbeState::computes[1]);
}

// REQ: RUNTIME-007
// REQ: RUNTIME-012
TEST(RunSessionTest, ClockPredicateEveryTrueEvaluationFiresWithExactSnapshot) {
  test::RuntimeProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_clock_predicate_plan(
      test::RuntimeClockPredicatePattern::kAlwaysTrue, PredicateFiring::kOnEveryTrueEvaluation);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.predicate_triggers().size(), 1U);
  EXPECT_EQ(plan.predicate_triggers().front().firing, PredicateFiring::kOnEveryTrueEvaluation);
  EXPECT_TRUE(plan.predicate_inputs().empty());
  EXPECT_TRUE(plan.input_selections().empty());
  test::RuntimeProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeClockPredicateProbeState::wait_until_compute_count(2U));
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  ASSERT_EQ(test::RuntimeClockPredicateProbeState::evaluation_count.load(std::memory_order_acquire),
            2U);
  ASSERT_EQ(test::RuntimeClockPredicateProbeState::compute_count.load(std::memory_order_acquire),
            2U);
  const auto& evaluations = test::RuntimeClockPredicateProbeState::evaluations;
  EXPECT_EQ(evaluations[0].state_count, 1U);
  EXPECT_EQ(evaluations[1].state_count, 2U);
  EXPECT_TRUE(evaluations[0].result);
  EXPECT_TRUE(evaluations[1].result);
  expect_clock_predicate_snapshot(evaluations[0],
                                  test::RuntimeClockPredicateProbeState::computes[0]);
  expect_clock_predicate_snapshot(evaluations[1],
                                  test::RuntimeClockPredicateProbeState::computes[1]);
}

// REQ: RUNTIME-008
// REQ: RUNTIME-012
TEST(RunSessionTest, ClockPredicateTruthAndStateResetAfterPublishedSelfRestart) {
  test::RuntimeProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_clock_predicate_plan(
      test::RuntimeClockPredicatePattern::kAlwaysTrue, PredicateFiring::kOnFalseToTrue, true);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.predicate_triggers().size(), 1U);
  ASSERT_EQ(plan.lifecycle_restart_targets().size(), 1U);
  EXPECT_TRUE(plan.predicate_inputs().empty());
  EXPECT_TRUE(plan.input_selections().empty());
  test::RuntimeProbeState::reset();
  test::RuntimeClockPredicateProbeState::reset();

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeClockPredicateProbeState::wait_until_compute_count(2U));
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  ASSERT_EQ(test::RuntimeClockPredicateProbeState::evaluation_count.load(std::memory_order_acquire),
            2U);
  ASSERT_EQ(test::RuntimeClockPredicateProbeState::compute_count.load(std::memory_order_acquire),
            2U);
  EXPECT_TRUE(
      test::RuntimeClockPredicateProbeState::restart_accepted.load(std::memory_order_acquire));
  EXPECT_EQ(test::RuntimeClockPredicateProbeState::restart_error.load(std::memory_order_relaxed),
            ErrorCode::kSuccess);
  EXPECT_NE(
      test::RuntimeClockPredicateProbeState::restart_receipt_id.load(std::memory_order_relaxed),
      0U);
  EXPECT_EQ(test::RuntimeClockPredicateProbeState::restart_requested_stage.load(
                std::memory_order_relaxed),
            LifecycleStage::kConfigure);
  const auto& evaluations = test::RuntimeClockPredicateProbeState::evaluations;
  EXPECT_EQ(evaluations[0].state_count, 1U);
  EXPECT_EQ(evaluations[1].state_count, 1U);
  EXPECT_TRUE(evaluations[0].result);
  EXPECT_TRUE(evaluations[1].result);
  EXPECT_LT(evaluations[0].tag.generation, evaluations[1].tag.generation);
  expect_clock_predicate_snapshot(evaluations[0],
                                  test::RuntimeClockPredicateProbeState::computes[0]);
  expect_clock_predicate_snapshot(evaluations[1],
                                  test::RuntimeClockPredicateProbeState::computes[1]);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-012
// REQ: RUNTIME-028
TEST(RunSessionTest, PredicateFailureRetainsExactPrivateCandidateContext) {
  test::RuntimeProbeState::reset();
  const auto sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      test::make_runtime_clock_predicate_error_plan(), sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.predicate_triggers().size(), 1U);
  ASSERT_EQ(plan.deadlines().size(), 1U);
  ASSERT_TRUE(plan.input_selections().empty()) << plan.json();
  ASSERT_EQ(plan.clock_triggers().size(), 1U) << plan.json();
  ASSERT_EQ(plan.clock_triggers().front().phase_origin,
            ClockPhaseOrigin::kCapturedAtSuccessfulStart)
      << plan.json();
  ASSERT_FALSE(plan.clock_triggers().front().explicit_phase) << plan.json();
  const PredicateTriggerPlanRecord trigger = plan.predicate_triggers().front();
  test::RuntimeProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  ASSERT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
  EXPECT_NE(session.diagnostics().front().token, "DEADLINE_MISS");
  EXPECT_EQ(session.diagnostics().front().service, "predicate_evaluator");
  EXPECT_TRUE(static_cast<bool>(session.diagnostics().front().source.vertex));
  EXPECT_TRUE(session.diagnostics().front().endpoint.empty());
  EXPECT_TRUE(session.diagnostics().front().json_pointer.empty());

  const auto published = sink->snapshot_details();
  ASSERT_EQ(published.size(), 1U);
  ASSERT_EQ(published.front().size(), 1U);
  const detail::RuntimeDiagnosticDetail& detail = published.front().front();
  EXPECT_TRUE(detail.present);
  EXPECT_EQ(detail.error.code, ErrorCode::kProtocolError);
  EXPECT_EQ(detail.error.message, "runtime predicate failure detail");
  EXPECT_EQ(detail.predicate_id, trigger.id);
  EXPECT_EQ(detail.operator_id, trigger.owner.id);
  EXPECT_TRUE(static_cast<bool>(detail.source));
  EXPECT_EQ(detail.source_sequence, 1U);
  EXPECT_EQ(detail.tag, (Tag{1U, 1U, 0U}));
  EXPECT_EQ(detail.tag_time, (TagTime{0, 0U, 0U}));
  EXPECT_TRUE(detail.reference_time.valid());
  EXPECT_EQ(detail.reference_time.timestamp_ns, detail.tag_time.logical_ns);
  EXPECT_EQ(detail.state_generation, detail.tag.generation);
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// REQ: RUNTIME-002
// REQ: RUNTIME-004
TEST(RunSessionTest, ClockReadFailureAfterStartStillRunsReverseCleanup) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kThrowClockRead);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_CLOCK_READ_FAILED"));
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 13}));
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
}
#endif

// REQ: RUNTIME-002
// REQ: RUNTIME-004
TEST(RunSessionTest, RejectsUninjectedReadinessBeforeLifecycleCallbacks) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_test_plan();
  test::RuntimeProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_READINESS_BINDING_UNAVAILABLE"));
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
  EXPECT_EQ(test::RuntimeProbeState::active.load(std::memory_order_acquire), 0U);
}

// REQ: RUNTIME-010
TEST(RunSessionTest, ReportsMaterializationFailureBeforeDeadlineCallback) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan baseline = test::enable_direct_runtime(test::make_runtime_test_plan());
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::mutate_materialization_strategy(
      baseline, "holoscan.runtime.unknown/v1");
  test::RuntimeProbeState::reset();

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "PROFILE_MATERIALIZATION_FAILED"));
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
}

// REQ: RUNTIME-010
TEST(RunSessionTest, RejectsEveryPackageMismatchBeforeOperatorReconstruction) {
  using ProfileMutation = detail::RuntimeTestPlanAccess::ProfileMutation;
  struct Case {
    ProfileMutation mutation;
    std::string_view token;
  };
  constexpr std::array cases{
      Case{ProfileMutation::kPlanSchema, "PROFILE_PLAN_SCHEMA_UNSUPPORTED"},
      Case{ProfileMutation::kProfilePolicy, "PROFILE_POLICY_UNSUPPORTED"},
      Case{ProfileMutation::kRuntimeAbi, "PROFILE_RUNTIME_ABI_INCOMPATIBLE"},
      Case{ProfileMutation::kTarget, "PROFILE_TARGET_INCOMPATIBLE"},
      Case{ProfileMutation::kCapability, "PROFILE_CAPABILITY_UNSATISFIED"},
      Case{ProfileMutation::kEvidence, "PROFILE_EVIDENCE_UNSATISFIED"},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.token);
    test::RuntimeProbeState::reset();
    const ExecutionPlan baseline = test::enable_direct_runtime(test::make_runtime_test_plan());
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::mutate_profile(baseline, test_case.mutation);
    ASSERT_TRUE(plan.ok());
    test::RuntimeProbeState::reset();

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, test_case.token));
    EXPECT_EQ(test::RuntimeProbeState::constructed.load(std::memory_order_acquire), 0U);
    EXPECT_FALSE(test::RuntimeProbeState::setup_entered.load(std::memory_order_acquire));
    EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  }
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-008
TEST(RunSessionTest, RejectsMalformedLifecyclePlanBeforeCallbacks) {
  using LifecycleMutation = detail::RuntimeTestPlanAccess::LifecycleMutation;
  constexpr std::array mutations{
      LifecycleMutation::kCheckpointMissing,
      LifecycleMutation::kCheckpointRequired,
      LifecycleMutation::kCheckpointLevel,
      LifecycleMutation::kCheckpointOrder,
      LifecycleMutation::kDependency,
      LifecycleMutation::kRestartPermission,
      LifecycleMutation::kMutableField,
      LifecycleMutation::kTransactionCapacity,
      LifecycleMutation::kRestartTargetMissing,
      LifecycleMutation::kRestartTargetIdentity,
      LifecycleMutation::kRestartTargetAuthority,
      LifecycleMutation::kRestartClosureMissing,
  };

  for (const LifecycleMutation mutation : mutations) {
    SCOPED_TRACE(static_cast<unsigned>(mutation));
    test::RuntimeProbeState::reset();
    const ExecutionPlan baseline = test::enable_direct_runtime(test::make_runtime_test_plan());
    const ExecutionPlan plan = detail::RuntimeTestPlanAccess::mutate_lifecycle(baseline, mutation);
    test::RuntimeProbeState::reset();

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_LIFECYCLE_PLAN_INVALID"));
    EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
    EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
  }
}

// REQ: RUNTIME-001
// REQ: RUNTIME-002
// REQ: RUNTIME-003
TEST(RunSessionTest, RetainsPlanStateAfterPlanWrapperIsDestroyed) {
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  RunSession session = [] {
    const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
    test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
    return run_async(plan);
  }();
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());

  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
}

// REQ: RUNTIME-003
TEST(RunSessionTest, MoveLeavesSourceInDefinedInertState) {
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);

  RunSession source = run_async(plan);
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());
  const OperatorRef plan_ref = plan.lifecycle_checkpoints().front().operator_instance;
  RunSession destination{std::move(source)};

  // Exercising the documented inert moved-from state is the purpose of this test.
  expect_inert_moved_from_session(
      source, plan_ref);  // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)

  destination.request_stop();
  destination.wait();
  EXPECT_EQ(destination.termination(), RunTermination::kStopped);
}

// REQ: RUNTIME-003
TEST(RunSessionTest, ConcurrentStopRequestsAndWaitersShareOneCompletion) {
  // Key setup: 8 threads call session.wait() and 8 threads call request_stop()
  // simultaneously; the wait interleaving control ensures peer waiters are
  // registered before the join owner is released.
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  auto wait_control = std::make_shared<detail::RuntimeWaitTestControl>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_wait_interleaving(
      test::enable_direct_runtime(test::make_runtime_test_plan()), wait_control);
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());

  std::vector<std::thread> waiters;
  waiters.reserve(8U);
  waiters.emplace_back([&] { session.wait(); });
  EXPECT_TRUE(wait_control->wait_for_join_owner());
  for (int index = 1; index < 8; ++index) {
    waiters.emplace_back([&] { session.wait(); });
  }
  EXPECT_TRUE(wait_control->wait_for_peer_waiters(7U));

  std::vector<std::thread> stoppers;
  stoppers.reserve(8U);
  for (int index = 0; index < 8; ++index) {
    stoppers.emplace_back([&] { session.request_stop(); });
  }
  for (auto& stopper : stoppers) {
    stopper.join();
  }
  wait_control->release_join_owner();
  for (auto& waiter : waiters) {
    waiter.join();
  }

  EXPECT_FALSE(session.is_running());
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_EQ(test::RuntimeProbeState::max_active.load(std::memory_order_relaxed), 1U);
  // Key assertion: join_owner_count==1 proves exactly one thread performed the
  // underlying thread join regardless of how many concurrent waiters arrived.
  EXPECT_EQ(wait_control->join_owner_count(), 1U);
}

// REQ: RUNTIME-003
// REQ: RUNTIME-004
TEST(RunSessionTest, DestructorRequestsStopAndJoinsWorker) {
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);

  {
    const RunSession session = run_async(plan);
    ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());
  }

  EXPECT_EQ(test::RuntimeProbeState::active.load(std::memory_order_acquire), 0U);
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
}

// REQ: RUNTIME-004
TEST(RunSessionTest, ComputeFailureIsTerminalAndNotRetried) {
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kFail);
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kFail);
  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_OPERATOR_CALLBACK_FAILED"));
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
  EXPECT_NE(run(plan, RunSignalPolicy::kCallerManaged), 0);
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// REQ: RUNTIME-004
TEST(RunSessionTest, SetupExceptionFailsStartupBeforeLifecycleCallbacks) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kThrowSetup);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_OPERATOR_SETUP_FAILED"));
  ASSERT_FALSE(session.diagnostics().empty());
  const auto& failure_detail = session.diagnostics().front().detail;
  if (!failure_detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  EXPECT_EQ(failure_detail->operation, DiagnosticOperation::kSetup);
  EXPECT_EQ(failure_detail->error.message, "runtime test setup failure");
  EXPECT_TRUE(session.diagnostics().front().source.vertex);
  EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
}

// REQ: RUNTIME-003
// REQ: RUNTIME-004
TEST(RunSessionTest, StartExceptionStopsOnlyPreviouslyStartedOperators) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan(true));
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kThrowStart);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_LIFECYCLE_CALLBACK_FAILED"));
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 21, 13}));
  EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_TRUE(session.diagnostics().front().source.vertex);
  const auto report = format_diagnostic(session.diagnostics().front(), plan.graph());
  EXPECT_NE(report.find("runtime test start failure"), std::string::npos) << report;
  EXPECT_NE(report.find("start"), std::string::npos) << report;
}

// REQ: RUNTIME-004
TEST(RunSessionTest, ComputeExceptionBecomesTerminalCallbackFailure) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kThrowCompute);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_OPERATOR_CALLBACK_FAILED"));
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
  EXPECT_EQ(test::RuntimeProbeState::active.load(std::memory_order_acquire), 0U);
  ASSERT_FALSE(session.diagnostics().empty());
  const auto report = format_diagnostic(session.diagnostics().front(), plan.graph());
  EXPECT_NE(report.find("runtime test compute failure"), std::string::npos) << report;
}

// REQ: RUNTIME-003
// REQ: RUNTIME-004
TEST(RunSessionTest, StopExceptionCompletesCleanupAndMarksRunFailed) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kThrowStop);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_LIFECYCLE_CALLBACK_FAILED"));
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{11, 12, 13}));
  ASSERT_FALSE(session.diagnostics().empty());
  const auto& failure_detail = session.diagnostics().front().detail;
  if (!failure_detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  EXPECT_EQ(failure_detail->stage, LifecycleStage::kStop);
  EXPECT_EQ(failure_detail->phase, LifecycleTransitionPhase::kShutdownTeardown);
  EXPECT_EQ(failure_detail->error.message, "runtime test stop failure");
}
#endif

// REQ: RUNTIME-004
TEST(RuntimeLifecycleExecutorTest, StartupPreparationFailureStopsOnlyPreviouslyStartedOperators) {
  // Key setup: pin count for z-second is set to near-overflow before lifecycle
  // startup, forcing a parameter-preparation failure after a-first has started.
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();
  const ExecutionPlan compiled = make_two_legacy_parameter_startup_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(compiled);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);

  constexpr RunGenerationId kRunGeneration = 17U;
  auto reconstructed = detail::GraphExec::reconstruct(accepted, kRunGeneration);
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  auto exec = std::move(reconstructed).value();
  ASSERT_EQ(exec->operators().size(), 2U);
  ASSERT_EQ(exec->operators()[0].operator_id, 1U);
  ASSERT_EQ(exec->operators()[1].operator_id, 2U);

  auto created_lanes =
      detail::ExecutionLaneSet::create(accepted->scheduler_plan, accepted->execution_lanes);
  ASSERT_TRUE(created_lanes) << created_lanes.error().message;
  auto lanes = std::move(created_lanes).value();
  ASSERT_TRUE(lanes->start());
  const std::atomic_bool cancellation{};
  ASSERT_TRUE(exec->initialize_lifecycle(kRunGeneration, &cancellation, lanes.get()));

  const OperatorRef failing_owner{
      .id = 2U,
      .plan_generation = accepted->plan_generation,
      .run_generation = kRunGeneration,
  };
  detail::ParameterUpdateCoordinatorTestAccess::set_published_pin_count(
      exec->parameter_updates(), failing_owner, std::numeric_limits<std::uint64_t>::max() >> 1U);
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();

  // Key assertion: lifecycle_startup() fails and only the stop callbacks for
  // a-first (901, 903) appear in the snapshot — z-second was never started.
  EXPECT_FALSE(exec->lifecycle_startup());
  EXPECT_EQ(exec->lifecycle_failure_token(), "RUNTIME_LIFECYCLE_CALLBACK_FAILED");
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{901, 903}));
  EXPECT_TRUE(test::RuntimeParamProbeState::stop_observed_params.load(std::memory_order_acquire));

  detail::ParameterUpdateCoordinatorTestAccess::set_published_pin_count(
      exec->parameter_updates(), failing_owner, 0U);

  lanes->request_stop();
  EXPECT_TRUE(lanes->join());
}

// REQ: RUNTIME-004
// REQ: RUNTIME-008
TEST(RuntimeLifecycleExecutorTest, AcceptsCanonicalTopologicalCheckpointOrder) {
  const ExecutionPlan compiled = make_reverse_topological_lifecycle_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  std::vector<OperatorInstanceId> configure_order;
  for (const auto& checkpoint : compiled.lifecycle_checkpoints()) {
    if (checkpoint.stage == LifecycleStage::kConfigure) {
      configure_order.push_back(checkpoint.operator_instance.id);
    }
  }
  ASSERT_EQ(configure_order, (std::vector<OperatorInstanceId>{2U, 1U}));

  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(compiled);
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);
  auto reconstructed = detail::GraphExec::reconstruct(accepted, 17U);
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
}

// REQ: RUNTIME-005
TEST(RunSessionTest, OperatorRefBindsOnlyMatchingPlanIdentity) {
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  const OperatorRef plan_ref = plan.lifecycle_checkpoints().front().operator_instance;
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());

  const auto bound = session.operator_ref(plan_ref);
  ASSERT_TRUE(bound);
  EXPECT_TRUE(bound->run_bound());
  EXPECT_EQ(bound->id, plan_ref.id);
  EXPECT_EQ(bound->plan_generation, plan_ref.plan_generation);
  EXPECT_NE(bound->run_generation, 0U);

  OperatorRef already_bound = plan_ref;
  already_bound.run_generation = 1U;
  EXPECT_EQ(session.operator_ref(already_bound).error().code, ErrorCode::kStaleOperatorRef);
  OperatorRef colliding_foreign = plan_ref;
  ++colliding_foreign.plan_generation;
  ASSERT_EQ(colliding_foreign.id, plan_ref.id);
  EXPECT_EQ(session.operator_ref(colliding_foreign).error().code, ErrorCode::kStaleOperatorRef);
  EXPECT_EQ(session.operator_ref(OperatorRef{}).error().code, ErrorCode::kStaleOperatorRef);
  OperatorRef unresolved = plan_ref;
  unresolved.id = std::numeric_limits<OperatorInstanceId>::max();
  EXPECT_EQ(session.operator_ref(unresolved).error().code, ErrorCode::kStaleOperatorRef);

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.operator_ref(plan_ref).error().code, ErrorCode::kStaleOperatorRef);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-005
TEST(RunSessionTest, ConcurrentSessionsReceiveUniqueRunGenerations) {
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  const OperatorRef plan_ref = plan.lifecycle_checkpoints().front().operator_instance;
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);

  RunSession first = run_async(plan);
  RunSession second = run_async(plan);
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_active(2U));

  const auto first_ref = first.operator_ref(plan_ref);
  const auto second_ref = second.operator_ref(plan_ref);
  ASSERT_TRUE(first_ref);
  ASSERT_TRUE(second_ref);
  EXPECT_NE(first_ref->run_generation, 0U);
  EXPECT_NE(second_ref->run_generation, 0U);
  EXPECT_NE(first_ref->run_generation, second_ref->run_generation);

  first.request_stop();
  second.request_stop();
  first.wait();
  second.wait();
}

// A clock-driven source that publishes one sample per explicit request. Between requests it
// keeps ticking without emitting, so a parked downstream owner receives no further wake.
class OwnerPollContentionSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (!emit_requested.exchange(false, std::memory_order_acq_rel)) {
      return {};
    }
    const std::uint32_t value = published.fetch_add(1U, std::memory_order_acq_rel) + 1U;
    published_ns.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                       std::memory_order_release);
    return output.emit(value);
  }

  static void reset() noexcept {
    emit_requested.store(false, std::memory_order_relaxed);
    published.store(0U, std::memory_order_relaxed);
    published_ns.store(0, std::memory_order_relaxed);
  }

  static inline std::atomic_bool emit_requested{};
  static inline std::atomic<std::uint32_t> published{};
  static inline std::atomic<std::int64_t> published_ns{};
  Output<std::uint32_t> output;
};

// One of two fanout sinks, each on its own partition so both are pump-fed HoloMQ owners.
class OwnerPollContentionSink final : public Operator<> {
 public:
  explicit OwnerPollContentionSink(const std::size_t index) noexcept : index_(index) {}

  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    received_ns[index_].store(std::chrono::steady_clock::now().time_since_epoch().count(),
                              std::memory_order_release);
    received[index_].fetch_add(1U, std::memory_order_acq_rel);
    return {};
  }

  static void reset() noexcept {
    for (auto& count : received) {
      count.store(0U, std::memory_order_relaxed);
    }
    for (auto& stamp : received_ns) {
      stamp.store(0, std::memory_order_relaxed);
    }
  }

  static inline std::array<std::atomic<std::uint32_t>, 2> received{};
  static inline std::array<std::atomic<std::int64_t>, 2> received_ns{};
  Input<std::uint32_t> input;

 private:
  std::size_t index_{};
};

// Request one publication and wait until both sinks have consumed it.
[[nodiscard]] bool publish_owner_poll_contention_sample(const std::uint32_t expected_deliveries) {
  OwnerPollContentionSource::emit_requested.store(true, std::memory_order_release);
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (OwnerPollContentionSink::received[0].load(std::memory_order_acquire) ==
            expected_deliveries &&
        OwnerPollContentionSink::received[1].load(std::memory_order_acquire) ==
            expected_deliveries) {
      return true;
    }
    std::this_thread::sleep_for(20us);
  }
  return false;
}

// A sink owner whose ordinary readiness poll loses its input cache lock has already consumed the
// wake token that announced the sample, and the lock holder never signals again. Without a retry
// the owner parks for the whole one-millisecond fallback with the sample retained; the retry must
// convert the published revision within the holder's bounded critical section instead.
// REQ: RUNTIME-025
// REQ: RUNTIME-026
TEST(RunSessionTest, ContendedReadinessPollRetriesInsteadOfParkingForTheFallbackWait) {
  OwnerPollContentionSource::reset();
  OwnerPollContentionSink::reset();

  Graph graph{"runtime-owner-poll-contention"};
  const auto source = graph.op<OwnerPollContentionSource>("source");
  const auto first_sink = graph.op<OwnerPollContentionSink>("sink-1", std::size_t{0});
  const auto second_sink = graph.op<OwnerPollContentionSink>("sink-2", std::size_t{1});
  graph.add_flow(source->output, first_sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(source->output, second_sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.partition("runtime-owner-poll-contention-source").add(source);
  graph.partition("runtime-owner-poll-contention-sink-1").add(first_sink);
  graph.partition("runtime-owner-poll-contention-sink-2").add(second_sink);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-owner-poll-contention-clock"));

  // Key setup: the control is armed only after a warm-up delivery, so exactly one steady-state
  // ordinary poll reports the contended decline while nothing else can wake that owner.
  const auto control = std::make_shared<detail::RuntimeChannelIncompletePollTestControl>(
      0U, detail::RuntimeChannelIncompletePollKind::kNonblocking);
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_channel_incomplete_poll_test_control(
          detail::RuntimeTestPlanAccess::with_phase_counters(compile(graph), counters), control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  ASSERT_TRUE(publish_owner_poll_contention_sample(1U)) << "warm-up sample was not delivered";

  // Three contended publications; the minimum latency is the oracle so one preempted lane on a
  // loaded host cannot fail the run, while an unfixed owner parks >= 1.06 ms on every trial.
  constexpr std::uint32_t kTrials = 3U;
  std::array<std::int64_t, kTrials> latencies_ns{};
  bool delivered = true;
  for (std::uint32_t trial = 0U; trial < kTrials && delivered; ++trial) {
    control->arm(1U);
    delivered = publish_owner_poll_contention_sample(2U + trial);
    const std::int64_t published_ns =
        OwnerPollContentionSource::published_ns.load(std::memory_order_acquire);
    const std::int64_t completed_ns =
        std::max(OwnerPollContentionSink::received_ns[0].load(std::memory_order_acquire),
                 OwnerPollContentionSink::received_ns[1].load(std::memory_order_acquire));
    latencies_ns[trial] = completed_ns - published_ns;
  }
  session.request_stop();
  session.wait();

  ASSERT_TRUE(delivered) << "contended sample was not delivered";
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const auto snapshot = counters->snapshot();
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  // Key assertion: every forced decline happened exactly once per trial and each was answered by
  // one retry.
  EXPECT_EQ(control->forced(), kTrials);
  EXPECT_GE(count(detail::RuntimePhaseCounter::kReadinessPollContentionRetries), kTrials);
  EXPECT_GE(count(detail::RuntimePhaseCounter::kChannelPollIncomplete), kTrials);
  // Key assertion: the retained sample completed within the holder's critical section, not after
  // the one-millisecond fallback (which measures ~1.06 ms on a 50 us timer-slack host).
  const std::int64_t min_latency_ns = *std::min_element(latencies_ns.begin(), latencies_ns.end());
  EXPECT_GT(min_latency_ns, 0);
  EXPECT_LT(min_latency_ns, 800'000)
      << "contended-poll completion latencies " << latencies_ns[0] << " / " << latencies_ns[1]
      << " / " << latencies_ns[2] << " ns";
}
}  // namespace
}  // namespace holoscan
