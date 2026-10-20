/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/synthetic_clock.hpp>

#include "common/allocation_counter.hpp"
#include "common/diagnostic_test_environment.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_placement_internal.hpp"
#include "run_session_test_support.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/parameter_update_coordinator.hpp"
#include "runtime/physical_readiness_binding.hpp"
#include "runtime/runtime_callback_context.hpp"
#include "runtime/runtime_partition_test_plan_access.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_worker.hpp"
#include "runtime/source_wait.hpp"
#include "runtime_channel_binding_test_support.hpp"
#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"
#include "support/optional_assertions.hpp"

namespace holoscan {

namespace {

using namespace std::chrono_literals;

using namespace test::run_session;

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

#if GTEST_HAS_DEATH_TEST
// Triggers a self-join death scenario: publish() spins until the bound session
// pointer is set, then calls session->wait() from the runtime's own thread,
// which must terminate under external supervision.
// Used by: RunOwnedThreadSelfJoinTerminatesUnderExternalSupervision
class SelfJoinDiagnosticSink final : public detail::RuntimeStructuralDiagnosticSink {
 public:
  explicit SelfJoinDiagnosticSink(bool fail_formatting = false)
      : fail_formatting_(fail_formatting) {}
  void publish(std::span<const Diagnostic>) override {
    RunSession* session = nullptr;
    while ((session = session_.load(std::memory_order_acquire)) == nullptr) {
      std::this_thread::yield();
    }
    std::optional<test::ScopedAllocationFailure> failure;
    if (fail_formatting_)
      failure.emplace(0U);
    session->wait();
  }

  void bind(RunSession& session) noexcept { session_.store(&session, std::memory_order_release); }

 private:
  std::atomic<RunSession*> session_{};
  bool fail_formatting_{};
};
#endif

struct OrdinaryPollStatusProbe {
  std::atomic<std::size_t> poll_calls{};
  std::atomic<std::size_t> incomplete_status_calls{};
  ErrorCode poll_failure{ErrorCode::kSuccess};
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
    if (probe_->poll_failure != ErrorCode::kSuccess) {
      return make_unexpected(Error{probe_->poll_failure});
    }
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

// A readiness poll fails before any Operator callback runs. The report names the Runtime service
// and operation so an application author does not read it as the receiving Operator's failure.
// REQ: RUNTIME-060
TEST(RunSessionTest, ReadinessPollFailureNamesRuntimeService) {
  test::RuntimeProbeState::reset();
  const auto probe = std::make_shared<OrdinaryPollStatusProbe>();
  probe->poll_failure = ErrorCode::kProtocolError;
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_port_binding_factory(
      test::make_runtime_test_plan(), std::make_shared<OrdinaryPollStatusFactory>(probe));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto operator_vertex = std::ranges::find_if(
      plan.graph().vertices(),
      [](const VertexRecord& vertex) { return vertex.kind == VertexKind::kOperator; });
  ASSERT_NE(operator_vertex, plan.graph().vertices().end());

  RunSession session = run_async(plan);
  session.wait();

  ASSERT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  const Diagnostic& diagnostic = session.diagnostics().front();
  EXPECT_EQ(diagnostic.token, "RUNTIME_READINESS_BINDING_UNAVAILABLE");
  EXPECT_EQ(diagnostic.service, "readiness_poll");
  EXPECT_EQ(diagnostic.source.vertex, operator_vertex->id);
  const auto& detail = diagnostic.detail;
  HOLOSCAN_ASSERT_HAS_VALUE(detail);
  EXPECT_EQ(detail->operation, DiagnosticOperation::kRuntime);
  EXPECT_EQ(detail->error.code, ErrorCode::kProtocolError);
  EXPECT_GT(probe->poll_calls.load(std::memory_order_relaxed), 0U);
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// REQ: RUNTIME-001
// REQ: RUNTIME-004
// REQ: RUNTIME-035
// REQ: RUNTIME-054
// REQ: RUNTIME-060
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

// REQ: RUNTIME-058
TEST(RunSessionDeathTest, BoundedWaitFromDispatcherContextTerminatesUnderExternalSupervision) {
  EXPECT_DEATH(
      {
        test::RuntimeProbeState::reset();
        const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
        RunSession session = run_async(plan);
        const detail::RuntimeCallbackContextScope dispatcher_context{
            detail::RuntimeCallbackContext::kDispatcher};

        // Key assertion: The bounded form preserves the fatal callback-context contract.
        static_cast<void>(session.wait_for(std::chrono::milliseconds{1}));
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

// Bind each notification source to a real physical readiness binding and keep its control, so a
// test can publish to the source or fail it from its own thread.
class PhysicalNotificationFactory final : public detail::RuntimePortBindingFactory {
 public:
  struct Bound {
    SourceId source;
    detail::PhysicalReadinessControl control;
    ClockId reference_clock;
  };

  [[nodiscard]] expected<std::unique_ptr<detail::PortBinding>, Error> bind(
      const detail::RuntimePortBindingRequest& request,
      TimePoint start_coordinate) noexcept override {
    if (request.kind != detail::RuntimeEventKind::kNotification) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const detail::PhysicalReadinessBindingDescriptor descriptor{
        .request = request,
        .execution_lane = request.execution_lane,
    };
    auto physical = detail::PhysicalReadinessBindingFactory::create(
        std::span<const detail::PhysicalReadinessBindingDescriptor>{&descriptor, 1U});
    if (!physical) {
      return make_unexpected(std::move(physical.error()));
    }
    auto binding = (*physical)->bind(request, start_coordinate);
    if (!binding) {
      return make_unexpected(std::move(binding.error()));
    }
    auto control = (*physical)->control(
        request.source, request.plan_generation, request.run_generation, request.source_generation);
    if (!control) {
      return make_unexpected(std::move(control.error()));
    }
    {
      const std::lock_guard lock(mutex_);
      factories_.push_back(std::move(*physical));
      bound_.push_back(Bound{.source = request.source,
                             .control = std::move(*control),
                             .reference_clock = request.reference_clock});
    }
    cv_.notify_all();
    return std::move(*binding);
  }

  [[nodiscard]] std::vector<Bound> wait_for_bound(std::size_t count) const {
    std::unique_lock lock(mutex_);
    static_cast<void>(cv_.wait_for(lock, 5s, [&] { return bound_.size() >= count; }));
    return bound_;
  }

 private:
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  std::vector<std::shared_ptr<detail::PhysicalReadinessBindingFactory>> factories_;
  std::vector<Bound> bound_;
};

// Run once per notification of a source the test controls.
class LateDrainNotifiedProbe final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.notification_source(notification_, "notification")
        .capacity(1U)
        .sender_reference_capacity(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnNotified{.event = notification_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    compute_count.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }

  static inline std::atomic<std::size_t> compute_count{};

 private:
  NotificationSource notification_;
};

struct LateDrainNotificationRun {
  ExecutionPlan plan;
  std::shared_ptr<PhysicalNotificationFactory> factory;
  std::shared_ptr<detail::RuntimeLateReadinessDrainTestControl> late_drain;
};

// One notified probe per partition; one partition runs on the shared owner, more on partition
// owners. With `shared_probes`, that many probes instead share the shared owner. An authored
// partition holds one independent root, so they share one include-all partition.
[[nodiscard]] LateDrainNotificationRun make_late_drain_notification_run(
    std::size_t partitions, std::size_t shared_probes = 0U) {
  Graph graph{"late-drain-notification"};
  if (shared_probes != 0U) {
    for (std::size_t index = 0U; index < shared_probes; ++index) {
      static_cast<void>(graph.op<LateDrainNotifiedProbe>("probe" + std::to_string(index)));
    }
    graph.partition("worker").include_all();
  }
  for (std::size_t index = 0U; shared_probes == 0U && index < partitions; ++index) {
    const std::string name = "probe" + std::to_string(index);
    const auto probe = graph.op<LateDrainNotifiedProbe>(name);
    graph.partition("worker" + std::to_string(index)).add(probe);
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  LateDrainNotificationRun run{
      .plan = compile(graph),
      .factory = std::make_shared<PhysicalNotificationFactory>(),
      .late_drain = std::make_shared<detail::RuntimeLateReadinessDrainTestControl>(),
  };
  run.plan = detail::RuntimeTestPlanAccess::with_late_readiness_drain_test_control(
      detail::RuntimeTestPlanAccess::with_port_binding_factory(run.plan, run.factory),
      run.late_drain);
  // An owner reaches the armed late-drain pause only on a scan round, and arming it does not wake
  // the owner, so keep the 1 ms fallback rather than a park until the next deadline.
  run.plan = detail::RuntimeTestPlanAccess::with_owner_park_cap(run.plan, 1ms);
  return run;
}

// A source that fails after its owner's last drain is closed, and closed() alone does not tell a
// failure from a normal close. Fail the last open source while its owner is paused before the late
// drain, which an idle round now skips: the run must fail, not complete.
// REQ: RUNTIME-012
TEST(RunSessionTest, SourceFailureAfterTheLastDrainFailsTheRun) {
  for (const std::size_t partitions : {std::size_t{1U}, std::size_t{2U}}) {
    SCOPED_TRACE(testing::Message() << "partitions=" << partitions);
    LateDrainNotifiedProbe::compute_count.store(0U);
    const LateDrainNotificationRun run = make_late_drain_notification_run(partitions);
    ASSERT_TRUE(run.plan.ok()) << run.plan.json();

    RunSession session = run_async(run.plan);
    const auto bound = run.factory->wait_for_bound(partitions);
    ASSERT_EQ(bound.size(), partitions);
    // Key setup: pause the failing source's owner after an ordinary pass that found nothing.
    run.late_drain->arm(bound.front().source);
    const bool paused = run.late_drain->wait_until_paused();
    bound.front().control.fail(ErrorCode::kProtocolError);
    run.late_drain->release();
    // Any other source closes normally, so only the failure can keep the run from completing.
    for (std::size_t index = 1U; index < bound.size(); ++index) {
      bound[index].control.close();
    }
    const bool terminal = session.wait_for(5s);
    if (!terminal) {
      session.request_stop();
    }
    session.wait();

    // Key assertion: the failure, observed only through the closed binding, fails the run.
    ASSERT_TRUE(paused);
    EXPECT_TRUE(terminal);
    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    const auto& diagnostics = session.diagnostics();
    EXPECT_TRUE(std::ranges::any_of(diagnostics, [](const auto& diagnostic) {
      return diagnostic.token == "RUNTIME_READINESS_BINDING_UNAVAILABLE";
    }));
    EXPECT_EQ(LateDrainNotifiedProbe::compute_count.load(), 0U);
  }
}

// A source failure outranks a stop. Fail one of two sources in the same owner while that owner is
// paused before the late drain, which an idle round now skips, then stop the run. The open peer
// keeps the owner from auditing its sources as closed, so only the shutdown audit can still see the
// failure. Without the failure, the same stop ends the run stopped.
// REQ: RUNTIME-046
TEST(RunSessionTest, SourceFailureBeforeAStopFailsTheRunWhileAPeerStaysOpen) {
  for (const bool fail : {true, false}) {
    SCOPED_TRACE(testing::Message() << "fail=" << fail);
    LateDrainNotifiedProbe::compute_count.store(0U);
    const LateDrainNotificationRun run = make_late_drain_notification_run(1U, 2U);
    ASSERT_TRUE(run.plan.ok()) << run.plan.json();

    RunSession session = run_async(run.plan);
    const auto bound = run.factory->wait_for_bound(2U);
    ASSERT_EQ(bound.size(), 2U);
    run.late_drain->arm(bound.front().source);
    const bool paused = run.late_drain->wait_until_paused();
    if (fail) {
      bound.front().control.fail(ErrorCode::kProtocolError);
    }
    // Key setup: the peer source stays open when the stop arrives.
    session.request_stop();
    run.late_drain->release();
    const bool terminal = session.wait_for(5s);
    session.wait();

    ASSERT_TRUE(paused);
    EXPECT_TRUE(terminal);
    const bool binding_failed =
        std::ranges::any_of(session.diagnostics(), [](const auto& diagnostic) {
          return diagnostic.token == "RUNTIME_READINESS_BINDING_UNAVAILABLE";
        });
    // Key assertion: the failure that preceded the stop wins.
    EXPECT_EQ(session.termination(), fail ? RunTermination::kFailed : RunTermination::kStopped);
    EXPECT_EQ(binding_failed, fail);
    EXPECT_EQ(LateDrainNotifiedProbe::compute_count.load(), 0U);
  }
}

// A notification published while its owner is paused before the late drain is still delivered:
// the idle round skips the late drain, and the post's wake starts the next round, which finds it.
// REQ: RUNTIME-012
// REQ: RUNTIME-025
TEST(RunSessionTest, NotificationDuringASkippedLateDrainReachesCompute) {
  test::RuntimeNotificationProbeState::reset();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto late_drain = std::make_shared<detail::RuntimeLateReadinessDrainTestControl>();
  late_drain->arm();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_late_readiness_drain_test_control(
      detail::RuntimeTestPlanAccess::with_phase_counters(
          test::make_runtime_notification_test_plan(), counters),
      late_drain);
  ASSERT_TRUE(plan.ok()) << plan.json();

  // Use the production notification registry and its sleeping owner, including the wake callback.
  RunSession session = run_async(plan);
  const bool paused = late_drain->wait_until_paused();
  const bool sender_ready = paused && test::RuntimeNotificationProbeState::wait_until_sender();
  const auto before = counters->snapshot();
  const NotificationSender sender = test::RuntimeNotificationProbeState::sender_copy();
  const auto posted = sender_ready
                          ? sender.post()
                          : expected<void, Error>{make_unexpected(Error{ErrorCode::kNotReady})};
  const auto after_post = counters->snapshot();
  late_drain->release();
  const bool entered = posted && test::RuntimeNotificationProbeState::wait_until_compute_enters();
  // Compute holds the owner here, excluding later idle waits from this exact interval.
  const auto at_compute = counters->snapshot();
  test::RuntimeNotificationProbeState::release_blocked_compute();
  const bool computed =
      entered && test::RuntimeNotificationProbeState::wait_until_compute_count(1U);
  session.request_stop();
  session.wait();

  // Check before teardown signals can mask a missing post wake. The held idle scan drained all
  // earlier tokens, so the notification must publish a signal and the next wait must consume it.
  ASSERT_TRUE(paused);
  ASSERT_TRUE(sender_ready);
  ASSERT_TRUE(posted) << static_cast<int>(posted.error().code);
  EXPECT_TRUE(entered);
  EXPECT_TRUE(computed);
  const auto index = [](detail::RuntimePhaseCounter counter) {
    return detail::RuntimePhaseCounters::index(counter);
  };
  EXPECT_EQ(after_post[index(detail::RuntimePhaseCounter::kWakeSignals)],
            before[index(detail::RuntimePhaseCounter::kWakeSignals)] + 1U);
  EXPECT_EQ(at_compute[index(detail::RuntimePhaseCounter::kWakeReturns)],
            before[index(detail::RuntimePhaseCounter::kWakeReturns)] + 1U);
  EXPECT_EQ(at_compute[index(detail::RuntimePhaseCounter::kTimeoutReturns)],
            before[index(detail::RuntimePhaseCounter::kTimeoutReturns)]);
  EXPECT_EQ(at_compute[index(detail::RuntimePhaseCounter::kLateReadinessDrainsElided)],
            before[index(detail::RuntimePhaseCounter::kLateReadinessDrainsElided)] + 1U);
  {
    const std::lock_guard lock{test::RuntimeNotificationProbeState::mutex};
    EXPECT_EQ(test::RuntimeNotificationProbeState::compute_count, 1U);
  }
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
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
// REQ: RUNTIME-076
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
// REQ: RUNTIME-081
TEST(RunSessionTest, PredicateFailureRetainsExactPrivateCandidateContext) {
  test::RuntimeProbeState::reset();
  const auto sink = std::make_shared<RecordingDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      test::make_runtime_clock_predicate_error_plan(), sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.predicate_triggers().size(), 1U);
  ASSERT_EQ(plan.max_start_delays().size(), 1U);
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
  EXPECT_NE(session.diagnostics().front().token, "MAX_START_DELAY_MISS");
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

// REQ: RUNTIME-058
TEST(RunSessionTest, BoundedWaitReportsTimeoutThenTerminalPublication) {
  // Key setup: Register a long bounded waiter before requesting the session stop.
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  auto wait_control = std::make_shared<detail::RuntimeWaitTestControl>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_wait_interleaving(
      test::enable_direct_runtime(test::make_runtime_test_plan()), wait_control);
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());
  EXPECT_FALSE(session.wait_for(std::chrono::nanoseconds::zero()));
  EXPECT_FALSE(session.wait_for(std::chrono::milliseconds{1}));

  bool terminal{};
  std::chrono::steady_clock::duration elapsed{};
  std::thread waiter([&] {
    const auto start = std::chrono::steady_clock::now();
    terminal = session.wait_for(std::chrono::seconds{30});
    elapsed = std::chrono::steady_clock::now() - start;
  });
  EXPECT_TRUE(wait_control->wait_for_bounded_waiters(1U));
  session.request_stop();
  waiter.join();

  // Key assertion: Terminal publication wakes the registered waiter instead of awaiting expiry.
  EXPECT_TRUE(terminal);
  EXPECT_LT(elapsed, std::chrono::seconds{2});
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  wait_control->release_join_owner();
  session.wait();
}

// REQ: RUNTIME-058
TEST(RunSessionTest, WaitAdmissionRejectsCallbackContextWithoutTerminating) {
  // Key setup: Query wait admission from ordinary and simulated dispatcher contexts.
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset(test::RuntimeProbeMode::kWaitForStop);
  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());

  EXPECT_TRUE(session.wait_admission());
  {
    const detail::RuntimeCallbackContextScope dispatcher_context{
        detail::RuntimeCallbackContext::kDispatcher};
    const auto rejected = session.wait_admission();

    // Key assertion: Callback-context waits fail as data without terminating the process.
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(rejected.error().message, "RUNTIME_SELF_JOIN_FORBIDDEN callback_context");
  }

  session.request_stop();
  session.wait();
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

#if GTEST_HAS_DEATH_TEST
void exercise_diagnostic_self_join(int context, bool fail_formatting) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan direct = test::enable_direct_runtime(test::make_runtime_test_plan());
  std::set_terminate([] { std::_Exit(std::current_exception() ? 91 : 90); });
  if (context != 0) {
    RunSession session = run_async(direct);
    const detail::RuntimeCallbackContextScope callback{detail::RuntimeCallbackContext::kDispatcher};
    std::optional<test::ScopedAllocationFailure> failure;
    if (fail_formatting) {
      std::size_t admission_allocations = 0U;
      if (context == 2) {
        // Bounded wait constructs its admission Error before reaching the fatal logger. Inject
        // after that work, so this scenario exercises reporting rather than Error allocation.
        const test::ScopedAllocationCounter allocations;
        const auto admission = session.wait_admission();
        if (admission)
          std::_Exit(93);
        admission_allocations = allocations.count();
      }
      failure.emplace(admission_allocations);
    }
    if (context == 2) {
      static_cast<void>(session.wait_for(std::chrono::nanoseconds{0}));
    } else {
      session.wait();
    }
  } else {
    const auto sink = std::make_shared<SelfJoinDiagnosticSink>(fail_formatting);
    const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(direct, sink);
    RunSession session = run_async(plan);
    sink->bind(session);
    session.wait();
  }
  std::_Exit(92);
}

// REQ: EXPECTED-032
// REQ: RUNTIME-003
TEST(RunSessionDeathTest, SelfJoinReportsBypassAutomaticPolicy) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (const int context : {0, 1, 2}) {
    for (const int policy : {0, 1, 2}) {
      EXPECT_EXIT(
          {
            test::set_test_environment("HOLOSCAN_LOG_LEVEL", nullptr);
            test::set_test_environment("HOLOSCAN_LOG_FORMAT", nullptr);
            test::set_test_environment("HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING", nullptr);
            if (policy == 2)
              test::set_test_environment("HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING", "OFF");
            set_automatic_diagnostic_logging_enabled(policy != 1);
            set_log_level(LogLevel::ERROR);
            exercise_diagnostic_self_join(context, false);
          },
          testing::ExitedWithCode(90),
          "RUNTIME_SELF_JOIN_FORBIDDEN");
    }
  }
}

// REQ: EXPECTED-032
// REQ: RUNTIME-003
TEST(RunSessionDeathTest, SelfJoinTerminationSurvivesFilteringAndFormattingFailure) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (const int context : {0, 1, 2}) {
    for (const bool fail_formatting : {false, true}) {
      EXPECT_EXIT(
          {
            test::set_test_environment("HOLOSCAN_LOG_LEVEL", nullptr);
            test::set_test_environment("HOLOSCAN_LOG_FORMAT", nullptr);
            test::set_test_environment("HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING", nullptr);
            set_automatic_diagnostic_logging_enabled(true);
            set_log_level(fail_formatting ? LogLevel::ERROR : LogLevel::OFF);
            exercise_diagnostic_self_join(context, fail_formatting);
          },
          testing::ExitedWithCode(90),
          "^$");
    }
  }
}
#endif

}  // namespace
}  // namespace holoscan
