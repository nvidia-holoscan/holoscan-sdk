/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/channel/runtime.hpp>

#include "common/allocation_counter.hpp"
#include "common/cuda_test_utils.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/domain/tensor_access.hpp"
#include "core/graph_compiler_access.hpp"
#include "run_session_tensor_test_support.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_fanout_test_control.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_tensor_resources.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_test_graph.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

using namespace test::run_session_tensor;

// Host raw-Tensor fanout separates successful delivery from publication pressure. Allocation
// exhaustion consumes a compute call, but neither a payload sequence nor a publication attempt.
struct RuntimeTensorFanoutExperimentProbe {
  static constexpr std::uint32_t kComputeCalls = 64U;
  static inline std::atomic<std::uint32_t> emitted{};
  static constexpr std::uint32_t kMaximumPressureEmits = 12U;
  static constexpr int kNoOutcome = -1;
  static constexpr int kNoPressure = -2;
  static inline std::atomic<bool> pressure{};
  static inline std::atomic<bool> release_slow{};
  static inline std::atomic<int> outcome{kNoOutcome};
  static inline std::atomic<std::uint32_t> attempts{};
  static inline std::atomic<std::uint32_t> compute_calls{};
  static inline std::atomic<std::uint32_t> allocation_attempts{};
  static inline std::atomic<std::uint32_t> allocation_skips{};
  static inline std::atomic<std::uint32_t> post_outcome_computes{};
  static inline std::array<std::atomic<std::uint32_t>, 2U> received{};
  static inline std::array<std::atomic<std::uint64_t>, 2U> last_value{};
  static inline std::atomic<bool> order_violation{};

  static void reset(const bool under_pressure) {
    pressure.store(under_pressure);
    release_slow.store(false);
    outcome.store(kNoOutcome);
    attempts.store(0U);
    compute_calls.store(0U);
    allocation_attempts.store(0U);
    allocation_skips.store(0U);
    post_outcome_computes.store(0U);
    emitted.store(0U);
    for (auto& count : received) {
      count.store(0U);
    }
    for (auto& value : last_value) {
      value.store(0U);
    }
    order_violation.store(false);
  }

  // Publish the outcome before releasing the sink that deliberately holds its first sample.
  static void settle(const int code) {
    outcome.store(code);
    release_slow.store(true);
  }

  template <typename Predicate>
  [[nodiscard]] static bool wait_for(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (!predicate()) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(100us);
    }
    return true;
  }
};

class RuntimeTensorFanoutExperimentSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                    .rank = 0U,
                },
            .bounds = tensor_bounds(8U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms})
        .complete_after(ComputeCalls{.limit = RuntimeTensorFanoutExperimentProbe::kComputeCalls});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    using Probe = RuntimeTensorFanoutExperimentProbe;
    Probe::compute_calls.fetch_add(1U);
    const bool pressure = Probe::pressure.load();
    if (pressure) {
      if (Probe::outcome.load() != Probe::kNoOutcome) {
        Probe::post_outcome_computes.fetch_add(1U);
        return {};
      }
      // Keep the fast recipient drained so only the gated recipient can limit publication.
      if (!Probe::wait_for([] { return Probe::received[0U].load() == Probe::emitted.load(); })) {
        Probe::settle(static_cast<int>(ErrorCode::kTimeout));
        return make_unexpected(Error{ErrorCode::kTimeout});
      }
    }
    Probe::allocation_attempts.fetch_add(1U);
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      if (loan.error().code == ErrorCode::kResourceExhausted) {
        Probe::allocation_skips.fetch_add(1U);
        return {};
      }
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write_host();
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      return make_unexpected(Error{writer ? ErrorCode::kFailure : writer.error().code});
    }
    *writer->data_as<std::uint64_t>() = ++value_;
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return make_unexpected(std::move(committed.error()));
    }
    Probe::attempts.fetch_add(1U);
    auto published = output.emit(std::move(*loan));
    if (!published) {
      if (!pressure) {
        return published;
      }
      const ErrorCode code = published.error().code;
      Probe::settle(static_cast<int>(code));
      const bool post_submit =
          code == ErrorCode::kPublicationPartial || code == ErrorCode::kPublicationIndeterminate;
      if (code != ErrorCode::kBackpressured && !post_submit) {
        return published;
      }
      // Let both recipients observe exactly what reached them before propagating the error.
      // The failed sample is never retried or counted as a successful publication.
      const std::uint32_t successes = Probe::emitted.load();
      if (!Probe::wait_for([&] {
            return Probe::received[1U].load() == successes &&
                   Probe::received[0U].load() == successes + (post_submit ? 1U : 0U);
          })) {
        return make_unexpected(Error{ErrorCode::kTimeout});
      }
      if (post_submit) {
        return published;
      }
      return {};
    }
    if (Probe::emitted.fetch_add(1U) + 1U >= Probe::kMaximumPressureEmits && pressure) {
      Probe::settle(Probe::kNoPressure);
    }
    return {};
  }

  Output<Tensor> output;

 private:
  std::uint64_t value_{};
};

template <std::size_t kSink, bool kSlow>
class RuntimeTensorFanoutExperimentSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(TensorInputSpec{.representation = {
                                            .memory_kind = MemoryKind::kHost,
                                            .dtype = DLDataType{kDLUInt, 64U, 1U},
                                            .rank = 0U,
                                        }});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    using Probe = RuntimeTensorFanoutExperimentProbe;
    auto sample = input.receive();
    if (!sample || sample->data.data_as<std::uint64_t>() == nullptr) {
      return make_unexpected(Error{sample ? ErrorCode::kFailure : sample.error().code});
    }
    const std::uint64_t value = *sample->data.data_as<std::uint64_t>();
    if (value <= Probe::last_value[kSink].exchange(value)) {
      Probe::order_violation.store(true);
    }
    Probe::received[kSink].fetch_add(1U);
    if constexpr (kSlow) {
      if (Probe::pressure.load()) {
        if (!Probe::wait_for([] { return Probe::release_slow.load(); })) {
          return make_unexpected(Error{ErrorCode::kTimeout});
        }
      } else {
        std::this_thread::sleep_for(2ms);
      }
    }
    return {};
  }

  Input<Tensor> input;
};

enum class TensorFanoutArm : std::uint8_t { kPump, kEmit, kEmitConcurrent };

[[nodiscard]] ExecutionPlan make_runtime_tensor_fanout_experiment_plan(
    const TensorFanoutArm arm, const std::uint32_t queue_depth,
    const std::shared_ptr<detail::RuntimePhaseCounters>& counters) {
  Graph graph{unique_runtime_graph_name("runtime-tensor-fanout-experiment")};
  auto source = graph.op<RuntimeTensorFanoutExperimentSourceOperator>("source");
  auto fast = graph.op<RuntimeTensorFanoutExperimentSinkOperator<0U, false>>("fast");
  auto slow = graph.op<RuntimeTensorFanoutExperimentSinkOperator<1U, true>>("slow");
  graph.add_flow(source->output, fast->input, ConnectionOptions{.queue_depth = queue_depth});
  graph.add_flow(source->output, slow->input, ConnectionOptions{.queue_depth = queue_depth});
  graph.partition("source-partition").add(source);
  graph.partition("fast-partition").add(fast);
  graph.partition("slow-partition").add(slow);
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(clock);
  ExecutionPlan plan =
      arm == TensorFanoutArm::kPump
          ? detail::compile_with_process_local_pump_delivery(graph, CompileOptions{})
          : detail::compile_with_process_local_fanout_experiment(
                graph, CompileOptions{}, detail::ProcessLocalFanoutExperiment::kEmitRecipients);
  if (!plan.ok()) {
    return plan;
  }
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  if (arm == TensorFanoutArm::kEmitConcurrent) {
    auto control = std::make_shared<detail::RuntimeFanoutTestControl>();
    control->concurrent_admission = true;
    plan = detail::RuntimeTestPlanAccess::with_fanout_test_control(plan, std::move(control));
  }
  return plan;
}

class RuntimeTensorFanoutExperiment : public ::testing::TestWithParam<TensorFanoutArm> {
 protected:
  [[nodiscard]] static RunSession run(
      const TensorFanoutArm arm, const std::uint32_t queue_depth,
      const std::shared_ptr<detail::RuntimePhaseCounters>& counters) {
    using Probe = RuntimeTensorFanoutExperimentProbe;
    const ExecutionPlan plan =
        make_runtime_tensor_fanout_experiment_plan(arm, queue_depth, counters);
    EXPECT_TRUE(plan.ok()) << plan.json();
    RunSession session = run_async(plan);
    if (!session.wait_for(10s)) {
      // Report where a stalled run stopped before stopping it, instead of hanging the suite.
      using Counter = detail::RuntimePhaseCounter;
      const auto snapshot = counters->snapshot();
      std::string counts;
      for (const auto& [name, counter] : std::array<std::pair<const char*, Counter>, 7U>{{
               {"known_blocked_skips", Counter::kChannelKnownBlockedSkips},
               {"scoped_retries", Counter::kChannelActivationCutScopedRetries},
               {"receive_backpressure", Counter::kChannelReceiveBackpressured},
               {"cuts_terminal", Counter::kChannelActivationCutsTerminal},
               {"miss_blocked", Counter::kChannelActivationCutMissBlockedLatch},
               {"miss_in_flight", Counter::kChannelActivationCutMissInFlight},
               {"miss_undelivered", Counter::kChannelActivationCutMissUndelivered},
           }}) {
        counts +=
            fmt::format(" {}={}", name, snapshot[detail::RuntimePhaseCounters::index(counter)]);
      }
      ADD_FAILURE() << "session did not complete: emitted=" << Probe::emitted.load()
                    << " allocation_skips=" << Probe::allocation_skips.load()
                    << " fast=" << Probe::received[0U].load()
                    << " slow=" << Probe::received[1U].load() << counts;
      Probe::release_slow.store(true);
      session.request_stop();
    }
    session.wait();

    return session;
  }

  static void record_counts() {
    using Probe = RuntimeTensorFanoutExperimentProbe;
    RecordProperty("compute_calls", static_cast<int>(Probe::compute_calls.load()));
    RecordProperty("allocation_attempts", static_cast<int>(Probe::allocation_attempts.load()));
    RecordProperty("allocation_skips", static_cast<int>(Probe::allocation_skips.load()));
    RecordProperty("post_outcome_computes", static_cast<int>(Probe::post_outcome_computes.load()));
    RecordProperty("publication_attempts", static_cast<int>(Probe::attempts.load()));
    RecordProperty("successful_publications", static_cast<int>(Probe::emitted.load()));
    RecordProperty("fast_received", static_cast<int>(Probe::received[0U].load()));
    RecordProperty("slow_received", static_cast<int>(Probe::received[1U].load()));
  }
};

// Queue capacity covers all 64 compute calls plus EOS. Pool exhaustion remains a pre-publication
// skip; every successful allocation must publish and reach both recipients in order.
// REQ: CHAN-136
TEST_P(RuntimeTensorFanoutExperiment, DeliversEveryPublishedHostTensorAcrossPartitions) {
  using Probe = RuntimeTensorFanoutExperimentProbe;
  Probe::reset(false);
  const TensorFanoutArm arm = GetParam();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const RunSession session = run(arm, Probe::kComputeCalls + 1U, counters);
  record_counts();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const std::uint32_t emitted = Probe::emitted.load();
  EXPECT_GT(emitted, 0U);
  EXPECT_EQ(Probe::compute_calls.load(), Probe::kComputeCalls);
  EXPECT_EQ(Probe::allocation_attempts.load(), Probe::kComputeCalls);
  EXPECT_EQ(Probe::attempts.load() + Probe::allocation_skips.load(), Probe::kComputeCalls);
  EXPECT_EQ(Probe::attempts.load(), emitted);
  EXPECT_EQ(Probe::post_outcome_computes.load(), 0U);
  EXPECT_EQ(Probe::received[0U].load(), emitted);
  EXPECT_EQ(Probe::received[1U].load(), emitted);
  EXPECT_EQ(Probe::last_value[0U].load(), emitted);
  EXPECT_EQ(Probe::last_value[1U].load(), emitted);
  EXPECT_FALSE(Probe::order_violation.load());
  const auto pump_waits = counters->snapshot()[detail::RuntimePhaseCounters::index(
      detail::RuntimePhaseCounter::kChannelPumpWaits)];
  if (arm == TensorFanoutArm::kPump) {
    EXPECT_GT(pump_waits, 0U);
  } else {
    EXPECT_EQ(pump_waits, 0U);
  }
}

// Hold the slow sink's first sample and keep the fast sink drained. The next blocked publication
// must report an explicit pre-submit or post-submit outcome. Opening the gate preserves exact
// recipient visibility and ordering; no later compute retries or replaces the failed sample.
// REQ: CHAN-108
// REQ: CHAN-136
TEST_P(RuntimeTensorFanoutExperiment, DepthOnePressureReportsTheBlockedPublicationOutcome) {
  using Probe = RuntimeTensorFanoutExperimentProbe;
  Probe::reset(true);
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const RunSession session = run(GetParam(), 1U, counters);

  const int outcome = Probe::outcome.load();
  const std::uint32_t emitted = Probe::emitted.load();
  RecordProperty("publication_outcome", outcome);
  record_counts();
  ASSERT_TRUE(outcome == static_cast<int>(ErrorCode::kBackpressured) ||
              outcome == static_cast<int>(ErrorCode::kPublicationPartial) ||
              outcome == static_cast<int>(ErrorCode::kPublicationIndeterminate))
      << "outcome=" << outcome << " emitted=" << emitted;
  EXPECT_GE(emitted, 1U);
  EXPECT_LT(emitted, Probe::kMaximumPressureEmits);
  EXPECT_EQ(Probe::attempts.load(), emitted + 1U);
  EXPECT_EQ(Probe::allocation_attempts.load(),
            Probe::attempts.load() + Probe::allocation_skips.load());
  EXPECT_EQ(Probe::compute_calls.load(),
            Probe::allocation_attempts.load() + Probe::post_outcome_computes.load());
  const bool post_submit = outcome != static_cast<int>(ErrorCode::kBackpressured);
  const std::uint32_t fast_count = emitted + (post_submit ? 1U : 0U);
  EXPECT_EQ(Probe::received[0U].load(), fast_count);
  EXPECT_EQ(Probe::last_value[0U].load(), fast_count);
  EXPECT_EQ(Probe::received[1U].load(), emitted);
  EXPECT_EQ(Probe::last_value[1U].load(), emitted);
  EXPECT_FALSE(Probe::order_violation.load());
  if (post_submit) {
    EXPECT_EQ(Probe::post_outcome_computes.load(), 0U);
    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    ASSERT_FALSE(session.diagnostics().empty());
    EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
  } else {
    EXPECT_EQ(Probe::compute_calls.load(), Probe::kComputeCalls);
    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  }
}

// Arm names give each case a readable test identity.
INSTANTIATE_TEST_SUITE_P(Arms, RuntimeTensorFanoutExperiment,
                         ::testing::Values(TensorFanoutArm::kPump, TensorFanoutArm::kEmit,
                                           TensorFanoutArm::kEmitConcurrent),
                         [](const ::testing::TestParamInfo<TensorFanoutArm>& info) {
                           switch (info.param) {
                             case TensorFanoutArm::kPump:
                               return std::string{"Pump"};
                             case TensorFanoutArm::kEmit:
                               return std::string{"Emit"};
                             case TensorFanoutArm::kEmitConcurrent:
                               return std::string{"EmitConcurrent"};
                           }
                           return std::string{"Unknown"};
                         });

// Host raw-Tensor fanout across execution partitions for the private fanout experiment. Each sink
// requires strictly increasing values; the slow sink holds its only typed slot until the producer
// has begun its next publication, so that publication meets a blocked head.
struct RuntimeTensorFanoutCreditProbe {
  static constexpr std::uint32_t kComputeCalls = 64U;
  static inline std::atomic<int> publication_error{-1};
  static inline std::atomic<std::uint32_t> emitted{};
  static inline std::atomic<std::uint32_t> skipped{};
  static inline std::array<std::atomic<std::uint32_t>, 2U> received{};
  static inline std::array<std::atomic<std::uint64_t>, 2U> last_value{};
  static inline std::atomic<bool> order_violation{};
  static inline std::atomic<std::uint64_t> publishing{};
  static inline std::atomic<bool> final_call{};

  static void reset() {
    publication_error.store(-1);
    emitted.store(0U);
    skipped.store(0U);
    publishing.store(0U);
    final_call.store(false);
    for (auto& count : received) {
      count.store(0U);
    }
    for (auto& value : last_value) {
      value.store(0U);
    }
    order_violation.store(false);
  }

  // Hold the slow sink's slot until the source has begun a later publication, so that
  // publication meets a blocked head, then briefly longer so it reaches the occupied slot.
  // Releasing on source progress rather than after a fixed sleep keeps each blocked send far
  // inside the 5 ms graph retry budget, even when CI delays the sink's thread.
  static void hold_slot_until_later_publication(const std::uint64_t value) {
    const auto deadline = std::chrono::steady_clock::now() + 20ms;
    while (publishing.load() <= value && !final_call.load() &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(50us);
    }
    std::this_thread::sleep_for(200us);
  }
};

class RuntimeTensorFanoutCreditSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation =
                TensorRepresentation{
                    .memory_kind = MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                    .rank = 0U,
                },
            .bounds = tensor_bounds(8U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms})
        .complete_after(ComputeCalls{.limit = RuntimeTensorFanoutCreditProbe::kComputeCalls});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    using Probe = RuntimeTensorFanoutCreditProbe;
    // On the final attempt, release the slow sink's last hold only when this callback exits:
    // after any publication was recorded, and also on an early return without one.
    struct FinalCallSignal {
      bool last;
      ~FinalCallSignal() {
        if (last) {
          Probe::final_call.store(true);
        }
      }
    } const final_call_signal{++compute_calls_ == Probe::kComputeCalls};
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      if (loan.error().code == ErrorCode::kResourceExhausted) {
        Probe::skipped.fetch_add(1U);
        return {};
      }
      return make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write_host();
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      return make_unexpected(Error{writer ? ErrorCode::kFailure : writer.error().code});
    }
    *writer->data_as<std::uint64_t>() = ++value_;
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return make_unexpected(std::move(committed.error()));
    }
    Probe::publishing.store(value_);
    auto published = output.emit(std::move(*loan));
    if (!published) {
      // Pre-submit pressure delivered nothing; drop the sample explicitly, as a lossy source would.
      if (published.error().code == ErrorCode::kBackpressured) {
        Probe::skipped.fetch_add(1U);
        return {};
      }
      Probe::publication_error.store(static_cast<int>(published.error().code));
      return published;
    }
    Probe::emitted.fetch_add(1U);
    return {};
  }

  Output<Tensor> output;

 private:
  std::uint64_t value_{};
  std::uint32_t compute_calls_{};
};

// kStallAtValue, when nonzero, stalls the sink for 20 ms on that value: four graph blocking
// budgets.
template <std::size_t kSink, bool kSlow, std::uint64_t kStallAtValue = 0U>
class RuntimeTensorFanoutCreditSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(TensorInputSpec{.representation = {
                                            .memory_kind = MemoryKind::kHost,
                                            .dtype = DLDataType{kDLUInt, 64U, 1U},
                                            .rank = 0U,
                                        }});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    using Probe = RuntimeTensorFanoutCreditProbe;
    auto sample = input.receive();
    if (!sample || sample->data.data_as<std::uint64_t>() == nullptr) {
      return make_unexpected(Error{sample ? ErrorCode::kFailure : sample.error().code});
    }
    const std::uint64_t value = *sample->data.data_as<std::uint64_t>();
    if (value <= Probe::last_value[kSink].exchange(value)) {
      Probe::order_violation.store(true);
    }
    Probe::received[kSink].fetch_add(1U);
    if constexpr (kSlow) {
      if (value == kStallAtValue) {
        std::this_thread::sleep_for(20ms);
      } else {
        Probe::hold_slot_until_later_publication(value);
      }
    }
    return {};
  }

  Input<Tensor> input;
};

template <std::uint64_t kStallAtValue = 0U>
[[nodiscard]] ExecutionPlan make_runtime_tensor_fanout_credit_plan(
    const TensorFanoutArm arm, const std::shared_ptr<detail::RuntimePhaseCounters>& counters,
    const bool disable_publication_credits = false) {
  Graph graph{unique_runtime_graph_name("runtime-tensor-fanout-experiment")};
  auto source = graph.op<RuntimeTensorFanoutCreditSourceOperator>("source");
  auto fast = graph.op<RuntimeTensorFanoutCreditSinkOperator<0U, false>>("fast");
  auto slow = graph.op<RuntimeTensorFanoutCreditSinkOperator<1U, true, kStallAtValue>>("slow");
  graph.add_flow(source->output, fast->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(source->output, slow->input, ConnectionOptions{.queue_depth = 1U});
  graph.partition("source-partition").add(source);
  graph.partition("fast-partition").add(fast);
  graph.partition("slow-partition").add(slow);
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(clock);
  ExecutionPlan plan =
      arm == TensorFanoutArm::kPump
          ? detail::compile_with_process_local_pump_delivery(graph, CompileOptions{})
          : detail::compile_with_process_local_fanout_experiment(
                graph, CompileOptions{}, detail::ProcessLocalFanoutExperiment::kEmitRecipients);
  if (!plan.ok()) {
    return plan;
  }
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  if (arm == TensorFanoutArm::kEmitConcurrent || disable_publication_credits) {
    auto control = std::make_shared<detail::RuntimeFanoutTestControl>();
    control->concurrent_admission = arm == TensorFanoutArm::kEmitConcurrent;
    control->disable_publication_credits = disable_publication_credits;
    plan = detail::RuntimeTestPlanAccess::with_fanout_test_control(plan, std::move(control));
  }
  return plan;
}

class RuntimeTensorFanoutCredits : public ::testing::TestWithParam<TensorFanoutArm> {};

// Publication credits, on by default: a recipient stalled beyond the graph blocking budget turns
// into pre-submit backpressure for its producer, delivering nothing to any recipient, instead of a
// partial or indeterminate fanout that fails the run. The source drops each rejected sample
// explicitly, and both sinks receive exactly the accepted publications, in order.
// REQ: CHAN-136
// REQ: CHAN-166
TEST_P(RuntimeTensorFanoutCredits, StalledRecipientRejectsPublicationBeforeSubmit) {
  using Probe = RuntimeTensorFanoutCreditProbe;
  Probe::reset();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan = make_runtime_tensor_fanout_credit_plan<10U>(GetParam(), counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  if (!session.wait_for(10s)) {
    ADD_FAILURE() << "session did not complete: emitted=" << Probe::emitted.load()
                  << " skipped=" << Probe::skipped.load();
    session.request_stop();
  }
  session.wait();

  // Key assertion: the stall produced pre-submit rejections, never a failed publication.
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const auto snapshot = counters->snapshot();
  const auto count = [&](const detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_GE(count(detail::RuntimePhaseCounter::kChannelPublicationCreditRejections), 1U);
  EXPECT_GE(count(detail::RuntimePhaseCounter::kChannelPublicationCreditWaits),
            count(detail::RuntimePhaseCounter::kChannelPublicationCreditRejections));
  // Key assertion: every accepted publication reached both sinks; rejected ones reached neither.
  const std::uint32_t emitted = Probe::emitted.load();
  EXPECT_GE(Probe::skipped.load(), 1U);
  EXPECT_EQ(emitted + Probe::skipped.load(), Probe::kComputeCalls);
  EXPECT_EQ(Probe::received[0U].load(), emitted);
  EXPECT_EQ(Probe::received[1U].load(), emitted);
  EXPECT_FALSE(Probe::order_violation.load());
}

// Control for the test above: with credits turned off, the same stall leaves HoloMQ to give up on
// the slow recipient after the fast one accepted, and the partial publication fails the run.
// REQ: CHAN-166
TEST_P(RuntimeTensorFanoutCredits, StalledRecipientFailsTheRunWithoutCredits) {
  using Probe = RuntimeTensorFanoutCreditProbe;
  Probe::reset();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      make_runtime_tensor_fanout_credit_plan<10U>(GetParam(), counters, true);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  if (!session.wait_for(10s)) {
    ADD_FAILURE() << "session did not fail on the blocked publication";
    session.request_stop();
  }
  session.wait();

  // A startup or unrelated callback failure must not satisfy this publication control.
  const int publication_error = Probe::publication_error.load();
  RecordProperty("publication_error", publication_error);
  RecordProperty("emitted", Probe::emitted.load());
  RecordProperty("fast_received", Probe::received[0U].load());
  RecordProperty("slow_received", Probe::received[1U].load());
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_FALSE(session.diagnostics().empty());
  EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
  EXPECT_TRUE(publication_error == static_cast<int>(ErrorCode::kPublicationPartial) ||
              publication_error == static_cast<int>(ErrorCode::kPublicationIndeterminate));
  EXPECT_GT(Probe::emitted.load(), 0U);
  EXPECT_GT(Probe::received[0U].load(), 0U);
  EXPECT_GT(Probe::received[1U].load(), 0U);
  EXPECT_FALSE(Probe::order_violation.load());
  const auto snapshot = counters->snapshot();
  EXPECT_EQ(snapshot[detail::RuntimePhaseCounters::index(
                detail::RuntimePhaseCounter::kChannelPublicationCreditWaits)],
            0U);
}

INSTANTIATE_TEST_SUITE_P(Arms, RuntimeTensorFanoutCredits,
                         ::testing::Values(TensorFanoutArm::kPump, TensorFanoutArm::kEmit,
                                           TensorFanoutArm::kEmitConcurrent),
                         [](const ::testing::TestParamInfo<TensorFanoutArm>& info) {
                           switch (info.param) {
                             case TensorFanoutArm::kPump:
                               return std::string{"Pump"};
                             case TensorFanoutArm::kEmit:
                               return std::string{"Emit"};
                             case TensorFanoutArm::kEmitConcurrent:
                               return std::string{"EmitConcurrent"};
                           }
                           return std::string{"Unknown"};
                         });

// What the mixed-locality Tensor export test's source and local sink observed.
struct RuntimeMixedExportTensorProbe {
  static void reset() noexcept {
    emit_code.store(-1, std::memory_order_release);
    settled_value.store(0U, std::memory_order_release);
    source_data.store(nullptr, std::memory_order_release);
    local_count.store(0U, std::memory_order_release);
    local_value.store(0U, std::memory_order_release);
    local_data.store(nullptr, std::memory_order_release);
  }

  // A recycled address is insufficient; match the successful attempt even if delivery won
  // the race with settlement. Acquiring emit_code also observes its settled_value publication.
  [[nodiscard]] static bool local_received_settled_publication() noexcept {
    if (emit_code.load(std::memory_order_acquire) != static_cast<int>(ErrorCode::kSuccess)) {
      return false;
    }
    return local_count.load(std::memory_order_acquire) != 0U &&
           local_value.load(std::memory_order_acquire) ==
               settled_value.load(std::memory_order_acquire);
  }

  static inline std::atomic<std::uint64_t> settled_value{};

  // The ErrorCode of the source's last emit(), or -1 before it settled.
  static inline std::atomic<int> emit_code{-1};
  static inline std::atomic<const void*> source_data{};
  static inline std::atomic<std::size_t> local_count{};
  static inline std::atomic<std::uint64_t> local_value{};
  static inline std::atomic<const void*> local_data{};
};

// A recycled backing address must not identify a different successful publication.
// REQ: CHAN-168
TEST(RunSessionTest, MixedExportObservationRejectsReusedBacking) {
  using Probe = RuntimeMixedExportTensorProbe;
  Probe::reset();
  const std::uint64_t backing{};
  Probe::source_data.store(&backing, std::memory_order_release);
  Probe::local_data.store(&backing, std::memory_order_release);
  Probe::local_value.store(41U, std::memory_order_release);
  Probe::local_count.store(1U, std::memory_order_release);
  Probe::settled_value.store(42U, std::memory_order_release);
  Probe::emit_code.store(static_cast<int>(ErrorCode::kSuccess), std::memory_order_release);
  EXPECT_FALSE(Probe::local_received_settled_publication());
  Probe::reset();
}

// Delivery may complete before the source publishes the successful settlement record.
// REQ: CHAN-168
TEST(RunSessionTest, MixedExportObservationAcceptsDeliveryBeforeSettlement) {
  using Probe = RuntimeMixedExportTensorProbe;
  Probe::reset();
  const std::uint64_t backing{};
  Probe::source_data.store(&backing, std::memory_order_release);
  Probe::local_data.store(&backing, std::memory_order_release);
  Probe::local_value.store(42U, std::memory_order_release);
  Probe::local_count.store(1U, std::memory_order_release);
  Probe::settled_value.store(42U, std::memory_order_release);
  EXPECT_FALSE(Probe::local_received_settled_publication());
  Probe::emit_code.store(static_cast<int>(ErrorCode::kSuccess), std::memory_order_release);
  EXPECT_TRUE(Probe::local_received_settled_publication());
  Probe::reset();
}

// A settled source waits for the matching delivery, without sampling a new count baseline.
// REQ: CHAN-168
TEST(RunSessionTest, MixedExportObservationWaitsForDeliveryAfterSettlement) {
  using Probe = RuntimeMixedExportTensorProbe;
  Probe::reset();
  EXPECT_FALSE(Probe::local_received_settled_publication());
  const std::uint64_t backing{};
  Probe::source_data.store(&backing, std::memory_order_release);
  Probe::settled_value.store(42U, std::memory_order_release);
  Probe::emit_code.store(static_cast<int>(ErrorCode::kSuccess), std::memory_order_release);
  EXPECT_FALSE(Probe::local_received_settled_publication());
  Probe::local_data.store(&backing, std::memory_order_release);
  Probe::local_value.store(42U, std::memory_order_release);
  Probe::local_count.store(1U, std::memory_order_release);
  EXPECT_TRUE(Probe::local_received_settled_publication());
  Probe::reset();
}

// Publishes one pool-backed host scalar. It retries a publication that its required export had no
// peer for yet, and stops after a success or a deterministic rejection.
class RuntimeMixedExportTensorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 64U, 1U},
                               .rank = 0U},
            .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = sizeof(std::uint64_t)},
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (settled_) {
      return {};
    }
    auto loan = output.allocate_tensor(
        TensorLoanRequest{.shape = TensorShapeView{}, .dtype = DLDataType{kDLUInt, 64U, 1U}});
    if (!loan) {
      return loan.error().code == ErrorCode::kResourceExhausted
                 ? expected<void, Error>{}
                 : make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write_host();
    if (!writer || writer->data_as<std::uint64_t>() == nullptr) {
      return make_unexpected(writer ? Error{ErrorCode::kFailure} : std::move(writer).error());
    }
    // Give each attempted publication a distinct scalar without changing its Tensor contract.
    const std::uint64_t value = kValue + attempt_++;
    *writer->data_as<std::uint64_t>() = value;
    RuntimeMixedExportTensorProbe::source_data.store(writer->data(), std::memory_order_release);
    if (auto committed = std::move(*writer).commit(); !committed) {
      return make_unexpected(std::move(committed).error());
    }
    auto emitted = output.emit(std::move(*loan));
    const ErrorCode code = emitted ? ErrorCode::kSuccess : emitted.error().code;
    settled_ = code == ErrorCode::kSuccess || code == ErrorCode::kInvalidArgument;
    if (settled_) {
      if (code == ErrorCode::kSuccess) {
        RuntimeMixedExportTensorProbe::settled_value.store(value, std::memory_order_release);
      }
      RuntimeMixedExportTensorProbe::emit_code.store(static_cast<int>(code),
                                                     std::memory_order_release);
    }
    return {};
  }

  Output<Tensor> output;
  static constexpr std::uint64_t kValue = 0x0123456789abcdefULL;

 private:
  bool settled_{};
  std::uint64_t attempt_{};
};

class RuntimeMixedExportTensorSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 64U, 1U},
                               .rank = 0U},
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample || sample->data.data_as<std::uint64_t>() == nullptr) {
      return make_unexpected(sample ? Error{ErrorCode::kFailure} : std::move(sample.error()));
    }
    RuntimeMixedExportTensorProbe::local_data.store(sample->data.data(), std::memory_order_release);
    RuntimeMixedExportTensorProbe::local_value.store(*sample->data.data_as<std::uint64_t>(),
                                                     std::memory_order_release);
    RuntimeMixedExportTensorProbe::local_count.fetch_add(1U, std::memory_order_acq_rel);
    return {};
  }

  Input<Tensor> input;
};

// A drop-oldest raw Tensor family with a Graph-internal member and an export keeps the internal
// member direct (CHAN-168). The output's pool is native, so the export copies each publication into
// its interface pool before the local store. With kCopyIfNeeded both arms receive the value and the
// local sink holds the producer's own backing; with kRequireExportable the publication fails before
// any store, and neither arm receives anything.
//
//   source.output --[drop-oldest]--> export      (copied into the interface pool)
//                 `-[drop-oldest]--> local.input (direct, the producer's backing)
//
// REQ: CHAN-168
TEST(RunSessionTest, MixedLocalityTensorExportCopiesBeforeTheLocalStore) {
  for (const InterfaceTensorEmitPolicy policy :
       {InterfaceTensorEmitPolicy::kCopyIfNeeded, InterfaceTensorEmitPolicy::kRequireExportable}) {
    const bool copy = policy == InterfaceTensorEmitPolicy::kCopyIfNeeded;
    SCOPED_TRACE(copy ? "copy if needed" : "require exportable");
    RuntimeMixedExportTensorProbe::reset();
    const std::string graph_name = unique_runtime_graph_name("runtime-mixed-locality-export");
    const std::string endpoint_name = graph_name + "-topic";
    Graph graph{graph_name};
    graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));
    const auto source = graph.op<RuntimeMixedExportTensorSource>("source");
    const auto local = graph.op<RuntimeMixedExportTensorSink>("local");
    TopicOptions topic_options;
    topic_options.endpoint_name = endpoint_name;
    topic_options.attach = TopicAttachPolicy{
        .peer = PeerRequirement::kRequired,
        .max_wait = 10s,
    };
    topic_options.bounds.subscriber = SubscriberEndpointBounds{
        .queue_depth = 1U,
        .max_borrowed_samples = 1U,
    };
    topic_options.tensor = TensorBoundaryContract{
        .representation = {.memory_kind = MemoryKind::kHost,
                           .dtype = DLDataType{kDLUInt, 64U, 1U},
                           .rank = 0U,
                           .strides = TensorStridePolicy::kContiguousOnly},
        .bounds = TensorBounds{.max_rank = 1U, .max_byte_span = sizeof(std::uint64_t)},
        .transfer = TensorTransferPolicy{.foreign_emit_policy = policy},
    };
    auto exported =
        graph.export_topic<Tensor>("holomq://" + endpoint_name, std::move(topic_options));
    const ConnectionOptions drop_oldest{.queue_policy = QueuePolicy::kDropOldest,
                                        .queue_depth = 1U};
    graph.add_flow(source->output, exported.input(), drop_oldest);
    graph.add_flow(source->output, local->input, drop_oldest);

    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto state = detail::ExecutionPlanAccess::retain(plan);
    ASSERT_NE(state, nullptr);
    ASSERT_EQ(state->process_local_channel_services.size(), 1U);
    EXPECT_EQ(state->process_local_channel_services.front().delivery_kind,
              detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
    auto client_runtime = ChannelRuntime::create();
    ASSERT_TRUE(client_runtime) << runtime_error_message(client_runtime.error());

    RunSession session = run_async(plan);
    auto subscriber =
        client_runtime->create_subscriber<Tensor>(plan.endpoint_catalog(), endpoint_name);
    const auto settled = [] {
      const auto deadline = std::chrono::steady_clock::now() + 5s;
      while (RuntimeMixedExportTensorProbe::emit_code.load(std::memory_order_acquire) < 0 &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(1ms);
      }
      return RuntimeMixedExportTensorProbe::emit_code.load(std::memory_order_acquire);
    }();
    expected<Sample<Tensor>, Error> received = make_unexpected(Error{ErrorCode::kNotReady});
    std::uint64_t exported_value{};
    if (subscriber && copy) {
      const auto deadline = std::chrono::steady_clock::now() + 5s;
      while (!received && received.error().code == ErrorCode::kNotReady &&
             std::chrono::steady_clock::now() < deadline) {
        received = subscriber->receive_sample();
        if (!received) {
          std::this_thread::sleep_for(1ms);
        }
      }
      if (received && received->data.data_as<std::uint64_t>() != nullptr) {
        exported_value = *received->data.data_as<std::uint64_t>();
      }
      // Earlier attempts that found no export peer were stored locally too; wait for the local
      // sink to take the settled publication.
      const auto local_deadline = std::chrono::steady_clock::now() + 5s;
      while (!RuntimeMixedExportTensorProbe::local_received_settled_publication() &&
             std::chrono::steady_clock::now() < local_deadline) {
        std::this_thread::sleep_for(1ms);
      }
    }
    session.request_stop();
    session.wait();

    // Key assertion: a copy reaches the export and the local sink holds the producer's backing;
    // an unexportable publication is rejected before any store.
    ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    if (copy) {
      EXPECT_EQ(settled, static_cast<int>(ErrorCode::kSuccess));
      ASSERT_TRUE(received) << runtime_error_message(received.error());
      const std::uint64_t settled_value =
          RuntimeMixedExportTensorProbe::settled_value.load(std::memory_order_acquire);
      EXPECT_GE(settled_value, RuntimeMixedExportTensorSource::kValue);
      EXPECT_EQ(exported_value, settled_value);
      EXPECT_TRUE(RuntimeMixedExportTensorProbe::local_received_settled_publication());
      EXPECT_GE(RuntimeMixedExportTensorProbe::local_count.load(std::memory_order_acquire), 1U);
      EXPECT_EQ(RuntimeMixedExportTensorProbe::local_value.load(std::memory_order_acquire),
                settled_value);
      EXPECT_EQ(RuntimeMixedExportTensorProbe::local_data.load(std::memory_order_acquire),
                RuntimeMixedExportTensorProbe::source_data.load(std::memory_order_acquire));
    } else {
      EXPECT_EQ(settled, static_cast<int>(ErrorCode::kInvalidArgument));
      EXPECT_EQ(RuntimeMixedExportTensorProbe::local_count.load(std::memory_order_acquire), 0U);
      auto nothing = subscriber->receive_sample();
      EXPECT_FALSE(nothing);
    }
    received = make_unexpected(Error{ErrorCode::kNotReady});
    EXPECT_TRUE(subscriber->close());
  }
}

}  // namespace
}  // namespace holoscan
