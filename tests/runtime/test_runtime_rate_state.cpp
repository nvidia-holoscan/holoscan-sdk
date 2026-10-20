/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../support/optional_assertions.hpp"
#include "common/allocation_counter.hpp"
#include "compiler/closed_island_activation_plan.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/inline_partition_execution_plan.hpp"
#include "compiler/partition_runner_plan.hpp"
#include "compiler/process_placement_internal.hpp"
#include "core/diagnostic_registry.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_rate_state.hpp"
#include "runtime/runtime_worker.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

class RuntimeRateFixedSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").nominal_rate(Rate::hz(30U));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

class RuntimeRateReplaySource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .nominal_rate(Rate::hz(numerator.load(std::memory_order_relaxed),
                               denominator.load(std::memory_order_relaxed)));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
  static inline std::atomic<std::uint64_t> numerator{30U};
  static inline std::atomic<std::uint64_t> denominator{1U};
};

class RuntimeRateFollower final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output").follows(input, RateRatio{2U, 1U});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
};

class RuntimeRateLifecycleSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .lifecycle_nominal_rate(LifecycleStage::kDiscover)
        .max_nominal_rate(Rate::hz(120U));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

class RuntimeRateRemoteLifecycleSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .lifecycle_nominal_rate(LifecycleStage::kDiscover)
        .max_nominal_rate(Rate::hz(120U));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

class RuntimeRateClockSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s}).input(input, Exact{}).on_missing(input, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
};

class RuntimeRateDrainAllClockSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s}).input(input, DrainAll{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
};

class RuntimeRateKilohertzSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").nominal_rate(Rate::hz(1000U));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

class RuntimeRateCameraDrivenFusion final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(camera, "camera");
    spec.input(imu, "imu");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{camera}).input(imu, DrainAll{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> camera;
  Input<std::uint32_t> imu;
};

class RuntimeRateExpectedOnEachSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").expected_rate(Rate::hz(60U));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
};

class RuntimeRateUnspecifiedSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> output;
};

class RuntimeRateExpectedClockSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").expected_rate(Rate::hz(60U));
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s}).input(input, Exact{}).on_missing(input, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
};

class RuntimeRateWindowSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(anchor, "anchor");
    spec.input(window, "window");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnWindow{.anchor = anchor, .max_wait = 1s, .max_pending_windows = 2U})
        .input(anchor, Exact{})
        .input(window, BracketedWindow{.lookbehind = 1U, .lookahead = 1U, .max_samples = 3U})
        .on_missing(window, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> anchor;
  Input<std::uint32_t> window;
};

class RuntimeRateLifecycleTensorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .lifecycle_nominal_rate(LifecycleStage::kDiscover)
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U,
                               .strides = TensorStridePolicy::kContiguousOnly},
            .bounds = tensor_bounds(64U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

class RuntimeRateFixedTensorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .nominal_rate(Rate::hz(30U))
        .produces_tensor(TensorOutputSpec{
            .representation = {.memory_kind = MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U,
                               .strides = TensorStridePolicy::kContiguousOnly},
            .bounds = tensor_bounds(64U),
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

class RuntimeRateTensorClockSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1s}).input(input, Exact{}).on_missing(input, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
};

template <typename OperatorT>
[[nodiscard]] OperatorT* find_runtime_operator(detail::GraphExec& exec) noexcept {
  for (auto& instance : exec.operators()) {
    if (auto* const result = dynamic_cast<OperatorT*>(instance.op.get()); result != nullptr) {
      return result;
    }
  }
  return nullptr;
}

template <typename OperatorT>
[[nodiscard]] OperatorInstanceId find_runtime_operator_id(detail::GraphExec& exec) noexcept {
  for (auto& instance : exec.operators()) {
    if (dynamic_cast<OperatorT*>(instance.op.get()) != nullptr) {
      return instance.operator_id;
    }
  }
  return {};
}

[[nodiscard]] expected<RateGenerationSnapshot, Error> capture_complete_tensor_rate_snapshot(
    const ExecutionPlan& plan, RunGenerationId run_generation) {
  auto exec =
      detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan), run_generation);
  if (!exec) {
    return make_unexpected(std::move(exec.error()));
  }
  auto rates =
      detail::RuntimeRateState::create((*exec)->plan(), (*exec)->operators(), run_generation, 1U);
  if (!rates) {
    return make_unexpected(std::move(rates.error()));
  }
  auto* const source = find_runtime_operator<RuntimeRateLifecycleTensorSource>(**exec);
  const OperatorInstanceId owner =
      find_runtime_operator_id<RuntimeRateLifecycleTensorSource>(**exec);
  if (source == nullptr || owner == 0U) {
    return make_unexpected(Error{ErrorCode::kFailure, "RATE_SNAPSHOT_TEST_SOURCE_MISSING"});
  }
  if (auto started = (*rates)->begin_startup_candidate(); !started) {
    return make_unexpected(std::move(started.error()));
  }
  if (auto bound = (*rates)->bind_nominal_rate(owner,
                                               LifecycleStage::kDiscover,
                                               LifecycleTransitionPhase::kStartupForward,
                                               &source->output,
                                               Rate::hz(60U));
      !bound) {
    return make_unexpected(std::move(bound.error()));
  }
  if (auto closed = (*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                                  LifecycleTransitionPhase::kStartupForward);
      !closed) {
    return make_unexpected(std::move(closed.error()));
  }
  std::uint8_t visitor_state{};
  auto visited = (*rates)->visit_candidate_materialization(
      &visitor_state,
      [](void* state,
         const detail::RuntimeRateMaterializationView& view) noexcept -> expected<void, Error> {
        if (state == nullptr || view.provider_results.size() != 1U ||
            view.pool_capacities.size() != 1U) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
        view.provider_results.front() = detail::RuntimeProviderMaterializationResult{
            .provider = view.pool_capacities.front().provider,
            .action = AllocationAction::kFresh,
            .materialized_entries = view.pool_capacities.front().required_entries,
            .reported = true,
        };
        return {};
      });
  if (!visited) {
    return make_unexpected(std::move(visited.error()));
  }
  if (auto preflight = (*rates)->preflight_candidate_publication(); !preflight) {
    return make_unexpected(std::move(preflight.error()));
  }
  if (auto published = (*rates)->publish_candidate(); !published) {
    return make_unexpected(std::move(published.error()));
  }
  return (*rates)->capture_snapshot();
}

// REQ: LIFECYCLE-012
// Publication is a checked transaction boundary: missing and unpreflighted candidates fail
// explicitly, while one fully preflighted candidate publishes exactly once.
TEST(RuntimeRateStateTest, CandidatePublicationFailsClosedUntilPreflight) {
  Graph graph{"runtime-rate-publication-guard"};
  graph.op<RuntimeRateFixedSource>("source");
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{72U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{72U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;

  auto missing = (*rates)->publish_candidate();
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().code, ErrorCode::kLifecycleRunState);
  EXPECT_EQ(missing.error().message, "PORT_RATE_CANDIDATE_MISSING");

  ASSERT_TRUE((*rates)->begin_startup_candidate());
  auto unpreflighted = (*rates)->publish_candidate();
  ASSERT_FALSE(unpreflighted);
  EXPECT_EQ(unpreflighted.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(unpreflighted.error().message, "PORT_RATE_PUBLICATION_MISSING");

  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());
  auto repeated = (*rates)->publish_candidate();
  ASSERT_FALSE(repeated);
  EXPECT_EQ(repeated.error().code, ErrorCode::kLifecycleRunState);
  EXPECT_EQ(repeated.error().message, "PORT_RATE_CANDIDATE_MISSING");
}

// REQ: LIFECYCLE-012
// Storage-binding identity capacity is part of preflight, before any other restart participant can
// commit. Exhaustion must poison the private candidate and leave publication unavailable.
TEST(RuntimeRateStateTest, CandidatePreflightRejectsExhaustedStorageBindingIdentity) {
  Graph graph{"runtime-rate-publication-capacity"};
  const auto source = graph.op<RuntimeRateLifecycleTensorSource>("source");
  const auto sink = graph.op<RuntimeRateTensorClockSink>("sink");
  graph.add_flow(source->output, sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{73U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{73U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  auto* const runtime_source = find_runtime_operator<RuntimeRateLifecycleTensorSource>(**exec);
  const OperatorInstanceId owner =
      find_runtime_operator_id<RuntimeRateLifecycleTensorSource>(**exec);
  ASSERT_NE(runtime_source, nullptr);
  ASSERT_NE(owner, 0U);

  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kStartupForward,
                                          &runtime_source->output,
                                          Rate::hz(60U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kStartupForward));
  std::uint8_t visitor_state{};
  ASSERT_TRUE((*rates)->visit_candidate_materialization(
      &visitor_state,
      [](void*,
         const detail::RuntimeRateMaterializationView& view) noexcept -> expected<void, Error> {
        if (view.provider_results.size() != 1U || view.pool_capacities.size() != 1U) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
        view.provider_results.front() = detail::RuntimeProviderMaterializationResult{
            .provider = view.pool_capacities.front().provider,
            .action = AllocationAction::kFresh,
            .materialized_entries = view.pool_capacities.front().required_entries,
            .reported = true,
        };
        return {};
      }));

  detail::RuntimeRateStateTestAccess::set_next_storage_binding_generation(**rates, 0U);
  auto preflight = (*rates)->preflight_candidate_publication();
  ASSERT_FALSE(preflight);
  EXPECT_EQ(preflight.error().code, ErrorCode::kResourceExhausted);
  EXPECT_EQ(preflight.error().message, "QUEUE_ALLOCATION_GENERATION_INVALID");

  auto publish = (*rates)->publish_candidate();
  ASSERT_FALSE(publish);
  EXPECT_EQ(publish.error().code, preflight.error().code);
  EXPECT_EQ(publish.error().message, preflight.error().message);
}

// REQ: LIFECYCLE-012
TEST(RuntimeRateStateTest, ResolvesConnectedInputsAndOwnerLocalFollowsTransitively) {
  Graph graph{"runtime-rate-propagation"};
  const auto source = graph.op<RuntimeRateFixedSource>("source");
  const auto follower = graph.op<RuntimeRateFollower>("follower");
  graph.add_flow(source->output, follower->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{73U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{73U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());

  auto* const runtime_source = find_runtime_operator<RuntimeRateFixedSource>(**exec);
  auto* const runtime_follower = find_runtime_operator<RuntimeRateFollower>(**exec);
  ASSERT_NE(runtime_source, nullptr);
  ASSERT_NE(runtime_follower, nullptr);
  auto source_pin =
      (*rates)->pin_published(find_runtime_operator_id<RuntimeRateFixedSource>(**exec));
  auto follower_pin =
      (*rates)->pin_published(find_runtime_operator_id<RuntimeRateFollower>(**exec));
  ASSERT_TRUE(source_pin);
  ASSERT_TRUE(follower_pin);

  auto source_view =
      detail::RuntimeRateGenerationPin::resolve(&*source_pin, &runtime_source->output, true);
  auto input_view =
      detail::RuntimeRateGenerationPin::resolve(&*follower_pin, &runtime_follower->input, false);
  auto output_view =
      detail::RuntimeRateGenerationPin::resolve(&*follower_pin, &runtime_follower->output, true);
  ASSERT_TRUE(source_view);
  ASSERT_TRUE(input_view);
  ASSERT_TRUE(output_view);
  EXPECT_EQ(source_view->resolved_nominal_rate, Rate::hz(30U));
  EXPECT_EQ(source_view->max_nominal_rate, Rate::hz(30U));
  EXPECT_EQ(source_view->origin, ResolvedRateOrigin::kOutputNominal);
  EXPECT_EQ(input_view->resolved_nominal_rate, Rate::hz(30U));
  EXPECT_EQ(input_view->max_nominal_rate, Rate::hz(30U));
  EXPECT_EQ(input_view->origin, ResolvedRateOrigin::kConnectedOutput);
  EXPECT_EQ(output_view->resolved_nominal_rate, Rate::hz(60U));
  EXPECT_EQ(output_view->max_nominal_rate, Rate::hz(60U));
  EXPECT_EQ(output_view->origin, ResolvedRateOrigin::kFollowsExpression);
  EXPECT_EQ(source_view->rate_generation, input_view->rate_generation);
  EXPECT_EQ(input_view->rate_generation, output_view->rate_generation);

  const auto measure_failed_query = [](auto&& query) {
    const test::ScopedAllocationCounter allocations;
    auto result = query();
    return std::pair{std::move(result), allocations.count()};
  };
  auto [wrong_direction, wrong_direction_allocations] = measure_failed_query([&] {
    return detail::RuntimeRateGenerationPin::resolve(
        &*follower_pin, &runtime_follower->input, true);
  });
  ASSERT_FALSE(wrong_direction);
  EXPECT_EQ(wrong_direction.error().code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(wrong_direction.error().message.empty());
  EXPECT_EQ(wrong_direction_allocations, 0U);

  auto [foreign_declaration, foreign_declaration_allocations] = measure_failed_query([&] {
    return detail::RuntimeRateGenerationPin::resolve(&*source_pin, &runtime_follower->output, true);
  });
  ASSERT_FALSE(foreign_declaration);
  EXPECT_EQ(foreign_declaration.error().code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(foreign_declaration.error().message.empty());
  EXPECT_EQ(foreign_declaration_allocations, 0U);

  auto retained_source_pin = std::move(*source_pin);
  auto [moved_from_pin, moved_from_pin_allocations] = measure_failed_query([&] {
    return detail::RuntimeRateGenerationPin::resolve(&*source_pin, &runtime_source->output, true);
  });
  ASSERT_FALSE(moved_from_pin);
  EXPECT_EQ(moved_from_pin.error().code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(moved_from_pin.error().message.empty());
  EXPECT_EQ(moved_from_pin_allocations, 0U);
  auto retained_source_view = detail::RuntimeRateGenerationPin::resolve(
      &retained_source_pin, &runtime_source->output, true);
  ASSERT_TRUE(retained_source_view);
  EXPECT_EQ(*retained_source_view, *source_view);
}

// Lifecycle bindings are accepted only from the declared owner's exact forward-stage callback and
// only for a valid lifecycle-selected output value. Unauthorized attempts fail the private
// candidate without changing the declaration graph.
// REQ: LIFECYCLE-012
TEST(RuntimeRateStateTest, RejectsInvalidOrUnauthorizedLifecycleBindings) {
  Graph graph{"runtime-rate-binding-authorization"};
  graph.op<RuntimeRateLifecycleSource>("lifecycle-source");
  const auto fixed_source = graph.op<RuntimeRateFixedSource>("fixed-source");
  const auto follower = graph.op<RuntimeRateFollower>("follower");
  graph.add_flow(fixed_source->output, follower->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{74U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{74U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  auto* const lifecycle_source = find_runtime_operator<RuntimeRateLifecycleSource>(**exec);
  auto* const runtime_fixed_source = find_runtime_operator<RuntimeRateFixedSource>(**exec);
  auto* const runtime_follower = find_runtime_operator<RuntimeRateFollower>(**exec);
  const OperatorInstanceId lifecycle_owner =
      find_runtime_operator_id<RuntimeRateLifecycleSource>(**exec);
  const OperatorInstanceId fixed_owner = find_runtime_operator_id<RuntimeRateFixedSource>(**exec);
  const OperatorInstanceId follower_owner = find_runtime_operator_id<RuntimeRateFollower>(**exec);
  ASSERT_NE(lifecycle_source, nullptr);
  ASSERT_NE(runtime_fixed_source, nullptr);
  ASSERT_NE(runtime_follower, nullptr);
  ASSERT_NE(lifecycle_owner, 0U);
  ASSERT_NE(fixed_owner, 0U);
  ASSERT_NE(follower_owner, 0U);

  ASSERT_TRUE((*rates)->begin_startup_candidate());
  auto wrong_phase = (*rates)->bind_nominal_rate(lifecycle_owner,
                                                 LifecycleStage::kDiscover,
                                                 LifecycleTransitionPhase::kForwardRollback,
                                                 &lifecycle_source->output,
                                                 Rate::hz(60U));
  ASSERT_FALSE(wrong_phase);
  EXPECT_EQ(wrong_phase.error().code, ErrorCode::kLifecycleInvalidCallingContext);
  EXPECT_EQ(wrong_phase.error().message, "PORT_RATE_BIND_NOT_AUTHORIZED");
  ASSERT_TRUE((*rates)->bind_nominal_rate(lifecycle_owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kStartupForward,
                                          &lifecycle_source->output,
                                          Rate::hz(60U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kStartupForward));
  auto late_binding = (*rates)->bind_nominal_rate(lifecycle_owner,
                                                  LifecycleStage::kDiscover,
                                                  LifecycleTransitionPhase::kStartupForward,
                                                  &lifecycle_source->output,
                                                  Rate::hz(60U));
  ASSERT_FALSE(late_binding);
  EXPECT_EQ(late_binding.error().message, "PORT_RATE_BIND_NOT_AUTHORIZED");
  (*rates)->discard_candidate();

  const auto expect_rejected_binding = [&](OperatorInstanceId owner,
                                           LifecycleStage stage,
                                           const void* declaration,
                                           Rate value,
                                           ErrorCode code,
                                           std::string_view diagnostic) {
    ASSERT_TRUE((*rates)->begin_startup_candidate());
    auto result = (*rates)->bind_nominal_rate(
        owner, stage, LifecycleTransitionPhase::kStartupForward, declaration, value);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, code);
    EXPECT_EQ(result.error().message, diagnostic);

    // A forward-phase validation failure rejects the whole private candidate. A valid fallback
    // cannot appear to repair it; the first failure remains authoritative until discard.
    auto retry = (*rates)->bind_nominal_rate(lifecycle_owner,
                                             LifecycleStage::kDiscover,
                                             LifecycleTransitionPhase::kStartupForward,
                                             &lifecycle_source->output,
                                             Rate::hz(60U));
    ASSERT_FALSE(retry);
    EXPECT_EQ(retry.error().code, code);
    EXPECT_EQ(retry.error().message, diagnostic);
    EXPECT_FALSE((*rates)->preflight_candidate_publication());
    (*rates)->discard_candidate();
  };

  expect_rejected_binding(lifecycle_owner,
                          LifecycleStage::kDiscover,
                          &lifecycle_source->output,
                          Rate::hz(0U),
                          ErrorCode::kRateBindingInvalid,
                          "PORT_RATE_BINDING_INVALID");
  expect_rejected_binding(lifecycle_owner,
                          LifecycleStage::kDiscover,
                          &lifecycle_source->output,
                          Rate::hz(1U, 0U),
                          ErrorCode::kRateBindingInvalid,
                          "PORT_RATE_BINDING_INVALID");
  expect_rejected_binding(fixed_owner,
                          LifecycleStage::kDiscover,
                          &lifecycle_source->output,
                          Rate::hz(60U),
                          ErrorCode::kPermissionDenied,
                          "PORT_RATE_BIND_NOT_AUTHORIZED");
  expect_rejected_binding(lifecycle_owner,
                          LifecycleStage::kAllocate,
                          &lifecycle_source->output,
                          Rate::hz(60U),
                          ErrorCode::kPermissionDenied,
                          "PORT_RATE_BIND_NOT_AUTHORIZED");
  expect_rejected_binding(fixed_owner,
                          LifecycleStage::kDiscover,
                          &runtime_fixed_source->output,
                          Rate::hz(60U),
                          ErrorCode::kPermissionDenied,
                          "PORT_RATE_BIND_NOT_AUTHORIZED");
  expect_rejected_binding(follower_owner,
                          LifecycleStage::kDiscover,
                          &runtime_follower->input,
                          Rate::hz(60U),
                          ErrorCode::kPermissionDenied,
                          "PORT_RATE_BIND_NOT_AUTHORIZED");
  expect_rejected_binding(follower_owner,
                          LifecycleStage::kDiscover,
                          &runtime_follower->output,
                          Rate::hz(60U),
                          ErrorCode::kPermissionDenied,
                          "PORT_RATE_BIND_NOT_AUTHORIZED");
  const std::uint32_t foreign_declaration = 0U;
  expect_rejected_binding(lifecycle_owner,
                          LifecycleStage::kDiscover,
                          &foreign_declaration,
                          Rate::hz(60U),
                          ErrorCode::kPermissionDenied,
                          "PORT_RATE_BIND_NOT_AUTHORIZED");
}

// Frozen queue sizing is valid only for the exact replayed rate declaration. Reconstruction must
// reject semantic drift before lifecycle or compute, while accepting a normalized-equivalent rate.
// REQ: LIFECYCLE-012
TEST(RuntimeRateStateTest, RuntimeReplayAuthenticatesRateDeclarationSemantics) {
  struct Reset {
    ~Reset() {
      RuntimeRateReplaySource::numerator.store(30U, std::memory_order_relaxed);
      RuntimeRateReplaySource::denominator.store(1U, std::memory_order_relaxed);
    }
  };
  const Reset reset;

  Graph graph{"runtime-rate-replay"};
  graph.op<RuntimeRateReplaySource>("source");
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RuntimeRateReplaySource::numerator.store(120U, std::memory_order_relaxed);
  RuntimeRateReplaySource::denominator.store(4U, std::memory_order_relaxed);
  auto equivalent = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                   RunGenerationId{75U});
  ASSERT_TRUE(equivalent) << equivalent.error().message;

  RuntimeRateReplaySource::numerator.store(31U, std::memory_order_relaxed);
  RuntimeRateReplaySource::denominator.store(1U, std::memory_order_relaxed);
  detail::RuntimeReplayMismatchDetail mismatch;
  auto changed = detail::GraphExec::reconstruct(
      detail::ExecutionPlanAccess::retain(plan), RunGenerationId{76U}, {}, &mismatch);
  ASSERT_FALSE(changed);
  EXPECT_EQ(changed.error().message, "RUNTIME_PORT_BINDING_UNAVAILABLE");
  EXPECT_TRUE(mismatch.present);
  EXPECT_EQ(mismatch.category, detail::RuntimeReplayMismatchCategory::kPorts);
  EXPECT_NE(mismatch.expected, mismatch.observed);
}

// Runtime resolution consumes the compiler-frozen finite-horizon evidence. An expected input
// cannot manufacture its own OnEach driver cadence when the input is disconnected.
// REQ: LIFECYCLE-012
TEST(RuntimeRateStateTest, ExpectedInputWithoutFiniteHorizonRemainsUnresolved) {
  Graph graph{"runtime-rate-expected-without-horizon"};
  graph.op<RuntimeRateExpectedOnEachSink>("sink");
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{77U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{77U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());

  auto* const sink = find_runtime_operator<RuntimeRateExpectedOnEachSink>(**exec);
  ASSERT_NE(sink, nullptr);
  auto pin =
      (*rates)->pin_published(find_runtime_operator_id<RuntimeRateExpectedOnEachSink>(**exec));
  ASSERT_TRUE(pin);
  auto view = detail::RuntimeRateGenerationPin::resolve(&*pin, &sink->input, false);
  ASSERT_TRUE(view);
  EXPECT_EQ(view->state, PortRateViewState::kUnresolved);
  EXPECT_FALSE(view->resolved_nominal_rate.has_value());
  EXPECT_FALSE(view->sizing_rate.has_value());
}

// An expected input becomes the sizing authority only when its connected producer is genuinely
// undeclared and the compiler froze a finite accumulation horizon. The generated queue record must
// join that input-rate record rather than silently reverting to the producer's missing metadata.
// REQ: LIFECYCLE-012
// REQ: COMPILE-070
TEST(RuntimeRateStateTest, ExpectedInputFallbackSizesItsConnectedQueueFromTheFrozenHorizon) {
  Graph graph{"runtime-rate-expected-fallback"};
  const auto source = graph.op<RuntimeRateUnspecifiedSource>("source");
  const auto sink = graph.op<RuntimeRateExpectedClockSink>("sink");
  graph.add_flow(source->output, sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.payload_edges().size(), 1U);
  EXPECT_EQ(plan.payload_edges().front().queue_depth, 120U);

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{78U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{78U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());

  auto* const runtime_sink = find_runtime_operator<RuntimeRateExpectedClockSink>(**exec);
  ASSERT_NE(runtime_sink, nullptr);
  auto pin =
      (*rates)->pin_published(find_runtime_operator_id<RuntimeRateExpectedClockSink>(**exec));
  ASSERT_TRUE(pin);
  auto input_view = detail::RuntimeRateGenerationPin::resolve(&*pin, &runtime_sink->input, false);
  ASSERT_TRUE(input_view);
  EXPECT_EQ(input_view->resolved_nominal_rate, Rate::hz(60U));
  EXPECT_EQ(input_view->sizing_rate, Rate::hz(60U));
  EXPECT_EQ(input_view->origin, ResolvedRateOrigin::kInputExpectationFallback);
  EXPECT_EQ(input_view->state, PortRateViewState::kResolved);

  auto snapshot = (*rates)->capture_snapshot();
  ASSERT_TRUE(snapshot) << snapshot.error().message;
  ASSERT_NE(snapshot->published(), nullptr);
  const auto queues = snapshot->queue_sizings(*snapshot->published());
  ASSERT_EQ(queues.size(), 1U);
  EXPECT_EQ(queues.front().nominal_arrivals, 60U);
  EXPECT_EQ(queues.front().rate_recommendation, 120U);
  EXPECT_EQ(queues.front().effective_logical_depth, 120U);
  EXPECT_EQ(queues.front().stability, StabilityClass::kPredictedSustainedBlocking);
  EXPECT_EQ(queues.front().stability_reason, detail::diagnostic_id("RATE_SUSTAINED_BACKPRESSURE"));
  const auto& producer_rate = queues.front().producer_rate;
  HOLOSCAN_ASSERT_HAS_VALUE(producer_rate);
  const auto port_rates = snapshot->port_rates(*snapshot->published());
  const auto sizing_record =
      std::ranges::find(port_rates, *producer_rate, &PortRateGenerationRecord::id);
  ASSERT_NE(sizing_record, port_rates.end());
  EXPECT_EQ(sizing_record->port, plan.payload_edges().front().target);
  EXPECT_EQ(sizing_record->resolved_origin, ResolvedRateOrigin::kInputExpectationFallback);

  const auto json = render_rate_queue_inspection(plan,
                                                 *snapshot,
                                                 snapshot->published()->rate_generation,
                                                 plan.payload_edges().front().edge,
                                                 RateQueueInspectionFormat::kCanonicalJson);
  ASSERT_TRUE(json) << json.error().message;
  EXPECT_NE(json->find(R"("schema":"holoscan.rate_queue_inspection.canonical_json.v1")"),
            std::string::npos);
  EXPECT_NE(json->find(R"("source":"source.output")"), std::string::npos);
  EXPECT_NE(json->find(R"("target":"sink.input")"), std::string::npos);
  EXPECT_NE(json->find(R"("resolved_origin":"input_expectation_fallback")"), std::string::npos);
  EXPECT_NE(json->find(R"("rate_recommendation":120)"), std::string::npos);
  EXPECT_NE(json->find(R"("effective_logical_depth":120)"), std::string::npos);
  EXPECT_NE(json->find(R"("provider_allocations":[])"), std::string::npos);
  EXPECT_NE(
      json->find(R"("paste_ready_exact_pin":"holoscan::ConnectionOptions{.queue_depth = 120U}")"),
      std::string::npos);

  const auto human = render_rate_queue_inspection(plan,
                                                  *snapshot,
                                                  snapshot->published()->rate_generation,
                                                  plan.payload_edges().front().edge,
                                                  RateQueueInspectionFormat::kHumanText);
  ASSERT_TRUE(human) << human.error().message;
  EXPECT_NE(human->find("edge: source.output -> sink.input"), std::string::npos);
  EXPECT_NE(human->find("producer rate: 60/1 Hz (input_expectation_fallback)"), std::string::npos);
  EXPECT_NE(human->find("provider allocations: none"), std::string::npos);
  EXPECT_NE(human->find("paste-ready exact pin: holoscan::ConnectionOptions{.queue_depth = 120U}"),
            std::string::npos);

  Graph foreign_graph{"runtime-rate-inspection-foreign"};
  const auto foreign_source = foreign_graph.op<RuntimeRateUnspecifiedSource>("foreign-source");
  const auto foreign_sink = foreign_graph.op<RuntimeRateExpectedClockSink>("foreign-sink");
  foreign_graph.add_flow(foreign_source->output, foreign_sink->input);
  foreign_graph.set_default_clock(
      foreign_graph.add_clock<RealtimeClock>("runtime-rate-foreign-clock"));
  const ExecutionPlan foreign_plan = compile(foreign_graph);
  ASSERT_TRUE(foreign_plan.ok()) << foreign_plan.json();
  const auto mismatched_plan =
      render_rate_queue_inspection(foreign_plan,
                                   *snapshot,
                                   snapshot->published()->rate_generation,
                                   foreign_plan.payload_edges().front().edge,
                                   RateQueueInspectionFormat::kCanonicalJson);
  ASSERT_FALSE(mismatched_plan);
  EXPECT_EQ(mismatched_plan.error().message, "RATE_QUEUE_INSPECTION_PLAN_MISMATCH");

  const auto invalid_format =
      render_rate_queue_inspection(plan,
                                   *snapshot,
                                   snapshot->published()->rate_generation,
                                   plan.payload_edges().front().edge,
                                   static_cast<RateQueueInspectionFormat>(255U));
  ASSERT_FALSE(invalid_format);
  EXPECT_EQ(invalid_format.error().message, "RATE_QUEUE_INSPECTION_JOIN_INVALID");

  const auto unknown_generation =
      render_rate_queue_inspection(plan,
                                   *snapshot,
                                   snapshot->published()->rate_generation + 1U,
                                   plan.payload_edges().front().edge,
                                   RateQueueInspectionFormat::kCanonicalJson);
  ASSERT_FALSE(unknown_generation);
  EXPECT_EQ(unknown_generation.error().message, "RATE_QUEUE_INSPECTION_GENERATION_UNKNOWN");

  const auto unknown_edge = render_rate_queue_inspection(plan,
                                                         *snapshot,
                                                         snapshot->published()->rate_generation,
                                                         EdgeId{999U},
                                                         RateQueueInspectionFormat::kHumanText);
  ASSERT_FALSE(unknown_edge);
  EXPECT_EQ(unknown_edge.error().message, "RATE_QUEUE_INSPECTION_EDGE_UNKNOWN");
}

// An explicit depth remains authoritative and must never be offered back as an automatic-sizing
// pin, even when rate metadata also produced an advisory recommendation.
// REQ: COMPILE-070
TEST(RuntimeRateStateTest, InspectionDoesNotOfferAnAlreadyExplicitDepthAsAnAutomaticPin) {
  Graph graph{"runtime-rate-explicit-inspection"};
  const auto source = graph.op<RuntimeRateFixedSource>("source");
  const auto sink = graph.op<RuntimeRateClockSink>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 7U});
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{82U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{82U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());
  auto snapshot = (*rates)->capture_snapshot();
  ASSERT_TRUE(snapshot) << snapshot.error().message;
  ASSERT_NE(snapshot->published(), nullptr);

  const auto json = render_rate_queue_inspection(plan,
                                                 *snapshot,
                                                 snapshot->published()->rate_generation,
                                                 plan.payload_edges().front().edge,
                                                 RateQueueInspectionFormat::kCanonicalJson);
  ASSERT_TRUE(json) << json.error().message;
  EXPECT_NE(json->find(R"("explicit_depth":7)"), std::string::npos);
  EXPECT_NE(json->find(R"("paste_ready_exact_pin":null)"), std::string::npos);
  EXPECT_NE(json->find(R"("paste_ready_exact_pin_effect":"already_explicit")"), std::string::npos);
}

// Bracketed windows retain a non-destructive snapshot, so their maximum selected-sample count is
// not a per-activation advancement capacity. Runtime generation evidence must preserve the same
// unresolved stability result selected by compile instead of treating retention as consumption.
// REQ: LIFECYCLE-012
TEST(RuntimeRateStateTest, StabilityDoesNotTreatWindowRetentionAsAdvancementCapacity) {
  Graph graph{"runtime-rate-window-stability"};
  const auto anchor_source = graph.op<RuntimeRateFixedSource>("anchor-source");
  const auto window_source = graph.op<RuntimeRateFixedSource>("window-source");
  const auto sink = graph.op<RuntimeRateWindowSink>("sink");
  graph.add_flow(anchor_source->output, sink->anchor);
  graph.add_flow(window_source->output, sink->window);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto window_port = std::ranges::find_if(plan.ports(), [](const auto& port) {
    return port.direction == PortDirection::kInput && port.local_name == "window";
  });
  ASSERT_NE(window_port, plan.ports().end());
  const auto window_edge =
      std::ranges::find(plan.payload_edges(), window_port->id, &PayloadEdgePlanRecord::target);
  ASSERT_NE(window_edge, plan.payload_edges().end());
  const auto window_recipe = std::ranges::find(
      plan.queue_sizing_recipes(), window_edge->edge, &QueueSizingRecipePlanRecord::edge);
  ASSERT_NE(window_recipe, plan.queue_sizing_recipes().end());
  const auto window_horizon = std::ranges::find(
      plan.accumulation_horizons(), window_edge->edge, &AccumulationHorizonPlanRecord::edge);
  ASSERT_NE(window_horizon, plan.accumulation_horizons().end());
  ASSERT_TRUE(window_horizon->horizon.has_value());
  EXPECT_EQ(window_horizon->origin, AccumulationHorizonOrigin::kTemporalWindowBound);
  EXPECT_EQ(window_horizon->horizon,
            (ExactDuration{.numerator_ns = 1'000'000'000U, .denominator = 1U}));

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{81U});
  ASSERT_TRUE(exec) << exec.error().message;
  // Exercise the generation classifier against the compiler-authored finite window horizon. The
  // Runtime consumes this immutable record; it must not reinterpret selected-sample retention as
  // destructive advancement.
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{81U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());

  auto snapshot = (*rates)->capture_snapshot();
  ASSERT_TRUE(snapshot) << snapshot.error().message;
  ASSERT_NE(snapshot->published(), nullptr);
  const auto queues = snapshot->queue_sizings(*snapshot->published());
  const auto generated =
      std::ranges::find(queues, window_recipe->id, &QueueSizingGenerationRecord::recipe);
  ASSERT_NE(generated, queues.end());
  ASSERT_TRUE(generated->nominal_arrivals);
  EXPECT_EQ(generated->stability, StabilityClass::kUnresolvedBoundedEvidence);
  EXPECT_FALSE(generated->stability_reason);
}

// REQ: LIFECYCLE-012
TEST(RuntimeRateStateTest, PublishesFreshRateGenerationAndKeepsPinnedPredecessorImmutable) {
  Graph graph{"runtime-rate-restart"};
  graph.op<RuntimeRateLifecycleSource>("source");
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{79U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{79U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  auto* const source = find_runtime_operator<RuntimeRateLifecycleSource>(**exec);
  const OperatorInstanceId owner = find_runtime_operator_id<RuntimeRateLifecycleSource>(**exec);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(owner, 0U);

  ASSERT_TRUE((*rates)->begin_startup_candidate());
  EXPECT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kStartupForward,
                                          &source->output,
                                          Rate::hz(60U)));
  EXPECT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kStartupForward,
                                          &source->output,
                                          Rate::hz(120U, 2U)));
  auto pending = (*rates)->resolve_lifecycle(owner,
                                             LifecycleStage::kDiscover,
                                             LifecycleTransitionPhase::kStartupForward,
                                             &source->output,
                                             true);
  ASSERT_TRUE(pending);
  EXPECT_EQ(pending->state, PortRateViewState::kPendingBarrier);
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kStartupForward));
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());

  auto old_pin = (*rates)->pin_published(owner);
  ASSERT_TRUE(old_pin);
  auto old_view = detail::RuntimeRateGenerationPin::resolve(&*old_pin, &source->output, true);
  ASSERT_TRUE(old_view);
  ASSERT_EQ(old_view->resolved_nominal_rate, Rate::hz(60U));

  const std::array members{owner};
  ASSERT_TRUE((*rates)->begin_restart_candidate(members, LifecycleStage::kDiscover));
  ASSERT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kRestartForward,
                                          &source->output,
                                          Rate::hz(30U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kRestartForward));
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());

  auto new_pin = (*rates)->pin_published(owner);
  ASSERT_TRUE(new_pin);
  auto new_view = detail::RuntimeRateGenerationPin::resolve(&*new_pin, &source->output, true);
  auto retained_old_view =
      detail::RuntimeRateGenerationPin::resolve(&*old_pin, &source->output, true);
  ASSERT_TRUE(new_view);
  ASSERT_TRUE(retained_old_view);
  EXPECT_EQ(new_view->resolved_nominal_rate, Rate::hz(30U));
  EXPECT_EQ(retained_old_view->resolved_nominal_rate, Rate::hz(60U));
  EXPECT_EQ(new_view->run_generation, old_view->run_generation);
  EXPECT_NE(new_view->rate_generation, old_view->rate_generation);

  auto snapshot = (*rates)->capture_snapshot();
  ASSERT_TRUE(snapshot) << snapshot.error().message;
  ASSERT_NE(snapshot->published(), nullptr);
  EXPECT_EQ(snapshot->published()->rate_generation, new_view->rate_generation);
  EXPECT_EQ(snapshot->run_generation(), RunGenerationId{79U});
  EXPECT_EQ(snapshot->port_rates().size(), plan.port_rates().size());
  EXPECT_EQ(snapshot->published()->port_rate_count, plan.port_rates().size());
  EXPECT_EQ(snapshot->port_rates(*snapshot->published()).size(), plan.port_rates().size());
  EXPECT_EQ(snapshot->queue_sizings(*snapshot->published()).size(),
            snapshot->published()->queue_sizing_count);
  EXPECT_EQ(snapshot->pool_contributions(*snapshot->published()).size(),
            snapshot->published()->pool_contribution_count);
  EXPECT_EQ(snapshot->pool_capacities(*snapshot->published()).size(),
            snapshot->published()->pool_capacity_count);
  EXPECT_EQ(snapshot->provider_allocations(*snapshot->published()).size(),
            snapshot->published()->provider_allocation_count);
  EXPECT_EQ(snapshot->last_rejected(), nullptr);
}

// A compute callback that never calls ExecutionContext::rate() must not take a reader pin. The
// callback-local adapter acquires its generation only on the first query, then retains that exact
// predecessor through callback return even if a newer generation is published.
// REQ: LIFECYCLE-012
TEST(RuntimeRateStateTest, CallbackBindingPinsPublishedGenerationOnlyOnFirstRateQuery) {
  Graph graph{"runtime-rate-lazy-callback-binding"};
  graph.op<RuntimeRateLifecycleSource>("source");
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{80U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{80U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  auto* const source = find_runtime_operator<RuntimeRateLifecycleSource>(**exec);
  const OperatorInstanceId owner = find_runtime_operator_id<RuntimeRateLifecycleSource>(**exec);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(owner, 0U);

  detail::RuntimeRateCallbackBinding callback_binding{rates->get(), owner};

  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kStartupForward,
                                          &source->output,
                                          Rate::hz(60U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kStartupForward));
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());

  const auto measure_query = [&] {
    const test::ScopedAllocationCounter allocations;
    auto result =
        detail::RuntimeRateCallbackBinding::resolve(&callback_binding, &source->output, true);
    return std::pair{std::move(result), allocations.count()};
  };
  auto [first, first_query_allocations] = measure_query();
  ASSERT_TRUE(first) << first.error().message;
  EXPECT_EQ(first_query_allocations, 0U);
  EXPECT_EQ(first->resolved_nominal_rate, Rate::hz(60U));

  auto [same_callback, repeated_query_allocations] = measure_query();
  ASSERT_TRUE(same_callback) << same_callback.error().message;
  EXPECT_EQ(repeated_query_allocations, 0U);
  EXPECT_EQ(same_callback->rate_generation, first->rate_generation);

  const std::array members{owner};
  ASSERT_TRUE((*rates)->begin_restart_candidate(members, LifecycleStage::kDiscover));
  ASSERT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kRestartForward,
                                          &source->output,
                                          Rate::hz(30U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kRestartForward));
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());

  auto retained =
      detail::RuntimeRateCallbackBinding::resolve(&callback_binding, &source->output, true);
  ASSERT_TRUE(retained) << retained.error().message;
  EXPECT_EQ(retained->resolved_nominal_rate, Rate::hz(60U));
  EXPECT_EQ(retained->rate_generation, first->rate_generation);

  detail::RuntimeRateCallbackBinding next_callback{rates->get(), owner};
  auto current = detail::RuntimeRateCallbackBinding::resolve(&next_callback, &source->output, true);
  ASSERT_TRUE(current) << current.error().message;
  EXPECT_EQ(current->resolved_nominal_rate, Rate::hz(30U));
  EXPECT_NE(current->rate_generation, retained->rate_generation);
}

// Every compute-time rate-query failure is code-only so invalid or not-yet-published state cannot
// allocate while the noexcept callback adapter is active.
// REQ: LIFECYCLE-012
TEST(RuntimeRateStateTest, CallbackRateQueryFailuresAreAllocationFreeAndCodeOnly) {
  Graph graph{"runtime-rate-callback-failures"};
  graph.op<RuntimeRateLifecycleSource>("source");
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{84U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{84U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  auto* const source = find_runtime_operator<RuntimeRateLifecycleSource>(**exec);
  const OperatorInstanceId owner = find_runtime_operator_id<RuntimeRateLifecycleSource>(**exec);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(owner, 0U);

  const auto measure_failure = [](auto&& query) {
    const test::ScopedAllocationCounter allocations;
    auto result = query();
    return std::pair{std::move(result), allocations.count()};
  };

  auto [invalid_binding, invalid_binding_allocations] = measure_failure(
      [] { return detail::RuntimeRateCallbackBinding::resolve(nullptr, nullptr, true); });
  ASSERT_FALSE(invalid_binding);
  EXPECT_EQ(invalid_binding.error().code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(invalid_binding.error().message.empty());
  EXPECT_EQ(invalid_binding_allocations, 0U);

  detail::RuntimeRateCallbackBinding unpublished_binding{rates->get(), owner};
  auto [unpublished, unpublished_allocations] = measure_failure([&] {
    return detail::RuntimeRateCallbackBinding::resolve(&unpublished_binding, &source->output, true);
  });
  ASSERT_FALSE(unpublished);
  EXPECT_EQ(unpublished.error().code, ErrorCode::kNotReady);
  EXPECT_TRUE(unpublished.error().message.empty());
  EXPECT_EQ(unpublished_allocations, 0U);

  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kStartupForward,
                                          &source->output,
                                          Rate::hz(60U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kStartupForward));
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());

  detail::RuntimeRateCallbackBinding published_binding{rates->get(), owner};
  auto [wrong_direction, wrong_direction_allocations] = measure_failure([&] {
    return detail::RuntimeRateCallbackBinding::resolve(&published_binding, &source->output, false);
  });
  ASSERT_FALSE(wrong_direction);
  EXPECT_EQ(wrong_direction.error().code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(wrong_direction.error().message.empty());
  EXPECT_EQ(wrong_direction_allocations, 0U);
}

// REQ: LIFECYCLE-012
TEST(RuntimeRateStateTest, FailedBindingsPoisonOnlyThePrivateCandidate) {
  Graph graph{"runtime-rate-failed-candidate"};
  graph.op<RuntimeRateLifecycleSource>("source");
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{83U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{83U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  auto* const source = find_runtime_operator<RuntimeRateLifecycleSource>(**exec);
  const OperatorInstanceId owner = find_runtime_operator_id<RuntimeRateLifecycleSource>(**exec);
  ASSERT_NE(source, nullptr);

  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kStartupForward,
                                          &source->output,
                                          Rate::hz(60U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kStartupForward));
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());
  auto published = (*rates)->pin_published(owner);
  ASSERT_TRUE(published);
  auto published_view =
      detail::RuntimeRateGenerationPin::resolve(&*published, &source->output, true);
  ASSERT_TRUE(published_view);

  const std::array members{owner};
  ASSERT_TRUE((*rates)->begin_restart_candidate(members, LifecycleStage::kDiscover));
  ASSERT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kRestartForward,
                                          &source->output,
                                          Rate::hz(30U)));
  auto conflict = (*rates)->bind_nominal_rate(owner,
                                              LifecycleStage::kDiscover,
                                              LifecycleTransitionPhase::kRestartForward,
                                              &source->output,
                                              Rate::hz(40U));
  ASSERT_FALSE(conflict);
  EXPECT_EQ(conflict.error().code, ErrorCode::kRateBindingConflict);
  EXPECT_EQ(conflict.error().message, "PORT_RATE_BINDING_CONFLICT");
  auto conflict_retry = (*rates)->bind_nominal_rate(owner,
                                                    LifecycleStage::kDiscover,
                                                    LifecycleTransitionPhase::kRestartForward,
                                                    &source->output,
                                                    Rate::hz(30U));
  ASSERT_FALSE(conflict_retry);
  EXPECT_EQ(conflict_retry.error().code, conflict.error().code);
  EXPECT_EQ(conflict_retry.error().message, conflict.error().message);
  EXPECT_FALSE((*rates)->preflight_candidate_publication());
  (*rates)->reject_candidate(conflict.error());

  auto conflict_snapshot = (*rates)->capture_snapshot();
  ASSERT_TRUE(conflict_snapshot) << conflict_snapshot.error().message;
  ASSERT_NE(conflict_snapshot->published(), nullptr);
  ASSERT_NE(conflict_snapshot->last_rejected(), nullptr);
  const RateGenerationId conflict_generation = conflict_snapshot->last_rejected()->rate_generation;
  EXPECT_EQ(conflict_snapshot->last_rejected()->error_code, ErrorCode::kRateBindingConflict);
  EXPECT_EQ(conflict_snapshot->last_rejected()->diagnostic.token, "PORT_RATE_BINDING_CONFLICT");
  EXPECT_EQ(conflict_snapshot->last_rejected()->diagnostic.id,
            detail::diagnostic_id("PORT_RATE_BINDING_CONFLICT"));
  EXPECT_EQ(conflict_snapshot->last_rejected()->diagnostic.provided, "PORT_RATE_BINDING_CONFLICT");
  const auto conflict_ports = conflict_snapshot->port_rates(*conflict_snapshot->last_rejected());
  ASSERT_FALSE(conflict_ports.empty());
  EXPECT_TRUE(std::ranges::all_of(conflict_ports, [](const auto& record) {
    return record.reason == detail::diagnostic_id("PORT_RATE_BINDING_CONFLICT");
  }));

  auto unchanged = (*rates)->pin_published(owner);
  ASSERT_TRUE(unchanged);
  auto unchanged_view =
      detail::RuntimeRateGenerationPin::resolve(&*unchanged, &source->output, true);
  ASSERT_TRUE(unchanged_view);
  EXPECT_EQ(*unchanged_view, *published_view);

  ASSERT_TRUE((*rates)->begin_restart_candidate(members, LifecycleStage::kDiscover));
  auto above_max = (*rates)->bind_nominal_rate(owner,
                                               LifecycleStage::kDiscover,
                                               LifecycleTransitionPhase::kRestartForward,
                                               &source->output,
                                               Rate::hz(121U));
  ASSERT_FALSE(above_max);
  EXPECT_EQ(above_max.error().code, ErrorCode::kRateBindingAboveMaximum);
  EXPECT_EQ(above_max.error().message, "PORT_RATE_BINDING_ABOVE_MAXIMUM");
  auto maximum_retry = (*rates)->bind_nominal_rate(owner,
                                                   LifecycleStage::kDiscover,
                                                   LifecycleTransitionPhase::kRestartForward,
                                                   &source->output,
                                                   Rate::hz(120U));
  ASSERT_FALSE(maximum_retry);
  EXPECT_EQ(maximum_retry.error().code, above_max.error().code);
  EXPECT_EQ(maximum_retry.error().message, above_max.error().message);
  (*rates)->reject_candidate(above_max.error());

  auto maximum_snapshot = (*rates)->capture_snapshot();
  ASSERT_TRUE(maximum_snapshot) << maximum_snapshot.error().message;
  ASSERT_NE(maximum_snapshot->last_rejected(), nullptr);
  EXPECT_GT(maximum_snapshot->last_rejected()->rate_generation, conflict_generation);
  EXPECT_EQ(maximum_snapshot->last_rejected()->diagnostic.token, "PORT_RATE_BINDING_ABOVE_MAXIMUM");
  EXPECT_EQ(maximum_snapshot->last_rejected()->diagnostic.id,
            detail::diagnostic_id("PORT_RATE_BINDING_ABOVE_MAXIMUM"));
  EXPECT_EQ(maximum_snapshot->last_rejected()->diagnostic.provided,
            "PORT_RATE_BINDING_ABOVE_MAXIMUM");

  ASSERT_TRUE((*rates)->begin_restart_candidate(members, LifecycleStage::kDiscover));
  auto missing = (*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                               LifecycleTransitionPhase::kRestartForward);
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().message, "PORT_RATE_BINDING_MISSING");
  (*rates)->reject_candidate(missing.error());

  auto missing_snapshot = (*rates)->capture_snapshot();
  ASSERT_TRUE(missing_snapshot) << missing_snapshot.error().message;
  ASSERT_NE(missing_snapshot->published(), nullptr);
  ASSERT_NE(missing_snapshot->last_rejected(), nullptr);
  EXPECT_EQ(missing_snapshot->published()->rate_generation, published_view->rate_generation);
  EXPECT_EQ(missing_snapshot->last_rejected()->diagnostic.token, "PORT_RATE_BINDING_MISSING");
  EXPECT_EQ(missing_snapshot->last_rejected()->diagnostic.id,
            detail::diagnostic_id("PORT_RATE_BINDING_MISSING"));
  EXPECT_EQ(missing_snapshot->last_rejected()->diagnostic.provided, "PORT_RATE_BINDING_MISSING");
}

// Compiler and lifecycle adapters share the same exact rational arithmetic. A fractional fixed
// rate and the identical lifecycle-selected rate must therefore produce the same rounded queue
// recommendation rather than drifting at their separate record/materialization boundaries.
// REQ: COMPILE-069
// REQ: LIFECYCLE-012
TEST(RuntimeRateStateTest, FractionalQueueRecommendationMatchesCompileConcreteResult) {
  RuntimeRateReplaySource::numerator.store(119U, std::memory_order_relaxed);
  RuntimeRateReplaySource::denominator.store(2U, std::memory_order_relaxed);
  Graph fixed_graph{"runtime-rate-fractional-fixed"};
  const auto fixed_source = fixed_graph.op<RuntimeRateReplaySource>("source");
  const auto fixed_sink = fixed_graph.op<RuntimeRateClockSink>("sink");
  fixed_graph.add_flow(fixed_source->output, fixed_sink->input);
  fixed_graph.set_default_clock(fixed_graph.add_clock<RealtimeClock>("fixed-clock"));
  const ExecutionPlan fixed_plan = compile(fixed_graph);
  RuntimeRateReplaySource::numerator.store(30U, std::memory_order_relaxed);
  RuntimeRateReplaySource::denominator.store(1U, std::memory_order_relaxed);
  ASSERT_TRUE(fixed_plan.ok()) << fixed_plan.json();
  ASSERT_EQ(fixed_plan.payload_edges().size(), 1U);
  const std::uint32_t compile_recommendation = fixed_plan.payload_edges().front().queue_depth;
  EXPECT_EQ(compile_recommendation, 120U);

  Graph lifecycle_graph{"runtime-rate-fractional-lifecycle"};
  const auto lifecycle_source = lifecycle_graph.op<RuntimeRateLifecycleSource>("source");
  const auto lifecycle_sink = lifecycle_graph.op<RuntimeRateClockSink>("sink");
  lifecycle_graph.add_flow(lifecycle_source->output, lifecycle_sink->input);
  lifecycle_graph.set_default_clock(lifecycle_graph.add_clock<RealtimeClock>("lifecycle-clock"));
  const ExecutionPlan lifecycle_plan = compile(lifecycle_graph);
  ASSERT_TRUE(lifecycle_plan.ok()) << lifecycle_plan.json();

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(lifecycle_plan),
                                             RunGenerationId{889U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{889U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  auto* const runtime_source = find_runtime_operator<RuntimeRateLifecycleSource>(**exec);
  const OperatorInstanceId owner = find_runtime_operator_id<RuntimeRateLifecycleSource>(**exec);
  ASSERT_NE(runtime_source, nullptr);

  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kStartupForward,
                                          &runtime_source->output,
                                          Rate::hz(119U, 2U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kStartupForward));
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());
  auto snapshot = (*rates)->capture_snapshot();
  ASSERT_TRUE(snapshot) << snapshot.error().message;
  ASSERT_NE(snapshot->published(), nullptr);
  const auto queues = snapshot->queue_sizings(*snapshot->published());
  ASSERT_EQ(queues.size(), 1U);
  EXPECT_EQ(queues.front().nominal_arrivals, 60U);
  EXPECT_EQ(queues.front().rate_recommendation, compile_recommendation);
  EXPECT_EQ(queues.front().effective_logical_depth, compile_recommendation);
}

// REQ: LIFECYCLE-012
TEST(RuntimeRateStateTest, EvaluatesLifecycleCandidateActiveDepthBelowFiniteBackingCeiling) {
  Graph graph{"runtime-rate-active-depth"};
  const auto source = graph.op<RuntimeRateLifecycleSource>("source");
  const auto sink = graph.op<RuntimeRateClockSink>("sink");
  graph.add_flow(source->output, sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.payload_edges().size(), 1U);
  EXPECT_EQ(plan.payload_edges().front().queue_depth, 240U);

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{89U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{89U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  auto* const runtime_source = find_runtime_operator<RuntimeRateLifecycleSource>(**exec);
  const OperatorInstanceId owner = find_runtime_operator_id<RuntimeRateLifecycleSource>(**exec);
  ASSERT_NE(runtime_source, nullptr);

  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kStartupForward,
                                          &runtime_source->output,
                                          Rate::hz(60U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kStartupForward));
  struct ObservedDepth {
    std::uint32_t value{};
    bool update_required{};
  } active_depth;
  using QueueDepths = std::span<const detail::RuntimeQueueDepthGeneration>;
  auto visited = (*rates)->visit_candidate_queue_depths(
      &active_depth, [](void* state, QueueDepths depths) noexcept -> expected<void, Error> {
        if (state == nullptr || depths.size() != 1U) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        auto& observed = *static_cast<ObservedDepth*>(state);
        observed.value = depths.front().active_logical_depth;
        observed.update_required = depths.front().active_depth_update_required;
        return {};
      });
  ASSERT_TRUE(visited) << visited.error().message;
  EXPECT_EQ(active_depth.value, 120U);
  EXPECT_TRUE(active_depth.update_required);

  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());
  const std::array members{owner};
  ASSERT_TRUE((*rates)->begin_restart_candidate(members, LifecycleStage::kDiscover));
  ASSERT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kRestartForward,
                                          &runtime_source->output,
                                          Rate::hz(60U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kRestartForward));
  active_depth = {};
  visited = (*rates)->visit_candidate_queue_depths(
      &active_depth, [](void* state, QueueDepths depths) noexcept -> expected<void, Error> {
        if (state == nullptr || depths.size() != 1U) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        auto& observed = *static_cast<ObservedDepth*>(state);
        observed.value = depths.front().active_logical_depth;
        observed.update_required = depths.front().active_depth_update_required;
        return {};
      });
  ASSERT_TRUE(visited) << visited.error().message;
  EXPECT_EQ(active_depth.value, 120U);
  EXPECT_FALSE(active_depth.update_required);
  (*rates)->discard_candidate();

  ASSERT_TRUE((*rates)->begin_restart_candidate(members, LifecycleStage::kDiscover));
  ASSERT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kRestartForward,
                                          &runtime_source->output,
                                          Rate::hz(30U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kRestartForward));
  active_depth = {};
  visited = (*rates)->visit_candidate_queue_depths(
      &active_depth, [](void* state, QueueDepths depths) noexcept -> expected<void, Error> {
        if (state == nullptr || depths.size() != 1U) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        auto& observed = *static_cast<ObservedDepth*>(state);
        observed.value = depths.front().active_logical_depth;
        observed.update_required = depths.front().active_depth_update_required;
        return {};
      });
  ASSERT_TRUE(visited) << visited.error().message;
  EXPECT_EQ(active_depth.value, 60U);
  EXPECT_TRUE(active_depth.update_required);
}

// A process-group worker owns only its reconstructed declarations and resources. A disconnected
// lifecycle root in another worker remains plan-visible, but it must not become this worker's
// binding, queue-update, or publication obligation.
// REQ: LIFECYCLE-012
TEST(RuntimeRateStateTest, WorkerProjectionIgnoresDisconnectedRemoteLifecycleRoot) {
  Graph graph{"runtime-rate-worker-projection"};
  const auto local_source = graph.op<RuntimeRateLifecycleSource>("local-source");
  const auto local_sink = graph.op<RuntimeRateClockSink>("local-sink");
  const auto remote_source = graph.op<RuntimeRateRemoteLifecycleSource>("remote-source");
  const auto remote_sink = graph.op<RuntimeRateClockSink>("remote-sink");
  graph.add_flow(local_source->output, local_sink->input);
  graph.add_flow(remote_source->output, remote_sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));

  constexpr std::array local_members{
      detail::GraphObjectSelector{.graph_path = "local-source"},
      detail::GraphObjectSelector{.graph_path = "local-sink"},
  };
  constexpr std::array remote_members{
      detail::GraphObjectSelector{.graph_path = "remote-source"},
      detail::GraphObjectSelector{.graph_path = "remote-sink"},
  };
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("local",
                         detail::ProcessPlacement{
                             .members = local_members,
                             .launch_group = "local-process",
                         })
      .bind_process_role("remote",
                         detail::ProcessPlacement{
                             .members = remote_members,
                             .launch_group = "remote-process",
                         });
  const ExecutionPlan plan =
      detail::compile_with_process_placement(graph, CompileOptions{}, std::move(placement));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  const auto local_vertex = std::ranges::find_if(
      state->graph.vertices(), [](const auto& vertex) { return vertex.name == "local-source"; });
  ASSERT_NE(local_vertex, state->graph.vertices().end());
  const ProcessGroupId local_group = detail::process_group_for(*state, local_vertex->id);
  ASSERT_TRUE(local_group);

  auto exec = detail::GraphExec::reconstruct(state, RunGenerationId{94U}, local_group);
  ASSERT_TRUE(exec) << exec.error().message;
  ASSERT_EQ((*exec)->operators().size(), 2U);
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{94U}, 1U, local_group);
  ASSERT_TRUE(rates) << rates.error().message;
  auto* const runtime_source = find_runtime_operator<RuntimeRateLifecycleSource>(**exec);
  const OperatorInstanceId owner = find_runtime_operator_id<RuntimeRateLifecycleSource>(**exec);
  ASSERT_NE(runtime_source, nullptr);

  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kStartupForward,
                                          &runtime_source->output,
                                          Rate::hz(60U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kStartupForward));
  struct Projection {
    std::size_t updates{};
    std::size_t resolved{};
  } projection;
  using QueueDepths = std::span<const detail::RuntimeQueueDepthGeneration>;
  auto visited = (*rates)->visit_candidate_queue_depths(
      &projection, [](void* opaque, QueueDepths depths) noexcept -> expected<void, Error> {
        auto& result = *static_cast<Projection*>(opaque);
        result.updates = static_cast<std::size_t>(std::ranges::count(
            depths, true, &detail::RuntimeQueueDepthGeneration::active_depth_update_required));
        result.resolved = static_cast<std::size_t>(std::ranges::count_if(
            depths, [](const auto& depth) { return depth.active_logical_depth != 0U; }));
        return {};
      });
  ASSERT_TRUE(visited) << visited.error().message;
  EXPECT_EQ(projection.updates, 1U);
  EXPECT_EQ(projection.resolved, 1U);
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());
  auto snapshot = (*rates)->capture_snapshot();
  ASSERT_TRUE(snapshot) << snapshot.error().message;
  ASSERT_NE(snapshot->published(), nullptr);
  const auto queues = snapshot->queue_sizings(*snapshot->published());
  ASSERT_EQ(queues.size(), 1U);
  EXPECT_LT(snapshot->port_rates(*snapshot->published()).size(), plan.port_rates().size());
}

// DrainAll with a retained-capacity-derived batch bound advances the candidate's complete active
// prefix. Runtime stability must use that evaluated bound rather than the compile-time pending
// placeholder.
// REQ: COMPILE-069
// REQ: LIFECYCLE-012
TEST(RuntimeRateStateTest, LifecycleDrainAllStabilityUsesCandidateActiveDepth) {
  Graph graph{"runtime-rate-drain-all-stability"};
  const auto source = graph.op<RuntimeRateLifecycleSource>("source");
  const auto sink = graph.op<RuntimeRateDrainAllClockSink>("sink");
  graph.add_flow(source->output, sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.input_selections().size(), 1U);
  EXPECT_EQ(plan.input_selections().front().max_selected_samples_origin,
            SelectedSampleBoundOrigin::kResolvedRetainedCapacity);

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{91U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{91U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  auto* const runtime_source = find_runtime_operator<RuntimeRateLifecycleSource>(**exec);
  const OperatorInstanceId owner = find_runtime_operator_id<RuntimeRateLifecycleSource>(**exec);
  ASSERT_NE(runtime_source, nullptr);

  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kStartupForward,
                                          &runtime_source->output,
                                          Rate::hz(60U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kStartupForward));
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());

  auto snapshot = (*rates)->capture_snapshot();
  ASSERT_TRUE(snapshot) << snapshot.error().message;
  ASSERT_NE(snapshot->published(), nullptr);
  const auto queues = snapshot->queue_sizings(*snapshot->published());
  ASSERT_EQ(queues.size(), 1U);
  EXPECT_EQ(queues.front().nominal_arrivals, 60U);
  EXPECT_EQ(queues.front().effective_logical_depth, 120U);
  EXPECT_EQ(queues.front().stability, StabilityClass::kStableNominalEstimate);
  EXPECT_FALSE(queues.front().stability_reason);
}

// A secondary 1 kHz input accumulates over the interarrival horizon of the lifecycle-selected
// camera driver. Runtime must evaluate the immutable inverse-driver recipe for every generation.
// REQ: COMPILE-069
// REQ: LIFECYCLE-012
TEST(RuntimeRateStateTest, LifecycleOnEachDriverReevaluatesSecondaryQueuePerGeneration) {
  Graph graph{"runtime-rate-lifecycle-driver-horizon"};
  const auto camera = graph.op<RuntimeRateLifecycleSource>("camera");
  const auto imu = graph.op<RuntimeRateKilohertzSource>("imu");
  const auto fusion = graph.op<RuntimeRateCameraDrivenFusion>("fusion");
  graph.add_flow(camera->output, fusion->camera);
  graph.add_flow(imu->output, fusion->imu);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto imu_input =
      std::ranges::find(plan.ports(), std::string{"imu"}, &PortPlanRecord::local_name);
  ASSERT_NE(imu_input, plan.ports().end());
  ASSERT_EQ(imu_input->direction, PortDirection::kInput);
  const auto camera_input =
      std::ranges::find(plan.ports(), std::string{"camera"}, &PortPlanRecord::local_name);
  ASSERT_NE(camera_input, plan.ports().end());
  ASSERT_EQ(camera_input->direction, PortDirection::kInput);
  const auto imu_edge =
      std::ranges::find(plan.payload_edges(), imu_input->id, &PayloadEdgePlanRecord::target);
  ASSERT_NE(imu_edge, plan.payload_edges().end());
  const auto imu_recipe = std::ranges::find(
      plan.queue_sizing_recipes(), imu_edge->edge, &QueueSizingRecipePlanRecord::edge);
  ASSERT_NE(imu_recipe, plan.queue_sizing_recipes().end());
  EXPECT_EQ(imu_recipe->latest_binding_stage, LifecycleStage::kDiscover);
  const auto imu_horizon = std::ranges::find(
      plan.accumulation_horizons(), imu_edge->edge, &AccumulationHorizonPlanRecord::edge);
  ASSERT_NE(imu_horizon, plan.accumulation_horizons().end());
  EXPECT_EQ(imu_horizon->origin, AccumulationHorizonOrigin::kDriverInterarrival);
  EXPECT_EQ(imu_horizon->inverse_rate_port, camera_input->id);
  EXPECT_FALSE(imu_horizon->horizon.has_value());
  const std::string plan_json{plan.json()};
  EXPECT_NE(plan_json.find("\"horizon\":null"), std::string::npos);
  EXPECT_NE(plan_json.find("\"inverse_rate_port\":" + std::to_string(camera_input->id.value())),
            std::string::npos);

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{93U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{93U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  auto* const runtime_camera = find_runtime_operator<RuntimeRateLifecycleSource>(**exec);
  const OperatorInstanceId camera_owner =
      find_runtime_operator_id<RuntimeRateLifecycleSource>(**exec);
  ASSERT_NE(runtime_camera, nullptr);

  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->bind_nominal_rate(camera_owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kStartupForward,
                                          &runtime_camera->output,
                                          Rate::hz(60U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kStartupForward));
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());
  auto initial = (*rates)->capture_snapshot();
  ASSERT_TRUE(initial) << initial.error().message;
  ASSERT_NE(initial->published(), nullptr);
  const auto initial_queues = initial->queue_sizings(*initial->published());
  const auto initial_imu =
      std::ranges::find(initial_queues, imu_recipe->id, &QueueSizingGenerationRecord::recipe);
  ASSERT_NE(initial_imu, initial_queues.end());
  EXPECT_EQ(initial_imu->nominal_arrivals, 17U);
  EXPECT_EQ(initial_imu->rate_recommendation, 34U);
  EXPECT_EQ(initial_imu->effective_logical_depth, 34U);
  EXPECT_FALSE(initial_imu->admitted_depth_ceiling.has_value());

  const std::array members{
      find_runtime_operator_id<RuntimeRateLifecycleSource>(**exec),
      find_runtime_operator_id<RuntimeRateKilohertzSource>(**exec),
      find_runtime_operator_id<RuntimeRateCameraDrivenFusion>(**exec),
  };
  ASSERT_TRUE((*rates)->begin_restart_candidate(members, LifecycleStage::kDiscover));
  ASSERT_TRUE((*rates)->bind_nominal_rate(camera_owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kRestartForward,
                                          &runtime_camera->output,
                                          Rate::hz(30U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kRestartForward));
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());
  auto restarted = (*rates)->capture_snapshot();
  ASSERT_TRUE(restarted) << restarted.error().message;
  ASSERT_NE(restarted->published(), nullptr);
  const auto restarted_queues = restarted->queue_sizings(*restarted->published());
  const auto restarted_imu =
      std::ranges::find(restarted_queues, imu_recipe->id, &QueueSizingGenerationRecord::recipe);
  ASSERT_NE(restarted_imu, restarted_queues.end());
  EXPECT_EQ(restarted_imu->nominal_arrivals, 34U);
  EXPECT_EQ(restarted_imu->rate_recommendation, 68U);
  EXPECT_EQ(restarted_imu->effective_logical_depth, 68U);
  EXPECT_FALSE(restarted_imu->admitted_depth_ceiling.has_value());
  EXPECT_EQ(restarted->published()->run_generation, initial->published()->run_generation);
  EXPECT_NE(restarted->published()->rate_generation, initial->published()->rate_generation);
}

// Restart materialization is a depth-delta transaction, not a whole-plan queue transaction. A
// disconnected fixed-rate queue and provider must not acquire candidate mutation/evidence merely
// because another component publishes a new lifecycle-selected rate generation.
// REQ: COMPILE-069
// REQ: LIFECYCLE-012
TEST(RuntimeRateStateTest, RestartMaterializationScopesQueueAndProviderUpdatesToRateDelta) {
  Graph graph{"runtime-rate-disconnected-restart-materialization"};
  const auto lifecycle_source = graph.op<RuntimeRateLifecycleTensorSource>("lifecycle-source");
  const auto lifecycle_sink = graph.op<RuntimeRateTensorClockSink>("lifecycle-sink");
  const auto fixed_source = graph.op<RuntimeRateFixedTensorSource>("fixed-source");
  const auto fixed_sink = graph.op<RuntimeRateTensorClockSink>("fixed-sink");
  graph.add_flow(lifecycle_source->output, lifecycle_sink->input);
  graph.add_flow(fixed_source->output, fixed_sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.queue_sizing_recipes().size(), 2U);
  ASSERT_EQ(plan.provider_allocation_plans().size(), 2U);

  const auto lifecycle_recipe = std::ranges::find_if(
      plan.queue_sizing_recipes(), [](const QueueSizingRecipePlanRecord& recipe) {
        return recipe.latest_binding_stage == LifecycleStage::kDiscover;
      });
  ASSERT_NE(lifecycle_recipe, plan.queue_sizing_recipes().end());
  const auto fixed_recipe = std::ranges::find_if(
      plan.queue_sizing_recipes(), [&](const QueueSizingRecipePlanRecord& recipe) {
        return recipe.id != lifecycle_recipe->id && !recipe.latest_binding_stage;
      });
  ASSERT_NE(fixed_recipe, plan.queue_sizing_recipes().end());

  const auto lifecycle_provider = std::ranges::find(plan.provider_allocation_plans(),
                                                    AllocationGeometryMode::kLifecycleCandidate,
                                                    &ProviderAllocationPlanRecord::geometry_mode);
  ASSERT_NE(lifecycle_provider, plan.provider_allocation_plans().end());
  const auto fixed_provider = std::ranges::find_if(
      plan.provider_allocation_plans(), [&](const ProviderAllocationPlanRecord& provider) {
        return provider.id != lifecycle_provider->id &&
               provider.geometry_mode == AllocationGeometryMode::kCompileConcrete;
      });
  ASSERT_NE(fixed_provider, plan.provider_allocation_plans().end());

  auto exec = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                             RunGenerationId{95U});
  ASSERT_TRUE(exec) << exec.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{95U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  auto* const runtime_source = find_runtime_operator<RuntimeRateLifecycleTensorSource>(**exec);
  const OperatorInstanceId lifecycle_owner =
      find_runtime_operator_id<RuntimeRateLifecycleTensorSource>(**exec);
  ASSERT_NE(runtime_source, nullptr);
  ASSERT_NE(lifecycle_owner, 0U);

  struct MaterializationObservation {
    EdgeId lifecycle_edge{};
    EdgeId fixed_edge{};
    ProviderAllocationPlanRecordId lifecycle_provider{};
    ProviderAllocationPlanRecordId fixed_provider{};
    std::uint32_t lifecycle_depth{};
    std::uint32_t fixed_depth{};
    std::uint32_t lifecycle_physical_entries{};
    bool lifecycle_update_required{};
    bool fixed_update_required{};
    bool lifecycle_provider_was_unreported{};
    bool fixed_provider_was_unreported{};
  } startup_observation{
      .lifecycle_edge = lifecycle_recipe->edge,
      .fixed_edge = fixed_recipe->edge,
      .lifecycle_provider = lifecycle_provider->id,
      .fixed_provider = fixed_provider->id,
  };

  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->bind_nominal_rate(lifecycle_owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kStartupForward,
                                          &runtime_source->output,
                                          Rate::hz(60U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kStartupForward));
  auto startup_visited = (*rates)->visit_candidate_materialization(
      &startup_observation,
      [](void* state,
         const detail::RuntimeRateMaterializationView& view) noexcept -> expected<void, Error> {
        if (state == nullptr || view.queue_depths.size() != 2U ||
            view.pool_capacities.size() != 2U || view.tensor_allocations.size() != 1U ||
            view.provider_results.size() != 2U) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        auto& observed = *static_cast<MaterializationObservation*>(state);
        const auto lifecycle_depth = std::ranges::find(
            view.queue_depths, observed.lifecycle_edge, &detail::RuntimeQueueDepthGeneration::edge);
        const auto fixed_depth = std::ranges::find(
            view.queue_depths, observed.fixed_edge, &detail::RuntimeQueueDepthGeneration::edge);
        const auto lifecycle_allocation =
            std::ranges::find(view.tensor_allocations,
                              observed.lifecycle_provider,
                              &detail::RuntimeTensorAllocationGeneration::provider);
        auto lifecycle_result =
            std::ranges::find(view.provider_results,
                              observed.lifecycle_provider,
                              &detail::RuntimeProviderMaterializationResult::provider);
        auto fixed_result =
            std::ranges::find(view.provider_results,
                              observed.fixed_provider,
                              &detail::RuntimeProviderMaterializationResult::provider);
        const auto fixed_capacity =
            std::ranges::find(view.pool_capacities,
                              observed.fixed_provider,
                              &detail::RuntimePoolCapacityGeneration::provider);
        if (lifecycle_depth == view.queue_depths.end() || fixed_depth == view.queue_depths.end() ||
            lifecycle_allocation == view.tensor_allocations.end() ||
            lifecycle_result == view.provider_results.end() ||
            fixed_result == view.provider_results.end() ||
            fixed_capacity == view.pool_capacities.end()) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        observed.lifecycle_depth = lifecycle_depth->active_logical_depth;
        observed.fixed_depth = fixed_depth->active_logical_depth;
        observed.lifecycle_update_required = lifecycle_depth->active_depth_update_required;
        observed.fixed_update_required = fixed_depth->active_depth_update_required;
        observed.lifecycle_physical_entries = lifecycle_allocation->allocation.slot_count;
        observed.lifecycle_provider_was_unreported = !lifecycle_result->reported;
        observed.fixed_provider_was_unreported = !fixed_result->reported;
        *lifecycle_result = detail::RuntimeProviderMaterializationResult{
            .provider = observed.lifecycle_provider,
            .action = AllocationAction::kFresh,
            .materialized_entries = observed.lifecycle_physical_entries,
            .reported = true,
        };
        *fixed_result = detail::RuntimeProviderMaterializationResult{
            .provider = observed.fixed_provider,
            .action = AllocationAction::kFresh,
            .materialized_entries = fixed_capacity->required_entries,
            .reported = true,
        };
        return {};
      });
  ASSERT_TRUE(startup_visited) << startup_visited.error().message;
  EXPECT_EQ(startup_observation.lifecycle_depth, 120U);
  EXPECT_EQ(startup_observation.fixed_depth, 60U);
  EXPECT_TRUE(startup_observation.lifecycle_update_required);
  EXPECT_FALSE(startup_observation.fixed_update_required);
  EXPECT_TRUE(startup_observation.lifecycle_provider_was_unreported);
  EXPECT_TRUE(startup_observation.fixed_provider_was_unreported);
  ASSERT_GT(startup_observation.lifecycle_physical_entries, 0U);
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());

  MaterializationObservation restart_observation{
      .lifecycle_edge = lifecycle_recipe->edge,
      .fixed_edge = fixed_recipe->edge,
      .lifecycle_provider = lifecycle_provider->id,
      .fixed_provider = fixed_provider->id,
      .lifecycle_physical_entries = startup_observation.lifecycle_physical_entries,
  };
  const std::array members{lifecycle_owner};
  ASSERT_TRUE((*rates)->begin_restart_candidate(members, LifecycleStage::kDiscover));
  ASSERT_TRUE((*rates)->bind_nominal_rate(lifecycle_owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kRestartForward,
                                          &runtime_source->output,
                                          Rate::hz(30U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kRestartForward));
  auto restart_visited = (*rates)->visit_candidate_materialization(
      &restart_observation,
      [](void* state,
         const detail::RuntimeRateMaterializationView& view) noexcept -> expected<void, Error> {
        if (state == nullptr || view.queue_depths.size() != 2U ||
            view.pool_capacities.size() != 2U || view.provider_results.size() != 2U) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        auto& observed = *static_cast<MaterializationObservation*>(state);
        const auto lifecycle_depth = std::ranges::find(
            view.queue_depths, observed.lifecycle_edge, &detail::RuntimeQueueDepthGeneration::edge);
        const auto fixed_depth = std::ranges::find(
            view.queue_depths, observed.fixed_edge, &detail::RuntimeQueueDepthGeneration::edge);
        auto lifecycle_result =
            std::ranges::find(view.provider_results,
                              observed.lifecycle_provider,
                              &detail::RuntimeProviderMaterializationResult::provider);
        const auto fixed_result =
            std::ranges::find(view.provider_results,
                              observed.fixed_provider,
                              &detail::RuntimeProviderMaterializationResult::provider);
        if (lifecycle_depth == view.queue_depths.end() || fixed_depth == view.queue_depths.end() ||
            lifecycle_result == view.provider_results.end() ||
            fixed_result == view.provider_results.end()) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        observed.lifecycle_depth = lifecycle_depth->active_logical_depth;
        observed.fixed_depth = fixed_depth->active_logical_depth;
        observed.lifecycle_update_required = lifecycle_depth->active_depth_update_required;
        observed.fixed_update_required = fixed_depth->active_depth_update_required;
        observed.lifecycle_provider_was_unreported = !lifecycle_result->reported;
        observed.fixed_provider_was_unreported = !fixed_result->reported;
        *lifecycle_result = detail::RuntimeProviderMaterializationResult{
            .provider = observed.lifecycle_provider,
            .action = AllocationAction::kReusedLarger,
            .materialized_entries = observed.lifecycle_physical_entries,
            .reported = true,
        };
        return {};
      });
  ASSERT_TRUE(restart_visited) << restart_visited.error().message;
  EXPECT_EQ(restart_observation.lifecycle_depth, 60U);
  EXPECT_EQ(restart_observation.fixed_depth, startup_observation.fixed_depth);
  EXPECT_TRUE(restart_observation.lifecycle_update_required);
  EXPECT_FALSE(restart_observation.fixed_update_required);
  EXPECT_TRUE(restart_observation.lifecycle_provider_was_unreported);
  EXPECT_TRUE(restart_observation.fixed_provider_was_unreported);
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());

  auto restarted = (*rates)->capture_snapshot();
  ASSERT_TRUE(restarted) << restarted.error().message;
  ASSERT_NE(restarted->published(), nullptr);
  const auto providers = restarted->provider_allocations(*restarted->published());
  ASSERT_EQ(providers.size(), 1U);
  EXPECT_EQ(providers.front().recipe, lifecycle_provider->id);
  EXPECT_EQ(providers.front().action, AllocationAction::kReusedLarger);
}

// REQ: LIFECYCLE-012
// REQ: COMPILE-070
TEST(RuntimeRateStateTest, EvaluatesNoMaximumTensorCandidateWithoutMutatingThePlan) {
  Graph graph{"runtime-rate-tensor-candidate"};
  const auto source = graph.op<RuntimeRateLifecycleTensorSource>("source");
  const auto sink = graph.op<RuntimeRateTensorClockSink>("sink");
  graph.add_flow(source->output, sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(plan.tensor_allocations().empty());
  ASSERT_EQ(plan.provider_allocation_plans().size(), 1U);
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(detail::valid_closed_island_activation_plan(*state));
  ASSERT_TRUE(detail::valid_inline_partition_execution_plan(*state));
  ASSERT_TRUE(detail::valid_partition_runner_plan(*state));

  detail::RuntimeReplayMismatchDetail replay_mismatch;
  auto exec = detail::GraphExec::reconstruct(
      detail::ExecutionPlanAccess::retain(plan), RunGenerationId{97U}, {}, &replay_mismatch);
  ASSERT_TRUE(exec) << exec.error().message
                    << ": category=" << static_cast<std::uint32_t>(replay_mismatch.category);
  auto rates = detail::RuntimeRateState::create(
      (*exec)->plan(), (*exec)->operators(), RunGenerationId{97U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;
  auto* const runtime_source = find_runtime_operator<RuntimeRateLifecycleTensorSource>(**exec);
  const OperatorInstanceId owner =
      find_runtime_operator_id<RuntimeRateLifecycleTensorSource>(**exec);
  ASSERT_NE(runtime_source, nullptr);

  ASSERT_TRUE((*rates)->begin_startup_candidate());
  ASSERT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kStartupForward,
                                          &runtime_source->output,
                                          Rate::hz(60U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kStartupForward));

  struct Observed {
    std::uint32_t queue_depth{};
    std::uint32_t pool_entries{};
    std::uint32_t tensor_slots{};
    std::uint64_t tensor_bytes{};
  } observed;
  auto visited = (*rates)->visit_candidate_materialization(
      &observed,
      [](void* state,
         const detail::RuntimeRateMaterializationView& view) noexcept -> expected<void, Error> {
        if (state == nullptr || view.queue_depths.size() != 1U ||
            view.pool_capacities.size() != 1U || view.tensor_allocations.size() != 1U) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        auto& result = *static_cast<Observed*>(state);
        result.queue_depth = view.queue_depths.front().active_logical_depth;
        result.pool_entries = view.pool_capacities.front().required_entries;
        result.tensor_slots = view.tensor_allocations.front().allocation.slot_count;
        result.tensor_bytes = view.tensor_allocations.front().allocation.aggregate_byte_capacity;
        view.provider_results.front() = detail::RuntimeProviderMaterializationResult{
            .provider = view.pool_capacities.front().provider,
            .action = AllocationAction::kFresh,
            .materialized_entries = result.tensor_slots,
            .reported = true,
        };
        return {};
      });
  ASSERT_TRUE(visited) << visited.error().message;
  EXPECT_EQ(observed.queue_depth, 120U);
  // This graph retains delivery and the clock-selected sample simultaneously. The two
  // rate-dependent groups therefore contribute 120 each, plus one producer loan and one
  // retirement slot.
  EXPECT_EQ(observed.pool_entries, 242U);
  EXPECT_EQ(observed.tensor_slots, 242U);
  EXPECT_EQ(observed.tensor_bytes, 242U * 64U);
  EXPECT_TRUE(plan.tensor_allocations().empty());

  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());
  auto startup = (*rates)->capture_snapshot();
  ASSERT_TRUE(startup) << startup.error().message;
  ASSERT_NE(startup->published(), nullptr);
  const auto startup_providers = startup->provider_allocations(*startup->published());
  ASSERT_EQ(startup_providers.size(), 1U);
  const auto startup_provider = startup_providers.front();
  EXPECT_EQ(startup_provider.action, AllocationAction::kFresh);
  ASSERT_TRUE(startup_provider.allocation_generation);
  ASSERT_TRUE(startup_provider.storage_binding_generation);
  EXPECT_FALSE(startup_provider.provider_stride_bytes_per_entry);
  EXPECT_FALSE(startup_provider.provider_reserved_bytes_total);

  const std::array members{owner};
  ASSERT_TRUE((*rates)->begin_restart_candidate(members, LifecycleStage::kDiscover));
  ASSERT_TRUE((*rates)->bind_nominal_rate(owner,
                                          LifecycleStage::kDiscover,
                                          LifecycleTransitionPhase::kRestartForward,
                                          &runtime_source->output,
                                          Rate::hz(30U)));
  ASSERT_TRUE((*rates)->close_stage_barrier(LifecycleStage::kDiscover,
                                            LifecycleTransitionPhase::kRestartForward));
  struct RestartObserved {
    std::uint32_t required_entries{};
    std::uint32_t physical_entries{};
  } restart_observed{.physical_entries = observed.tensor_slots};
  auto restart_visited = (*rates)->visit_candidate_materialization(
      &restart_observed,
      [](void* state,
         const detail::RuntimeRateMaterializationView& view) noexcept -> expected<void, Error> {
        if (state == nullptr || view.pool_capacities.size() != 1U ||
            view.provider_results.size() != 1U) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        auto& observed = *static_cast<RestartObserved*>(state);
        observed.required_entries = view.pool_capacities.front().required_entries;
        view.provider_results.front() = detail::RuntimeProviderMaterializationResult{
            .provider = view.pool_capacities.front().provider,
            .action = AllocationAction::kReusedLarger,
            .materialized_entries = observed.physical_entries,
            .reported = true,
        };
        return {};
      });
  ASSERT_TRUE(restart_visited) << restart_visited.error().message;
  EXPECT_LT(restart_observed.required_entries, observed.tensor_slots);
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());

  auto reused = (*rates)->capture_snapshot();
  ASSERT_TRUE(reused) << reused.error().message;
  ASSERT_NE(reused->published(), nullptr);
  const auto reused_providers = reused->provider_allocations(*reused->published());
  ASSERT_EQ(reused_providers.size(), 1U);
  const auto& reused_provider = reused_providers.front();
  EXPECT_EQ(reused_provider.action, AllocationAction::kReusedLarger);
  EXPECT_EQ(reused_provider.allocation_generation, startup_provider.allocation_generation);
  ASSERT_TRUE(reused_provider.storage_binding_generation);
  EXPECT_NE(reused_provider.storage_binding_generation,
            startup_provider.storage_binding_generation);
  EXPECT_EQ(reused_provider.previous_decision, startup_provider.id);
  EXPECT_EQ(reused_provider.previous_allocation_generation, startup_provider.allocation_generation);
  EXPECT_EQ(reused_provider.rate_generation, reused->published()->rate_generation);

  const std::string plan_json_before_inspection{plan.json()};
  const auto inspection = render_rate_queue_inspection(plan,
                                                       *reused,
                                                       reused->published()->rate_generation,
                                                       plan.payload_edges().front().edge,
                                                       RateQueueInspectionFormat::kCanonicalJson);
  ASSERT_TRUE(inspection) << inspection.error().message;
  EXPECT_NE(inspection->find(R"("action":"reused_larger")"), std::string::npos);
  const auto& allocation_generation = reused_provider.allocation_generation;
  const auto& storage_binding_generation = reused_provider.storage_binding_generation;
  HOLOSCAN_ASSERT_HAS_VALUE(allocation_generation);
  HOLOSCAN_ASSERT_HAS_VALUE(storage_binding_generation);
  EXPECT_NE(
      inspection->find("\"allocation_generation\":" + std::to_string(allocation_generation->value)),
      std::string::npos);
  EXPECT_NE(inspection->find("\"storage_binding_generation\":" +
                             std::to_string(storage_binding_generation->value)),
            std::string::npos);
  EXPECT_NE(inspection->find("\"required_pool_entries\":" +
                             std::to_string(restart_observed.required_entries)),
            std::string::npos);
  EXPECT_NE(inspection->find("\"logical_payload_capacity_bytes\":" +
                             std::to_string(restart_observed.required_entries * 64U)),
            std::string::npos);
  EXPECT_NE(inspection->find(R"("max_payload_byte_span_per_slot":64)"), std::string::npos);
  EXPECT_NE(inspection->find(R"("admitted_pool_entry_ceiling":null)"), std::string::npos);
  EXPECT_NE(inspection->find(R"("admitted_logical_payload_capacity_ceiling_bytes":null)"),
            std::string::npos);
  EXPECT_NE(inspection->find(R"("materialized_entries":242)"), std::string::npos);
  EXPECT_NE(inspection->find(R"("stride_bytes_per_entry":null)"), std::string::npos);
  EXPECT_NE(inspection->find(R"("reserved_bytes_total":null)"), std::string::npos);
  EXPECT_EQ(plan.json(), plan_json_before_inspection);
  EXPECT_EQ(reused->provider_allocations(*reused->published()).front().id, reused_provider.id);

  // A restart after the selectable rate's binding stage retains the published binding and
  // backing. It must not demand or fabricate a provider allocation decision.
  ASSERT_TRUE((*rates)->begin_restart_candidate(members, LifecycleStage::kArm));
  ASSERT_TRUE((*rates)->preflight_candidate_publication());
  ASSERT_TRUE((*rates)->publish_candidate());
  auto retained = (*rates)->capture_snapshot();
  ASSERT_TRUE(retained) << retained.error().message;
  ASSERT_NE(retained->published(), nullptr);
  EXPECT_TRUE(retained->provider_allocations(*retained->published()).empty());
}

// Public range helpers must accept only the exact evidence record owned by their snapshot. Session-
// local child IDs intentionally collide across runs, so foreign or copied parents must fail closed;
// owned parents must also describe the exact child range and generation.
// REQ: OPERATOR-048
TEST(RuntimeRateStateTest, SnapshotRangesRejectForeignAndCraftedGenerationEvidence) {
  Graph graph{"runtime-rate-snapshot-range-ownership"};
  const auto source = graph.op<RuntimeRateLifecycleTensorSource>("source");
  const auto sink = graph.op<RuntimeRateTensorClockSink>("sink");
  graph.add_flow(source->output, sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-rate-clock"));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  auto first = capture_complete_tensor_rate_snapshot(plan, RunGenerationId{201U});
  auto second = capture_complete_tensor_rate_snapshot(plan, RunGenerationId{202U});
  ASSERT_TRUE(first) << first.error().message;
  ASSERT_TRUE(second) << second.error().message;
  ASSERT_NE(first->published(), nullptr);
  ASSERT_NE(second->published(), nullptr);
  ASSERT_FALSE(second->port_rates(*second->published()).empty());
  ASSERT_FALSE(second->queue_sizings(*second->published()).empty());
  ASSERT_FALSE(second->pool_contributions(*second->published()).empty());
  ASSERT_FALSE(second->pool_capacities(*second->published()).empty());
  ASSERT_FALSE(second->provider_allocations(*second->published()).empty());

  // Both sessions begin at rate generation one, so address ownership—not session-local child IDs—
  // is what prevents a foreign parent from selecting this snapshot's colliding ranges.
  ASSERT_EQ(first->published()->rate_generation, second->published()->rate_generation);
  EXPECT_TRUE(second->port_rates(*first->published()).empty());
  EXPECT_TRUE(second->queue_sizings(*first->published()).empty());
  EXPECT_TRUE(second->pool_contributions(*first->published()).empty());
  EXPECT_TRUE(second->pool_capacities(*first->published()).empty());
  EXPECT_TRUE(second->provider_allocations(*first->published()).empty());

  const RateGenerationEvidenceRecord copied_parent = *second->published();
  EXPECT_TRUE(second->port_rates(copied_parent).empty());
  EXPECT_TRUE(second->queue_sizings(copied_parent).empty());
  EXPECT_TRUE(second->pool_contributions(copied_parent).empty());
  EXPECT_TRUE(second->pool_capacities(copied_parent).empty());
  EXPECT_TRUE(second->provider_allocations(copied_parent).empty());

  // Copy construction creates an independent immutable snapshot. Its exact parent is accepted,
  // while the corresponding record from the source snapshot remains foreign by address.
  const RateGenerationSnapshot copied_snapshot{*second};
  ASSERT_NE(copied_snapshot.published(), nullptr);
  EXPECT_FALSE(copied_snapshot.port_rates(*copied_snapshot.published()).empty());
  EXPECT_TRUE(copied_snapshot.port_rates(*second->published()).empty());

  auto forged_contribution_link = *second;
  const auto linked_contribution = std::ranges::find_if(
      forged_contribution_link.pool_contributions(),
      [](const auto& record) { return record.edge_sizing_generation.has_value(); });
  ASSERT_NE(linked_contribution, forged_contribution_link.pool_contributions().end());
  auto& linked_queue = const_cast<PoolCapacityContributionGenerationRecord&>(*linked_contribution)
                           .edge_sizing_generation;
  HOLOSCAN_ASSERT_HAS_VALUE(linked_queue);
  linked_queue->value += (std::uint64_t{1U} << detail::kRateGenerationRecordLocalBits);
  EXPECT_TRUE(
      forged_contribution_link.pool_contributions(*forged_contribution_link.published()).empty());

  auto forged_port_range = *second;
  auto& forged_port_parent =
      const_cast<RateGenerationEvidenceRecord&>(*forged_port_range.published());
  ++forged_port_parent.port_rate_count;
  EXPECT_TRUE(forged_port_range.port_rates(forged_port_parent).empty());

  auto forged_queue_range = *second;
  auto& forged_queue_parent =
      const_cast<RateGenerationEvidenceRecord&>(*forged_queue_range.published());
  ++forged_queue_parent.queue_sizing_count;
  EXPECT_TRUE(forged_queue_range.queue_sizings(forged_queue_parent).empty());

  auto forged_contribution_range = *second;
  auto& forged_contribution_parent =
      const_cast<RateGenerationEvidenceRecord&>(*forged_contribution_range.published());
  ++forged_contribution_parent.pool_contribution_count;
  EXPECT_TRUE(forged_contribution_range.pool_contributions(forged_contribution_parent).empty());

  auto forged_pool_range = *second;
  auto& forged_pool_parent =
      const_cast<RateGenerationEvidenceRecord&>(*forged_pool_range.published());
  ++forged_pool_parent.pool_capacity_count;
  EXPECT_TRUE(forged_pool_range.pool_capacities(forged_pool_parent).empty());

  auto forged_provider_range = *second;
  auto& forged_provider_parent =
      const_cast<RateGenerationEvidenceRecord&>(*forged_provider_range.published());
  ++forged_provider_parent.provider_allocation_count;
  EXPECT_TRUE(forged_provider_range.provider_allocations(forged_provider_parent).empty());

  auto forged_children = *second;
  const auto* forged_generation = forged_children.published();
  ASSERT_NE(forged_generation, nullptr);
  auto& port = const_cast<PortRateGenerationRecord&>(forged_children.port_rates().front());
  port.run_generation = RunGenerationId{};
  auto& queue = const_cast<QueueSizingGenerationRecord&>(forged_children.queue_sizings().front());
  ++queue.rate_generation;
  auto& contribution = const_cast<PoolCapacityContributionGenerationRecord&>(
      forged_children.pool_contributions().front());
  contribution.id.value += (std::uint64_t{1U} << 32U);
  auto& pool = const_cast<PoolCapacityGenerationRecord&>(forged_children.pool_capacities().front());
  pool.run_generation = RunGenerationId{};
  auto& provider = const_cast<ProviderAllocationGenerationRecord&>(
      forged_children.provider_allocations().front());
  ++provider.rate_generation;
  EXPECT_TRUE(forged_children.port_rates(*forged_generation).empty());
  EXPECT_TRUE(forged_children.queue_sizings(*forged_generation).empty());
  EXPECT_TRUE(forged_children.pool_contributions(*forged_generation).empty());
  EXPECT_TRUE(forged_children.pool_capacities(*forged_generation).empty());
  EXPECT_TRUE(forged_children.provider_allocations(*forged_generation).empty());
}

}  // namespace
}  // namespace holoscan
