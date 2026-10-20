/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>

#include <holoscan/logger/logger.hpp>
#include <holoscan/time/clock.hpp>
#include <holoscan/time/manual_clock.hpp>

#include "channel/endpoint_catalog_state.hpp"
#include "compile_test_graph.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/compile_platform.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/runtime_package_capabilities.hpp"

namespace holoscan {
namespace {

// Copy-counting capture type parameterized by Tag so operator and clock captures
// can be tracked independently. Throws when copies exceed copy_limit, which makes
// it easy to assert that compile() never duplicates a factory snapshot.
template <std::uint32_t Tag>
struct CountedFactoryCapture {
  CountedFactoryCapture() = default;

  CountedFactoryCapture(const CountedFactoryCapture&) {
    ++copies;
    if (copy_limit != 0U && copies > copy_limit) {
      throw std::runtime_error{"unexpected executable factory copy"};
    }
  }

  CountedFactoryCapture& operator=(const CountedFactoryCapture&) = default;

  static void reset(std::size_t limit = 0U) noexcept {
    copies = 0U;
    copy_limit = limit;
  }

  static inline std::size_t copies{};
  static inline std::size_t copy_limit{};
};

using OperatorFactoryCapture = CountedFactoryCapture<1U>;
using ClockFactoryCapture = CountedFactoryCapture<2U>;

// Minimal relay operator (queue_depth=2, max_emits=1) that accepts an
// OperatorFactoryCapture at construction. Each instance records copies so the
// test can verify compile() takes at most one factory snapshot per operator.
class CountedFactoryOperator final : public Operator<> {
 public:
  explicit CountedFactoryOperator(const OperatorFactoryCapture&) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2);
    spec.output(output, "output").max_emits_per_compute(1);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
};

// Stub Clock that accepts a ClockFactoryCapture and always returns timestamp 0.
// Used alongside CountedFactoryOperator in CapturesEachExecutableFactoryOnce
// to verify at most one factory copy per clock.
class CountedFactoryClock final : public Clock {
 public:
  explicit CountedFactoryClock(const ClockFactoryCapture&) {}

  [[nodiscard]] double time() const override { return timestamp_to_time(timestamp()); }
  [[nodiscard]] std::int64_t timestamp() const override { return 0; }
  [[nodiscard]] expected<void, Error> sleep_for(std::int64_t) override { return {}; }
  [[nodiscard]] expected<void, Error> sleep_until(std::int64_t) override { return {}; }
};

// REQ: COMPILE-009
// REQ: COMPILE-095
TEST(CompilePurity, HasNoRuntimeArtifactOrOsMutationSideEffects) {
  test::CompileTestOperator::compute_calls = 0U;
  test::CompileTestOperator::lifecycle_calls = 0U;
  // source --> target
  Graph graph{"pure-compile"};
  test::add_linear_graph(graph);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(test::CompileTestOperator::compute_calls, 0U);
  EXPECT_EQ(test::CompileTestOperator::lifecycle_calls, 0U);
  const auto* catalog_state = detail::ChannelEndpointCatalogAccess::state(plan.endpoint_catalog());
  EXPECT_TRUE(catalog_state == nullptr || catalog_state->endpoints.empty());
  EXPECT_FALSE(plan.lifecycle_transaction_capacity().has_value());
  EXPECT_EQ(plan.scheduler_plan().worker_count, 1U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 1U);
  EXPECT_FALSE(plan.scheduler_plan().activation_overlap);
}

// REQ: COMPILE-009
// REQ: COMPILE-095
// GoogleTest fatal assertions terminate this test before optional access, but clang-tidy does not
// model the macro's control flow.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
TEST(CompilePurity, CapturesEachExecutableFactoryOnce) {
  // source --> target  (both CountedFactoryOperator, plus one CountedFactoryClock)
  Graph graph{"single-factory-capture"};
  const auto source = graph.op<CountedFactoryOperator>("source", OperatorFactoryCapture{});
  const auto target = graph.op<CountedFactoryOperator>("target", OperatorFactoryCapture{});
  graph.add_flow(source->output, target->input);
  const ClockKey clock = graph.add_clock<CountedFactoryClock>("clock", ClockFactoryCapture{});
  graph.set_default_clock(clock);

  // Key setup: limits set to 2 per operator and 1 per clock; a second inventory
  // snapshot would throw before compile() returns.
  OperatorFactoryCapture::reset(2U);
  ClockFactoryCapture::reset(1U);
  std::optional<ExecutionPlan> plan;
  EXPECT_NO_THROW(plan.emplace(compile(graph)));
  // Key assertion: exactly 2 operator copies (one per op) and 1 clock copy.
  EXPECT_EQ(OperatorFactoryCapture::copies, 2U);
  EXPECT_EQ(ClockFactoryCapture::copies, 1U);
  OperatorFactoryCapture::reset();
  ClockFactoryCapture::reset();

  ASSERT_TRUE(plan.has_value());
  EXPECT_TRUE(plan->ok());
}
// NOLINTEND(bugprone-unchecked-optional-access)

// REQ: COMPILE-002
// REQ: COMPILE-015
// REQ: COMPILE-060
// REQ: COMPILE-023
// REQ: COMPILE-043
TEST(CompileIngress, RetainsExecutableFactoriesAfterTheSourceGraphIsDestroyed) {
  // Key setup: the source Graph is destroyed when the lambda returns; factories must
  // outlive it via retained state.
  //
  //   source --> target  (plus ManualClock)
  //
  const ExecutionPlan plan = [] {
    Graph graph{"factory-retention"};
    test::add_linear_graph_without_default_clock(graph);
    const ClockKey clock = graph.add_clock<ManualClock>("factory-clock", std::int64_t{0});
    graph.set_default_clock(clock);
    return detail::compile_with_runtime_evidence(
        graph,
        {},
        detail::RuntimeCompileEvidenceView{
            .package_capabilities = detail::runtime_package_capability_view(),
            .endpoint_clock_health = {},
        });
  }();

  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->executable_factories.operators.size(), 2U);
  ASSERT_EQ(state->executable_factories.clocks.size(), 1U);
  // Key assertion: each retained factory can still create a valid instance after
  // the source Graph is gone.
  for (const auto& record : state->executable_factories.operators) {
    ASSERT_NE(record.factory, nullptr);
    const auto created = record.factory->create(nullptr);
    ASSERT_TRUE(created);
    EXPECT_NE(created.value(), nullptr);
  }
  const auto& clock_record = state->executable_factories.clocks.front();
  ASSERT_NE(clock_record.factory, nullptr);
  EXPECT_EQ(clock_record.shared, nullptr);
  const auto created_clock = clock_record.factory->create(nullptr);
  ASSERT_TRUE(created_clock);
  EXPECT_NE(created_clock.value(), nullptr);
}

// REQ: COMPILE-002
// REQ: COMPILE-009
// REQ: COMPILE-095
TEST(CompileDeterminism, LoggingThresholdDoesNotChangePlanBytes) {
  const LogLevel original = log_level();
  // source --> target
  Graph graph{"logging-invariant"};
  test::add_linear_graph(graph);

  set_log_level(LogLevel::OFF);
  const ExecutionPlan quiet = compile(graph);

  set_log_level(LogLevel::TRACE);
  const ExecutionPlan trace = compile(graph);
  set_log_level(original);

  ASSERT_TRUE(quiet.ok());
  ASSERT_TRUE(trace.ok());
  EXPECT_EQ(quiet.digest(), trace.digest());
  EXPECT_EQ(quiet.json(), trace.json());
  EXPECT_EQ(quiet.diagnostics().size(), trace.diagnostics().size());
}

// REQ: COMPILE-009
// REQ: COMPILE-095
// REQ: COMPILE-014
// REQ: COMPILE-059
// REQ: COMPILE-101
TEST(CompilePortability, DistinguishesGnuAndUnknownLinuxUserspace) {
  using detail::CompileTargetArchitecture;
  using detail::CompileTargetPlatform;
  EXPECT_EQ(detail::compile_target_triple_for(CompileTargetArchitecture::kX86_64,
                                              CompileTargetPlatform::kLinuxGnu),
            std::string_view{"x86_64-unknown-linux-gnu"});
  EXPECT_EQ(detail::compile_target_triple_for(CompileTargetArchitecture::kX86_64,
                                              CompileTargetPlatform::kLinuxUnknown),
            std::string_view{"x86_64-unknown-linux-unknown"});
  EXPECT_EQ(detail::compile_target_triple_for(CompileTargetArchitecture::kAArch64,
                                              CompileTargetPlatform::kLinuxUnknown),
            std::string_view{"aarch64-unknown-linux-unknown"});
}

}  // namespace
}  // namespace holoscan
