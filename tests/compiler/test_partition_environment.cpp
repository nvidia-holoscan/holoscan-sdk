/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

#include <holoscan/time/manual_clock.hpp>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "compile_test_graph.hpp"
#include "compiler/partitioning/partition_planner_catalog.hpp"
#include "compiler/partitioning/partitioning_pass.hpp"
#include "partition_test_helpers.hpp"

namespace holoscan {
namespace {

constexpr const char* kPartitionEnvironment = "HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM";

// Cross-platform reader for HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM. Returns nullopt
// when the variable is absent; returns an empty string when present-but-empty (Windows).
[[nodiscard]] std::optional<std::string> read_partition_environment() {
#if defined(_WIN32)
  ::SetLastError(ERROR_SUCCESS);
  const DWORD required = ::GetEnvironmentVariableA(kPartitionEnvironment, nullptr, 0U);
  if (required == 0U) {
    return ::GetLastError() == ERROR_ENVVAR_NOT_FOUND ? std::nullopt
                                                      : std::optional<std::string>{std::string{}};
  }
  std::string value(required, '\0');
  ::SetLastError(ERROR_SUCCESS);
  const DWORD written = ::GetEnvironmentVariableA(kPartitionEnvironment, value.data(), required);
  if (written == 0U) {
    return ::GetLastError() == ERROR_SUCCESS ? std::optional<std::string>{std::string{}}
                                             : std::nullopt;
  }
  if (written >= required) {
    return std::nullopt;
  }
  value.resize(written);
  return value;
#else
  const char* const value = std::getenv(kPartitionEnvironment);
  return value == nullptr ? std::nullopt : std::optional<std::string>{value};
#endif
}

// Sets or clears HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM cross-platform.
// Passing nullopt removes the variable. Returns false on OS-level failure.
[[nodiscard]] bool update_partition_environment(std::optional<std::string_view> value) {
#if defined(_WIN32)
  if (!value.has_value()) {
    return ::SetEnvironmentVariableA(kPartitionEnvironment, nullptr) != 0;
  }
  const std::string owned{*value};
  return ::SetEnvironmentVariableA(kPartitionEnvironment, owned.c_str()) != 0;
#else
  return value.has_value() ? setenv(kPartitionEnvironment, std::string(*value).c_str(), 1) == 0
                           : unsetenv(kPartitionEnvironment) == 0;
#endif
}

// Fatal-asserting wrapper: aborts the current GTest if the OS env update fails,
// so callers do not need to inspect the return value.
void set_partition_environment(std::optional<std::string_view> value) {
  ASSERT_TRUE(update_partition_environment(value));
}

// RAII guard that saves the current partition-environment value on construction
// and restores it on destruction. Every test in this file instantiates one to
// prevent env-var mutations from leaking across test cases.
class ScopedPartitionEnvironment {
 public:
  ScopedPartitionEnvironment() : original_(read_partition_environment()) {}
  ~ScopedPartitionEnvironment() {
    if (original_.has_value()) {
      static_cast<void>(update_partition_environment(std::string_view{*original_}));
    } else {
      static_cast<void>(update_partition_environment(std::nullopt));
    }
  }

  ScopedPartitionEnvironment(const ScopedPartitionEnvironment&) = delete;
  ScopedPartitionEnvironment& operator=(const ScopedPartitionEnvironment&) = delete;

 private:
  std::optional<std::string> original_;
};

class UnusedReadinessInputOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.readable_source(unused, "unused");
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  ReadableSource unused;
};

class SelectedReadinessInputOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.readable_source(selected, "selected");
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnReadable{selected});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  ReadableSource selected;
};

class UnusedReadinessLeaf final : public Graph {
 public:
  Input<std::uint32_t> input{"input"};

 protected:
  void compose() override {
    const auto target = op<UnusedReadinessInputOperator>("target");
    export_input(input, target->input);
    const ClockKey clock = add_clock<ManualClock>("clock", std::int64_t{0});
    set_default_clock(clock);
  }
};

// Builds and compiles the canonical two-operator linear graph under the given profile.
//
//   source --> target
//
// Used by all PartitionEnvironmentSelectionTest cases.
[[nodiscard]] ExecutionPlan compile_linear_with_profile(Profile profile = Profile::kStandard) {
  Graph graph{"partition-environment"};
  test::add_linear_graph(graph);
  CompileOptions options;
  options.profile = profile;
  return compile(graph, options);
}

// Builds and compiles a one-source-to-two-sinks fanout graph:
//
//   source --> left
//          --> right
//
// Used in EnvironmentSelectedCoverCreatesNoImplicitTensorOrDeviceResources to
// compare partition counts across different algorithm selections.
[[nodiscard]] ExecutionPlan compile_fanout() {
  Graph graph{"partition-resource-separation"};
  test::add_compile_test_default_clock(graph);
  auto source = graph.op<test::CompileTestOperator>("source");
  auto left = graph.op<test::CompileTestOperator>("left");
  auto right = graph.op<test::CompileTestOperator>("right");
  graph.add_flow(source->output, left->input);
  graph.add_flow(source->output, right->input);
  return compile(graph);
}

// Declaration-only readiness sources remain inert under production source-chain planning. The
// planner must resolve the embedded interface predecessor and preserve
// one unnamed serial partition rather than isolating the inner Operator.
TEST(PartitionEnvironmentSelectionTest,
     ProductionSourceChainIgnoresUnusedReadinessAcrossEmbeddedInputBoundary) {
  // REQ: COMPILE-068
  const ScopedPartitionEnvironment guard;
  set_partition_environment(std::nullopt);

  Graph graph{"source-chain-unused-readiness"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto leaf = graph.embed<UnusedReadinessLeaf>("leaf");
  graph.add_flow(source->output, leaf->input);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_FALSE(plan.execution_partitions().front().authored_name.has_value());
  EXPECT_EQ(plan.execution_partitions().front().members.size(), 2U);
  EXPECT_TRUE(test::has_diagnostic(plan, "READINESS_SOURCE_UNUSED"));
}

// REQ: COMPILE-068
TEST(PartitionEnvironmentSelectionTest,
     ProductionSourceChainIsolatesSelectedIndependentReadinessRoot) {
  const ScopedPartitionEnvironment guard;
  set_partition_environment(std::nullopt);

  Graph graph{"source-chain-selected-readiness"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto target = graph.op<SelectedReadinessInputOperator>("target");
  graph.add_flow(source->output, target->input);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  EXPECT_TRUE(std::any_of(plan.partition_decisions().begin(),
                          plan.partition_decisions().end(),
                          [](const auto& decision) {
                            return std::find(decision.reasons.begin(),
                                             decision.reasons.end(),
                                             PartitionDecisionReason::kReadinessIsolation) !=
                                   decision.reasons.end();
                          }));
}

// REQ: COMPILE-005, COMPILE-068
TEST(PartitionEnvironmentSelectionTest, AbsentUsesProductionSourceChainDefault) {
  const ScopedPartitionEnvironment restore_environment;
  set_partition_environment(std::nullopt);
  const ExecutionPlan absent = compile_linear_with_profile();
  ASSERT_TRUE(absent.ok());
  const auto& absent_algorithm = absent.partitioning_plan().accepted_algorithm;
  if (!absent_algorithm.has_value()) {
    ADD_FAILURE() << "expected the default partition algorithm";
    return;
  }
  EXPECT_EQ(absent_algorithm->id, "holoscan.partition.source_chain");
  EXPECT_EQ(absent.partitioning_plan().selection_source, PlanSelectionSource::kCoreDefault);
  EXPECT_FALSE(test::has_diagnostic(absent, "PARTITION_PLANNER_EXPERIMENTAL_SELECTED"));
}

// REQ: COMPILE-005, COMPILE-068
TEST(PartitionEnvironmentSelectionTest, EveryPresentValueFailsClosedWithoutValueAuthority) {
  const ScopedPartitionEnvironment restore_environment;
  constexpr std::array present_values{
      "", "serial_baseline/v1", "source_chain/v1", "not-a-selector", "secret-value"};
  std::optional<PlanDigest> first_digest;
  std::optional<std::string> first_json;
  for (std::string_view present : present_values) {
    set_partition_environment(present);
    const ExecutionPlan plan = compile_linear_with_profile();
    EXPECT_FALSE(plan.ok()) << present;
    EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_ENVIRONMENT_SELECTOR_REMOVED")) << present;
    EXPECT_EQ(plan.partitioning_plan().selection_source, PlanSelectionSource::kCoreDefault);
    ASSERT_TRUE(plan.partitioning_plan().requested_algorithm.has_value());
    EXPECT_EQ(plan.partitioning_plan().requested_algorithm->id, "holoscan.partition.source_chain");
    EXPECT_FALSE(plan.partitioning_plan().accepted_algorithm.has_value());
    EXPECT_TRUE(static_cast<bool>(plan.partitioning_plan().reason));
    EXPECT_NE(plan.partitioning_plan().config_digest, PlanDigest{});
    const auto diagnostic =
        std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
          return item.token == "PARTITION_ENVIRONMENT_SELECTOR_REMOVED";
        });
    ASSERT_NE(diagnostic, plan.diagnostics().end());
    EXPECT_EQ(plan.partitioning_plan().reason, diagnostic->id);
    const std::string json{plan.json()};
    if (!present.empty()) {
      EXPECT_EQ(json.find(present), std::string::npos);
    }
    if (!first_digest.has_value()) {
      first_digest = plan.digest();
      first_json = json;
    } else {
      EXPECT_EQ(plan.digest(), *first_digest);
      EXPECT_EQ(json, *first_json);
    }
  }

  // The removed product selector does not remove the compiler-private, test-only
  // selection seam used to validate a planner in isolation.
  auto catalog = detail::partitioning::PartitionPlannerCatalog::create_default_builtins();
  ASSERT_TRUE(catalog);
  const test::PartitionProblemStorage storage{1U};
  const auto request = detail::partitioning::source_chain_algorithm();
  const detail::partitioning::PartitionSelectionInput conflicting{
      request, request, Profile::kStandard, {}};
  auto result =
      detail::partitioning::PartitioningPass(*catalog).run(conflicting, storage.problem());
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().message, "PARTITION_SELECTION_CONFLICT");
}

// REQ: COMPILE-058, COMPILE-068
TEST(PartitionEnvironmentSelectionTest, ManualCoverAlsoRejectsRetiredEnvironmentSelector) {
  const ScopedPartitionEnvironment restore_environment;
  Graph graph{"manual-partition-selection-conflict"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto target = graph.op<test::CompileTestOperator>("target");
  graph.add_flow(source->output, target->input);
  graph.partition("serial").add(source).add(target);

  for (const std::string_view selection : {"serial_baseline/v1", "source_chain/v1"}) {
    set_partition_environment(selection);
    const ExecutionPlan plan = compile(graph);
    EXPECT_FALSE(plan.ok()) << selection;
    EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_ENVIRONMENT_SELECTOR_REMOVED")) << selection;
    EXPECT_EQ(plan.partitioning_plan().selection_source, PlanSelectionSource::kCoreDefault);
    EXPECT_FALSE(plan.partitioning_plan().accepted_algorithm.has_value());
  }
}

// REQ: COMPILE-005, COMPILE-068
TEST(PartitionEnvironmentSelectionTest, RuntimeNeverRereadsRemovedEnvironmentSelector) {
  const ScopedPartitionEnvironment restore_environment;
  set_partition_environment(std::nullopt);
  const ExecutionPlan baseline = compile_linear_with_profile();
  ASSERT_TRUE(baseline.ok());

  set_partition_environment("source_chain/v1");
  const ExecutionPlan rejected = compile_linear_with_profile();
  ASSERT_FALSE(rejected.ok());
  EXPECT_TRUE(test::has_diagnostic(rejected, "PARTITION_ENVIRONMENT_SELECTOR_REMOVED"));
  const PlanDigest accepted_digest = baseline.digest();
  const std::string accepted_json{baseline.json()};

  set_partition_environment("invalid-after-compile");
  EXPECT_TRUE(baseline.ok());
  EXPECT_EQ(baseline.digest(), accepted_digest);
  EXPECT_EQ(baseline.json(), accepted_json);
  const auto& accepted_algorithm = baseline.partitioning_plan().accepted_algorithm;
  if (!accepted_algorithm.has_value()) {
    ADD_FAILURE() << "expected the frozen production source-chain partition algorithm";
    return;
  }
  EXPECT_EQ(accepted_algorithm->id, "holoscan.partition.source_chain");
}

// REQ: COMPILE-005
// REQ: COMPILE-048
// REQ: COMPILE-064
// REQ: COMPILE-068
TEST(PartitionResourceSeparationTest,
     ProductionSourceChainCreatesNoImplicitTensorOrDeviceResources) {
  const ScopedPartitionEnvironment restore_environment;
  set_partition_environment(std::nullopt);
  const ExecutionPlan source_chain = compile_fanout();
  ASSERT_TRUE(source_chain.ok());
  EXPECT_EQ(source_chain.execution_partitions().size(), 3U);
  EXPECT_TRUE(source_chain.tensor_allocations().empty());
  EXPECT_TRUE(source_chain.tensor_interface_pools().empty());
  EXPECT_TRUE(source_chain.tensor_device_resources().empty());

  const PlanDigest accepted_digest = source_chain.digest();
  set_partition_environment("retired-after-compile");
  EXPECT_EQ(source_chain.digest(), accepted_digest);
  EXPECT_TRUE(source_chain.tensor_device_resources().empty());
}

}  // namespace
}  // namespace holoscan
