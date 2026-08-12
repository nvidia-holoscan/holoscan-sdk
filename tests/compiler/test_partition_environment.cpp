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

// Declaration-only readiness sources remain inert under every planner. In particular, the
// experimental source-chain planner must resolve the embedded interface predecessor and preserve
// one unnamed serial partition rather than isolating the inner Operator.
TEST(PartitionEnvironmentSelectionTest,
     SourceChainIgnoresUnusedReadinessAcrossEmbeddedInputBoundary) {
  const ScopedPartitionEnvironment guard;
  set_partition_environment("source_chain/v1");

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

// REQ: COMPILE-005
TEST(PartitionEnvironmentSelectionTest, AbsentUsesCoreDefaultAndExactValuesOverride) {
  const ScopedPartitionEnvironment restore_environment;
  // Key setup: absent env selects the core-default algorithm; explicit tag strings
  // activate kEnvironmentOverride and select the named algorithm.
  set_partition_environment(std::nullopt);
  const ExecutionPlan absent = compile_linear_with_profile();
  ASSERT_TRUE(absent.ok());
  const auto& absent_algorithm = absent.partitioning_plan().accepted_algorithm;
  if (!absent_algorithm.has_value()) {
    ADD_FAILURE() << "expected the default partition algorithm";
    return;
  }
  EXPECT_EQ(absent_algorithm->id, "holoscan.partition.serial_baseline");
  EXPECT_EQ(absent.partitioning_plan().selection_source, PlanSelectionSource::kCoreDefault);

  set_partition_environment("serial_baseline/v1");
  const ExecutionPlan explicit_baseline = compile_linear_with_profile();
  ASSERT_TRUE(explicit_baseline.ok());
  EXPECT_EQ(explicit_baseline.partitioning_plan().selection_source,
            PlanSelectionSource::kEnvironmentOverride);
  const auto& explicit_algorithm = explicit_baseline.partitioning_plan().accepted_algorithm;
  if (!explicit_algorithm.has_value()) {
    ADD_FAILURE() << "expected the explicit baseline partition algorithm";
    return;
  }
  EXPECT_EQ(explicit_algorithm->id, "holoscan.partition.serial_baseline");

  set_partition_environment("source_chain/v1");
  const ExecutionPlan source_chain = compile_linear_with_profile();
  ASSERT_TRUE(source_chain.ok());
  const auto& source_chain_algorithm = source_chain.partitioning_plan().accepted_algorithm;
  if (!source_chain_algorithm.has_value()) {
    ADD_FAILURE() << "expected the source-chain partition algorithm";
    return;
  }
  EXPECT_EQ(source_chain_algorithm->id, "holoscan.partition.source_chain");
  EXPECT_EQ(source_chain.partitioning_plan().selection_source,
            PlanSelectionSource::kEnvironmentOverride);
  // Key assertion: source_chain is experimental and must emit exactly one warning diagnostic.
  const auto warnings = std::count_if(
      source_chain.diagnostics().begin(), source_chain.diagnostics().end(), [](const auto& item) {
        return item.token == "PARTITION_PLANNER_EXPERIMENTAL_SELECTED" &&
               item.severity == DiagnosticSeverity::kWarning;
      });
  EXPECT_EQ(warnings, 1);
}

// REQ: COMPILE-005
TEST(PartitionEnvironmentSelectionTest, RejectsInvalidValuesConflictsAndForbiddenProfiles) {
  const ScopedPartitionEnvironment restore_environment;
  // Key setup: 5 syntactically invalid strings, 3 forbidden profiles, and a direct
  // PartitioningPass conflict scenario are all exercised in a single test body.
  constexpr std::array invalid_values{
      "", "source_chain", "source_chain/v2", " source_chain/v1", "SOURCE_CHAIN/v1"};
  for (std::string_view invalid : invalid_values) {
    set_partition_environment(invalid);
    const ExecutionPlan plan = compile_linear_with_profile();
    EXPECT_FALSE(plan.ok()) << invalid;
    EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_SELECTION_ENV_VALUE_INVALID")) << invalid;
    EXPECT_EQ(plan.partitioning_plan().selection_source, PlanSelectionSource::kEnvironmentOverride);
    EXPECT_FALSE(plan.partitioning_plan().requested_algorithm.has_value());
    EXPECT_FALSE(plan.partitioning_plan().accepted_algorithm.has_value());
    EXPECT_TRUE(static_cast<bool>(plan.partitioning_plan().reason));
    EXPECT_NE(plan.partitioning_plan().config_digest, PlanDigest{});
    const auto diagnostic =
        std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
          return item.token == "PARTITION_SELECTION_ENV_VALUE_INVALID";
        });
    ASSERT_NE(diagnostic, plan.diagnostics().end());
    EXPECT_EQ(plan.partitioning_plan().reason, diagnostic->id);
  }

  for (const Profile forbidden : {Profile::kRobotics, Profile::kRealTime, Profile::kRegulated}) {
    set_partition_environment("serial_baseline/v1");
    const ExecutionPlan plan = compile_linear_with_profile(forbidden);
    EXPECT_FALSE(plan.ok());
    EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_SELECTION_PROFILE_FORBIDDEN"));
    const auto& requested_algorithm = plan.partitioning_plan().requested_algorithm;
    if (!requested_algorithm.has_value()) {
      ADD_FAILURE() << "expected forbidden profile evidence to retain the requested algorithm";
      return;
    }
    EXPECT_EQ(requested_algorithm->id, "holoscan.partition.serial_baseline");
    EXPECT_EQ(plan.partitioning_plan().selection_source, PlanSelectionSource::kEnvironmentOverride);
    EXPECT_FALSE(plan.partitioning_plan().accepted_algorithm.has_value());
    EXPECT_TRUE(static_cast<bool>(plan.partitioning_plan().reason));
  }

  for (const Profile forbidden : {Profile::kRobotics, Profile::kRealTime, Profile::kRegulated}) {
    Graph graph{"partition-environment-empty-region"};
    test::add_linear_graph(graph);
    graph.region("empty-constrained-region").profile(forbidden);
    CompileOptions options;
    options.profile = Profile::kStandard;

    set_partition_environment("serial_baseline/v1");
    const ExecutionPlan plan = compile(graph, options);
    EXPECT_FALSE(plan.ok());
    EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_SELECTION_PROFILE_FORBIDDEN"));
    EXPECT_FALSE(plan.partitioning_plan().accepted_algorithm.has_value());
  }

  auto catalog = detail::partitioning::PartitionPlannerCatalog::create_ea1_builtins();
  ASSERT_TRUE(catalog);
  const test::PartitionProblemStorage storage{1U};
  const auto request = detail::partitioning::serial_baseline_algorithm();
  const std::string_view captured{"serial_baseline/v1"};
  const detail::partitioning::PartitionSelectionInput conflicting{
      captured, request, std::nullopt, Profile::kStandard, {}};
  auto result =
      detail::partitioning::PartitioningPass(*catalog).run(conflicting, storage.problem());
  ASSERT_FALSE(result);
  // Key assertion: direct PartitioningPass invocation with a conflict reports the error
  // message directly rather than through the diagnostics list.
  EXPECT_EQ(result.error().message, "PARTITION_SELECTION_CONFLICT");
}

// REQ: COMPILE-058
TEST(PartitionEnvironmentSelectionTest, ManualCoverRejectsEveryExplicitPlannerSelection) {
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
    EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_SELECTION_CONFLICT")) << selection;
    EXPECT_EQ(plan.partitioning_plan().selection_source, PlanSelectionSource::kEnvironmentOverride);
    EXPECT_FALSE(plan.partitioning_plan().accepted_algorithm.has_value());
  }
}

// REQ: COMPILE-005
TEST(PartitionEnvironmentSelectionTest, CaptureIsPartOfIdentityAndRuntimeNeverRereadsEnvironment) {
  const ScopedPartitionEnvironment restore_environment;
  set_partition_environment(std::nullopt);
  const ExecutionPlan baseline = compile_linear_with_profile();
  ASSERT_TRUE(baseline.ok());

  // Key setup: compile with source_chain, then mutate the env to an invalid value
  // after compile to verify that the frozen plan is unaffected.
  set_partition_environment("source_chain/v1");
  const ExecutionPlan source_chain = compile_linear_with_profile();
  ASSERT_TRUE(source_chain.ok());
  const PlanDigest accepted_digest = source_chain.digest();
  const std::string accepted_json{source_chain.json()};
  EXPECT_NE(baseline.digest(), accepted_digest);

  set_partition_environment("invalid-after-compile");
  // Key assertion: post-compile env mutation cannot alter a frozen ExecutionPlan's
  // digest, JSON, or accepted algorithm.
  EXPECT_TRUE(source_chain.ok());
  EXPECT_EQ(source_chain.digest(), accepted_digest);
  EXPECT_EQ(source_chain.json(), accepted_json);
  const auto& accepted_algorithm = source_chain.partitioning_plan().accepted_algorithm;
  if (!accepted_algorithm.has_value()) {
    ADD_FAILURE() << "expected the frozen source-chain partition algorithm";
    return;
  }
  EXPECT_EQ(accepted_algorithm->id, "holoscan.partition.source_chain");
}

// REQ: COMPILE-005
// REQ: COMPILE-048
TEST(PartitionResourceSeparationTest,
     EnvironmentSelectedCoverCreatesNoImplicitTensorOrDeviceResources) {
  const ScopedPartitionEnvironment restore_environment;
  // Key setup: same fanout graph compiled under serial_baseline and source_chain yields
  // a different number of execution partitions, proving the cover selection took effect.
  set_partition_environment("serial_baseline/v1");
  const ExecutionPlan baseline = compile_fanout();
  ASSERT_TRUE(baseline.ok());

  set_partition_environment("source_chain/v1");
  const ExecutionPlan source_chain = compile_fanout();
  ASSERT_TRUE(source_chain.ok());
  EXPECT_NE(baseline.execution_partitions().size(), source_chain.execution_partitions().size());
  // Key assertion: neither cover allocates implicit Tensor or device resources
  // for a pure-scalar fanout graph.
  EXPECT_TRUE(baseline.tensor_allocations().empty());
  EXPECT_TRUE(source_chain.tensor_allocations().empty());
  EXPECT_TRUE(baseline.tensor_interface_pools().empty());
  EXPECT_TRUE(source_chain.tensor_interface_pools().empty());
  EXPECT_TRUE(baseline.tensor_device_resources().empty());
  EXPECT_TRUE(source_chain.tensor_device_resources().empty());

  const PlanDigest accepted_digest = source_chain.digest();
  set_partition_environment("serial_baseline/v1");
  EXPECT_EQ(source_chain.digest(), accepted_digest);
  EXPECT_TRUE(source_chain.tensor_device_resources().empty());
}

}  // namespace
}  // namespace holoscan
