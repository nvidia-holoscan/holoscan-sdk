/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/time/manual_clock.hpp>

#include "artifact_test_support.hpp"
#include "compile_test_graph.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/plan_artifact_serialization.hpp"
#include "compiler/plan_generation.hpp"

namespace holoscan {
namespace {

// Operator that requests clock resolution via the default clock key rather than a
// named clock; used in ResolvesDefaultReadinessClockOrRejectsAtomically.
class DefaultClockReadinessOperator : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.notification_source(notification, "notification").capacity(1);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  NotificationSource notification;
};

class CudaGraphCaptureEligibleOperator : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.cuda_graph_capture_eligible(); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Lightweight recursive-descent JSON validator; verifies plan.json() output is
// syntactically valid without a full JSON library dependency.
// Used in AcceptsInsignificantWhitespaceAndRejectsTrailingData and
// CanonicalizesTopologyReadinessAndClockDomains.
class JsonValidator {
 public:
  explicit JsonValidator(std::string_view input) : input_(input) {}

  [[nodiscard]] bool parse() noexcept {
    consume_whitespace();
    const bool parsed = parse_value();
    consume_whitespace();
    return parsed && position_ == input_.size();
  }

 private:
  void consume_whitespace() noexcept {
    while (position_ < input_.size() &&
           std::isspace(static_cast<unsigned char>(input_[position_])) != 0) {
      ++position_;
    }
  }

  [[nodiscard]] bool consume(char expected) noexcept {
    if (position_ >= input_.size() || input_[position_] != expected) {
      return false;
    }
    ++position_;
    return true;
  }

  [[nodiscard]] bool parse_value() noexcept {
    consume_whitespace();
    if (position_ >= input_.size()) {
      return false;
    }
    switch (input_[position_]) {
      case '{':
        return parse_object();
      case '[':
        return parse_array();
      case '"':
        return parse_string();
      case 't':
        return parse_literal("true");
      case 'f':
        return parse_literal("false");
      case 'n':
        return parse_literal("null");
      default:
        return parse_number();
    }
  }

  [[nodiscard]] bool parse_object() noexcept {
    ++position_;
    consume_whitespace();
    if (consume('}')) {
      return true;
    }
    do {
      consume_whitespace();
      if (!parse_string()) {
        return false;
      }
      consume_whitespace();
      if (!consume(':') || !parse_value()) {
        return false;
      }
      consume_whitespace();
      if (consume('}')) {
        return true;
      }
      if (!consume(',')) {
        return false;
      }
    } while (true);
  }

  [[nodiscard]] bool parse_array() noexcept {
    ++position_;
    consume_whitespace();
    if (consume(']')) {
      return true;
    }
    do {
      if (!parse_value()) {
        return false;
      }
      consume_whitespace();
      if (consume(']')) {
        return true;
      }
      if (!consume(',')) {
        return false;
      }
    } while (true);
  }

  [[nodiscard]] bool parse_string() noexcept {
    if (!consume('"')) {
      return false;
    }
    while (position_ < input_.size()) {
      const unsigned char byte = static_cast<unsigned char>(input_[position_++]);
      if (byte == '"') {
        return true;
      }
      if (byte < 0x20U) {
        return false;
      }
      if (byte != '\\') {
        continue;
      }
      if (position_ >= input_.size()) {
        return false;
      }
      const char escape = input_[position_++];
      if (escape == 'u') {
        for (std::size_t count = 0U; count < 4U; ++count) {
          if (position_ >= input_.size() ||
              std::isxdigit(static_cast<unsigned char>(input_[position_++])) == 0) {
            return false;
          }
        }
      } else if (std::string_view{"\"\\/bfnrt"}.find(escape) == std::string_view::npos) {
        return false;
      }
    }
    return false;
  }

  [[nodiscard]] bool parse_literal(std::string_view literal) noexcept {
    if (input_.substr(position_, literal.size()) != literal) {
      return false;
    }
    position_ += literal.size();
    return true;
  }

  [[nodiscard]] bool parse_number() noexcept {
    const std::size_t start = position_;
    if (position_ < input_.size() && input_[position_] == '-') {
      ++position_;
    }
    if (position_ >= input_.size()) {
      return false;
    }
    if (input_[position_] == '0') {
      ++position_;
    } else {
      if (input_[position_] < '1' || input_[position_] > '9') {
        return false;
      }
      while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') {
        ++position_;
      }
    }
    if (position_ < input_.size() && input_[position_] == '.') {
      ++position_;
      const std::size_t fraction = position_;
      while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') {
        ++position_;
      }
      if (position_ == fraction) {
        return false;
      }
    }
    if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
      ++position_;
      if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      const std::size_t exponent = position_;
      while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') {
        ++position_;
      }
      if (position_ == exponent) {
        return false;
      }
    }
    return position_ > start;
  }

  std::string_view input_;
  std::size_t position_{};
};

// REQ: COMPILE-009
// REQ: COMPILE-095
// REQ: COMPILE-014
// REQ: COMPILE-059
// REQ: COMPILE-101
TEST(CompileArtifactJsonValidation, AcceptsInsignificantWhitespaceAndRejectsTrailingData) {
  EXPECT_TRUE(JsonValidator{" \n { \"items\" : [ 1, true, null ] } \t"}.parse());
  EXPECT_FALSE(JsonValidator{"{} trailing"}.parse());
}

// Graph with one multi-source readiness operator and two explicitly-declared clocks.
// A clock conversion links notification to doorbell; notification is the default root.
//
//   notification_clock --[affine offset=3ns]--> doorbell_clock
//
// REQ: COMPILE-007
TEST(CompileLowering, CanonicalizesTopologyReadinessAndClockDomains) {
  Graph graph{"compile-lowering"};
  (void)graph.op<test::CompileReadinessOperator>("ready");
  const ClockKey notification = graph.add_clock<ManualClock>("notification_clock", std::int64_t{0});
  const ClockKey doorbell = graph.add_clock<ManualClock>("doorbell_clock", std::int64_t{0});
  graph.set_default_clock(notification);
  // Key setup: affine conversion makes doorbell a derived clock of the default root.
  graph.add_clock_conversion(notification, doorbell, AffineClockConversion{.offset_ns = 3});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok());
  const ClockCatalogView catalog = plan.clock_catalog();
  ASSERT_NE(catalog.plan_generation, 0U);
  ASSERT_EQ(catalog.clocks.size(), 2U);
  ASSERT_TRUE(catalog.root_default_clock.valid());
  for (std::size_t index = 0U; index < catalog.clocks.size(); ++index) {
    EXPECT_EQ(catalog.clocks[index].id.value, index + 1U);
    EXPECT_FALSE(catalog.clocks[index].key.is_default());
  }
  const auto default_clock =
      std::find_if(catalog.clocks.begin(), catalog.clocks.end(), [](const ClockPlanRecord& clock) {
        return clock.is_default;
      });
  ASSERT_NE(default_clock, catalog.clocks.end());
  EXPECT_EQ(default_clock->key, notification);
  EXPECT_EQ(catalog.root_default_clock, default_clock->id);

  ASSERT_EQ(plan.clock_conversions().size(), 1U);
  const ClockConversionRecord& conversion = plan.clock_conversions().front();
  EXPECT_EQ(conversion.id, 1U);
  EXPECT_EQ(conversion.plan_generation, catalog.plan_generation);
  EXPECT_EQ(conversion.graph, default_clock->graph);
  EXPECT_EQ(conversion.from, default_clock->id);
  EXPECT_NE(conversion.to, conversion.from);
  EXPECT_EQ(conversion.scale, ClockRateRatio{});
  EXPECT_EQ(conversion.offset_ns, 3);

  ASSERT_EQ(plan.readiness().size(), 2U);
  EXPECT_LT(plan.readiness()[0].name, plan.readiness()[1].name);
  for (const auto& source : plan.readiness()) {
    EXPECT_TRUE(static_cast<bool>(source.source_id));
    EXPECT_EQ(source.plan_generation, catalog.plan_generation);
    EXPECT_EQ(source.effective_profile, Profile::kStandard);
    EXPECT_TRUE(static_cast<bool>(source.owner));
    EXPECT_FALSE(source.referenced);
    EXPECT_FALSE(source.materialized);
    EXPECT_EQ(source.initial_source_generation, 1U);
    EXPECT_EQ(source.initial_source_sequence, 0U);
    EXPECT_TRUE(static_cast<bool>(source.disposition_reason));
    EXPECT_EQ(source.requested_capacity, source.accepted_capacity);
    EXPECT_EQ(source.requested_sender_reference_capacity,
              source.accepted_sender_reference_capacity);
    EXPECT_EQ(source.owner_path, "%root/operator/ready");
    EXPECT_GT(source.accepted_capacity, 0U);
    EXPECT_FALSE(source.requested_clock.is_default());
    EXPECT_TRUE(source.resolved_clock.valid());
    // Key assertion: each source's resolved_clock matches the catalog entry for its key.
    EXPECT_TRUE(std::any_of(catalog.clocks.begin(), catalog.clocks.end(), [&](const auto& clock) {
      return clock.id == source.resolved_clock && clock.key == source.requested_clock;
    }));
  }
  EXPECT_NE(plan.readiness()[0].source_id, plan.readiness()[1].source_id);
  EXPECT_TRUE(test::has_diagnostic(plan, "READINESS_SOURCE_UNUSED"));
  EXPECT_NE(plan.json().find("\"clock_catalog\""), std::string_view::npos);
  EXPECT_NE(plan.json().find("\"clock_conversions\""), std::string_view::npos);
  EXPECT_NE(plan.json().find("\"requested_clock\""), std::string_view::npos);
  EXPECT_NE(plan.json().find("\"resolved_clock\""), std::string_view::npos);
  EXPECT_NE(plan.json().find("\"plan_generation\""), std::string_view::npos);
  EXPECT_NE(plan.json().find("\"initial_source_generation\""), std::string_view::npos);
  EXPECT_NE(plan.json().find("\"initial_source_sequence\""), std::string_view::npos);
  EXPECT_NE(plan.json().find("\"disposition_reason\""), std::string_view::npos);
  EXPECT_TRUE(JsonValidator{plan.json()}.parse());
}

// REQ: COMPILE-007
TEST(CompileLowering, RejectsMissingReadinessClockDeclarations) {
  Graph graph{"missing-clock"};
  (void)graph.op<test::CompileReadinessOperator>("ready");
  const ExecutionPlan plan = compile(graph);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "READINESS_CLOCK_UNDECLARED"));
  EXPECT_TRUE(plan.clock_catalog().clocks.empty());
  EXPECT_TRUE(plan.readiness().empty());
}

// Accepted graph calls set_default_clock; rejected graph declares a clock but never
// designates it as default — the readiness source's default-key request is unresolvable.
//
//   accepted: op[ready] <-- default_clock  (set_default_clock called)
//   rejected: op[ready] <-- not_default    (no set_default_clock; default key absent)
//
// REQ: COMPILE-007
TEST(CompileLowering, ResolvesDefaultReadinessClockOrRejectsAtomically) {
  Graph accepted_graph{"default-clock"};
  (void)accepted_graph.op<DefaultClockReadinessOperator>("ready");
  const ClockKey declared = accepted_graph.add_clock<ManualClock>("default_clock", std::int64_t{0});
  accepted_graph.set_default_clock(declared);

  const ExecutionPlan accepted = compile(accepted_graph);
  ASSERT_TRUE(accepted.ok());
  ASSERT_EQ(accepted.readiness().size(), 1U);
  ASSERT_EQ(accepted.clock_catalog().clocks.size(), 1U);
  EXPECT_TRUE(accepted.readiness().front().requested_clock.is_default());
  // Key assertion: default-key source resolves to the graph's root_default_clock.
  EXPECT_EQ(accepted.readiness().front().resolved_clock,
            accepted.clock_catalog().root_default_clock);

  // Key setup: rejected graph adds a clock but never calls set_default_clock.
  Graph rejected_graph{"missing-default-clock"};
  (void)rejected_graph.op<DefaultClockReadinessOperator>("ready");
  (void)rejected_graph.add_clock<ManualClock>("not_default", std::int64_t{0});

  const ExecutionPlan rejected = compile(rejected_graph);
  EXPECT_FALSE(rejected.ok());
  EXPECT_TRUE(test::has_diagnostic(rejected, "READINESS_DEFAULT_CLOCK_UNDECLARED"));
  EXPECT_NE(rejected.clock_catalog().plan_generation, 0U);
  EXPECT_TRUE(rejected.clock_catalog().clocks.empty());
  EXPECT_TRUE(rejected.clock_conversions().empty());
  EXPECT_TRUE(rejected.readiness().empty());
}

// Two independently-built graphs with the same name and one identically-named clock;
// their digest and JSON must be byte-equal, confirming plan identity is structure-only.
//
// REQ: COMPILE-002
// REQ: COMPILE-007
TEST(CompileLowering, EquivalentClockCatalogsReproducePlanIdentityAndBytes) {
  Graph first_graph{"clock-scope"};
  (void)first_graph.op<test::CompileTestOperator>("operator");
  (void)first_graph.add_clock<ManualClock>("clock", std::int64_t{0});
  Graph second_graph{"clock-scope"};
  (void)second_graph.op<test::CompileTestOperator>("operator");
  (void)second_graph.add_clock<ManualClock>("clock", std::int64_t{0});

  const ExecutionPlan first = compile(first_graph);
  const ExecutionPlan second = compile(second_graph);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  ASSERT_EQ(first.clock_catalog().clocks.size(), 1U);
  ASSERT_EQ(second.clock_catalog().clocks.size(), 1U);
  EXPECT_EQ(first.clock_catalog().clocks.front().id, second.clock_catalog().clocks.front().id);
  EXPECT_EQ(first.clock_catalog().plan_generation, second.clock_catalog().plan_generation);
  EXPECT_EQ(first.digest(), second.digest());
  EXPECT_EQ(first.json(), second.json());
}

// REQ: COMPILE-002
// REQ: COMPILE-007
TEST(CompileLowering, CanonicalSemanticDigestQualifiesPlanGenerationIdentity) {
  detail::ExecutionPlanState first_state;
  detail::ExecutionPlanState second_state;
  first_state.digest.value.front() = std::byte{0x11};
  second_state.digest.value.front() = std::byte{0x22};
  const PlanGenerationId first = detail::derive_plan_generation_id(first_state);
  const PlanGenerationId repeated = detail::derive_plan_generation_id(first_state);
  const PlanGenerationId second = detail::derive_plan_generation_id(second_state);

  EXPECT_NE(first, 0U);
  EXPECT_EQ(first, repeated);
  EXPECT_NE(first, second);
}

// REQ: COMPILE-002
// REQ: COMPILE-007
TEST(CompileLowering, EqualNumericClockIdsRequireMatchingCatalogSemantics) {
  Graph first_graph{"clock-scope"};
  (void)first_graph.op<test::CompileTestOperator>("operator");
  (void)first_graph.add_clock<ManualClock>("first_clock", std::int64_t{0});
  Graph second_graph{"clock-scope"};
  (void)second_graph.op<test::CompileTestOperator>("operator");
  (void)second_graph.add_clock<ManualClock>("second_clock", std::int64_t{0});

  const ExecutionPlan first = compile(first_graph);
  const ExecutionPlan second = compile(second_graph);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  ASSERT_EQ(first.clock_catalog().clocks.size(), 1U);
  ASSERT_EQ(second.clock_catalog().clocks.size(), 1U);
  EXPECT_EQ(first.clock_catalog().clocks.front().id, second.clock_catalog().clocks.front().id);
  EXPECT_NE(first.clock_catalog().plan_generation, second.clock_catalog().plan_generation);

  Graph first_readiness_graph{"readiness-clock-scope"};
  (void)first_readiness_graph.op<test::CompileReadinessOperator>("ready");
  (void)first_readiness_graph.add_clock<ManualClock>("notification_clock", std::int64_t{0});
  (void)first_readiness_graph.add_clock<ManualClock>("doorbell_clock", std::int64_t{0});
  Graph second_readiness_graph{"readiness-clock-scope"};
  (void)second_readiness_graph.op<test::CompileReadinessOperator>("ready");
  (void)second_readiness_graph.add_clock<ManualClock>("notification_clock", std::int64_t{0});
  (void)second_readiness_graph.add_clock<ManualClock>("doorbell_clock", std::int64_t{0});
  const ExecutionPlan first_readiness = compile(first_readiness_graph);
  CompileOptions changed_options;
  changed_options.profile = Profile::kResearch;
  const ExecutionPlan second_readiness = compile(second_readiness_graph, changed_options);
  ASSERT_TRUE(first_readiness.ok());
  ASSERT_TRUE(second_readiness.ok());
  ASSERT_FALSE(first_readiness.readiness().empty());
  ASSERT_FALSE(second_readiness.readiness().empty());
  EXPECT_EQ(first_readiness.readiness().front().resolved_clock,
            second_readiness.readiness().front().resolved_clock);
  EXPECT_NE(first_readiness.readiness().front().plan_generation,
            second_readiness.readiness().front().plan_generation);
}

// REQ: COMPILE-002
// REQ: COMPILE-010
TEST(CompileLowering, RejectsUnsupportedConverterDisableRatherThanIgnoringIt) {
  Graph default_graph{"compile-options-identity"};
  (void)default_graph.op<test::CompileTestOperator>("operator");
  Graph changed_graph{"compile-options-identity"};
  (void)changed_graph.op<test::CompileTestOperator>("operator");
  CompileOptions changed_options;
  changed_options.auto_insert_converters = false;

  const ExecutionPlan defaults = compile(default_graph);
  const ExecutionPlan changed = compile(changed_graph, changed_options);
  ASSERT_TRUE(defaults.ok());
  ASSERT_FALSE(changed.ok());
  EXPECT_TRUE(test::has_diagnostic(changed, "CONVERTER_AUTO_INSERTION_DISABLE_UNSUPPORTED"));
  EXPECT_NE(defaults.clock_catalog().plan_generation, changed.clock_catalog().plan_generation);
  EXPECT_NE(defaults.digest(), changed.digest());
  EXPECT_NE(defaults.json(), changed.json());
}

// REQ: GRAPH-038
TEST(CompileLowering, PublishesValidatedCUDAGraphCapturePlan) {
  Graph graph{"cuda-graph-region"};
  test::add_compile_test_default_clock(graph);
  auto op = graph.op<CudaGraphCaptureEligibleOperator>("capture");
  graph.region("capture").add(op).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "CUDA_GRAPH_EXECUTION_REGION_INVALID"));
  EXPECT_FALSE(test::has_diagnostic(plan, "CUDA_GRAPH_REGION_INVALID"));
  EXPECT_FALSE(test::has_diagnostic(plan, "SPIN_UNPLACED"));
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->cuda_graph_capture_plans.size(), 1U);
  const ExecutionGraphView plan_graph = plan.graph();
  ASSERT_EQ(plan_graph.vertices().size(), 1U);
  ASSERT_EQ(plan_graph.regions().size(), 1U);
  EXPECT_TRUE(plan_graph.vertices().front().cuda_graph_capture_eligible);
  ASSERT_EQ(plan_graph.regions().front().constraints.size(), 1U);
  EXPECT_EQ(plan_graph.regions().front().constraints.front().kind(),
            RegionConstraintKind::kCudaGraph);
  EXPECT_TRUE(JsonValidator{plan.json()}.parse());
}

// REQ: GRAPH-038
TEST(CompileLowering, CudaGraphDeclarationArtifactMatchesNormalizedBaseline) {
  Graph graph{"cuda-graph-region"};
  test::add_compile_test_default_clock(graph);
  auto op = graph.op<CudaGraphCaptureEligibleOperator>("capture");
  graph.region("capture").add(op).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  ASSERT_FALSE(plan.ok());
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(accepted);
  auto normalized = std::make_shared<detail::ExecutionPlanState>(*accepted);
  detail::rebind_plan_generation(*normalized, PlanGenerationId{1U});
  normalized->profile_manifest.target_triple = "normalized-target-triple";
  const auto finalized = detail::finalize_plan_artifact(*normalized);
  ASSERT_TRUE(finalized) << runtime_error_message(finalized.error());
  const ExecutionPlan normalized_plan = detail::ExecutionPlanAccess::make(std::move(normalized));

  const std::string golden_path =
      std::string(HOLOSCAN_COMPILER_TEST_DATA_DIR) + "/cuda_graph_declaration.v9.normalized.json";
  std::ifstream golden_stream{golden_path, std::ios::binary};
  ASSERT_TRUE(golden_stream.is_open()) << golden_path;
  const std::string golden{std::istreambuf_iterator<char>{golden_stream},
                           std::istreambuf_iterator<char>{}};
  ASSERT_TRUE(golden_stream.good() || golden_stream.eof()) << golden_path;

  const std::string actual =
      detail::artifact_test::normalized_artifact_json(normalized_plan.json());
  EXPECT_EQ(actual, golden) << "review the complete artifact diff before updating " << golden_path;
}

// REQ: OPERATOR-051
// REQ: GRAPH-038
TEST(CompileLowering, CaptureEligibilityOutsideCudaGraphRegionKeepsOrdinaryExecution) {
  Graph graph{"cuda-graph-eligible-ordinary-execution"};
  test::add_compile_test_default_clock(graph);
  (void)graph.op<CudaGraphCaptureEligibleOperator>("ordinary");

  const ExecutionPlan plan = compile(graph);

  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_TRUE(plan.diagnostics().empty());
  const ExecutionGraphView plan_graph = plan.graph();
  ASSERT_EQ(plan_graph.vertices().size(), 1U);
  EXPECT_TRUE(plan_graph.vertices().front().cuda_graph_capture_eligible);
  EXPECT_TRUE(plan_graph.regions().empty());
}

// REQ: GRAPH-038
TEST(CompileLowering, IdentifiesEveryCudaGraphRegionWithoutPhysicalCUDAEvidence) {
  Graph graph{"multiple-cuda-graph-regions"};
  test::add_compile_test_default_clock(graph);
  auto first = graph.op<CudaGraphCaptureEligibleOperator>("first");
  auto second = graph.op<CudaGraphCaptureEligibleOperator>("second");
  graph.region("first-region").add(first).constrain(RegionConstraint::cuda_graph());
  graph.region("second-region").add(second).constrain(RegionConstraint::cuda_graph());

  const ExecutionPlan plan = compile(graph);
  ASSERT_FALSE(plan.ok());
  const auto* state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto& capture_plans = state->cuda_graph_capture_plans;
  ASSERT_EQ(capture_plans.size(), 2U);
  EXPECT_LT(capture_plans[0].region.value, capture_plans[1].region.value);
  std::size_t cuda_graph_diagnostics = 0U;
  bool found_first = false;
  bool found_second = false;
  constexpr std::string_view kRequiredRule = "single_cuda_device";
  for (const Diagnostic& diagnostic : plan.diagnostics()) {
    if (diagnostic.token != "CUDA_GRAPH_EXECUTION_REGION_INVALID") {
      continue;
    }
    ++cuda_graph_diagnostics;
    if (diagnostic.provided.starts_with("region_id=1,region_name=first-region,")) {
      found_first = diagnostic.json_pointer == "/graphs/0/regions/0/constraints/cuda_graph" &&
                    diagnostic.required == kRequiredRule;
    } else if (diagnostic.provided.starts_with("region_id=2,region_name=second-region,")) {
      found_second = diagnostic.json_pointer == "/graphs/0/regions/1/constraints/cuda_graph" &&
                     diagnostic.required == kRequiredRule;
    }
  }
  EXPECT_EQ(cuda_graph_diagnostics, 2U);
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);

  // Resolve both pointer indices against the canonical plan Graph instead of only comparing text.
  const ExecutionGraphView plan_graph = plan.graph();
  ASSERT_EQ(plan_graph.graphs().size(), 1U);
  ASSERT_EQ(plan_graph.regions().size(), 2U);
  EXPECT_EQ(plan_graph.graphs()[0].id.value, 1U);
  EXPECT_EQ(plan_graph.regions()[0].id.value, 1U);
  EXPECT_EQ(plan_graph.regions()[1].id.value, 2U);
}

// REQ: COMPILE-002
// REQ: COMPILE-007
TEST(CompileLowering, AssignsDensePlanLocalIdsAndPreservesAuthoringProvenance) {
  Graph graph{"dense-plan-ids"};
  (void)graph.op<test::CompileReadinessOperator>("ready");
  (void)graph.add_clock<ManualClock>("notification_clock", std::int64_t{0});
  (void)graph.add_clock<ManualClock>("doorbell_clock", std::int64_t{0});
  const GraphAuthoringIdentity authored = graph.view().identity();

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok());
  EXPECT_EQ(plan.graph_identity(), authored);
  EXPECT_EQ(plan.graph_hash(), authored.hash);
  ASSERT_EQ(plan.readiness().size(), 2U);
  EXPECT_TRUE(static_cast<bool>(plan.readiness()[0].source_id));
  EXPECT_TRUE(static_cast<bool>(plan.readiness()[1].source_id));
  EXPECT_NE(plan.readiness()[0].source_id, plan.readiness()[1].source_id);
  for (const auto& partition : plan.execution_partitions()) {
    EXPECT_TRUE(static_cast<bool>(partition.id));
    EXPECT_FALSE(partition.members.empty());
  }
  for (const auto& decision : plan.partition_decisions()) {
    EXPECT_TRUE(static_cast<bool>(decision.vertex));
    EXPECT_TRUE(static_cast<bool>(decision.partition));
  }
}

}  // namespace
}  // namespace holoscan
