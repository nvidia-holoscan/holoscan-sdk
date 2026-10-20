/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../common/allocation_counter.hpp"
#include "compiler/compiler_diagnostic.hpp"
#include "compiler/endpoint_catalog_lowering.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/lifecycle_plan_lowering.hpp"
#include "compiler/physical_plan_lowering.hpp"
#include "compiler/plan_lowering_diagnostic.hpp"
#include "compiler/temporal_plan_lowering.hpp"

namespace holoscan::detail {
namespace {

static_assert(std::is_same_v<EndpointCatalogDiagnostic, PlanLoweringDiagnostic>);
static_assert(std::is_same_v<PhysicalPlanDiagnostic, PlanLoweringDiagnostic>);
static_assert(std::is_same_v<TemporalPlanDiagnostic, PlanLoweringDiagnostic>);
static_assert(std::is_same_v<LifecyclePlanDiagnostic, PlanLoweringDiagnostic>);

[[nodiscard]] PlanLoweringDiagnostic diagnostic(std::string token, GraphGeneration generation,
                                                GraphId graph, VertexId vertex, PortId port) {
  return PlanLoweringDiagnostic{
      .token = std::move(token),
      .source =
          DiagnosticSource{
              .graph_generation = generation,
              .graph = graph,
              .vertex = vertex,
              .port = port,
          },
      .json_pointer = "/records/0",
      .required = "required",
      .provided = "provided",
  };
}

TEST(PlanLoweringDiagnostic, CanonicalizesAndDeduplicatesTheCompleteValue) {
  const auto first = diagnostic("A_TOKEN", 2U, GraphId{1U}, VertexId{2U}, PortId{3U});
  const auto second = diagnostic("B_TOKEN", 1U, GraphId{1U}, VertexId{2U}, PortId{3U});
  auto changed_source = first;
  changed_source.source.port = PortId{4U};
  std::vector diagnostics{second, changed_source, first, first};

  // Key setup: provide out-of-order records containing one exact duplicate and one record that
  // differs only in a nested source identity.
  canonicalize_plan_lowering_diagnostics(diagnostics);

  // Key assertion: canonical order covers all evidence and removes only the complete duplicate.
  ASSERT_EQ(diagnostics.size(), 3U);
  EXPECT_TRUE(plan_lowering_diagnostic_exactly_equal(diagnostics[0], first));
  EXPECT_TRUE(plan_lowering_diagnostic_exactly_equal(diagnostics[1], changed_source));
  EXPECT_TRUE(plan_lowering_diagnostic_exactly_equal(diagnostics[2], second));
}

TEST(PlanLoweringDiagnostic, AppendsOnlyCompleteUniqueValues) {
  const auto existing = diagnostic("TOKEN", 1U, GraphId{2U}, VertexId{3U}, PortId{4U});
  auto changed_evidence = existing;
  changed_evidence.provided = "other";
  std::vector diagnostics{existing};

  // Key setup: append one exact duplicate followed by a value differing in one evidence field.
  const bool duplicate_appended = append_unique_plan_lowering_diagnostic(diagnostics, existing);
  const bool changed_appended =
      append_unique_plan_lowering_diagnostic(diagnostics, changed_evidence);

  // Key assertion: the helper reports and suppresses only complete duplicates.
  EXPECT_FALSE(duplicate_appended);
  EXPECT_TRUE(changed_appended);
  ASSERT_EQ(diagnostics.size(), 2U);
  EXPECT_EQ(diagnostics.back().provided, "other");
}

// REQ: CHAN-140
TEST(PlanLoweringDiagnostic, PreservesCatalogNormalizationEvidenceByFormat) {
  const DiagnosticSource source{
      .graph_generation = 3U,
      .graph = GraphId{4U},
      .vertex = VertexId{5U},
      .port = PortId{6U},
  };

  const auto capacity = endpoint_catalog_normalization_diagnostic(
      "ENDPOINT_CATALOG_INLINE_CAPACITY_EXCEEDED|path=/endpoints/0/payload|endpoint=frames|"
      "application_capacity=32705|framing_overhead=64|required_frame=32769|maximum_frame=32768",
      source);
  EXPECT_EQ(capacity.token, "ENDPOINT_CATALOG_INLINE_CAPACITY_EXCEEDED");
  EXPECT_EQ(capacity.source.graph_generation, source.graph_generation);
  EXPECT_EQ(capacity.source.graph, source.graph);
  EXPECT_EQ(capacity.source.vertex, source.vertex);
  EXPECT_EQ(capacity.source.port, source.port);
  EXPECT_EQ(capacity.json_pointer, "/endpoints/0/payload");
  EXPECT_EQ(capacity.required, "required_frame<=maximum_frame;maximum_frame=32768");
  EXPECT_EQ(capacity.provided,
            "endpoint=frames;application_capacity=32705;framing_overhead=64;required_frame=32769");

  const auto resource = endpoint_catalog_normalization_diagnostic(
      "ENDPOINT_CATALOG_RESOURCE_LIMIT_EXCEEDED|path=/endpoints|resource=normalized_endpoints|"
      "limit=4096|actual=4097",
      source);
  EXPECT_EQ(resource.token, "ENDPOINT_CATALOG_RESOURCE_LIMIT_EXCEEDED");
  EXPECT_EQ(resource.json_pointer, "/endpoints");
  EXPECT_EQ(resource.required, "resource=normalized_endpoints;limit=4096");
  EXPECT_EQ(resource.provided, "actual=4097");

  const auto clock_health = endpoint_catalog_normalization_diagnostic(
      "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID|path=/endpoints/0/clock_health|endpoint=sensor|"
      "required=unavailable+not_required|status_reports+fresh_healthy_report|"
      "provided=capability=status_reports;requirement=not_required",
      source);
  EXPECT_EQ(clock_health.token, "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID");
  EXPECT_EQ(clock_health.json_pointer, "/endpoints/0/clock_health");
  EXPECT_EQ(clock_health.required, "unavailable+not_required|status_reports+fresh_healthy_report");
  EXPECT_EQ(clock_health.provided,
            "endpoint=sensor;capability=status_reports;requirement=not_required");
}

// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST(PlanLoweringDiagnostic, RetainsFirstCauseAndContextBeforeCanonicalPublication) {
  const auto structural = diagnostic("TOKEN", 1U, GraphId{2U}, VertexId{3U}, PortId{4U});
  auto context = structural;
  context.detail = DiagnosticDetail{.operation = DiagnosticOperation::kCompile,
                                    .error = Error{ErrorCode::kSuccess},
                                    .operator_path = "retained-owner",
                                    .port_name = "retained-output"};
  auto first = structural;
  first.detail = DiagnosticDetail{.operation = DiagnosticOperation::kSetup,
                                  .error = Error{ErrorCode::kInvalidArgument, "first cause"},
                                  .operator_path = "later-owner"};
  auto later = first;
  if (!later.detail.has_value()) {
    FAIL() << "Expected the copied first-cause fixture detail";
  }
  later.detail->error = Error{ErrorCode::kNotSupported, "later cause"};
  auto distinct = structural;
  distinct.provided = "distinct structural evidence";
  for (const bool canonicalize : {false, true}) {
    for (const bool with_context : {false, true}) {
      SCOPED_TRACE(canonicalize);
      SCOPED_TRACE(with_context);
      std::vector rows{with_context ? context : structural};
      if (canonicalize) {
        rows.insert(rows.end(), {first, later, distinct});
        canonicalize_plan_lowering_diagnostics(rows);
      } else {
        EXPECT_FALSE(append_unique_plan_lowering_diagnostic(rows, first));
        EXPECT_FALSE(append_unique_plan_lowering_diagnostic(rows, later));
        EXPECT_TRUE(append_unique_plan_lowering_diagnostic(rows, distinct));
      }
      ASSERT_EQ(rows.size(), 2U);
      const auto found = std::find_if(rows.begin(), rows.end(), [&](const auto& row) {
        return plan_lowering_diagnostic_exactly_equal(row, structural);
      });
      ASSERT_NE(found, rows.end());
      const auto& retained_detail = found->detail;
      ASSERT_TRUE(retained_detail.has_value());
      if (!retained_detail.has_value()) {
        FAIL() << "Expected the retained first cause after deduplication";
      }
      const auto& evidence = retained_detail.value();
      EXPECT_EQ(evidence.error.code, ErrorCode::kInvalidArgument);
      EXPECT_EQ(evidence.error.message, "first cause");
      EXPECT_EQ(evidence.operation, DiagnosticOperation::kSetup);
      EXPECT_EQ(evidence.operator_path, with_context ? "retained-owner" : "later-owner");
      if (with_context) {
        EXPECT_EQ(evidence.port_name, "retained-output");
      }
    }
  }
}

// REQ: COMPILE-076
// REQ: COMPILE-077
// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST(PlanLoweringDiagnostic, FinalPublicationPreservesFirstCauseAndOwnedContext) {
  for (const std::string token : {"PHYSICAL_TENSOR_MEMORY_KIND_VALUE_INVALID",
                                  "PHYSICAL_TENSOR_BYTE_SPAN_PROJECTION_FAILED"}) {
    SCOPED_TRACE(token);
    ExecutionPlanState state;
    auto row = diagnostic(token, 1U, GraphId{2U}, VertexId{3U}, PortId{4U});
    row.detail = DiagnosticDetail{.operation = DiagnosticOperation::kCompile,
                                  .error = Error{ErrorCode::kSuccess},
                                  .operator_path = "retained-owner",
                                  .port_name = "retained-output"};
    publish_derivation_diagnostic(state, row);
    ASSERT_EQ(state.diagnostics.size(), 1U);
    const Diagnostic original = state.diagnostics.front();
    row.detail->operation = DiagnosticOperation::kSetup;
    row.detail->error = Error{ErrorCode::kInvalidArgument, "first cause"};
    row.detail->operator_path = "later-owner";
    publish_derivation_diagnostic(state, row);
    row.detail->error = Error{ErrorCode::kNotSupported, "later cause"};
    publish_derivation_diagnostic(state, row);
    ASSERT_EQ(state.diagnostics.size(), 1U);
    const auto& retained = state.diagnostics.front();
    ASSERT_TRUE(retained.detail.has_value());
    if (!retained.detail.has_value()) {
      FAIL() << "Expected the published first cause";
    }
    EXPECT_EQ(retained.detail->operation, DiagnosticOperation::kSetup);
    EXPECT_EQ(retained.detail->error.code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(retained.detail->error.message, "first cause");
    EXPECT_EQ(retained.detail->operator_path, "retained-owner");
    EXPECT_EQ(retained.detail->port_name, "retained-output");
    EXPECT_EQ(retained.id.value, original.id.value);
    EXPECT_EQ(retained.token, token);
    EXPECT_EQ(retained.severity, original.severity);
    EXPECT_EQ(retained.source.graph_generation, original.source.graph_generation);
    EXPECT_EQ(retained.source.graph, original.source.graph);
    EXPECT_EQ(retained.service, original.service);
    EXPECT_EQ(retained.endpoint, original.endpoint);
    EXPECT_EQ(retained.cause_id, original.cause_id);
    EXPECT_EQ(retained.json_pointer, row.json_pointer);
    EXPECT_EQ(retained.required, row.required);
    EXPECT_EQ(retained.provided, row.provided);
    EXPECT_EQ(retained.source.vertex, row.source.vertex);
    EXPECT_EQ(retained.source.port, row.source.port);
  }
}

// REQ: COMPILE-078
// REQ: EXPECTED-030
TEST(PlanLoweringDiagnostic, OptionalAllocationFailureKeepsRetainedCodeAndContext) {
  const auto structural = diagnostic("TOKEN", 1U, GraphId{2U}, VertexId{3U}, PortId{4U});
  auto incoming = structural;
  incoming.detail =
      DiagnosticDetail{.operation = DiagnosticOperation::kSetup,
                       .error = Error{ErrorCode::kInvalidArgument, std::string(256U, 'p')},
                       .causes = {Error{ErrorCode::kNotSupported, std::string(256U, 'n')}}};
  for (const bool with_context : {false, true}) {
    std::vector rows{structural};
    if (with_context) {
      rows.front().detail =
          DiagnosticDetail{.error = Error{ErrorCode::kSuccess}, .operator_path = "retained-owner"};
    }
    auto candidate = incoming;
    bool appended{};
    {
      const test::ScopedAllocationExhaustion exhausted{0U};
      appended = append_unique_plan_lowering_diagnostic(rows, std::move(candidate));
    }
    EXPECT_FALSE(appended);
    ASSERT_EQ(rows.size(), 1U);
    EXPECT_TRUE(plan_lowering_diagnostic_exactly_equal(rows.front(), structural));
    const auto& retained_detail = rows.front().detail;
    ASSERT_TRUE(retained_detail.has_value());
    if (!retained_detail.has_value()) {
      FAIL() << "Expected retained code despite optional allocation failure";
    }
    const auto& retained = retained_detail.value();
    EXPECT_EQ(retained.error.code, ErrorCode::kInvalidArgument);
    EXPECT_TRUE(retained.truncated);
    if (with_context) {
      EXPECT_EQ(retained.operator_path, "retained-owner");
    }
  }
  DiagnosticCapture capture;
  capture.append(ErrorCode::kInvalidArgument, "primary cause");
  capture.append(ErrorCode::kNotSupported, std::string(256U, 'n'));
  std::optional<DiagnosticDetail> evidence;
  {
    // Primary text fits its string's short buffer. The nested cause allocation is optional.
    const test::ScopedAllocationExhaustion exhausted{0U};
    attach_diagnostic_capture(evidence, capture, DiagnosticOperation::kSetup);
  }
  ASSERT_TRUE(evidence.has_value());
  if (!evidence.has_value()) {
    FAIL() << "Expected retained primary capture evidence";
  }
  EXPECT_EQ(evidence->error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(evidence->error.message, "primary cause");
  EXPECT_TRUE(evidence->truncated);
  const DiagnosticDetail later{.error = Error{ErrorCode::kNotSupported, "replacement"}};
  {
    const test::ScopedAllocationExhaustion exhausted{0U};
    enrich_diagnostic_detail(evidence, &later);
  }
  if (!evidence.has_value()) {
    FAIL() << "Optional enrichment erased the retained primary";
  }
  EXPECT_EQ(evidence->error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(evidence->error.message, "primary cause");
}

}  // namespace
}  // namespace holoscan::detail
