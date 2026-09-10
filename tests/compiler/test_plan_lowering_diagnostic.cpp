/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "compiler/endpoint_catalog_lowering.hpp"
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

}  // namespace
}  // namespace holoscan::detail
