/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/core/compile.hpp>

#include "compiler/plan_lowering_diagnostic.hpp"
#include "core/graph_snapshot.hpp"

namespace holoscan::detail {

struct ResolvedQueueSizing;

/// Endpoint-catalog spelling of the common lowering diagnostic value.
using EndpointCatalogDiagnostic = PlanLoweringDiagnostic;

/// Hold endpoint-catalog admission evidence for the compile orchestrator.
struct EndpointCatalogLoweringResult {
  ChannelEndpointCatalog catalog;
  std::vector<EndpointCatalogDiagnostic> errors;
  std::vector<EndpointCatalogDiagnostic> warnings;

  [[nodiscard]] bool accepted() const noexcept { return errors.empty(); }
};

/// Convert one detailed endpoint-catalog normalization error into compiler evidence.
[[nodiscard]] EndpointCatalogDiagnostic endpoint_catalog_normalization_diagnostic(
    std::string_view message, DiagnosticSource source);

/// Return whether an endpoint is governed by a constrained execution profile.
[[nodiscard]] bool endpoint_profile_is_constrained(Profile profile) noexcept;

/// Build the diagnostic source for one authored external endpoint.
[[nodiscard]] DiagnosticSource make_endpoint_diagnostic_source(
    const GraphSnapshotData& snapshot, const ExternalTopicRecord& topic) noexcept;

/// Return the canonical generated name for one authored external endpoint.
[[nodiscard]] std::string generated_endpoint_name_for_topic(const GraphSnapshotData& snapshot,
                                                            const ExternalTopicRecord& topic);

/// Build one immutable endpoint catalog and retain exact generated-name admission evidence.
[[nodiscard]] EndpointCatalogLoweringResult lower_endpoint_catalog(
    const GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    std::span<const std::optional<Profile>> effective_vertex_profiles,
    const std::optional<ChannelEndpointCatalog>& supplied_catalog = std::nullopt,
    std::span<const ClockPlanRecord> clocks = {},
    std::span<const ClockConversionRecord> clock_conversions = {},
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations = {},
    std::span<const ResolvedQueueSizing> resolved_queue_sizings = {});

/// Build one immutable endpoint catalog under one uniform test profile.
///
/// Production compilation uses the per-vertex overload above. This overload
/// exists for source-tree lowering tests that construct a canonical snapshot
/// without running profile admission.
[[nodiscard]] EndpointCatalogLoweringResult lower_endpoint_catalog(
    const GraphSnapshotData& snapshot, const DeploymentBinding& deployment, Profile profile,
    const std::optional<ChannelEndpointCatalog>& supplied_catalog = std::nullopt,
    std::span<const ClockPlanRecord> clocks = {},
    std::span<const ClockConversionRecord> clock_conversions = {},
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations = {},
    std::span<const ResolvedQueueSizing> resolved_queue_sizings = {});

}  // namespace holoscan::detail
