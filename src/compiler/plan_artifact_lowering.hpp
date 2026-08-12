/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "compiler/execution_plan_state.hpp"
#include "core/graph_snapshot.hpp"

namespace holoscan::detail {

/// Pair one endpoint-catalog diagnostic token with its canonical plan location.
struct EndpointCatalogDiagnostic {
  std::string token;
  DiagnosticSource source{};
  std::string json_pointer;
  std::string required;
  std::string provided;
};

/// Hold endpoint-catalog admission evidence for the compile orchestrator.
struct EndpointCatalogLoweringResult {
  ChannelEndpointCatalog catalog;
  std::vector<EndpointCatalogDiagnostic> errors;
  std::vector<EndpointCatalogDiagnostic> warnings;

  [[nodiscard]] bool accepted() const noexcept { return errors.empty(); }
};

/// Hold normalized per-endpoint clock-health records and admission diagnostics.
struct EndpointClockHealthLoweringResult {
  std::vector<EndpointClockHealthPlanRecord> records;
  std::vector<EndpointCatalogDiagnostic> errors;

  [[nodiscard]] bool accepted() const noexcept { return errors.empty(); }
};

/// Hold one private, immutable monitor-path qualification supplied by deployment admission.
///
/// This record is not Graph authoring and cannot upgrade an ordinary endpoint
/// to monitored mode. It proves the finite monitor path only for an endpoint
/// that already declares `status_reports + fresh_healthy_report`.
struct EndpointClockHealthQualificationRecord {
  PortId endpoint{};                               ///< Exact plan-local external endpoint identity.
  std::chrono::nanoseconds monitor_period{};       ///< Positive finite reporting period.
  std::chrono::nanoseconds observation_to_call{};  ///< Finite sensor-to-API delay bound.
  std::chrono::nanoseconds delivery_delay{};       ///< Finite API-to-receiver delay bound.
  std::chrono::nanoseconds jitter_margin{};        ///< Nonnegative admitted jitter margin.
  bool bounded_transition_delivery{};   ///< Lossless bounded or sticky-unsafe handoff proof.
  bool trace_loss_fail_closed{};        ///< Structural evidence cannot fail open.
  bool unspecified_reason_qualified{};  ///< Regulated use separately qualifies this reason.
  std::string evidence_reference;       ///< Stable immutable qualification identity.
};

/// Hold one atomically validated clock catalog, direct conversions, and readiness table.
///
/// Every numeric clock identity in this result belongs to \ref plan_generation.
/// The compiler publishes the result only after validating every declaration,
/// so an error cannot expose a partial catalog or source table.
struct ClockAndReadinessLoweringResult {
  PlanGenerationId plan_generation{};
  ClockId root_default_clock{};
  std::vector<ClockPlanRecord> clocks;
  std::vector<ClockImportResolutionPlanRecord> clock_import_resolutions;
  std::vector<ClockConversionRecord> clock_conversions;
  std::vector<ReadinessPlanRecord> readiness;
  std::vector<DiagnosticSource> unused_source_warnings;
};

/// Derive the deterministic nonzero plan generation from canonical semantic plan bytes.
///
/// The state's artifact digest must have been finalized while every plan-generation
/// field used the compiler's fixed canonicalization seed. Protected resolved
/// Parameter value records are incorporated without exposing their digest bytes
/// in public JSON. Consequently, equivalent request spellings that lower to the
/// same semantic records share one identity, while addresses, authoring order,
/// logging state, and unused valid inputs do not contribute.
[[nodiscard]] PlanGenerationId derive_plan_generation_id(
    const ExecutionPlanState& canonical_state) noexcept;

/// Rebind every plan-generation-bearing record to one final identity.
///
/// This cold compiler pass changes provenance fields and the Parameter-catalog
/// compatibility digest only. All normalized semantic records remain unchanged.
void rebind_plan_generation(ExecutionPlanState& state, PlanGenerationId plan_generation);

/// Lower clocks, direct conversions, and readiness declarations into one plan-local identity space.
///
/// On failure the result is absent, so callers cannot accidentally publish a
/// partial clock catalog, conversion table, or source table.
[[nodiscard]] expected<ClockAndReadinessLoweringResult, Error> lower_clocks_and_readiness(
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation,
    std::span<const std::optional<Profile>> effective_vertex_profiles);

/// Build one immutable endpoint catalog and retain exact generated-name admission evidence.
[[nodiscard]] EndpointCatalogLoweringResult lower_endpoint_catalog(
    const GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    std::span<const std::optional<Profile>> effective_vertex_profiles,
    const std::optional<ChannelEndpointCatalog>& supplied_catalog = std::nullopt,
    std::span<const ClockPlanRecord> clocks = {},
    std::span<const ClockConversionRecord> clock_conversions = {});

/// Normalize every endpoint clock-health contract after clocks and profiles are resolved.
[[nodiscard]] EndpointClockHealthLoweringResult lower_endpoint_clock_health(
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation,
    std::span<const ClockPlanRecord> clocks,
    std::span<const ClockConversionRecord> clock_conversions,
    std::span<const std::optional<Profile>> effective_vertex_profiles,
    std::span<const InputSelectionPlanRecord> input_selections = {},
    std::span<const EndpointClockHealthQualificationRecord> qualifications = {});

/// Build one immutable endpoint catalog under one uniform test profile.
///
/// Production compilation uses the per-vertex overload above. This overload
/// exists for source-tree lowering tests that construct a canonical snapshot
/// without running profile admission.
[[nodiscard]] EndpointCatalogLoweringResult lower_endpoint_catalog(
    const GraphSnapshotData& snapshot, const DeploymentBinding& deployment, Profile profile,
    const std::optional<ChannelEndpointCatalog>& supplied_catalog = std::nullopt,
    std::span<const ClockPlanRecord> clocks = {},
    std::span<const ClockConversionRecord> clock_conversions = {});

/// Serialize normalized plan semantics and set the exact SHA-256 plan digest.
///
/// This operation performs no filesystem, environment, transport, runtime,
/// thread, CUDA, logging, or lifecycle mutation.
[[nodiscard]] expected<void, Error> finalize_plan_artifact(ExecutionPlanState& state);

}  // namespace holoscan::detail
