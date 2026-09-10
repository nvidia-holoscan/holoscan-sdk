/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <span>
#include <vector>

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/profile.hpp>
#include <holoscan/core/temporal_plan.hpp>

#include "compiler/lifecycle_plan_lowering.hpp"
#include "compiler/plan_lowering_diagnostic.hpp"
#include "compiler/profile_admission.hpp"
#include "core/graph_snapshot.hpp"

namespace holoscan::detail {

/// @brief Compile-cold resolved rate used only while evaluating immutable recipes.
struct ResolvedPortRate {
  PortId port{};
  std::optional<Rate> resolved_rate;
  std::optional<Rate> max_nominal_rate;
  std::optional<PortId> immediate_provenance_port;
  std::optional<PortId> root_provenance_port;
  ResolvedRateOrigin origin{ResolvedRateOrigin::kUnresolved};
  RateComparison expectation_comparison{RateComparison::kNotCompared};
  bool genuinely_absent{};
};

/// @brief Single downstream authority for one edge's effective logical depth.
struct ResolvedQueueSizing {
  EdgeId edge{};
  QueueSizingRecipeId recipe{};
  QueueDepthOrigin origin{QueueDepthOrigin::kFrameworkFallback};
  std::optional<std::uint32_t> explicit_depth;
  std::uint32_t hard_floor{1U};
  std::optional<std::uint32_t> nominal_arrivals;
  std::optional<std::uint32_t> rate_recommendation;
  std::optional<std::uint32_t> admitted_depth_ceiling;
  StabilityClass stability{StabilityClass::kUnresolvedBoundedEvidence};
  std::uint32_t effective_logical_depth{1U};
};

/// @brief Compiler-private owner relation used only to widen restart membership.
using RateRestartDependency = LifecycleRestartClosureDependency;

/// @brief Total immutable rate/sizing recipe projection plus compiler-only evaluated depths.
struct RatePlanLoweringResult {
  std::vector<PortRatePlanRecord> port_rates;
  std::vector<ResolvedPortRate> resolved_rates;
  std::vector<AccumulationHorizonPlanRecord> accumulation_horizons;
  std::vector<HardFloorContributorPlanRecord> hard_floor_contributors;
  std::vector<QueueSizingRecipePlanRecord> queue_sizing_recipes;
  std::vector<ResolvedQueueSizing> queue_sizings;
  std::vector<RateRestartDependency> restart_dependencies;
  std::vector<PlanLoweringDiagnostic> errors;
  std::vector<PlanLoweringDiagnostic> warnings;

  [[nodiscard]] bool accepted() const noexcept { return errors.empty(); }
};

/// @brief Resolve exact rate declarations and evaluate compile-known queue-sizing recipes.
[[nodiscard]] RatePlanLoweringResult lower_rate_plan(
    const GraphSnapshotData& snapshot, RateRatio headroom_ratio,
    std::span<const TemporalTriggerPlanRecord> temporal_triggers,
    std::span<const InputSelectionPlanRecord> input_selections,
    std::span<const BracketedWindowPlanRecord> bracketed_windows,
    std::span<const std::optional<Profile>> effective_profiles,
    std::span<const RateBoundedEvidenceRequirement> rate_evidence_requirements,
    PlanGenerationId plan_generation, std::span<const std::uint32_t> process_group_by_vertex = {},
    std::span<const WindowTriggerPlanRecord> window_triggers = {},
    std::span<const PredicateTriggerPlanRecord> predicate_triggers = {});

/// @brief Permissive test/helper overload for callers without a profile-policy projection.
[[nodiscard]] inline RatePlanLoweringResult lower_rate_plan(
    const GraphSnapshotData& snapshot, RateRatio headroom_ratio,
    std::span<const TemporalTriggerPlanRecord> temporal_triggers,
    std::span<const InputSelectionPlanRecord> input_selections,
    std::span<const BracketedWindowPlanRecord> bracketed_windows,
    std::span<const std::optional<Profile>> effective_profiles, PlanGenerationId plan_generation) {
  return lower_rate_plan(snapshot,
                         headroom_ratio,
                         temporal_triggers,
                         input_selections,
                         bracketed_windows,
                         effective_profiles,
                         {},
                         plan_generation,
                         {});
}

/// @brief Find one evaluated edge sizing without reconstructing precedence.
[[nodiscard]] const ResolvedQueueSizing* find_resolved_queue_sizing(
    std::span<const ResolvedQueueSizing> sizings, EdgeId edge) noexcept;

/// @brief Compose one edge's hard-floor scenarios with checked max/sum lifetime algebra.
[[nodiscard]] std::optional<std::uint32_t> compose_hard_floor(
    std::span<const HardFloorContributorPlanRecord> contributors, EdgeId edge) noexcept;

/// @brief Copy one evaluated logical edge depth into every temporal capacity view.
/// @note This does not replay or reinterpret an Operator Contract.
void finalize_temporal_queue_capacities(
    const GraphSnapshotData& snapshot, std::span<const ResolvedQueueSizing> sizings,
    std::span<InputSelectionPlanRecord> input_selections,
    std::span<BracketedWindowPlanRecord> bracketed_windows,
    std::span<AlignedTriggerPlanRecord> aligned_triggers,
    std::span<PredicateInputPlanRecord> predicate_inputs) noexcept;

}  // namespace holoscan::detail
