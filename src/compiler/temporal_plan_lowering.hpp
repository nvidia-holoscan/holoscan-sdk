/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <span>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>

#include "core/graph_compiler_access.hpp"
#include "core/graph_snapshot.hpp"
#include "plan_lowering_diagnostic.hpp"
#include "runtime_package_capabilities.hpp"

namespace holoscan::detail {

/// @brief Temporal-plan spelling of the common lowering diagnostic value.
using TemporalPlanDiagnostic = PlanLoweringDiagnostic;

/// @brief Hold the complete pointer-free temporal projection for one plan.
struct TemporalPlanLoweringResult {
  std::vector<OperatorCompletionPlanRecord> operator_completion_rules;
  std::vector<TemporalTriggerPlanRecord> triggers;
  std::vector<AlignedTriggerPlanRecord> aligned_triggers;
  std::vector<ClockTriggerPlanRecord> clock_triggers;
  std::vector<MaxStartDelayPlanRecord> max_start_delays;
  std::vector<ReadableTriggerPlanRecord> readable_triggers;
  std::vector<MissingAggregatePlanRecord> missing_aggregates;
  std::vector<WindowTriggerPlanRecord> window_triggers;
  std::vector<BracketedWindowPlanRecord> bracketed_windows;
  std::vector<InputSelectionPlanRecord> input_selections;
  std::vector<PredicateTriggerPlanRecord> predicate_triggers;
  std::vector<PredicateInputPlanRecord> predicate_inputs;
  std::vector<PredicateCallablePlanRecord> predicate_callables;
  std::vector<ParameterDerivedTermPlanRecord> parameter_derived_terms;
  std::vector<TemporalPlanDiagnostic> errors;
  std::vector<TemporalPlanDiagnostic> warnings;

  [[nodiscard]] bool accepted() const noexcept { return errors.empty(); }
};

/// @brief Validate and resolve every normalized Operator temporal contract.
/// @param normalized_contracts Compile-local contracts produced by the single factory replay.
/// @param snapshot Canonical pointer-free Graph snapshot.
/// @param plan_generation Nonzero identity shared by all plan-local records.
/// @param clocks Resolved plan-local clock catalog.
/// @param conversions Resolved direct authored clock conversions.
/// @param readiness Resolved logical readiness catalog; referenced rows are marked in place.
/// @param effective_profiles Canonical per-vertex profile disposition.
/// @param parameter_options Frozen Parameter bounds used by duration dependencies.
/// @param package_capabilities Injected immutable package/build capability evidence.
/// @return Total temporal records plus stable errors and warnings.
/// @note Borrowed replay facts are used only while creating the returned pointer-free records.
/// This pass constructs no runtime cache, Channel queue, Tensor allocation, scheduler, or backend.
[[nodiscard]] TemporalPlanLoweringResult lower_temporal_plan(
    std::span<const GraphCompilerAccess::TemporalReplay> normalized_contracts,
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation,
    std::span<const ClockPlanRecord> clocks, std::span<const ClockConversionRecord> conversions,
    std::vector<ReadinessPlanRecord>& readiness,
    std::span<const std::optional<Profile>> effective_profiles,
    const ParameterCompileOptions& parameter_options,
    const RuntimePackageCapabilityView& package_capabilities);

}  // namespace holoscan::detail
