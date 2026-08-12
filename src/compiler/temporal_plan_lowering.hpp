/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <span>
#include <string>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>

#include "core/graph_snapshot.hpp"
#include "runtime/executable_factory_inventory.hpp"
#include "runtime_package_capabilities.hpp"

namespace holoscan::detail {

/// @brief Pair one temporal admission token with its canonical plan location.
struct TemporalPlanDiagnostic {
  std::string token;
  DiagnosticSource source{};
  std::string json_pointer;
  std::string required;
  std::string provided;
};

/// @brief Hold the complete pointer-free temporal projection for one plan.
struct TemporalPlanLoweringResult {
  std::vector<TemporalTriggerPlanRecord> triggers;
  std::vector<AlignedTriggerPlanRecord> aligned_triggers;
  std::vector<ClockTriggerPlanRecord> clock_triggers;
  std::vector<DeadlinePlanRecord> deadlines;
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

/// @brief Replay, validate, and resolve every Operator temporal contract.
/// @param factories Frozen plan-owned Operator factories and canonical identity map.
/// @param snapshot Canonical pointer-free Graph snapshot.
/// @param plan_generation Nonzero identity shared by all plan-local records.
/// @param clocks Resolved plan-local clock catalog.
/// @param conversions Resolved direct authored clock conversions.
/// @param readiness Resolved logical readiness catalog; referenced rows are marked in place.
/// @param effective_profiles Canonical per-vertex profile disposition.
/// @param parameter_options Frozen Parameter bounds used by duration dependencies.
/// @param package_capabilities Injected immutable package/build capability evidence.
/// @return Total temporal records plus stable errors and warnings.
/// @note This cold pass allocates and replays factories. It constructs no
/// runtime cache, Channel queue, Tensor allocation, scheduler, or backend.
[[nodiscard]] TemporalPlanLoweringResult lower_temporal_plan(
    const ExecutableFactoryInventory& factories, const GraphSnapshotData& snapshot,
    PlanGenerationId plan_generation, std::span<const ClockPlanRecord> clocks,
    std::span<const ClockConversionRecord> conversions, std::vector<ReadinessPlanRecord>& readiness,
    std::span<const std::optional<Profile>> effective_profiles,
    const ParameterCompileOptions& parameter_options,
    const RuntimePackageCapabilityView& package_capabilities);

}  // namespace holoscan::detail
