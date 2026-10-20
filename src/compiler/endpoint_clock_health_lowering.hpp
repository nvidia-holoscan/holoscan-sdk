/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <span>
#include <string>
#include <vector>

#include "compiler/endpoint_catalog_lowering.hpp"
#include "core/graph_snapshot.hpp"

namespace holoscan::detail {

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
/// that already declares c status_reports and c fresh_healthy_report.
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

/// Normalize every endpoint clock-health contract after clocks and profiles are resolved.
[[nodiscard]] EndpointClockHealthLoweringResult lower_endpoint_clock_health(
    const GraphSnapshotData& snapshot, PlanGenerationId plan_generation,
    std::span<const ClockPlanRecord> clocks,
    std::span<const ClockConversionRecord> clock_conversions,
    std::span<const std::optional<Profile>> effective_vertex_profiles,
    std::span<const InputSelectionPlanRecord> input_selections = {},
    std::span<const EndpointClockHealthQualificationRecord> qualifications = {});

}  // namespace holoscan::detail
