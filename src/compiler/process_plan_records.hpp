/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/core/execution_plan.hpp>

namespace holoscan::detail {

/// @brief Select the process-isolation relationship for one private placement role.
enum class ProcessIsolation : std::uint8_t {
  kSameProcess = 0,
  kSeparateProcess = 1,
  kExternalProcess = 2,
};

/// @brief Identify one canonical source-private process-role request.
struct ProcessRoleId {
  std::uint32_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(ProcessRoleId, ProcessRoleId) noexcept = default;
};

/// @brief Classify one source-private process-object selector.
enum class ProcessSelectorKind : std::uint8_t {
  kGraphPath = 0,
  kRegion = 1,
  kExternalTopic = 2,
};

/// @brief Preserve one normalized source-private process selector.
struct ProcessSelectorPlanRecord {
  ProcessRoleId process_role{};
  std::uint32_t ordinal{};
  ProcessSelectorKind kind{ProcessSelectorKind::kGraphPath};
  std::string authored_value;
  std::string canonical_value;
  std::optional<GraphId> graph;
  std::optional<VertexId> vertex;
  std::optional<RegionId> region;
  std::optional<ExternalTopicId> external_topic;
  std::vector<VertexId> members;
  PlanDisposition disposition{PlanDisposition::kRejected};
  DiagnosticId reason{};
};

/// @brief Preserve one normalized source-private process-role request.
struct ProcessRolePlanRecord {
  ProcessRoleId id{};
  std::string role;
  ProcessIsolation requested_isolation{ProcessIsolation::kSameProcess};
  std::optional<ProcessIsolation> accepted_isolation;
  std::string requested_launch_group;
  std::string accepted_launch_group;
  bool require_locality_probe_success{true};
  bool runtime_seal_required{};
  std::vector<VertexId> members;
  PlanDisposition disposition{PlanDisposition::kRejected};
  DiagnosticId reason{};
};

/// @brief Map one canonical logical vertex into a private admitted process group.
struct VertexProcessPlacementPlanRecord {
  VertexId vertex{};
  ProcessGroupId process_group{};
  std::optional<ProcessRoleId> process_role;
  ProcessIsolation isolation{ProcessIsolation::kSameProcess};
  std::string launch_group;
  bool runtime_seal_required{};
};

}  // namespace holoscan::detail
