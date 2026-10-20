/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "process_plan_records.hpp"

namespace holoscan::detail {

/// @brief Select one stable Graph object family for source-private process placement.
struct GraphObjectSelector {
  std::string_view graph_path;
  std::string_view region_id;
  std::string_view external_topic;
};

/// @brief Describe one bounded source-private process-placement request.
struct ProcessPlacement {
  ProcessIsolation isolation{ProcessIsolation::kSameProcess};
  std::span<const GraphObjectSelector> members;
  std::string_view launch_group;
  bool require_locality_probe_success{true};
};

/// @brief Deep-own one source-private process-object selector.
struct GraphObjectSelectorRecord {
  std::string graph_path;
  std::string region_id;
  std::string external_topic;
};

/// @brief Deep-own one source-private process-placement request.
struct ProcessPlacementRecord {
  ProcessIsolation isolation{ProcessIsolation::kSameProcess};
  std::vector<GraphObjectSelectorRecord> members;
  std::string launch_group;
  bool require_locality_probe_success{true};
};

/// @brief Deep-own one source-private logical process-role binding.
struct ProcessRoleBinding {
  std::string role;
  ProcessPlacementRecord placement;
};

/// @brief Own source-private process placement supplied to compiler tests and benchmarks.
///
/// This type is declared only in a non-installed header. Public compilation
/// never receives this authority and therefore always lowers one process group.
class ProcessPlacementBinding {
 public:
  ProcessPlacementBinding& bind_process_role(std::string_view role, ProcessPlacement placement) {
    ProcessRoleBinding binding{
        .role = std::string(role),
        .placement =
            ProcessPlacementRecord{
                .isolation = placement.isolation,
                .launch_group = std::string(placement.launch_group),
                .require_locality_probe_success = placement.require_locality_probe_success,
            },
    };
    binding.placement.members.reserve(placement.members.size());
    for (const auto& selector : placement.members) {
      binding.placement.members.push_back(GraphObjectSelectorRecord{
          .graph_path = std::string(selector.graph_path),
          .region_id = std::string(selector.region_id),
          .external_topic = std::string(selector.external_topic),
      });
    }
    process_roles_.push_back(std::move(binding));
    return *this;
  }

  [[nodiscard]] std::span<const ProcessRoleBinding> process_roles() const noexcept {
    return process_roles_;
  }

 private:
  std::vector<ProcessRoleBinding> process_roles_;
};

/// @brief Inspect private process-plan records from source-tree tests and benchmarks.
class HOLOSCAN_CORE_EXPORT ProcessPlacementPlanAccess {
 public:
  [[nodiscard]] static std::span<const ProcessRolePlanRecord> process_roles(
      const ExecutionPlan& plan) noexcept;

  [[nodiscard]] static std::span<const ProcessSelectorPlanRecord> process_selectors(
      const ExecutionPlan& plan) noexcept;

  [[nodiscard]] static std::span<const VertexProcessPlacementPlanRecord> vertex_process_placements(
      const ExecutionPlan& plan) noexcept;
};

}  // namespace holoscan::detail
