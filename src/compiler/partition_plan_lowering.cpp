/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "partition_plan_lowering.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holoscan/core/parameter_schema.hpp>

#include "../core/diagnostic_registry.hpp"
#include "partition_cover_lowering.hpp"
#include "partition_problem_lowering.hpp"
#include "partitioning/partition_planner_catalog.hpp"
#include "partitioning/partitioning_pass.hpp"
#include "snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] PlanDigest digest_partition_config(
    Profile profile, std::span<const std::optional<Profile>> effective_profiles,
    std::span<const PartitionRecord> authored_partitions, bool automatic_partition_remaining) {
  StableHashBuilder hash;
  constexpr std::string_view domain = "holoscan.partition.config.v1";
  for (const char character : domain) {
    hash.add_byte(static_cast<std::uint8_t>(character));
  }
  hash.add(static_cast<std::uint8_t>(profile));
  hash.add(automatic_partition_remaining);
  hash.add(static_cast<std::uint64_t>(effective_profiles.size()));
  for (const auto effective : effective_profiles) {
    hash.add(effective.has_value());
    if (effective) {
      hash.add(static_cast<std::uint8_t>(*effective));
    }
  }
  hash.add(static_cast<std::uint64_t>(authored_partitions.size()));
  for (const auto& partition : authored_partitions) {
    hash.add(partition.id.value);
    hash.add(partition.graph.value);
    hash.add(partition.include_all);
    hash.add(static_cast<std::uint64_t>(partition.name.size()));
    for (const char character : partition.name) {
      hash.add_byte(static_cast<std::uint8_t>(character));
    }
    hash.add(static_cast<std::uint64_t>(partition.members.size()));
    for (const VertexId member : partition.members) {
      hash.add(member.value);
    }
  }
  return PlanDigest{hash.finish().value};
}

void initialize_partition_attempt(Profile root_profile,
                                  std::span<const std::optional<Profile>> effective_profiles,
                                  std::span<const PartitionRecord> authored_partitions,
                                  bool automatic_partition_remaining, ExecutionPlanState& output) {
  output.partitioning_plan = PartitioningPlanRecord{
      .requested_algorithm = partitioning::source_chain_algorithm(),
      .accepted_algorithm = std::nullopt,
      .interface_version = partitioning::kPartitionPlannerInterfaceVersion,
      .maturity = AlgorithmMaturity::kProduction,
      .disposition = PlanDisposition::kRejected,
      .selection_source = PlanSelectionSource::kCoreDefault,
      .config_digest = digest_partition_config(
          root_profile, effective_profiles, authored_partitions, automatic_partition_remaining),
      .reason = {},
  };
}

[[nodiscard]] unexpected<Error> publish_partition_failure(Error error, ExecutionPlanState& output) {
  output.partitioning_plan.accepted_algorithm.reset();
  output.partitioning_plan.disposition = PlanDisposition::kRejected;
  output.partitioning_plan.reason = diagnostic_id(error.message);
  return make_unexpected(std::move(error));
}

void append_schedule_edges(const GraphSnapshotData& snapshot, ExecutionPlanState& output) {
  std::unordered_map<std::uint64_t, ExecutionPartitionId> partition_by_vertex;
  for (const auto& partition : output.execution_partitions) {
    for (const VertexId vertex : partition.members) {
      partition_by_vertex.emplace(vertex.value, partition.id);
    }
  }
  for (const auto& edge : snapshot.edges) {
    const PortRecord* const source =
        find_port(snapshot, resolve_interface_port(snapshot, edge.source));
    const PortRecord* const target =
        find_port(snapshot, resolve_interface_port(snapshot, edge.target));
    if (source == nullptr || target == nullptr) {
      continue;
    }
    const auto source_partition = partition_by_vertex.find(source->owner.value);
    const auto target_partition = partition_by_vertex.find(target->owner.value);
    if (source_partition != partition_by_vertex.end() &&
        target_partition != partition_by_vertex.end() &&
        source_partition->second != target_partition->second) {
      output.schedule_edges.push_back(
          ScheduleEdgeRecord{edge.id, source_partition->second, target_partition->second});
    }
  }
}

// Publish all mutually constrained inspection indices as one cold atomic validation pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> publish_partition_inspection_indices(
    const GraphSnapshotData& snapshot, ExecutionPlanState& output) {
  const auto invalid = []() {
    return partition_lowering_error(ErrorCode::kInvalidArgument,
                                    "PARTITION_INSPECTION_PLAN_INVALID");
  };
  std::vector<std::uint32_t> partition_by_vertex(snapshot.vertices.size(), 0U);
  output.execution_partition_index_by_id.assign(output.execution_partitions.size(), 0U);
  output.execution_partition_index_by_operator_id.assign(
      output.executable_factories.operators.size(), 0U);
  output.execution_partition_index_by_root_operator.clear();
  output.execution_partition_index_by_authored_name.clear();
  output.execution_partition_index_by_operator_path.clear();

  for (std::size_t index = 0U; index < output.execution_partitions.size(); ++index) {
    const auto table_index = static_cast<std::uint32_t>(index + 1U);
    auto& partition = output.execution_partitions[index];
    if (partition.id.value != table_index) {
      return invalid();
    }
    output.execution_partition_index_by_id[index] = table_index;
    if (partition.authored_name) {
      output.execution_partition_index_by_authored_name.push_back(
          ExecutionPartitionStringLookupEntry{*partition.authored_name, table_index});
    }
    for (const VertexId member : partition.members) {
      if (!member || member.value > partition_by_vertex.size()) {
        return invalid();
      }
      const std::size_t vertex_index = static_cast<std::size_t>(member.value - 1U);
      if (partition_by_vertex[vertex_index] != 0U) {
        return invalid();
      }
      partition_by_vertex[vertex_index] = table_index;
      const VertexRecord& vertex = snapshot.vertices[vertex_index];
      if (vertex.id != member) {
        return invalid();
      }
      if (vertex.kind == VertexKind::kOperator) {
        auto operator_path = canonical_operator_path(snapshot, vertex);
        if (!operator_path) {
          return invalid();
        }
        partition.operator_paths.push_back(std::move(*operator_path));
        output.execution_partition_index_by_operator_path.push_back(
            ExecutionPartitionStringLookupEntry{partition.operator_paths.back(), table_index});
      }
    }
  }

  std::size_t root_operator_count = 0U;
  for (const auto& factory : output.executable_factories.operators) {
    if (factory.root_operator_index) {
      root_operator_count = std::max(root_operator_count, *factory.root_operator_index + 1U);
    }
  }
  output.execution_partition_index_by_root_operator.assign(root_operator_count, 0U);
  for (const auto& factory : output.executable_factories.operators) {
    if (factory.operator_id == 0U ||
        factory.operator_id > output.execution_partition_index_by_operator_id.size() ||
        !factory.vertex || factory.vertex.value > partition_by_vertex.size()) {
      return invalid();
    }
    const std::uint32_t table_index =
        partition_by_vertex[static_cast<std::size_t>(factory.vertex.value - 1U)];
    auto& operator_entry = output.execution_partition_index_by_operator_id[static_cast<std::size_t>(
        factory.operator_id - 1U)];
    if (table_index == 0U || operator_entry != 0U) {
      return invalid();
    }
    operator_entry = table_index;
    if (factory.root_operator_index) {
      auto& root_entry =
          output.execution_partition_index_by_root_operator[*factory.root_operator_index];
      if (root_entry != 0U) {
        return invalid();
      }
      root_entry = table_index;
    }
  }

  const auto sort_and_validate = [&](auto& entries) -> bool {
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.key < rhs.key;
    });
    return std::adjacent_find(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
             return lhs.key == rhs.key;
           }) == entries.end();
  };
  if (!sort_and_validate(output.execution_partition_index_by_authored_name) ||
      !sort_and_validate(output.execution_partition_index_by_operator_path)) {
    return invalid();
  }
  return {};
}

[[nodiscard]] expected<void, Error> publish_validated_plan(
    const partitioning::PartitioningPassResult& result, const GraphSnapshotData& snapshot,
    ExecutionPlanState& output) {
  output.partitioning_plan = PartitioningPlanRecord{
      .requested_algorithm = result.requested_algorithm,
      .accepted_algorithm = result.plan.algorithm,
      .interface_version = partitioning::kPartitionPlannerInterfaceVersion,
      .maturity = result.maturity,
      .disposition = PlanDisposition::kAccepted,
      .selection_source = result.selection_source,
      .config_digest = {},
      .reason = {},
      .mode = snapshot.partitions.empty() ? PartitioningMode::kAutomatic
                                          : (snapshot.automatic_partition_remaining
                                                 ? PartitioningMode::kConstrainedAutomatic
                                                 : PartitioningMode::kManual),
      .authored_partition_count = 0U,
      .automatic_partition_count = 0U,
  };

  output.execution_partitions.clear();
  output.execution_partitions.reserve(result.plan.partitions.size());
  for (const auto& partition : result.plan.partitions) {
    std::optional<std::string> authored_name;
    if (partition.authored_constraint.has_value()) {
      const auto authored = std::find_if(
          snapshot.partitions.begin(), snapshot.partitions.end(), [&](const auto& candidate) {
            return candidate.id.value == partition.authored_constraint->value;
          });
      if (authored == snapshot.partitions.end()) {
        return partition_lowering_error(ErrorCode::kInvalidArgument,
                                        "PARTITION_PROPOSAL_CONSTRAINT_VIOLATION");
      }
      authored_name = authored->name;
    }
    output.execution_partitions.push_back(ExecutionPartitionRecord{
        .id = ExecutionPartitionId{partition.canonical_id},
        .authored_name = std::move(authored_name),
        .members = partition.members,
        .operator_paths = {},
        .execution_lane = 0U,
    });
    if (output.execution_partitions.back().authored_name) {
      ++output.partitioning_plan.authored_partition_count;
    } else {
      ++output.partitioning_plan.automatic_partition_count;
    }
  }
  if (auto inspection = publish_partition_inspection_indices(snapshot, output); !inspection) {
    return inspection;
  }
  output.partition_decisions.clear();
  output.partition_decisions.reserve(result.plan.decisions.size());
  for (const auto& decision : result.plan.decisions) {
    output.partition_decisions.push_back(PartitionDecisionRecord{
        .vertex = decision.vertex,
        .partition = ExecutionPartitionId{decision.canonical_partition_id},
        .decision = decision.decision,
        .reasons = decision.reasons,
    });
  }

  append_schedule_edges(snapshot, output);
  return {};
}

}  // namespace

expected<void, Error> lower_partition_plan(
    const GraphSnapshotData& snapshot, Profile root_profile,
    std::span<const std::optional<Profile>> effective_profiles,
    bool removed_environment_selector_present,
    std::span<const std::uint32_t> process_group_by_vertex, const DeploymentBinding& deployment,
    std::span<const TensorDeviceBindingRecord> tensor_device_bindings, ExecutionPlanState& output) {
  initialize_partition_attempt(root_profile,
                               effective_profiles,
                               snapshot.partitions,
                               snapshot.automatic_partition_remaining,
                               output);
  if (removed_environment_selector_present) {
    return publish_partition_failure(
        Error{ErrorCode::kInvalidArgument, "PARTITION_ENVIRONMENT_SELECTOR_REMOVED"}, output);
  }
  const bool include_all =
      snapshot.partitions.size() == 1U && snapshot.partitions.front().include_all;
  const bool manual_cover = !snapshot.partitions.empty() && !snapshot.automatic_partition_remaining;
  const bool constrained_automatic =
      !snapshot.partitions.empty() && snapshot.automatic_partition_remaining;
  if (snapshot.automatic_partition_remaining && snapshot.partitions.empty()) {
    return publish_partition_failure(
        Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_AUTOMATIC_REMAINDER_EMPTY"}, output);
  }
  if (include_all && snapshot.automatic_partition_remaining) {
    return publish_partition_failure(
        Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_AUTOMATIC_REMAINDER_CONFLICT"}, output);
  }
  // Implicit/manual partitioning is finalized after temporal replay. Declaration-only
  // readiness sources must remain inert here; otherwise an unused source silently manufactures a
  // partition boundary before the compiler knows which source the Operator contract selected.
  auto storage_result = make_partition_problem_storage(
      snapshot, process_group_by_vertex, deployment, tensor_device_bindings, output);
  if (!storage_result) {
    return publish_partition_failure(std::move(storage_result).error(), output);
  }
  auto storage = std::move(storage_result).value();
  if (constrained_automatic) {
    auto constraints = append_authored_partition_constraints(snapshot, storage);
    if (!constraints) {
      return publish_partition_failure(std::move(constraints).error(), output);
    }
  }
  const partitioning::PartitionProblem problem{
      storage.vertices,
      storage.edges,
      storage.authored_constraints,
      static_cast<std::uint32_t>(storage.vertices.size()),
      partitioning::kAllPartitionPlannerCapabilities,
  };

  auto accepted = [&]() -> expected<partitioning::PartitioningPassResult, Error> {
    if (manual_cover) {
      return build_manual_partition_cover(snapshot, storage, process_group_by_vertex);
    }
    auto catalog = partitioning::PartitionPlannerCatalog::create_default_builtins();
    if (!catalog) {
      return make_unexpected(std::move(catalog).error());
    }
    const partitioning::PartitioningPass pass{catalog.value()};
    return pass.run(
        partitioning::PartitionSelectionInput{
            .compiler_service_request = std::nullopt,
            .test_override_request = std::nullopt,
            .root_profile = root_profile,
            .effective_profiles = effective_profiles,
            .effective_regions = output.profile_manifest.effective_regions,
        },
        problem);
  }();
  if (!accepted) {
    return publish_partition_failure(std::move(accepted).error(), output);
  }
  auto synthetics = attach_constrained_partition_synthetics(
      snapshot, storage, process_group_by_vertex, accepted.value().plan);
  if (!synthetics) {
    return publish_partition_failure(std::move(synthetics).error(), output);
  }
  auto final_cover = validate_final_accepted_partition_cover(
      snapshot, process_group_by_vertex, accepted.value().plan);
  if (!final_cover) {
    return publish_partition_failure(std::move(final_cover).error(), output);
  }
  auto published = publish_validated_plan(accepted.value(), snapshot, output);
  if (!published) {
    return publish_partition_failure(std::move(published).error(), output);
  }
  output.partitioning_plan.config_digest =
      digest_partition_config(root_profile,
                              effective_profiles,
                              snapshot.partitions,
                              snapshot.automatic_partition_remaining);
  const auto max_lane = std::max_element(
      output.execution_partitions.begin(),
      output.execution_partitions.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.execution_lane < rhs.execution_lane; });
  const std::uint32_t lane_count =
      max_lane == output.execution_partitions.end() ? 0U : max_lane->execution_lane + 1U;
  output.scheduler_plan = SchedulerPlanRecord{
      .ready_policy = AlgorithmKey{"holoscan.ready.deterministic_serial", 1U},
      .execution_lane_count = lane_count,
      .max_concurrent_activations = output.execution_partitions.empty() ? 0U : 1U,
      .activation_overlap = false,
      .worker_count = lane_count,
      .intra_path_parallelism = output.execution_partitions.empty() ? 0U : 1U,
      .reason = {},
  };
  return {};
}

}  // namespace holoscan::detail
