/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parameter_rebind_support.hpp"

#include <algorithm>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "compiler/compile_platform.hpp"
#include "compiler/deployment_plan_lowering.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/parameter_plan_lowering.hpp"
#include "compiler/partition_plan_lowering.hpp"
#include "compiler/process_plan_lowering.hpp"
#include "compiler/profile_admission.hpp"
#include "core/graph_compiler_access.hpp"

namespace holoscan::detail {
namespace {

expected<ParameterRebindGraphEvidence, Error> capture_parameter_rebind_graph_impl(
    Graph& graph, std::shared_ptr<const GraphSnapshotData> snapshot,
    std::span<const VertexId> selected_operators) noexcept {
  if (!snapshot ||
      (!selected_operators.empty() &&
       std::ranges::any_of(selected_operators, [](VertexId vertex) { return !vertex; }))) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "PARAM_REBIND_SCOPE_INVALID"});
  }
  try {
    auto profile = resolve_profile_admission(*snapshot, Profile::kStandard);
    if (!profile.accepted()) {
      return make_unexpected(Error{ErrorCode::kIncompatible, "PARAM_REBIND_PROFILE_REJECTED"});
    }
    profile.manifest.runtime_abi_version = "holoscan.runtime_abi.v1";
    profile.manifest.target_triple = std::string(compile_target_triple());
    profile.manifest.producer_build_id = "holoscan-core-5.0";
    const DeploymentBinding deployment_options;
    const ProcessPlacementBinding process_placement;
    std::vector<ExecutionPartitionRecord> partitions;
    if (snapshot->partitions.empty()) {
      std::vector<VertexId> members;
      for (const auto& vertex : snapshot->vertices) {
        if (vertex.kind == VertexKind::kOperator) {
          members.push_back(vertex.id);
        }
      }
      partitions.push_back(ExecutionPartitionRecord{
          .id = ExecutionPartitionId{1U},
          .members = std::move(members),
          .execution_lane = 0U,
      });
    } else {
      const auto process = lower_process_plan(*snapshot, process_placement);
      if (!process.error_tokens.empty()) {
        return make_unexpected(Error{ErrorCode::kIncompatible, "PARAM_REBIND_DEPLOYMENT_REJECTED"});
      }
      ExecutionPlanState partition_state;
      const std::vector<std::uint32_t> unresolved_lanes(snapshot->vertices.size(), 0U);
      auto partition = lower_partition_plan(*snapshot,
                                            Profile::kStandard,
                                            profile.effective_profiles(),
                                            std::nullopt,
                                            unresolved_lanes,
                                            process.process_group_by_vertex,
                                            partition_state);
      if (!partition) {
        return make_unexpected(Error{partition.error().code, "PARAM_REBIND_PARTITION_REJECTED"});
      }
      partitions = std::move(partition_state.execution_partitions);
    }
    auto deployment = lower_deployment_plan(
        *snapshot, deployment_options, profile.effective_profiles(), partitions);
    if (!deployment.error_tokens.empty()) {
      return make_unexpected(Error{ErrorCode::kIncompatible, "PARAM_REBIND_DEPLOYMENT_REJECTED"});
    }
    const ParamSchemaHash context = parameter_admission_context_digest(
        profile.manifest, deployment, empty_parameter_migration_catalog_digest());
    auto factories =
        selected_operators.empty()
            ? GraphCompilerAccess::executable_factories(graph, *snapshot)
            : GraphCompilerAccess::parameter_factories(graph, *snapshot, selected_operators);
    return ParameterRebindGraphEvidence{
        .snapshot = std::move(snapshot),
        .factories = std::move(factories),
        .admission_context_digest = context,
    };
  } catch (const RuntimeError& error) {
    return make_unexpected(make_error(error));
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted, "PARAM_REBIND_ALLOC_FAILED"});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure, "PARAM_REBIND_CAPTURE_FAILED"});
  }
}

}  // namespace

expected<ParameterRebindGraphEvidence, Error> capture_parameter_rebind_graph(
    Graph& graph) noexcept {
  try {
    return capture_parameter_rebind_graph_impl(
        graph, GraphCompilerAccess::canonical_snapshot(graph), {});
  } catch (const RuntimeError& error) {
    return make_unexpected(make_error(error));
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted, "PARAM_REBIND_ALLOC_FAILED"});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure, "PARAM_REBIND_CAPTURE_FAILED"});
  }
}

expected<ParameterRebindGraphEvidence, Error> capture_parameter_rebind_graph(
    Graph& graph, std::shared_ptr<const GraphSnapshotData> snapshot,
    std::span<const VertexId> selected_operators) noexcept {
  if (selected_operators.empty()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "PARAM_REBIND_SCOPE_EMPTY"});
  }
  return capture_parameter_rebind_graph_impl(graph, std::move(snapshot), selected_operators);
}

}  // namespace holoscan::detail
