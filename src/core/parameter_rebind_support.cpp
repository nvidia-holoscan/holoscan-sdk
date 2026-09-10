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
#include "compiler/parameter_identity.hpp"
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
    const ParamSchemaHash context = parameter_admission_context_digest(
        profile.manifest, deployment_options, empty_parameter_migration_catalog_digest());
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
