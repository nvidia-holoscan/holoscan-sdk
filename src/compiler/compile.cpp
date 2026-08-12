/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/compile.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <holoscan/logger/logger.hpp>

#include "../core/diagnostic_registry.hpp"
#include "../core/graph_compiler_access.hpp"
#include "closed_island_activation_plan.hpp"
#include "compile_internal.hpp"
#include "compile_platform.hpp"
#include "deployment_plan_lowering.hpp"
#include "detail/snapshot_lookup.hpp"
#include "execution_plan_state.hpp"
#include "inline_partition_execution_plan.hpp"
#include "lifecycle_plan_lowering.hpp"
#include "parameter_plan_lowering.hpp"
#include "partition_plan_lowering.hpp"
#include "partition_runner_plan.hpp"
#include "physical_plan_lowering.hpp"
#include "plan_artifact_lowering.hpp"
#include "process_plan_lowering.hpp"
#include "profile_admission.hpp"
#include "profile_manifest_lowering.hpp"
#include "temporal_plan_lowering.hpp"

namespace holoscan {
namespace {

[[nodiscard]] const detail::StructuralDiagnosticDescriptor* diagnostic_for_token(
    std::string_view token) noexcept {
  for (const auto& descriptor : detail::kStructuralDiagnosticRegistry) {
    if (descriptor.token == token) {
      return &descriptor;
    }
  }
  return nullptr;
}

[[nodiscard]] DiagnosticSource complete_diagnostic_source(const detail::ExecutionPlanState& state,
                                                          DiagnosticSource source) noexcept {
  if (source.graph_generation == 0U) {
    source.graph_generation = state.graph.generation();
  }
  if (!source.graph && source.vertex) {
    const auto owner = std::find_if(state.graph.vertices().begin(),
                                    state.graph.vertices().end(),
                                    [&](const auto& vertex) { return vertex.id == source.vertex; });
    if (owner != state.graph.vertices().end()) {
      source.graph = owner->graph;
    }
  }
  if (!source.graph && source.port) {
    const auto port = std::find_if(
        state.graph.ports().begin(), state.graph.ports().end(), [&](const auto& candidate) {
          return candidate.id.value == source.port.value();
        });
    if (port != state.graph.ports().end()) {
      source.vertex = port->owner;
      const auto owner = std::find_if(
          state.graph.vertices().begin(), state.graph.vertices().end(), [&](const auto& vertex) {
            return vertex.id == source.vertex;
          });
      if (owner != state.graph.vertices().end()) {
        source.graph = owner->graph;
      }
    }
  }
  if (!source.graph) {
    const auto root = std::find_if(state.graph.graphs().begin(),
                                   state.graph.graphs().end(),
                                   [](const auto& graph) { return !graph.parent.has_value(); });
    if (root != state.graph.graphs().end()) {
      source.graph = root->id;
    }
  }
  return source;
}

[[nodiscard]] std::uint32_t callback_partition_owner_count(
    const detail::ExecutionPlanState& state) noexcept {
  return static_cast<std::uint32_t>(std::count_if(
      state.execution_partitions.begin(),
      state.execution_partitions.end(),
      [&](const auto& partition) {
        return std::any_of(
            partition.members.begin(), partition.members.end(), [&](VertexId member) {
              return std::any_of(state.executable_factories.operators.begin(),
                                 state.executable_factories.operators.end(),
                                 [&](const auto& factory) { return factory.vertex == member; });
            });
      }));
}

void append_diagnostic(detail::ExecutionPlanState& state, std::string_view token,
                       DiagnosticSource source = {},
                       std::optional<DiagnosticSeverity> severity_override = std::nullopt,
                       std::string_view json_pointer = {}, std::string_view required = {},
                       std::string_view provided = {}) {
  source = complete_diagnostic_source(state, source);
  const auto* descriptor = diagnostic_for_token(token);
  if (descriptor == nullptr) {
    descriptor = diagnostic_for_token("COMPILER_DIAGNOSTIC_TOKEN_UNREGISTERED");
    token = descriptor->token;
  }
  if (std::any_of(state.diagnostics.begin(), state.diagnostics.end(), [&](const auto& item) {
        return item.token == token && item.source.graph_generation == source.graph_generation &&
               item.source.graph == source.graph && item.source.vertex == source.vertex &&
               item.source.port == source.port && item.json_pointer == json_pointer &&
               item.required == required && item.provided == provided;
      })) {
    return;
  }

  const auto severity = severity_override.value_or(
      descriptor->severity == detail::DiagnosticSeverity::kWarning ? DiagnosticSeverity::kWarning
                                                                   : DiagnosticSeverity::kError);
  state.diagnostics.push_back(Diagnostic{
      .id = DiagnosticId{static_cast<std::uint32_t>(descriptor->id)},
      .severity = severity,
      .token = std::string(token),
      .source = source,
      .json_pointer = std::string(json_pointer),
      .required = std::string(required),
      .provided = std::string(provided),
  });
}

void append_error(detail::ExecutionPlanState& state, const Error& error) {
  append_diagnostic(state, error.message.empty() ? error_code_name(error.code) : error.message);
}

[[nodiscard]] std::string bounded_placement_value(std::string_view value) {
  constexpr std::size_t kMaximumEvidenceCharacters = 128U;
  if (value.size() <= kMaximumEvidenceCharacters) {
    return std::string(value);
  }
  return "<oversized:" + std::to_string(value.size()) + ">";
}

// Resolve and diagnose the complete cold deployment binding set in one canonical pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::vector<detail::TensorDeviceBindingRecord> resolve_tensor_device_placements(
    const detail::GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    detail::ExecutionPlanState& state) {
  struct IndexedPlacement {
    const TensorOutputDevicePlacementRecord* record{};
    std::size_t authored_index{};
  };
  std::vector<IndexedPlacement> placements;
  placements.reserve(deployment.tensor_output_devices().size());
  for (std::size_t index = 0U; index < deployment.tensor_output_devices().size(); ++index) {
    placements.push_back(IndexedPlacement{
        .record = &deployment.tensor_output_devices()[index],
        .authored_index = index,
    });
  }
  std::sort(placements.begin(), placements.end(), [](const auto& left, const auto& right) {
    return std::tie(
               left.record->operator_path, left.record->output_port, left.record->device.value) <
           std::tie(
               right.record->operator_path, right.record->output_port, right.record->device.value);
  });

  std::vector<detail::TensorDeviceBindingRecord> result;
  result.reserve(placements.size());
  std::set<std::uint64_t> selected_ports;
  std::unordered_map<std::string, const VertexRecord*> operator_by_path;
  operator_by_path.reserve(snapshot.vertices.size());
  for (const auto& vertex : snapshot.vertices) {
    if (vertex.kind != VertexKind::kOperator) {
      continue;
    }
    auto [found, inserted] =
        operator_by_path.emplace(detail::canonical_vertex_path(snapshot, vertex), &vertex);
    if (!inserted) {
      found->second = nullptr;
    }
  }
  std::vector<std::unordered_map<std::string_view, const PortRecord*>> ports_by_owner(
      snapshot.vertices.size());
  for (const auto& port : snapshot.ports) {
    if (port.owner && port.owner.value <= ports_by_owner.size()) {
      ports_by_owner[static_cast<std::size_t>(port.owner.value - 1U)].emplace(port.name, &port);
    }
  }
  for (const auto& indexed : placements) {
    const auto& placement = *indexed.record;
    const std::string pointer =
        "/deployment/tensor_output_devices/" + std::to_string(indexed.authored_index);
    const auto reject_placement = [&](std::string_view reason, DiagnosticSource source = {}) {
      append_diagnostic(state,
                        "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID",
                        source,
                        std::nullopt,
                        pointer,
                        "one exact operator-owned Tensor-bearing output and nonnegative device",
                        "operator_path=" + bounded_placement_value(placement.operator_path) +
                            ",output_port=" + bounded_placement_value(placement.output_port) +
                            ",device=" + std::to_string(placement.device.value) +
                            ",reason=" + std::string(reason));
    };

    const auto owner_match = operator_by_path.find(placement.operator_path);
    if (owner_match == operator_by_path.end() || owner_match->second == nullptr) {
      reject_placement("unknown_or_ambiguous_operator");
      continue;
    }
    const VertexRecord* owner = owner_match->second;

    const PortRecord* output = nullptr;
    const PortRecord* named_port = nullptr;
    const auto& owner_ports = ports_by_owner[static_cast<std::size_t>(owner->id.value - 1U)];
    const auto port_match = owner_ports.find(placement.output_port);
    if (port_match != owner_ports.end()) {
      named_port = port_match->second;
      if (named_port->owner_kind == PortOwnerKind::kOperator &&
          named_port->direction == PortDirection::kOutput) {
        output = named_port;
      }
    }
    const DiagnosticSource source{
        .graph_generation = snapshot.generation,
        .graph = owner->graph,
        .vertex = owner->id,
        .port = output != nullptr ? PortId{output->id.value} : PortId{},
    };
    if (output == nullptr) {
      reject_placement(named_port == nullptr ? "unknown_output" : "not_operator_output", source);
      continue;
    }
    const bool tensor_bearing =
        output->payload_identity.kind == PayloadIdentityKind::kTensor ||
        (output->payload_identity.kind == PayloadIdentityKind::kSchema &&
         output->payload_identity.payload_kind == PayloadKind::kDescriptorBacked);
    if (!tensor_bearing) {
      reject_placement("non_tensor_output", source);
      continue;
    }
    if (!placement.device.valid()) {
      reject_placement("invalid_device", source);
      continue;
    }
    if (!selected_ports.insert(output->id.value).second) {
      reject_placement("duplicate_output", source);
      continue;
    }
    result.push_back(detail::TensorDeviceBindingRecord{
        .port = PortId{output->id.value},
        .device = placement.device,
    });
  }
  return result;
}

[[nodiscard]] std::optional<std::string> capture_partition_environment() {
  // Capture once at compile ingress. No downstream pass calls getenv, so an
  // accepted plan cannot change when the process environment changes later.
#if defined(_WIN32)
  constexpr const char* kName = "HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM";
  ::SetLastError(ERROR_SUCCESS);
  const DWORD required = ::GetEnvironmentVariableA(kName, nullptr, 0U);
  if (required == 0U) {
    return ::GetLastError() == ERROR_ENVVAR_NOT_FOUND ? std::nullopt
                                                      : std::optional<std::string>{std::string{}};
  }
  std::string value(required, '\0');
  ::SetLastError(ERROR_SUCCESS);
  const DWORD written = ::GetEnvironmentVariableA(kName, value.data(), required);
  if (written == 0U) {
    return ::GetLastError() == ERROR_SUCCESS
               ? std::optional<std::string>{std::string{}}
               : std::optional<std::string>{std::string{"<environment-changed-during-capture>"}};
  }
  if (written >= required) {
    // A concurrent process-environment mutation invalidated the captured size.
    // Preserve an invalid nonempty value rather than silently treating it as
    // absent or accepting a truncated selector.
    return std::string{"<environment-changed-during-capture>"};
  }
  value.resize(written);
  return value;
#else
  const char* value = std::getenv("HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM");
  return value != nullptr ? std::optional<std::string>{value} : std::nullopt;
#endif
}

void publish_lifecycle(detail::LifecyclePlanLowering&& lifecycle,
                       detail::ExecutionPlanState& state) {
  for (const auto& diagnostic : lifecycle.diagnostics) {
    append_diagnostic(state,
                      diagnostic.token,
                      diagnostic.source,
                      std::nullopt,
                      diagnostic.json_pointer,
                      diagnostic.required,
                      diagnostic.provided);
  }
  state.lifecycle_checkpoints = std::move(lifecycle.checkpoints);
  state.lifecycle_dependencies = std::move(lifecycle.dependencies);
  state.lifecycle_restart_permissions = std::move(lifecycle.restart_permissions);
  state.lifecycle_restart_targets = std::move(lifecycle.restart_targets);
  state.lifecycle_restart_closures = std::move(lifecycle.restart_closures);
  state.lifecycle_mutable_fields = std::move(lifecycle.mutable_fields);
  state.lifecycle_transaction_capacity = std::move(lifecycle.transaction_capacity);
}

void publish_catalog(detail::EndpointCatalogLoweringResult&& catalog,
                     detail::ExecutionPlanState& state) {
  for (const auto& diagnostic : catalog.errors) {
    append_diagnostic(state,
                      diagnostic.token,
                      diagnostic.source,
                      DiagnosticSeverity::kError,
                      diagnostic.json_pointer,
                      diagnostic.required,
                      diagnostic.provided);
  }
  for (const auto& diagnostic : catalog.warnings) {
    append_diagnostic(state,
                      diagnostic.token,
                      diagnostic.source,
                      DiagnosticSeverity::kWarning,
                      diagnostic.json_pointer,
                      diagnostic.required,
                      diagnostic.provided);
  }
  state.endpoint_catalog = std::move(catalog.catalog);
}

void publish_endpoint_clock_health(detail::EndpointClockHealthLoweringResult&& health,
                                   detail::ExecutionPlanState& state) {
  for (const auto& diagnostic : health.errors) {
    append_diagnostic(state,
                      diagnostic.token,
                      diagnostic.source,
                      DiagnosticSeverity::kError,
                      diagnostic.json_pointer,
                      diagnostic.required,
                      diagnostic.provided);
  }
  state.endpoint_clock_health = std::move(health.records);
}

void publish_clocks_and_readiness(detail::ClockAndReadinessLoweringResult&& lowered,
                                  detail::ExecutionPlanState& state) {
  state.root_default_clock = lowered.root_default_clock;
  state.clocks = std::move(lowered.clocks);
  state.clock_import_resolutions = std::move(lowered.clock_import_resolutions);
  state.clock_conversions = std::move(lowered.clock_conversions);
  state.readiness = std::move(lowered.readiness);
}

void publish_temporal(detail::TemporalPlanLoweringResult&& temporal,
                      detail::ExecutionPlanState& state) {
  for (const auto& diagnostic : temporal.errors) {
    append_diagnostic(state,
                      diagnostic.token,
                      diagnostic.source,
                      DiagnosticSeverity::kError,
                      diagnostic.json_pointer,
                      diagnostic.required,
                      diagnostic.provided);
  }
  for (const auto& diagnostic : temporal.warnings) {
    append_diagnostic(state,
                      diagnostic.token,
                      diagnostic.source,
                      DiagnosticSeverity::kWarning,
                      diagnostic.json_pointer,
                      diagnostic.required,
                      diagnostic.provided);
  }
  state.temporal_triggers = std::move(temporal.triggers);
  state.window_triggers = std::move(temporal.window_triggers);
  state.bracketed_windows = std::move(temporal.bracketed_windows);
  state.input_selections = std::move(temporal.input_selections);
  state.predicate_triggers = std::move(temporal.predicate_triggers);
  state.predicate_inputs = std::move(temporal.predicate_inputs);
  state.predicate_callables = std::move(temporal.predicate_callables);
  state.parameter_derived_terms = std::move(temporal.parameter_derived_terms);
  state.aligned_triggers = std::move(temporal.aligned_triggers);
  state.clock_triggers = std::move(temporal.clock_triggers);
  state.deadlines = std::move(temporal.deadlines);
  state.readable_triggers = std::move(temporal.readable_triggers);
  state.missing_aggregates = std::move(temporal.missing_aggregates);
}

void publish_physical(detail::PhysicalPlanLoweringResult&& physical,
                      detail::ExecutionPlanState& state) {
  for (const auto& diagnostic : physical.errors) {
    append_diagnostic(state,
                      diagnostic.token,
                      diagnostic.source,
                      DiagnosticSeverity::kError,
                      diagnostic.json_pointer,
                      diagnostic.required,
                      diagnostic.provided);
  }
  detail::publish_physical_plan_records(std::move(physical), state);
}

[[nodiscard]] bool has_error_diagnostic(const detail::ExecutionPlanState& state) noexcept {
  return std::any_of(state.diagnostics.begin(), state.diagnostics.end(), [](const auto& item) {
    return item.severity == DiagnosticSeverity::kError;
  });
}

void reject_unsupported_staged_inputs(const CompileOptions& options,
                                      detail::ExecutionPlanState& state) {
  // The public option is retained for the final converter lowerer, but the
  // foundation cannot yet prove a converter-free graph. Reject the one
  // non-default request instead of accepting a hash-only policy switch.
  if (!options.auto_insert_converters) {
    append_diagnostic(state, "CONVERTER_AUTO_INSERTION_DISABLE_UNSUPPORTED");
  }
}

[[nodiscard]] bool clock_health_qualification_shape_valid(
    const detail::RuntimeEndpointClockHealthQualification& qualification) noexcept {
  constexpr std::size_t kMaxEvidenceReferenceBytes = 1024U;
  const std::array<std::chrono::nanoseconds, 4U> terms{
      qualification.monitor_period,
      qualification.observation_to_call,
      qualification.delivery_delay,
      qualification.jitter_margin,
  };
  if (terms.front().count() <= 0 || qualification.evidence_reference.empty() ||
      qualification.evidence_reference.size() > kMaxEvidenceReferenceBytes ||
      !qualification.bounded_transition_delivery || !qualification.trace_loss_fail_closed) {
    return false;
  }
  std::int64_t total{};
  for (const auto term : terms) {
    if (term.count() < 0 || total > std::numeric_limits<std::int64_t>::max() - term.count()) {
      return false;
    }
    total += term.count();
  }
  return true;
}

[[nodiscard]] std::vector<detail::EndpointClockHealthQualificationRecord>
resolve_clock_health_qualifications(const detail::RuntimeCompileEvidenceView& evidence,
                                    const detail::ChannelEndpointCatalogState* endpoint_catalog,
                                    detail::ExecutionPlanState& state) {
  constexpr std::size_t kMaxEndpointQualifications = 4096U;
  std::vector<detail::EndpointClockHealthQualificationRecord> result;
  const std::size_t qualification_count =
      std::min(evidence.endpoint_clock_health.size(), kMaxEndpointQualifications);
  result.reserve(qualification_count);
  const auto& package = evidence.package_capabilities.manifest();
  std::set<std::string_view> seen_endpoints;
  if (evidence.endpoint_clock_health.size() > kMaxEndpointQualifications) {
    append_diagnostic(state, "PROFILE_EVIDENCE_UNSATISFIED");
  }

  for (std::size_t index = 0U; index < qualification_count; ++index) {
    const auto& qualification = evidence.endpoint_clock_health[index];
    const detail::ChannelEndpointRecord* endpoint{};
    if (endpoint_catalog != nullptr) {
      const auto found = std::lower_bound(
          endpoint_catalog->endpoints.begin(),
          endpoint_catalog->endpoints.end(),
          std::string_view{qualification.endpoint_name},
          [](const auto& candidate, std::string_view name) { return candidate.name < name; });
      if (found != endpoint_catalog->endpoints.end() &&
          found->name == qualification.endpoint_name) {
        endpoint = &*found;
      }
    }
    DiagnosticSource source{};
    if (endpoint != nullptr) {
      source.port = endpoint->boundary_port;
    }
    const bool duplicate = !seen_endpoints.insert(qualification.endpoint_name).second;
    const bool package_matches = qualification.package_id == package.package_id &&
                                 qualification.build_id == package.build_id &&
                                 qualification.package_digest == package.digest;
    const bool monitored_import =
        endpoint != nullptr && endpoint->direction == TopicDirection::kImport &&
        endpoint->clock_health.capability == ClockHealthCapability::kStatusReports &&
        endpoint->clock_health.requirement == ClockHealthRequirement::kFreshHealthyReport;
    if (endpoint == nullptr || duplicate || !package_matches || !monitored_import ||
        !clock_health_qualification_shape_valid(qualification)) {
      append_diagnostic(state, "PROFILE_EVIDENCE_UNSATISFIED", source);
      continue;
    }
    result.push_back(detail::EndpointClockHealthQualificationRecord{
        .endpoint = endpoint->boundary_port,
        .monitor_period = qualification.monitor_period,
        .observation_to_call = qualification.observation_to_call,
        .delivery_delay = qualification.delivery_delay,
        .jitter_margin = qualification.jitter_margin,
        .bounded_transition_delivery = qualification.bounded_transition_delivery,
        .trace_loss_fail_closed = qualification.trace_loss_fail_closed,
        .unspecified_reason_qualified = qualification.unspecified_reason_qualified,
        .evidence_reference = qualification.evidence_reference,
    });
  }
  return result;
}

}  // namespace

void detail::publish_physical_plan_records(detail::PhysicalPlanLoweringResult&& physical,
                                           detail::ExecutionPlanState& state) {
  state.ports = std::move(physical.ports);
  state.interface_port_bindings = std::move(physical.interface_bindings);
  state.payload_edges = std::move(physical.payload_edges);
  state.channel_locality = std::move(physical.channel_locality);
  state.channel_services = std::move(physical.channel_services);
  state.channel_publishers = std::move(physical.channel_publishers);
  state.channel_subscribers = std::move(physical.channel_subscribers);
  state.process_local_channel_services = std::move(physical.process_local_channel_services);
  state.process_local_channel_publishers = std::move(physical.process_local_channel_publishers);
  state.process_local_channel_subscribers = std::move(physical.process_local_channel_subscribers);
  state.cross_process_channel_services = std::move(physical.cross_process_channel_services);
  state.cross_process_channel_publishers = std::move(physical.cross_process_channel_publishers);
  state.cross_process_channel_subscribers = std::move(physical.cross_process_channel_subscribers);
  state.generic_output_loans = std::move(physical.generic_output_loans);
  state.tensor_allocations = std::move(physical.tensor_allocations);
  state.tensor_field_allocations = std::move(physical.tensor_field_allocations);
  state.tensor_interface_pools = std::move(physical.tensor_interface_pools);
  state.tensor_device_resources = std::move(physical.tensor_device_resources);
  state.tensor_inline_records = std::move(physical.tensor_inline_records);
  state.imported_generation_capacities = std::move(physical.imported_generation_capacities);
  state.tensor_transfer_plans = std::move(physical.tensor_transfer_plans);
  state.pool_capacity_contributors = std::move(physical.pool_capacity_contributors);
  state.publication_outcomes = std::move(physical.publication_outcomes);
}

// CompileOptions is intentionally captured by value so environment and caller-
// owned option state cannot change during total plan construction.
// NOLINTBEGIN(performance-unnecessary-value-param,readability-function-cognitive-complexity)
ExecutionPlan detail::compile_with_runtime_evidence_and_process_placement(
    Graph& graph, CompileOptions options, detail::RuntimeCompileEvidenceView evidence,
    detail::ProcessPlacementBinding placement) {
  // REQ: COMPILE-001, COMPILE-002, COMPILE-009
  const std::optional<std::string> captured_environment = capture_partition_environment();
  // All target-package decisions in one compile share this single validated
  // immutable view; no later pass re-reads packaging state or searches paths.
  const detail::RuntimePackageCapabilityView package_capabilities = evidence.package_capabilities;
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);

  auto state = std::make_shared<detail::ExecutionPlanState>();
  state->graph = detail::GraphCompilerAccess::canonical_view(snapshot);
  state->graph_identity = state->graph.identity();

  auto profile = detail::resolve_profile_admission(*snapshot, options.profile);
  const bool profile_accepted = profile.accepted();
  state->profile_manifest = std::move(profile.manifest);
  state->profile_manifest.runtime_abi_version = "holoscan.runtime_abi.v1";
  state->profile_manifest.target_triple = std::string(detail::compile_target_triple());
  state->profile_manifest.producer_build_id = "holoscan-core-5.0";
  for (const auto& token : profile.error_tokens) {
    append_diagnostic(*state, token);
  }

  reject_unsupported_staged_inputs(options, *state);

  auto process = detail::lower_process_plan(*snapshot, placement);
  for (const auto& token : process.error_tokens) {
    append_diagnostic(*state, token);
  }
  state->process_roles = std::move(process.roles);
  state->process_selectors = std::move(process.selectors);
  state->vertex_process_placements = std::move(process.vertices);
  const auto captured_view = captured_environment.has_value()
                                 ? std::optional<std::string_view>{*captured_environment}
                                 : std::nullopt;
  // Lower every record under one fixed nonzero seed. Once all compile inputs
  // have been normalized into semantic records, the seed-form artifact digest
  // determines the final generation and one cold pass rebinds provenance.
  // This prevents equivalent request spellings from becoming cache identity.
  constexpr PlanGenerationId plan_generation = 1U;
  state->plan_generation = plan_generation;
  // Capture once at the common source-private ingress so public compilation and provider-injected
  // evidence compilation retain the same executable inventory after the source Graph is gone.
  state->schema_packages = detail::GraphCompilerAccess::schema_packages(graph, *snapshot);
  state->executable_factories = detail::GraphCompilerAccess::executable_factories(graph, *snapshot);

  // Lower profile-independent facts even when profile admission rejects. A
  // failed profile must not erase an otherwise inspectable canonical plan.
  const std::vector<std::uint32_t> unresolved_lanes(snapshot->vertices.size(), 0U);
  auto partition = detail::lower_partition_plan(*snapshot,
                                                options.profile,
                                                profile.effective_profiles(),
                                                captured_view,
                                                unresolved_lanes,
                                                process.process_group_by_vertex,
                                                *state);
  if (!partition) {
    append_error(*state, partition.error());
  } else if (state->partitioning_plan.maturity == AlgorithmMaturity::kExperimental) {
    append_diagnostic(*state, "PARTITION_PLANNER_EXPERIMENTAL_SELECTED");
  }

  auto deployment = detail::lower_deployment_plan(
      *snapshot, options.deployment, profile.effective_profiles(), state->execution_partitions);
  for (const auto& token : deployment.error_tokens) {
    append_diagnostic(*state, token);
  }
  for (const auto& token : deployment.warning_tokens) {
    append_diagnostic(*state, token, {}, holoscan::DiagnosticSeverity::kWarning);
  }
  const ParamSchemaHash parameter_admission_context = detail::parameter_admission_context_digest(
      state->profile_manifest, deployment, detail::empty_parameter_migration_catalog_digest());
  state->placement_probes = std::move(deployment.placement_probes);
  state->partition_placements = std::move(deployment.placements);
  state->execution_lanes = std::move(deployment.lanes);
  state->scheduler_plan.execution_lane_count =
      static_cast<std::uint32_t>(state->execution_lanes.size());
  const std::uint32_t partition_owner_count = callback_partition_owner_count(*state);
  state->scheduler_plan.max_concurrent_activations = partition_owner_count;
  state->scheduler_plan.activation_overlap = partition_owner_count > 1U;
  state->scheduler_plan.worker_count = partition_owner_count;
  state->scheduler_plan.intra_path_parallelism = partition_owner_count == 0U ? 0U : 1U;

  auto clocks_and_readiness =
      detail::lower_clocks_and_readiness(*snapshot, plan_generation, profile.effective_profiles());
  if (!clocks_and_readiness) {
    append_error(*state, clocks_and_readiness.error());
  } else {
    publish_clocks_and_readiness(std::move(clocks_and_readiness).value(), *state);
  }

  // Resolve generation zero before replaying temporal declarations. An
  // accepted override set replaces only isolated factory clones, so every
  // later setup(), contract(), param_ref(), and lifecycle pass observes the
  // same exact resolved generation without mutating the source Graph.
  auto parameters = detail::lower_parameter_plan(*snapshot,
                                                 state->executable_factories,
                                                 profile.effective_profiles(),
                                                 plan_generation,
                                                 options.parameters,
                                                 options.lifecycle_transactions,
                                                 parameter_admission_context,
                                                 options.parameter_overrides);
  publish_temporal(detail::lower_temporal_plan(state->executable_factories,
                                               *snapshot,
                                               plan_generation,
                                               state->clocks,
                                               state->clock_conversions,
                                               state->readiness,
                                               profile.effective_profiles(),
                                               options.parameters,
                                               package_capabilities),
                   *state);
  auto partition_roots = detail::validate_partition_execution_roots(*snapshot, *state);
  if (!partition_roots) {
    append_error(*state, partition_roots.error());
  }
  // Temporal replay marks the readiness rows actually selected by each
  // contract. Validate busy-poll placement only after that replay so an
  // unreferenced SpinSource remains inert rather than rejecting the plan.
  for (const auto& source : state->readiness) {
    if (!source.referenced || source.kind != ReadinessSourceKind::kSpin || !source.owner ||
        source.owner.value > deployment.lane_by_vertex.size()) {
      continue;
    }
    const std::uint32_t lane =
        deployment.lane_by_vertex[static_cast<std::size_t>(source.owner.value - 1U)];
    const auto accepted_lane = std::find_if(
        state->execution_lanes.begin(), state->execution_lanes.end(), [lane](const auto& item) {
          return item.execution_lane == lane;
        });
    if (accepted_lane == state->execution_lanes.end() ||
        accepted_lane->accepted_placement.wake_mode != WakeMode::kBusyPoll) {
      append_diagnostic(*state, "SPIN_UNPLACED", DiagnosticSource{.vertex = source.owner});
    }
  }
  for (const auto& source : state->readiness) {
    if (!source.referenced) {
      append_diagnostic(*state,
                        "READINESS_SOURCE_UNUSED",
                        DiagnosticSource{.vertex = source.owner},
                        holoscan::DiagnosticSeverity::kWarning);
    }
  }

  // Parameter queue/storage facts are available before lifecycle lowering,
  // but exact fence/quiesce/rollback minima require dependency-expanded
  // restart closures. Build those closures once, finalize Parameter capacity,
  // then rebuild the lifecycle result from the final admitted Parameter set so
  // rejected capacity cannot leave stale parameter-origin restart authority.
  auto lifecycle = detail::lower_lifecycle_plan(*snapshot,
                                                profile.effective_profiles(),
                                                parameters.lifecycle_mutable_fields,
                                                parameters.capacity,
                                                options.lifecycle_transactions,
                                                plan_generation);
  detail::finalize_parameter_lifecycle_capacity(
      parameters, lifecycle, options.lifecycle_transactions);
  lifecycle = detail::lower_lifecycle_plan(*snapshot,
                                           profile.effective_profiles(),
                                           parameters.lifecycle_mutable_fields,
                                           parameters.capacity,
                                           options.lifecycle_transactions,
                                           plan_generation);
  for (const auto& diagnostic : parameters.diagnostics) {
    append_diagnostic(*state, diagnostic.token, diagnostic.source);
  }
  state->operator_parameters = std::move(parameters.operators);
  state->parameter_fields = std::move(parameters.fields);
  state->parameter_catalog = std::move(parameters.catalog);
  state->parameter_generation_zero_operators = std::move(parameters.generation_zero_operators);
  state->parameter_generation_zero_fields = std::move(parameters.generation_zero_fields);
  state->parameter_runtime_capacity = parameters.capacity;

  publish_lifecycle(std::move(lifecycle), *state);
  auto endpoint_catalog = detail::lower_endpoint_catalog(*snapshot,
                                                         options.deployment,
                                                         profile.effective_profiles(),
                                                         options.endpoint_catalog,
                                                         state->clocks,
                                                         state->clock_conversions);
  const auto* endpoint_catalog_state =
      detail::ChannelEndpointCatalogAccess::state(endpoint_catalog.catalog);
  const auto catalog_endpoints =
      endpoint_catalog_state != nullptr
          ? std::span<const detail::ChannelEndpointRecord>{endpoint_catalog_state->endpoints}
          : std::span<const detail::ChannelEndpointRecord>{};
  auto tensor_device_bindings =
      resolve_tensor_device_placements(*snapshot, options.deployment, *state);
  tensor_device_bindings.insert(tensor_device_bindings.end(),
                                evidence.tensor_device_bindings.begin(),
                                evidence.tensor_device_bindings.end());
  publish_physical(detail::lower_physical_plan(*snapshot,
                                               plan_generation,
                                               profile.effective_profiles(),
                                               package_capabilities,
                                               state->input_selections,
                                               state->bracketed_windows,
                                               catalog_endpoints,
                                               state->vertex_process_placements,
                                               state->lifecycle_restart_targets,
                                               state->execution_partitions,
                                               tensor_device_bindings,
                                               evidence.force_holomq_process_local_inline),
                   *state);
  detail::apply_process_edge_locality(
      process.locality_by_edge, state->payload_edges, state->channel_locality);
  state->closed_island_activations = detail::derive_closed_island_activation_plan(*state);
  const auto clock_health_qualifications =
      resolve_clock_health_qualifications(evidence, endpoint_catalog_state, *state);
  publish_endpoint_clock_health(detail::lower_endpoint_clock_health(*snapshot,
                                                                    plan_generation,
                                                                    state->clocks,
                                                                    state->clock_conversions,
                                                                    profile.effective_profiles(),
                                                                    state->input_selections,
                                                                    clock_health_qualifications),
                                *state);
  publish_catalog(std::move(endpoint_catalog), *state);

  auto manifest = detail::finalize_profile_manifest(options, *state, package_capabilities);
  if (!manifest) {
    append_error(*state, manifest.error());
  }

  state->admitted = profile_accepted && !has_error_diagnostic(*state);
  if (auto runners = detail::derive_partition_runner_plan(*state); runners) {
    state->partition_runners = std::move(*runners);
  } else {
    append_diagnostic(*state, "PARTITION_RESOURCE_REALIZATION_FAILED");
    state->admitted = false;
    state->partition_runners.clear();
  }
  state->inline_partition_execution = detail::derive_inline_partition_execution_plan(*state);
  auto canonicalized = detail::finalize_plan_artifact(*state);
  if (!canonicalized) {
    // Artifact construction is a cold deterministic pass. Preserve a total
    // rejected result whenever its failure still permits bounded state.
    append_error(*state, canonicalized.error());
    state->admitted = false;
    state->partition_runners.clear();
    state->inline_partition_execution.reset();
    auto retry = detail::finalize_plan_artifact(*state);
    if (!retry) {
      throw RuntimeError(retry.error());
    }
  }

  const PlanGenerationId resolved_generation = detail::derive_plan_generation_id(*state);
  detail::rebind_plan_generation(*state, resolved_generation);
  auto finalized = detail::finalize_plan_artifact(*state);
  if (!finalized) {
    // Rebinding changes only fixed-width provenance values, so failure here is
    // a process-resource failure rather than an admission disposition.
    throw RuntimeError(finalized.error());
  }

  HOLOSCAN_LOG_TRACE("Graph compile completed: admitted={} generation={} diagnostics={}",
                     state->admitted,
                     state->graph_identity.version,
                     state->diagnostics.size());
  return detail::ExecutionPlanAccess::make(std::move(state));
}
// NOLINTEND(performance-unnecessary-value-param,readability-function-cognitive-complexity)

ExecutionPlan detail::compile_with_runtime_evidence(Graph& graph, CompileOptions options,
                                                    detail::RuntimeCompileEvidenceView evidence) {
  return detail::compile_with_runtime_evidence_and_process_placement(
      graph, std::move(options), std::move(evidence), {});
}

ExecutionPlan detail::compile_with_process_placement(
    Graph& graph, CompileOptions options,
    detail::ProcessPlacementBinding placement) {  // NOLINT(performance-unnecessary-value-param)
  return detail::compile_with_runtime_evidence_and_process_placement(
      graph, std::move(options), detail::runtime_compile_evidence_view(), std::move(placement));
}

ExecutionPlan compile(Graph& graph,
                      CompileOptions options) {  // NOLINT(performance-unnecessary-value-param)
  // Capture the generated provider view exactly once before any compiler pass.
  return detail::compile_with_runtime_evidence(
      graph, std::move(options), detail::runtime_compile_evidence_view());
}

ExecutionPlan detail::compile_with_process_local_holomq(
    Graph& graph, CompileOptions options) {  // NOLINT(performance-unnecessary-value-param)
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.force_holomq_process_local_inline = true;
  return detail::compile_with_runtime_evidence(graph, std::move(options), std::move(evidence));
}

}  // namespace holoscan
