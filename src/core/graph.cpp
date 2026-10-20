/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Graph authoring. This file holds the helpers shared by the Graph sources (declared in
// graph_internal.hpp), the GraphState mutation transaction and its rollback, parameter input,
// lifecycle ordering, clocks, and the members of the public Graph, GraphView, and handle types.
// The other responsibilities live in graph_operators.cpp (Operator registration, lookup, and
// parameters), graph_flows.cpp (flows, interface bindings, port and flow configuration, and
// external topics), graph_composition.cpp (embedded Graphs, regions, and partitions), and
// graph_snapshot.cpp (the authoring hash and the published snapshot).

#include <holoscan/core/graph.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/time/synthetic_clock.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "canonical_text_encoding.hpp"
#include "core/diagnostic_reporting.hpp"
#include "diagnostic_context.hpp"
#include "graph_internal.hpp"
#include "graph_mutation_rollback.hpp"
#include "graph_state.hpp"
#include "parameter_authoring_target.hpp"
#include "parameter_input_internal.hpp"
#include "plan_record_contracts.hpp"

namespace holoscan {
namespace detail {

[[nodiscard]] std::string authored_operator_path(const GraphState& state,
                                                 const VertexRecord& vertex) {
  const auto graph =
      std::ranges::find(state.embedded_graph_records_, vertex.graph, &GraphRecord::id);
  const GraphRecord* const record =
      vertex.graph == state.graph_record_.id
          ? &state.graph_record_
          : (graph == state.embedded_graph_records_.end() ? nullptr : &*graph);
  if (record == nullptr)
    return {};
  std::string path;
  const auto append = [&](std::string_view segment) {
    if (!path.empty())
      path.push_back('/');
    append_percent_encoded_uri_component(path, segment);
  };
  if (record->path) {
    for (const auto& segment : record->path->segments())
      append(segment);
  }
  append(vertex.name);
  return path;
}

expected<ParameterAuthoringInventory, Error> ParameterInputAccess::authoring_targets(Graph& graph) {
  try {
    const std::shared_ptr<GraphState>& state = graph.state_;
    auto prepared = state->prepare_mutation();
    if (!prepared) {
      return make_unexpected(std::move(prepared.error()));
    }
    const std::lock_guard lock(state->mutex_);
    ParameterAuthoringInventory inventory{.root_graph = state->graph_record_.id};
    inventory.graphs.reserve(1U + state->embedded_graph_records_.size());
    inventory.targets.reserve(state->operators_.size() + state->embedded_operator_mappings_.size());
    inventory.graphs.push_back(ParameterAuthoringGraph{.graph = state->graph_record_.id});
    for (const GraphRecord& record : state->embedded_graph_records_) {
      std::string path;
      if (record.path) {
        for (const auto& segment : record.path->segments()) {
          if (!path.empty())
            path.push_back('/');
          append_percent_encoded_uri_component(path, segment);
        }
      }
      inventory.graphs.push_back(
          ParameterAuthoringGraph{.graph = record.id, .path = std::move(path)});
    }
    const auto append = [&](VertexId target, const std::shared_ptr<OperatorFactoryBase>& factory) {
      const auto vertex = std::ranges::find(state->vertices_, target, &VertexRecord::id);
      if (vertex == state->vertices_.end() || !factory) {
        throw RuntimeError{Error{ErrorCode::kFailure, "PARAM_REBIND_TARGET_OPERATOR_INVALID"}};
      }
      const ParamTypeDescriptor& type = factory->parameter_type();
      inventory.targets.push_back(ParameterAuthoringTarget{
          .vertex = target,
          .graph = vertex->graph,
          .path = authored_operator_path(*state, *vertex),
          .schema = factory->parameter_schema(),
          .type = &type,
      });
    };
    for (const auto& op : state->operators_)
      append(op.vertex, op.factory);
    for (const auto& mapping : state->embedded_operator_mappings_)
      append(mapping.vertex, mapping.factory);
    return inventory;
  } catch (const RuntimeError& error) {
    return make_unexpected(make_error(error));
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted, "PARAM_REBIND_ALLOC_FAILED"});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure, "PARAM_REBIND_TARGET_CAPTURE_FAILED"});
  }
}

expected<void, Error> ParameterInputAccess::attach_transaction(
    Graph& graph, std::shared_ptr<const ParameterTransactionState> transaction) {
  if (!transaction || transaction->operators.empty()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "GRAPH_PARAMETER_TRANSACTION_INVALID"});
  }
  const std::shared_ptr<GraphState>& state = graph.state_;
  auto prepared = state->prepare_mutation();
  if (!prepared) {
    return make_unexpected(std::move(prepared.error()));
  }
  const std::lock_guard lock(state->mutex_);
  if (state->parameter_transaction_) {
    return make_unexpected(Error{ErrorCode::kDuplicateName, "GRAPH_PARAMETER_TRANSACTION_EXISTS"});
  }
  state->parameter_transaction_ = std::move(transaction);
  state->mark_mutation();
  return {};
}

[[nodiscard]] ClockCapabilityAuthoringRecord clock_capabilities(const Clock& clock, GraphId graph,
                                                                const ClockKey& key) {
  ClockCapabilityAuthoringRecord result{
      .graph = graph, .key = key, .can_read_now = true, .can_wait_until = true};
  if (dynamic_cast<const RealtimeClock*>(&clock) != nullptr) {
    result.steady = true;
    result.independent_progress = true;
    result.finite_deadline = true;
    result.evidence_id = "holoscan.clock.realtime.v1";
  } else if (dynamic_cast<const ManualClock*>(&clock) != nullptr) {
    result.steady = true;
    result.evidence_id = "holoscan.clock.manual.external-progress-required.v1";
  } else if (dynamic_cast<const SyntheticClock*>(&clock) != nullptr) {
    result.steady = true;
    result.evidence_id = "holoscan.clock.synthetic.external-progress-required.v1";
  } else {
    // Clock's base interface proves readable/waitable operations, but it does
    // not prove monotonic or independently progressing deadline behavior.
    result.evidence_id = "holoscan.clock.custom.unqualified.v1";
  }
  return result;
}

Error graph_factory_error(std::string_view token, DiagnosticCapture* capture) noexcept {
  if (capture == nullptr) {
    return graph_factory_error(token);
  }
  capture->capture_exception();
  return Error{capture->causes[0].code};
}

Error graph_factory_error(std::string_view token) noexcept {
  DiagnosticCapture capture;
  capture.capture_exception();
  try {
    const auto evidence = capture.detail(DiagnosticOperation::kConstruct);
    std::string message(token);
    if (!evidence.error.message.empty()) {
      if (!message.empty()) {
        message += ": ";
      }
      message += evidence.error.message;
    }
    for (const auto& cause : evidence.causes) {
      message += "; caused by " + runtime_error_message(cause);
    }
    if (capture.truncated) {
      message += " [truncated]";
    }
    return Error{evidence.error.code, std::move(message)};
  } catch (...) {
    return Error{capture.count != 0U ? capture.causes[0].code : ErrorCode::kFailure};
  }
}

void graph_log_factory_exception(std::string_view context, std::string_view detail) noexcept {
  try {
    HOLOSCAN_LOG_DEBUG(
        "Graph {} factory exception detail: {}", context, diagnostic_display_text(detail));
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Logging is diagnostic-only and must never replace the stable typed factory failure.
  }
}

GraphMutationRollback::GraphMutationRollback(GraphState& state, CaptureExistingRecords)
    : GraphMutationRollback(state) {
  existing_records_ = std::make_unique<ExistingRecords>(state);
}

GraphMutationRollback::~GraphMutationRollback() noexcept {
  rollback();
}

void GraphMutationRollback::rollback() noexcept {
  if (committed_)
    return;
  if (existing_records_) {
    rollback_complete_mutation();
    assert(state_.dense_ids_valid());
    committed_ = true;
    return;
  }
  rollback_ports();
  rollback_name_indexes();
  rollback_adjacency();
  rollback_embedded_names();
  truncate_records();
  restore_identity();
  assert(state_.dense_ids_valid());
  committed_ = true;
}

void GraphMutationRollback::rollback_complete_mutation() noexcept {
  release_root_port_ownership();

  state_.embedded_graphs_.erase(
      state_.embedded_graphs_.begin() + static_cast<std::ptrdiff_t>(embedded_graphs_size_),
      state_.embedded_graphs_.end());
  state_.external_topics_.resize(external_topics_size_);

  ExistingRecords& existing = *existing_records_;
  state_.embedded_graph_records_.swap(existing.embedded_graph_records);
  state_.embedded_operator_mappings_.swap(existing.embedded_operator_mappings);
  state_.operators_.swap(existing.operators);
  state_.vertex_adjacency_.swap(existing.vertex_adjacency);
  state_.cycle_visit_marks_.swap(existing.cycle_visit_marks);
  state_.cycle_visit_epoch_ = existing.cycle_visit_epoch;
  state_.vertices_.swap(existing.vertices);
  state_.ports_.swap(existing.ports);
  state_.port_by_declaration_.swap(existing.port_by_declaration);
  state_.edges_.swap(existing.edges);
  state_.edge_handle_identities_.swap(existing.edge_handle_identities);
  state_.interface_bindings_.swap(existing.interface_bindings);
  state_.readiness_sources_.swap(existing.readiness_sources);
  state_.regions_.swap(existing.regions);
  state_.region_handle_identities_.swap(existing.region_handle_identities);
  state_.partitions_.swap(existing.partitions);
  state_.partition_handle_identities_.swap(existing.partition_handle_identities);
  state_.automatic_partition_remaining_ = existing.automatic_partition_remaining;
  state_.clocks_.swap(existing.clocks);
  state_.clock_imports_.swap(existing.clock_imports);
  state_.clock_bindings_.swap(existing.clock_bindings);
  state_.clock_conversions_.swap(existing.clock_conversions);
  state_.lifecycle_dependencies_.swap(existing.lifecycle_dependencies);
  state_.lifecycle_callbacks_.swap(existing.lifecycle_callbacks);
  state_.lifecycle_facts_.swap(existing.lifecycle_facts);
  state_.lifecycle_restart_permissions_.swap(existing.lifecycle_restart_permissions);
  state_.vertex_names_.swap(existing.vertex_names);
  state_.interface_port_names_.swap(existing.interface_port_names);
  state_.region_names_.swap(existing.region_names);
  state_.partition_names_.swap(existing.partition_names);
  state_.clock_names_.swap(existing.clock_names);
  state_.explicit_endpoint_names_.swap(existing.explicit_endpoint_names);
  state_.embedded_instance_names_.swap(existing.embedded_instance_names);
  state_.pod_identity_catalog_.swap(existing.pod_identity_catalog);
  std::swap(state_.default_clock_, existing.default_clock);
  state_.parameter_transaction_ = parameter_transaction_;
  restore_identity();
  claim_root_port_ownership();
}

void GraphMutationRollback::release_root_port_ownership() noexcept {
  for (const auto& port : state_.ports_) {
    if (port.declaration == nullptr || !is_root_vertex(state_, port.record.owner))
      continue;
    if (port.record.direction == PortDirection::kInput) {
      const auto& input = *static_cast<const InputBase*>(port.declaration);
      if (GraphPortOwnershipAccess::owner(input) == &state_)
        GraphPortOwnershipAccess::claim(input, nullptr);
    } else {
      const auto& output = *static_cast<const OutputBase*>(port.declaration);
      if (GraphPortOwnershipAccess::owner(output) == &state_)
        GraphPortOwnershipAccess::claim(output, nullptr);
    }
  }
}

void GraphMutationRollback::claim_root_port_ownership() noexcept {
  for (const auto& port : state_.ports_) {
    if (port.declaration == nullptr || !is_root_vertex(state_, port.record.owner))
      continue;
    if (port.record.direction == PortDirection::kInput) {
      GraphPortOwnershipAccess::claim(*static_cast<const InputBase*>(port.declaration), &state_);
    } else {
      GraphPortOwnershipAccess::claim(*static_cast<const OutputBase*>(port.declaration), &state_);
    }
  }
}

void GraphMutationRollback::rollback_ports() noexcept {
  for (std::size_t index = ports_size_; index < state_.ports_.size(); ++index) {
    const auto& port = state_.ports_[index];
    erase_declaration(state_.port_by_declaration_, port.declaration);
    const bool root_owned = is_root_vertex(state_, port.record.owner);
    const bool parent_owned = port.declaration != nullptr && root_owned;
    if (root_owned && port.record.owner_kind == PortOwnerKind::kGraphInterface) {
      erase_name(state_.interface_port_names_, port.record.name);
    }
    if (!parent_owned)
      continue;
    if (port.record.direction == PortDirection::kInput) {
      GraphPortOwnershipAccess::claim(*static_cast<const InputBase*>(port.declaration), nullptr);
    } else {
      GraphPortOwnershipAccess::claim(*static_cast<const OutputBase*>(port.declaration), nullptr);
    }
  }
}

void GraphMutationRollback::rollback_name_indexes() noexcept {
  for (std::size_t index = vertices_size_; index < state_.vertices_.size(); ++index) {
    const auto& vertex = state_.vertices_[index];
    if (vertex.graph == state_.graph_record_.id && vertex.kind == VertexKind::kOperator) {
      erase_name(state_.vertex_names_, vertex.name);
    }
  }
  for (std::size_t index = regions_size_; index < state_.regions_.size(); ++index) {
    if (state_.regions_[index].graph == state_.graph_record_.id) {
      erase_name(state_.region_names_, state_.regions_[index].name);
    }
  }
  for (std::size_t index = partitions_size_; index < state_.partitions_.size(); ++index) {
    if (state_.partitions_[index].graph == state_.graph_record_.id) {
      erase_name(state_.partition_names_, state_.partitions_[index].name);
    }
  }
  for (std::size_t index = clocks_size_; index < state_.clocks_.size(); ++index) {
    if (state_.clocks_[index].record.graph == state_.graph_record_.id) {
      erase_name(state_.clock_names_, state_.clocks_[index].record.key.name());
    }
  }
  for (std::size_t index = clock_imports_size_; index < state_.clock_imports_.size(); ++index) {
    if (state_.clock_imports_[index].graph == state_.graph_record_.id) {
      erase_name(state_.clock_names_, state_.clock_imports_[index].key.name());
    }
  }
  for (std::size_t index = external_topics_size_; index < state_.external_topics_.size(); ++index) {
    const auto& endpoint_name = state_.external_topics_[index].record.endpoint_name;
    if (endpoint_name) {
      erase_name(state_.explicit_endpoint_names_, *endpoint_name);
    }
  }
}

void GraphMutationRollback::rollback_adjacency() noexcept {
  for (std::size_t index = state_.interface_bindings_.size(); index > interface_bindings_size_;
       --index) {
    remove_binding_adjacency(state_, state_.interface_bindings_[index - 1U]);
  }
  for (std::size_t index = state_.edges_.size(); index > edges_size_; --index) {
    remove_edge_adjacency(state_, state_.edges_[index - 1U]);
  }
}

void GraphMutationRollback::rollback_embedded_names() noexcept {
  for (std::size_t index = embedded_graph_records_size_;
       index < state_.embedded_graph_records_.size();
       ++index) {
    const auto& graph = state_.embedded_graph_records_[index];
    if (graph.parent == state_.graph_record_.id) {
      erase_name(state_.embedded_instance_names_, graph.name);
    }
  }
}

void GraphMutationRollback::truncate_records() noexcept {
  state_.interface_bindings_.resize(interface_bindings_size_);
  state_.edges_.resize(edges_size_);
  state_.edge_handle_identities_.resize(edges_size_);
  state_.readiness_sources_.resize(readiness_sources_size_);
  state_.clock_bindings_.resize(clock_bindings_size_);
  state_.clock_imports_.resize(clock_imports_size_);
  state_.clock_conversions_.resize(clock_conversions_size_);
  state_.lifecycle_dependencies_.resize(lifecycle_dependencies_size_);
  state_.lifecycle_callbacks_.resize(lifecycle_callbacks_size_);
  state_.lifecycle_facts_.resize(lifecycle_facts_size_);
  state_.lifecycle_restart_permissions_.resize(lifecycle_permissions_size_);
  state_.ports_.resize(ports_size_);
  state_.vertices_.resize(vertices_size_);
  state_.external_topics_.resize(external_topics_size_);
  state_.operators_.resize(operators_size_);
  state_.embedded_operator_mappings_.resize(embedded_operator_mappings_size_);
  state_.embedded_graph_records_.resize(embedded_graph_records_size_);
  state_.embedded_graphs_.erase(
      state_.embedded_graphs_.begin() + static_cast<std::ptrdiff_t>(embedded_graphs_size_),
      state_.embedded_graphs_.end());
  state_.parameter_transaction_ = parameter_transaction_;
  state_.vertex_adjacency_.resize(vertices_size_);
  state_.cycle_visit_marks_.resize(vertices_size_);
  state_.regions_.resize(regions_size_);
  state_.region_handle_identities_.resize(regions_size_);
  state_.partitions_.resize(partitions_size_);
  state_.partition_handle_identities_.resize(partitions_size_);
  state_.clocks_.resize(clocks_size_);
  state_.pod_identity_catalog_.resize(pod_identity_catalog_size_);
}

void GraphMutationRollback::restore_identity() noexcept {
  state_.generation_ = generation_;
  state_.next_vertex_id_ = next_vertex_id_;
  state_.next_graph_id_ = next_graph_id_;
  state_.next_port_id_ = next_port_id_;
  state_.next_edge_id_ = next_edge_id_;
  state_.next_region_id_ = next_region_id_;
  state_.next_partition_id_ = next_partition_id_;
  state_.next_external_topic_id_ = next_external_topic_id_;
  state_.next_boundary_ordinal_ = next_boundary_ordinal_;
}

bool GraphMutationRollback::is_root_vertex(const GraphState& state, VertexId id) noexcept {
  if (!id || id.value > state.vertices_.size())
    return false;
  const auto& vertex = state.vertices_[static_cast<std::size_t>(id.value - 1U)];
  return vertex.id == id && vertex.graph == state.graph_record_.id;
}

void GraphMutationRollback::remove_adjacency(GraphState& state, VertexId source,
                                             VertexId target) noexcept {
  if (!source || !target || source.value > state.vertex_adjacency_.size())
    return;
  auto& outgoing = state.vertex_adjacency_[static_cast<std::size_t>(source.value - 1U)];
  const std::size_t target_index = static_cast<std::size_t>(target.value - 1U);
  if (!outgoing.empty() && outgoing.back() == target_index) {
    outgoing.pop_back();
  }
}

const PortStorage* GraphMutationRollback::rollback_port(const GraphState& state,
                                                        GraphPortId id) noexcept {
  if (!id || id.value > state.ports_.size())
    return nullptr;
  const auto& port = state.ports_[static_cast<std::size_t>(id.value - 1U)];
  return port.record.id == id ? &port : nullptr;
}

void GraphMutationRollback::remove_edge_adjacency(GraphState& state,
                                                  const EdgeRecord& edge) noexcept {
  const PortStorage* source = rollback_port(state, edge.source);
  const PortStorage* target = rollback_port(state, edge.target);
  if (source != nullptr && target != nullptr) {
    remove_adjacency(state, source->record.owner, target->record.owner);
  }
}

void GraphMutationRollback::remove_binding_adjacency(
    GraphState& state, const InterfaceBindingRecord& binding) noexcept {
  const PortStorage* public_port = rollback_port(state, binding.public_port);
  const PortStorage* internal_port = rollback_port(state, binding.internal_port);
  if (public_port == nullptr || internal_port == nullptr)
    return;
  if (binding.direction == PortDirection::kInput) {
    remove_adjacency(state, public_port->record.owner, internal_port->record.owner);
  } else {
    remove_adjacency(state, internal_port->record.owner, public_port->record.owner);
  }
}

void GraphMutationRollback::erase_declaration(
    std::unordered_map<const void*, GraphPortId>& declarations, const void* declaration) noexcept {
  const auto found = std::find_if(declarations.begin(), declarations.end(), [&](const auto& item) {
    return item.first == declaration;
  });
  if (found != declarations.end()) {
    declarations.erase(found);
  }
}

void GraphMutationRollback::erase_name(std::unordered_set<std::string>& names,
                                       std::string_view name) noexcept {
  const auto found = std::find_if(
      names.begin(), names.end(), [&](const std::string& item) { return item == name; });
  if (found != names.end()) {
    names.erase(found);
  }
}

namespace {

std::atomic<std::uint64_t> g_next_graph_instance{1U};

}  // namespace

[[nodiscard]] Error graph_error(ErrorCode code, std::string_view token) {
  return Error{code, std::string(token)};
}

[[nodiscard]] std::string graph_payload_identity_name(const PayloadIdentityRecord& identity) {
  if (identity.schema) {
    return fmt::format("{} epoch={} family_id={} file_identifier={} transport_profile_tag={}",
                       identity.schema->canonical_name,
                       identity.schema->compatibility_epoch,
                       lowercase_hex_text(identity.schema->family_id.value, HexPrefix::kNone),
                       std::string_view{identity.schema->file_identifier.value.data(),
                                        identity.schema->file_identifier.value.size()},
                       identity.transport_profile_tag);
  }
  if (identity.pod) {
    return fmt::format("{} v{} size={} alignment={}",
                       identity.pod->canonical_name,
                       identity.pod->version,
                       identity.pod->object_size,
                       identity.pod->object_alignment);
  }
  if (identity.scalar) {
    constexpr std::array names{"bool",
                               "uint8",
                               "int8",
                               "uint16",
                               "int16",
                               "uint32",
                               "int32",
                               "uint64",
                               "int64",
                               "float32",
                               "float64",
                               "local enum"};
    const auto index = static_cast<std::size_t>(identity.scalar->kind);
    if (index < names.size()) {
      return std::string{names[index]};
    }
  }
  return identity.kind == PayloadIdentityKind::kTensor ? std::string{"Tensor"}
                                                       : std::string{"unknown payload"};
}

namespace {

[[nodiscard]] std::string bounded_exception_context(std::string_view context) {
  constexpr std::size_t kMaximumContextBytes = 160U;
  const std::size_t count = std::min(context.size(), kMaximumContextBytes);
  std::string result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const unsigned char value = static_cast<unsigned char>(context[index]);
    result.push_back(value >= 0x20U && value != 0x7FU ? static_cast<char>(value) : '?');
  }
  return result;
}

}  // namespace

[[nodiscard]] bool edge_whitespace(std::string_view value) noexcept {
  if (value.empty())
    return false;
  const auto decode = [&](std::size_t index) {
    const auto first = static_cast<unsigned char>(value[index]);
    if (first < 0x80U)
      return static_cast<std::uint32_t>(first);
    const std::size_t remaining = (first & 0xE0U) == 0xC0U   ? 1U
                                  : (first & 0xF0U) == 0xE0U ? 2U
                                                             : 3U;
    std::uint32_t codepoint = first & (remaining == 1U ? 0x1FU : remaining == 2U ? 0x0FU : 0x07U);
    for (std::size_t offset = 1; offset <= remaining; ++offset) {
      codepoint = (codepoint << 6U) | (static_cast<unsigned char>(value[index + offset]) & 0x3FU);
    }
    return codepoint;
  };
  const auto whitespace = [](std::uint32_t codepoint) {
    return (codepoint >= 0x0009U && codepoint <= 0x000DU) || codepoint == 0x0020U ||
           codepoint == 0x0085U || codepoint == 0x00A0U || codepoint == 0x1680U ||
           (codepoint >= 0x2000U && codepoint <= 0x200AU) || codepoint == 0x2028U ||
           codepoint == 0x2029U || codepoint == 0x202FU || codepoint == 0x205FU ||
           codepoint == 0x3000U;
  };
  std::size_t last = value.size() - 1U;
  while (last > 0U && (static_cast<unsigned char>(value[last]) & 0xC0U) == 0x80U) {
    --last;
  }
  return whitespace(decode(0)) || whitespace(decode(last));
}

[[nodiscard]] expected<void, Error> validate_name(std::string_view name, std::string_view token,
                                                  bool allow_empty) {
  if ((!allow_empty && name.empty()) || name.size() > 255U || !valid_utf8(name) ||
      edge_whitespace(name) || std::any_of(name.begin(), name.end(), [](char value) {
        const auto byte = static_cast<unsigned char>(value);
        return byte == 0U || byte < 0x20U || byte == 0x7FU;
      })) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, token));
  }
  return {};
}

namespace {

[[nodiscard]] std::uint64_t pod_scalar_size(PodScalarKind kind) noexcept {
  switch (kind) {
    case PodScalarKind::kByte:
    case PodScalarKind::kUInt8:
    case PodScalarKind::kInt8:
      return 1;
    case PodScalarKind::kUInt16:
    case PodScalarKind::kInt16:
      return 2;
    case PodScalarKind::kUInt32:
    case PodScalarKind::kInt32:
    case PodScalarKind::kFloat32:
      return 4;
    case PodScalarKind::kUInt64:
    case PodScalarKind::kInt64:
    case PodScalarKind::kFloat64:
      return 8;
  }
  return 0;
}

}  // namespace

[[nodiscard]] expected<void, Error> validate_schema(const SchemaIdentityRecord& schema) {
  if (!valid_schema_name(schema.canonical_name)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_SCHEMA_NAME_INVALID"));
  }
  if (!valid_schema_identity(schema)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_SCHEMA_FAMILY_INVALID"));
  }
  return {};
}

namespace {

[[nodiscard]] expected<void, Error> validate_pod(const PodIdentityRecord& pod) {
  auto name = validate_name(pod.canonical_name, "GRAPH_POD_NAME_INVALID");
  if (!name)
    return name;
  if (pod.version == 0U || pod.object_size == 0U || pod.object_alignment == 0U ||
      (pod.object_alignment & (pod.object_alignment - 1U)) != 0U || pod.fields.empty()) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_POD_HEADER_INVALID"));
  }
  std::unordered_set<std::string> field_names;
  std::uint64_t previous_end = 0;
  for (const auto& field : pod.fields) {
    auto field_name = validate_name(field.name, "GRAPH_POD_FIELD_NAME_INVALID");
    if (!field_name)
      return field_name;
    if (!field_names.emplace(field.name).second || field.element_count == 0U) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_POD_FIELD_INVALID"));
    }
    const std::uint64_t element_size = pod_scalar_size(field.scalar_kind);
    const std::uint64_t byte_count = element_size * field.element_count;
    const std::uint64_t end = static_cast<std::uint64_t>(field.byte_offset) + byte_count;
    if (element_size == 0U || end < field.byte_offset || end > pod.object_size ||
        field.byte_offset != previous_end) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_POD_LAYOUT_INVALID"));
    }
    previous_end = end;
  }
  if (previous_end != pod.object_size) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_POD_LAYOUT_INVALID"));
  }
  return {};
}

}  // namespace

[[nodiscard]] expected<void, Error> validate_payload_identity(
    const PayloadIdentityRecord& identity) {
  if (!closed_enum(identity.kind) || !closed_enum(identity.payload_kind)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PAYLOAD_IDENTITY_ENUM_INVALID"));
  }
  switch (identity.kind) {
    case PayloadIdentityKind::kScalar: {
      const bool valid_native_size =
          identity.scalar && (identity.scalar->kind == ScalarIdentityKind::kLocalEnum
                                  ? identity.scalar->native_object_size == 1U ||
                                        identity.scalar->native_object_size == 2U ||
                                        identity.scalar->native_object_size == 4U ||
                                        identity.scalar->native_object_size == 8U
                                  : identity.scalar->native_object_size == 0U);
      if (identity.transport_profile_tag != 0U || !identity.scalar ||
          !closed_enum(identity.scalar->kind) || identity.schema || identity.pod ||
          !valid_native_size) {
        return make_unexpected(
            graph_error(ErrorCode::kInvalidArgument, "GRAPH_PAYLOAD_IDENTITY_INVALID"));
      }
      break;
    }
    case PayloadIdentityKind::kTensor:
      if (identity.transport_profile_tag != 0U || identity.scalar || identity.schema ||
          identity.pod || identity.payload_kind != PayloadKind::kDescriptorBacked) {
        return make_unexpected(
            graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_IDENTITY_INVALID"));
      }
      break;
    case PayloadIdentityKind::kSchema: {
      const bool supported_schema_payload =
          identity.payload_kind == PayloadKind::kSerializedBytes ||
          identity.payload_kind == PayloadKind::kDescriptorBacked ||
          (identity.payload_kind == PayloadKind::kInlineTyped &&
           is_py_object_payload_identity(identity));
      if (identity.scalar || !identity.schema || identity.pod || !supported_schema_payload) {
        return make_unexpected(
            graph_error(ErrorCode::kInvalidArgument, "GRAPH_SCHEMA_IDENTITY_INVALID"));
      }
      if ((identity.payload_kind == PayloadKind::kDescriptorBacked) !=
          (identity.transport_profile_tag != 0U)) {
        return make_unexpected(
            graph_error(ErrorCode::kInvalidArgument, "GRAPH_SCHEMA_PROFILE_INVALID"));
      }
      return validate_schema(*identity.schema);
    }
    case PayloadIdentityKind::kPod:
      if (identity.transport_profile_tag != 0U || identity.scalar || identity.schema ||
          !identity.pod) {
        return make_unexpected(
            graph_error(ErrorCode::kInvalidArgument, "GRAPH_POD_IDENTITY_INVALID"));
      }
      return validate_pod(*identity.pod);
  }
  return {};
}

namespace {

[[nodiscard]] bool tensor_bearing(const PayloadIdentityRecord& identity,
                                  const std::optional<TensorRepresentation>& layout) noexcept {
  return identity.kind == PayloadIdentityKind::kTensor || layout.has_value();
}

}  // namespace

[[nodiscard]] expected<void, Error> validate_payload_options(
    const PayloadOptionsRecord& options, const PayloadIdentityRecord& identity,
    const std::optional<TensorRepresentation>& layout) {
  const bool inline_policy = options.interface_tensor_emit_policy ==
                             InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded;
  if (!closed_enum(options.interface_tensor_emit_policy) ||
      (options.max_inline_bytes && *options.max_inline_bytes == 0U) ||
      (options.max_tensor_byte_span && *options.max_tensor_byte_span == 0U) ||
      (inline_policy != options.max_inline_bytes.has_value())) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_INLINE_BOUND_INVALID"));
  }
  if (options.schema) {
    auto valid = validate_schema(*options.schema);
    if (!valid)
      return valid;
    if (identity.kind == PayloadIdentityKind::kSchema) {
      if (!identity.schema || !schema_identity_exactly_equal(*options.schema, *identity.schema)) {
        return make_unexpected(
            graph_error(ErrorCode::kIncompatible, "GRAPH_PAYLOAD_SCHEMA_MISMATCH"));
      }
    } else if (identity.kind != PayloadIdentityKind::kTensor) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_PAYLOAD_SCHEMA_IRRELEVANT"));
    }
  }
  if (identity.kind == PayloadIdentityKind::kSchema &&
      (!options.schema || !identity.schema ||
       !schema_identity_exactly_equal(*options.schema, *identity.schema))) {
    return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_PAYLOAD_SCHEMA_MISMATCH"));
  }
  const bool nondefault_tensor_policy =
      options.interface_tensor_emit_policy != InterfaceTensorEmitPolicy::kRequireExportable ||
      options.max_inline_bytes.has_value() || options.max_tensor_byte_span.has_value();
  if (identity.kind == PayloadIdentityKind::kSchema && nondefault_tensor_policy && !layout) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_PAYLOAD_OPTIONS"));
  }
  if (identity.kind != PayloadIdentityKind::kSchema && !tensor_bearing(identity, layout)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_PAYLOAD_OPTIONS"));
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_pod_catalog(const GraphState& state,
                                                         const PayloadIdentityRecord& identity) {
  if (!identity.pod)
    return {};
  const auto& candidate = identity.pod.value();
  for (const auto& existing : state.pod_identity_catalog_) {
    if (!existing.pod)
      continue;
    const auto& prior = existing.pod.value();
    if (prior.canonical_name == candidate.canonical_name && prior.version == candidate.version) {
      if (!payload_identity_exactly_equal(existing, identity)) {
        return make_unexpected(
            graph_error(ErrorCode::kIncompatible, "GRAPH_POD_IDENTITY_CONFLICT"));
      }
      return {};
    }
  }
  return {};
}

void record_pod_identity(GraphState& state, const PayloadIdentityRecord& identity) {
  if (!identity.pod)
    return;
  const auto& candidate = identity.pod.value();
  const bool present = std::any_of(state.pod_identity_catalog_.begin(),
                                   state.pod_identity_catalog_.end(),
                                   [&](const auto& item) {
                                     return item.pod &&
                                            item.pod->canonical_name == candidate.canonical_name &&
                                            item.pod->version == candidate.version;
                                   });
  if (!present)
    state.pod_identity_catalog_.push_back(identity);
}

[[nodiscard]] PortStorage* find_port(GraphState& state, GraphPortId id) noexcept {
  if (!id || id.value > state.ports_.size())
    return nullptr;
  PortStorage& port = state.ports_[static_cast<std::size_t>(id.value - 1U)];
  if (port.record.id != id) {
    assert(false && "Graph port IDs must remain dense");
    return nullptr;
  }
  return &port;
}

[[nodiscard]] const PortStorage* find_port(const GraphState& state, GraphPortId id) noexcept {
  if (!id || id.value > state.ports_.size())
    return nullptr;
  const PortStorage& port = state.ports_[static_cast<std::size_t>(id.value - 1U)];
  if (port.record.id != id) {
    assert(false && "Graph port IDs must remain dense");
    return nullptr;
  }
  return &port;
}

GraphState::GraphState(Graph* owner, std::string name)
    : owner_(owner),
      graph_record_{.id = GraphId{1}, .parent = std::nullopt, .name = std::move(name)} {
  const std::uint64_t sequence = g_next_graph_instance.fetch_add(1U, std::memory_order_relaxed);
  for (std::size_t index = 0; index < sizeof(sequence); ++index) {
    instance_id_.value[index] =
        static_cast<std::byte>((sequence >> static_cast<unsigned>(index * 8U)) & 0xFFU);
  }
  // The second word is a fixed domain discriminator, not an address or implementation token.
  constexpr std::array<std::byte, 8> kDomain{std::byte{'H'},
                                             std::byte{'S'},
                                             std::byte{'D'},
                                             std::byte{'K'},
                                             std::byte{'G'},
                                             std::byte{'R'},
                                             std::byte{'P'},
                                             std::byte{'H'}};
  std::copy(kDomain.begin(), kDomain.end(), instance_id_.value.begin() + sizeof(sequence));
}

expected<void, Error> GraphState::prepare_mutation() {
  std::unique_lock lock(mutex_);
  while (composition_state_ == CompositionState::kComposing &&
         composing_thread_ != std::this_thread::get_id()) {
    composition_cv_.wait(lock);
  }
  if (composition_state_ == CompositionState::kComposed)
    return {};
  if (composition_state_ == CompositionState::kFailed) {
    return make_unexpected(composition_diagnostic_->primary);
  }
  if (composition_state_ == CompositionState::kComposing)
    return {};
  if (owner_ == nullptr) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  composition_state_ = CompositionState::kComposing;
  composing_thread_ = std::this_thread::get_id();
  Graph* owner = owner_;
  lock.unlock();

  Error failure{ErrorCode::kSuccess};
  std::optional<Error> failure_cause;
  std::string failure_context;
  const auto capture_failure = [&](auto&& capture) noexcept {
    try {
      if (fail_composition_diagnostic_capture_) {
        fail_composition_diagnostic_capture_ = false;
        throw std::bad_alloc{};
      }
      std::forward<decltype(capture)>(capture)();
    } catch (...) {
      failure.code = ErrorCode::kFailure;
      failure.message.clear();
      failure_cause.reset();
      failure_context.clear();
    }
  };
  try {
    owner->compose();
  } catch (const RuntimeError& error) {
    capture_failure([&] {
      graph_log_noexcept(
          [&] { HOLOSCAN_LOG_DEBUG("Graph compose RuntimeError detail: {}", error.what()); });
      Error cause = make_error(error);
      if (cause.message == "GRAPH_COMPOSE_REENTRANT" || cause.message == "GRAPH_EMBED_CYCLE") {
        failure = std::move(cause);
      } else {
        failure = graph_error(ErrorCode::kFailure, "GRAPH_COMPOSE_FAILED");
        failure_cause = std::move(cause);
        failure.message += fmt::format(": graph={} caused by {}",
                                       diagnostic_display_text(graph_record_.name),
                                       runtime_error_message(*failure_cause));
      }
    });
  } catch (const std::exception& exception) {
    capture_failure([&] {
      failure_context = bounded_exception_context(exception.what());
      graph_log_noexcept(
          [&] { HOLOSCAN_LOG_DEBUG("Graph compose exception detail: {}", failure_context); });
      failure = graph_factory_error("GRAPH_COMPOSE_EXCEPTION");
      failure.message += "; graph=" + diagnostic_display_text(graph_record_.name);
    });
  } catch (...) {
    capture_failure(
        [&] { failure = graph_error(ErrorCode::kFailure, "GRAPH_COMPOSE_UNKNOWN_EXCEPTION"); });
  }

  lock.lock();
  composing_thread_ = {};
  if (failure.code != ErrorCode::kSuccess) {
    static_assert(std::is_nothrow_move_constructible_v<GraphCompositionDiagnostic>);
    GraphCompositionDiagnostic diagnostic{
        .primary = std::move(failure),
        .cause = std::move(failure_cause),
        .context = std::move(failure_context),
    };
    composition_diagnostic_.emplace(std::move(diagnostic));
    composition_state_ = CompositionState::kFailed;
    clear_failed_draft();
    composition_cv_.notify_all();
    graph_log_noexcept([&] {
      HOLOSCAN_LOG_AUTOMATIC_ERROR(
          "Graph composition failed: {}",
          diagnostic_display_text(runtime_error_message(composition_diagnostic_->primary), 16384U));
    });
    return make_unexpected(composition_diagnostic_->primary);
  }
  composition_state_ = CompositionState::kComposed;
  ++generation_;
  composition_cv_.notify_all();
  graph_log_noexcept(
      [&] { HOLOSCAN_LOG_TRACE("Graph composition completed at generation {}", generation_); });
  return {};
}

expected<void, Error> GraphState::prepare_observation() {
  {
    const std::lock_guard lock(mutex_);
    if (composition_state_ == CompositionState::kComposing &&
        composing_thread_ == std::this_thread::get_id()) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_COMPOSE_REENTRANT"));
    }
  }
  return prepare_mutation();
}

void GraphState::mark_mutation() noexcept {
  if (!(composition_state_ == CompositionState::kComposing &&
        composing_thread_ == std::this_thread::get_id())) {
    ++generation_;
  }
}

void GraphState::detach() noexcept {
  const std::lock_guard lock(mutex_);
  owner_ = nullptr;
}

void GraphState::clear_failed_draft() noexcept {
  operators_.clear();
  vertex_adjacency_.clear();
  cycle_visit_marks_.clear();
  cycle_visit_epoch_ = 0U;
  embedded_graph_records_.clear();
  embedded_operator_mappings_.clear();
  embedded_graphs_.clear();
  vertices_.clear();
  ports_.clear();
  port_by_declaration_.clear();
  edges_.clear();
  edge_handle_identities_.clear();
  interface_bindings_.clear();
  external_topics_.clear();
  readiness_sources_.clear();
  regions_.clear();
  region_handle_identities_.clear();
  partitions_.clear();
  partition_handle_identities_.clear();
  automatic_partition_remaining_ = false;
  clocks_.clear();
  clock_imports_.clear();
  clock_bindings_.clear();
  clock_conversions_.clear();
  lifecycle_dependencies_.clear();
  lifecycle_callbacks_.clear();
  lifecycle_facts_.clear();
  lifecycle_restart_permissions_.clear();
  vertex_names_.clear();
  interface_port_names_.clear();
  region_names_.clear();
  partition_names_.clear();
  clock_names_.clear();
  default_clock_ = ClockKey{};
  explicit_endpoint_names_.clear();
  embedded_instance_names_.clear();
  pod_identity_catalog_.clear();
}

expected<void, Error> graph_prepare_authoring(const std::shared_ptr<GraphState>& state) {
  if (!state)
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  return state->prepare_mutation();
}

expected<void, Error> graph_prepare_embed(const std::shared_ptr<GraphState>& state,
                                          std::string_view instance_name) {
  if (!state)
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  auto valid = validate_name(instance_name, "GRAPH_EMBED_INSTANCE_NAME_INVALID");
  if (!valid)
    return valid;
  // Structured paths retain decoded segments. Dot segments would invite
  // normalization aliases, while the exact top-level `_root` spelling is
  // reserved for rendering the root boundary.
  if (instance_name == "." || instance_name == ".." || instance_name == "_root") {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_EMBED_INSTANCE_PATH_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (state->embedded_instance_names_.contains(std::string(instance_name))) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_EMBED_INSTANCE_DUPLICATE"));
  }
  return {};
}

namespace {

[[nodiscard]] bool lifecycle_cycle(const GraphState& state, VertexId before, VertexId after,
                                   LifecycleStage stage) {
  std::vector<VertexId> pending{after};
  std::unordered_set<std::uint64_t> visited;
  while (!pending.empty()) {
    const VertexId current = pending.back();
    pending.pop_back();
    if (current == before)
      return true;
    if (!visited.emplace(current.value).second)
      continue;
    for (const auto& dependency : state.lifecycle_dependencies_) {
      if (dependency.stage == stage && dependency.before == current) {
        pending.push_back(dependency.after);
      }
    }
  }
  return false;
}

[[nodiscard]] std::optional<VertexId> resolve_operator_vertex(const GraphState& parent,
                                                              const GraphState& source,
                                                              std::size_t source_index) noexcept {
  if (&parent == &source) {
    if (source_index >= parent.operators_.size())
      return std::nullopt;
    return parent.operators_[source_index].vertex;
  }
  const auto found =
      std::find_if(parent.embedded_operator_mappings_.begin(),
                   parent.embedded_operator_mappings_.end(),
                   [&](const EmbeddedOperatorMapping& mapping) {
                     return mapping.source_state == &source && mapping.source_index == source_index;
                   });
  return found == parent.embedded_operator_mappings_.end() ? std::nullopt
                                                           : std::optional<VertexId>{found->vertex};
}

}  // namespace

expected<void, Error> graph_lifecycle_before(const std::shared_ptr<GraphState>& state,
                                             std::size_t before_index, std::size_t after_index,
                                             LifecycleStage stage) {
  return graph_lifecycle_before_resolved(state, state, before_index, state, after_index, stage);
}

expected<void, Error> graph_lifecycle_before_resolved(
    const std::shared_ptr<GraphState>& state, const std::shared_ptr<GraphState>& before_state,
    std::size_t before_index, const std::shared_ptr<GraphState>& after_state,
    std::size_t after_index, LifecycleStage stage) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (!closed_enum(stage)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_STAGE_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  const std::optional<VertexId> before =
      resolve_operator_vertex(*state, *before_state, before_index);
  const std::optional<VertexId> after = resolve_operator_vertex(*state, *after_state, after_index);
  if (!before || !after) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_HANDLE_INVALID"));
  }
  if (*before == *after || lifecycle_cycle(*state, *before, *after, stage)) {
    return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_LIFECYCLE_CYCLE"));
  }
  const LifecycleDependencyAuthoringRecord candidate{
      .graph = state->graph_record_.id,
      .before = *before,
      .after = *after,
      .stage = stage,
  };
  if (std::any_of(state->lifecycle_dependencies_.begin(),
                  state->lifecycle_dependencies_.end(),
                  [&](const auto& existing) {
                    return existing.before == candidate.before &&
                           existing.after == candidate.after && existing.stage == candidate.stage;
                  })) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_LIFECYCLE_DEPENDENCY_DUPLICATE"));
  }
  state->lifecycle_dependencies_.push_back(candidate);
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_lifecycle_permission(const std::shared_ptr<GraphState>& state,
                                                 LifecycleRestartOrigin origin,
                                                 std::optional<std::size_t> requester,
                                                 std::size_t target_index) {
  return graph_lifecycle_permission_resolved(state, origin, state, requester, state, target_index);
}

expected<void, Error> graph_lifecycle_permission_resolved(
    const std::shared_ptr<GraphState>& state, LifecycleRestartOrigin origin,
    const std::shared_ptr<GraphState>& requester_state, std::optional<std::size_t> requester,
    const std::shared_ptr<GraphState>& target_state, std::size_t target_index) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (!closed_enum(origin)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_ORIGIN_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  const std::optional<VertexId> target =
      resolve_operator_vertex(*state, *target_state, target_index);
  const std::optional<VertexId> requester_vertex =
      requester ? resolve_operator_vertex(*state, *requester_state, *requester) : std::nullopt;
  if (!target || (requester && !requester_vertex)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_HANDLE_INVALID"));
  }
  if ((requester_vertex && *requester_vertex == *target) ||
      ((origin == LifecycleRestartOrigin::kOperatorRequest) != requester.has_value())) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_PERMISSION_INVALID"));
  }
  LifecycleRestartPermissionAuthoringRecord candidate{
      .origin = origin,
      .requester = requester_vertex,
      .target = *target,
  };
  if (std::any_of(state->lifecycle_restart_permissions_.begin(),
                  state->lifecycle_restart_permissions_.end(),
                  [&](const auto& existing) {
                    return existing.origin == candidate.origin &&
                           existing.requester == candidate.requester &&
                           existing.target == candidate.target;
                  })) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_LIFECYCLE_PERMISSION_DUPLICATE"));
  }
  state->lifecycle_restart_permissions_.push_back(std::move(candidate));
  state->mark_mutation();
  return {};
}

expected<ClockKey, Error> graph_add_clock(const std::shared_ptr<GraphState>& state,
                                          std::string_view name,
                                          std::shared_ptr<ClockFactoryBase> factory) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  auto valid_name = validate_name(name, "GRAPH_CLOCK_NAME_INVALID");
  if (!valid_name)
    return make_unexpected(std::move(valid_name.error()));
  if (!factory) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CLOCK_FACTORY_INVALID"));
  }
  auto candidate = factory->create(nullptr);
  if (!candidate)
    return make_unexpected(std::move(candidate.error()));
  if (!candidate.value()) {
    return make_unexpected(graph_error(ErrorCode::kFailure, "GRAPH_CLOCK_FACTORY_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (state->clock_names_.contains(std::string(name))) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_CLOCK_NAME_DUPLICATE"));
  }
  ClockKey key = ClockKey::named(name);
  ClockStorage storage{
      .record =
          ClockDeclarationRecord{
              .graph = state->graph_record_.id,
              .key = key,
              .shared = false,
              .is_default = false,
          },
      .capabilities = clock_capabilities(*candidate.value(), state->graph_record_.id, key),
      .factory = std::move(factory),
  };
  GraphMutationRollback rollback(*state);
  state->clocks_.push_back(std::move(storage));
  GraphMutationRollback::checkpoint(*state);
  state->clock_names_.emplace(name);
  GraphMutationRollback::checkpoint(*state);
  state->mark_mutation();
  rollback.commit();
  return key;
}

expected<ClockKey, Error> graph_bind_shared_clock(const std::shared_ptr<GraphState>& state,
                                                  std::string_view name,
                                                  std::shared_ptr<Clock> clock) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  auto valid_name = validate_name(name, "GRAPH_CLOCK_NAME_INVALID");
  if (!valid_name)
    return make_unexpected(std::move(valid_name.error()));
  if (!clock) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_SHARED_CLOCK_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (state->clock_names_.contains(std::string(name))) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_CLOCK_NAME_DUPLICATE"));
  }
  ClockKey key = ClockKey::named(name);
  ClockStorage storage{
      .record =
          ClockDeclarationRecord{
              .graph = state->graph_record_.id,
              .key = key,
              .shared = true,
              .is_default = false,
          },
      .capabilities = clock_capabilities(*clock, state->graph_record_.id, key),
      .shared = std::move(clock),
  };
  GraphMutationRollback rollback(*state);
  state->clocks_.push_back(std::move(storage));
  GraphMutationRollback::checkpoint(*state);
  state->clock_names_.emplace(name);
  GraphMutationRollback::checkpoint(*state);
  state->mark_mutation();
  rollback.commit();
  return key;
}

expected<ClockKey, Error> graph_import_clock(const std::shared_ptr<GraphState>& state,
                                             std::string_view name) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  auto valid_name = validate_name(name, "GRAPH_CLOCK_IMPORT_NAME_INVALID");
  if (!valid_name)
    return make_unexpected(std::move(valid_name.error()));
  const std::lock_guard lock(state->mutex_);
  if (state->clock_names_.contains(std::string(name))) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_CLOCK_NAME_DUPLICATE"));
  }
  ClockKey key = ClockKey::named(name);
  GraphMutationRollback rollback(*state);
  state->clock_imports_.push_back(ClockImportAuthoringRecord{
      .graph = state->graph_record_.id,
      .key = key,
      .is_default = false,
  });
  GraphMutationRollback::checkpoint(*state);
  state->clock_names_.emplace(name);
  GraphMutationRollback::checkpoint(*state);
  state->mark_mutation();
  rollback.commit();
  return key;
}

expected<void, Error> graph_bind_clock_import(const std::shared_ptr<GraphState>& state,
                                              std::size_t child_index, const void* handle_identity,
                                              ClockKey child_import, ClockKey parent_clock) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (child_import.is_default() || child_import.name().empty() || parent_clock.is_default() ||
      parent_clock.name().empty()) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CLOCK_BINDING_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (child_index >= state->embedded_graphs_.size() || handle_identity == nullptr ||
      state->embedded_graphs_[child_index].handle_identity.get() != handle_identity) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_CLOCK_BINDING_HANDLE_INVALID"));
  }
  const GraphId child_graph = state->embedded_graphs_[child_index].graph_id;
  const auto imported = std::find_if(
      state->clock_imports_.begin(), state->clock_imports_.end(), [&](const auto& candidate) {
        return candidate.graph == child_graph && candidate.key == child_import;
      });
  if (imported == state->clock_imports_.end()) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_CLOCK_IMPORT_NOT_FOUND"));
  }
  if (!state->clock_names_.contains(std::string(parent_clock.name()))) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_CLOCK_NOT_FOUND"));
  }
  if (std::any_of(
          state->clock_bindings_.begin(), state->clock_bindings_.end(), [&](const auto& existing) {
            return existing.child_graph == child_graph && existing.child_clock == child_import;
          })) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_CLOCK_BINDING_DUPLICATE"));
  }
  GraphMutationRollback rollback(*state);
  state->clock_bindings_.push_back(EmbeddedClockBindingAuthoringRecord{
      .child_graph = child_graph,
      .child_clock = std::move(child_import),
      .parent_graph = state->graph_record_.id,
      .parent_clock = std::move(parent_clock),
  });
  state->mark_mutation();
  rollback.commit();
  return {};
}

expected<void, Error> graph_set_default_clock(const std::shared_ptr<GraphState>& state,
                                              ClockKey key) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (key.is_default() || key.name().empty()) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_DEFAULT_CLOCK_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (!state->clock_names_.contains(std::string(key.name()))) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_CLOCK_NOT_FOUND"));
  }
  static_assert(std::is_nothrow_move_assignable_v<ClockKey>);
  GraphMutationRollback::checkpoint(*state);
  for (auto& clock : state->clocks_) {
    if (clock.record.graph == state->graph_record_.id) {
      clock.record.is_default = clock.record.key == key;
    }
  }
  for (auto& clock_import : state->clock_imports_) {
    if (clock_import.graph == state->graph_record_.id) {
      clock_import.is_default = clock_import.key == key;
    }
  }
  state->default_clock_ = std::move(key);
  state->mark_mutation();
  return {};
}

expected<ClockKey, Error> graph_default_clock(const std::shared_ptr<GraphState>& state) {
  auto prepared = state->prepare_observation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  const std::lock_guard lock(state->mutex_);
  return state->default_clock_;
}

expected<void, Error> graph_add_clock_conversion(const std::shared_ptr<GraphState>& state,
                                                 ClockKey from, ClockKey to,
                                                 AffineClockConversion conversion) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (from.is_default() || to.is_default() || from.name().empty() || to.name().empty() ||
      from == to || conversion.scale.numerator <= 0 || conversion.scale.denominator == 0U) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_CLOCK_CONVERSION_INVALID"));
  }
  const auto divisor = std::gcd(static_cast<std::uint64_t>(conversion.scale.numerator),
                                conversion.scale.denominator);
  conversion.scale.numerator /= static_cast<std::int64_t>(divisor);
  conversion.scale.denominator /= divisor;
  const std::lock_guard lock(state->mutex_);
  if (!state->clock_names_.contains(std::string(from.name())) ||
      !state->clock_names_.contains(std::string(to.name()))) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_CLOCK_NOT_FOUND"));
  }
  for (const auto& existing : state->clock_conversions_) {
    if (existing.graph != state->graph_record_.id)
      continue;
    if (existing.from == from && existing.to == to) {
      if (existing.conversion == conversion)
        return {};
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_CLOCK_CONVERSION_CONFLICT"));
    }
  }
  std::vector<ClockKey> pending{to};
  std::unordered_set<std::string> visited;
  while (!pending.empty()) {
    const ClockKey current = std::move(pending.back());
    pending.pop_back();
    if (current == from) {
      return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_CLOCK_CONVERSION_CYCLE"));
    }
    if (!visited.emplace(current.name()).second)
      continue;
    for (const auto& existing : state->clock_conversions_) {
      if (existing.graph == state->graph_record_.id && existing.from == current)
        pending.push_back(existing.to);
    }
  }
  state->clock_conversions_.push_back(ClockConversionAuthoringRecord{
      .graph = state->graph_record_.id,
      .from = std::move(from),
      .to = std::move(to),
      .conversion = conversion,
  });
  state->mark_mutation();
  return {};
}

std::shared_ptr<GraphState> make_graph_state(Graph* owner, std::string_view name) {
  auto valid = validate_name(name, "GRAPH_NAME_INVALID", true);
  if (!valid)
    throw RuntimeError(valid.error());
  return std::make_shared<GraphState>(owner, std::string(name));
}

expected<void, Error> graph_apply_atomic_mutation(const std::shared_ptr<GraphState>& state,
                                                  void* context,
                                                  GraphAtomicMutationOperation operation) {
  if (!state || operation == nullptr) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_MUTATION_INVALID"));
  }
  auto prepared = graph_prepare_authoring(state);
  if (!prepared) {
    return prepared;
  }

  const std::lock_guard lock(state->mutex_);
  auto rollback = std::make_unique<GraphMutationRollback>(
      *state, GraphMutationRollback::CaptureExistingRecords{});
  try {
    auto result = operation(context);
    if (result) {
      rollback->commit();
    } else {
      rollback->rollback();
    }
    return result;
  } catch (...) {
    rollback->rollback();
    throw;
  }
}

}  // namespace detail

RegionConstraint RegionConstraint::static_topology() noexcept {
  return RegionConstraint{RegionConstraintKind::kStaticTopology, {}};
}

RegionConstraint RegionConstraint::bounded_memory() noexcept {
  return RegionConstraint{RegionConstraintKind::kBoundedMemory, {}};
}

RegionConstraint RegionConstraint::device_compatible_ops() noexcept {
  return RegionConstraint{RegionConstraintKind::kDeviceCompatibleOperators, {}};
}

RegionConstraint RegionConstraint::cuda_graph() noexcept {
  return RegionConstraint{RegionConstraintKind::kCudaGraph, {}};
}

RegionConstraint RegionConstraint::latency_budget(std::chrono::nanoseconds budget) {
  if (budget.count() <= 0) {
    throw RuntimeError(Error{ErrorCode::kInvalidArgument, "GRAPH_REGION_LATENCY_INVALID"});
  }
  return RegionConstraint{RegionConstraintKind::kLatencyBudget, budget};
}

TensorOutputPortConfigurator& TensorOutputPortConfigurator::max_emits_per_compute(
    std::uint32_t count) {
  apply(detail::graph_configure_output_emits(state(), port_, handle_identity_.get(), count));
  return *this;
}

TensorOutputPortConfigurator& TensorOutputPortConfigurator::late_joiner_history_depth(
    std::uint32_t depth) {
  apply(detail::graph_configure_output_history(state(), port_, handle_identity_.get(), depth));
  return *this;
}

TensorOutputPortConfigurator& TensorOutputPortConfigurator::produces_tensor(
    const TensorOutputSpec& spec) {
  apply(detail::graph_configure_output_tensor_spec(state(), port_, handle_identity_.get(), spec));
  return *this;
}

TensorOutputPortConfigurator& TensorOutputPortConfigurator::payload_options(
    PayloadOptions options) {
  apply(detail::graph_configure_output_payload(
      state(), port_, handle_identity_.get(), std::move(options)));
  return *this;
}

TensorOutputPortConfigurator& TensorOutputPortConfigurator::diagnostic_alias(
    std::string_view alias) {
  apply(detail::graph_configure_port_alias(state(), port_, handle_identity_.get(), alias));
  return *this;
}

std::shared_ptr<detail::GraphState> TensorOutputPortConfigurator::state() const {
  auto value = state_.lock();
  if (!value) {
    detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
  }
  return value;
}

void TensorOutputPortConfigurator::apply(expected<void, Error> result) {
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
}

RegionHandle& RegionHandle::constrain(RegionConstraint constraint) {
  auto state = state_.lock();
  if (!state) {
    detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"});
  }
  auto result = detail::graph_region_add_constraint(state, id_, handle_identity_.get(), constraint);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return *this;
}

RegionHandle& RegionHandle::profile(Profile profile) {
  auto state = state_.lock();
  if (!state) {
    detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"});
  }
  auto result = detail::graph_region_set_profile(state, id_, handle_identity_.get(), profile);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return *this;
}

GraphGeneration GraphView::generation() const noexcept {
  return snapshot_ ? snapshot_->generation : 0U;
}

GraphAuthoringIdentity GraphView::identity() const noexcept {
  return snapshot_ ? snapshot_->identity : GraphAuthoringIdentity{};
}

std::span<const GraphRecord> GraphView::graphs() const noexcept {
  return snapshot_ ? std::span<const GraphRecord>{snapshot_->graphs}
                   : std::span<const GraphRecord>{};
}

std::span<const VertexRecord> GraphView::vertices() const noexcept {
  return snapshot_ ? std::span<const VertexRecord>{snapshot_->vertices}
                   : std::span<const VertexRecord>{};
}

std::span<const PortRecord> GraphView::ports() const noexcept {
  return snapshot_ ? std::span<const PortRecord>{snapshot_->ports} : std::span<const PortRecord>{};
}

std::span<const EdgeRecord> GraphView::edges() const noexcept {
  return snapshot_ ? std::span<const EdgeRecord>{snapshot_->edges} : std::span<const EdgeRecord>{};
}

std::span<const InterfaceBindingRecord> GraphView::interface_bindings() const noexcept {
  return snapshot_ ? std::span<const InterfaceBindingRecord>{snapshot_->interface_bindings}
                   : std::span<const InterfaceBindingRecord>{};
}

std::span<const ExternalTopicRecord> GraphView::external_topics() const noexcept {
  return snapshot_ ? std::span<const ExternalTopicRecord>{snapshot_->external_topics}
                   : std::span<const ExternalTopicRecord>{};
}

std::span<const ReadinessSourceDeclarationRecord> GraphView::readiness_sources() const noexcept {
  return snapshot_ ? std::span<const ReadinessSourceDeclarationRecord>{snapshot_->readiness_sources}
                   : std::span<const ReadinessSourceDeclarationRecord>{};
}

std::span<const RegionRecord> GraphView::regions() const noexcept {
  return snapshot_ ? std::span<const RegionRecord>{snapshot_->regions}
                   : std::span<const RegionRecord>{};
}

std::span<const PartitionRecord> GraphView::partitions() const noexcept {
  return snapshot_ ? std::span<const PartitionRecord>{snapshot_->partitions}
                   : std::span<const PartitionRecord>{};
}

bool GraphView::automatic_partition_remaining() const noexcept {
  return snapshot_ && snapshot_->automatic_partition_remaining;
}

std::span<const ClockDeclarationRecord> GraphView::clocks() const noexcept {
  return snapshot_ ? std::span<const ClockDeclarationRecord>{snapshot_->clocks}
                   : std::span<const ClockDeclarationRecord>{};
}

std::span<const ClockImportAuthoringRecord> GraphView::clock_imports() const noexcept {
  return snapshot_ ? std::span<const ClockImportAuthoringRecord>{snapshot_->clock_imports}
                   : std::span<const ClockImportAuthoringRecord>{};
}

std::span<const EmbeddedClockBindingAuthoringRecord> GraphView::clock_bindings() const noexcept {
  return snapshot_ ? std::span<const EmbeddedClockBindingAuthoringRecord>{snapshot_->clock_bindings}
                   : std::span<const EmbeddedClockBindingAuthoringRecord>{};
}

std::span<const ClockConversionAuthoringRecord> GraphView::clock_conversions() const noexcept {
  return snapshot_ ? std::span<const ClockConversionAuthoringRecord>{snapshot_->clock_conversions}
                   : std::span<const ClockConversionAuthoringRecord>{};
}

std::span<const LifecycleDependencyAuthoringRecord> GraphView::lifecycle_dependencies()
    const noexcept {
  return snapshot_
             ? std::span<const LifecycleDependencyAuthoringRecord>{snapshot_
                                                                       ->lifecycle_dependencies}
             : std::span<const LifecycleDependencyAuthoringRecord>{};
}

std::span<const LifecycleRestartPermissionAuthoringRecord>
GraphView::lifecycle_restart_permissions() const noexcept {
  if (snapshot_) {
    return std::span<const LifecycleRestartPermissionAuthoringRecord>{
        snapshot_->lifecycle_restart_permissions};
  }
  return {};
}

Graph::Graph(std::string_view name) : state_(detail::make_graph_state(this, name)) {}

Graph::~Graph() {
  state_->detach();
}

RegionHandle Graph::region(std::string_view name) {
  auto result = detail::graph_add_region(state_, name);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return RegionHandle{state_, result.value().id, std::move(result.value().handle_identity)};
}

PartitionHandle Graph::partition(std::string_view name) {
  auto result = detail::graph_add_partition(state_, name);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return PartitionHandle{state_, result.value().id, std::move(result.value().handle_identity)};
}

PartitionHandle& PartitionHandle::include_all() {
  auto state = state_.lock();
  if (!state) {
    detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"});
  }
  auto result = detail::graph_partition_include_all(state, id_, handle_identity_.get());
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return *this;
}

void Graph::auto_partition_remaining() {
  auto result = detail::graph_auto_partition_remaining(state_);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
}

GraphLifecycleBuilder Graph::lifecycle() {
  auto prepared = state_->prepare_mutation();
  if (!prepared)
    detail::throw_graph_error(std::move(prepared.error()));
  return GraphLifecycleBuilder{state_};
}

GraphView Graph::view() {
  return GraphView{detail::graph_snapshot(state_)};
}

ClockKey Graph::bind_shared_clock(std::string_view name, std::shared_ptr<Clock> clock) {
  auto result = detail::graph_bind_shared_clock(state_, name, std::move(clock));
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return std::move(result.value());
}

ClockKey Graph::import_clock(std::string_view name) {
  auto result = detail::graph_import_clock(state_, name);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return std::move(result.value());
}

Graph& Graph::set_default_clock(ClockKey key) {
  auto result = detail::graph_set_default_clock(state_, std::move(key));
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return *this;
}

ClockKey Graph::default_clock() {
  auto result = detail::graph_default_clock(state_);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return std::move(result.value());
}

Graph& Graph::add_clock_conversion(ClockKey from, ClockKey to, AffineClockConversion conversion) {
  auto result =
      detail::graph_add_clock_conversion(state_, std::move(from), std::move(to), conversion);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return *this;
}

}  // namespace holoscan
