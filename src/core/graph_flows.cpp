/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Graph flows, interface bindings, port and flow configuration, and external topics.

#include <holoscan/core/graph.hpp>
#include <holoscan/logger/logger.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "diagnostic_context.hpp"
#include "graph_internal.hpp"
#include "graph_mutation_rollback.hpp"
#include "graph_state.hpp"
#include "plan_record_contracts.hpp"
#include "tensor_contract_comparison.hpp"
#include "tensor_contract_resolution_internal.hpp"

namespace holoscan::detail {

/// Restore a deliberately pre-validation external-boundary ordinal only when
/// control leaves by exception. Ordinary typed validation failures return
/// normally and therefore retain their reserved declaration slot.
class ExceptionalBoundaryOrdinalRollback {
 public:
  ExceptionalBoundaryOrdinalRollback(std::uint32_t& next_ordinal,
                                     std::uint32_t reserved_ordinal) noexcept
      : next_ordinal_(next_ordinal),
        reserved_ordinal_(reserved_ordinal),
        uncaught_exceptions_(std::uncaught_exceptions()) {}

  ExceptionalBoundaryOrdinalRollback(const ExceptionalBoundaryOrdinalRollback&) = delete;
  ExceptionalBoundaryOrdinalRollback& operator=(const ExceptionalBoundaryOrdinalRollback&) = delete;

  ~ExceptionalBoundaryOrdinalRollback() noexcept {
    if (std::uncaught_exceptions() > uncaught_exceptions_)
      next_ordinal_ = reserved_ordinal_;
  }

 private:
  std::uint32_t& next_ordinal_;
  std::uint32_t reserved_ordinal_;
  int uncaught_exceptions_;
};

namespace {

constexpr std::string_view kGeneratedEndpointNamespace{"__holoscan_auto_endpoint/"};
constexpr std::string_view kHoloMqTopicPrefix{"holomq://"};

[[nodiscard]] Error descriptor_topic_field_graph_error(
    ErrorCode code, std::string_view token, const AuthoredDescriptorTensorTopicField& field) {
  std::string message(token);
  message.append(" (field_id=");
  message.append(std::to_string(field.field_id));
  message.append(", tensor_index=");
  message.append(std::to_string(field.tensor_index));
  message.push_back(')');
  return Error{code, std::move(message)};
}

[[nodiscard]] expected<void, Error> validate_endpoint_name(std::string_view name) {
  auto valid = validate_name(name, "GRAPH_ENDPOINT_NAME_INVALID");
  if (!valid)
    return valid;
  if (name.starts_with(kGeneratedEndpointNamespace)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_RESERVED_ENDPOINT_NAMESPACE"));
  }
  return {};
}

[[nodiscard]] std::optional<SchemaIdentityRecord> copy_schema(
    const std::optional<SchemaIdentity>& source) {
  if (!source)
    return std::nullopt;
  return make_schema_identity_record(*source);
}

[[nodiscard]] ConnectionOptionsRecord copy_connection_options(const ConnectionOptions& source) {
  return ConnectionOptionsRecord{
      .queue_policy = source.queue_policy,
      .queue_depth = source.queue_depth,
      .topic_name =
          source.topic_name ? std::optional<std::string>{*source.topic_name} : std::nullopt,
      .memory_conversion = source.memory_conversion,
      .max_inflight_samples = source.max_inflight_samples,
  };
}

[[nodiscard]] expected<void, Error> validate_connection_options(const ConnectionOptions& options) {
  if (!closed_enum(options.queue_policy) ||
      (options.memory_conversion && !closed_enum(*options.memory_conversion)) ||
      (options.queue_depth && *options.queue_depth == 0U) ||
      (options.max_inflight_samples && *options.max_inflight_samples == 0U)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_FLOW_BOUND_INVALID"));
  }
  if (options.topic_name) {
    return validate_name(*options.topic_name, "GRAPH_FLOW_TOPIC_NAME_INVALID");
  }
  return {};
}

[[nodiscard]] EdgeRecord* find_edge(GraphState& state, EdgeId id) noexcept {
  if (!id || id.value > state.edges_.size())
    return nullptr;
  EdgeRecord& edge = state.edges_[static_cast<std::size_t>(id.value - 1U)];
  if (edge.id != id) {
    assert(false && "Graph edge IDs must remain dense");
    return nullptr;
  }
  return &edge;
}

[[nodiscard]] expected<GraphPortId, Error> find_port_by_declaration(
    GraphState& state, const void* declaration, PortDirection direction,
    const PayloadIdentityRecord& payload_identity) {
  const auto found = state.port_by_declaration_.find(declaration);
  if (found == state.port_by_declaration_.end()) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_PORT_NOT_FOUND"));
  }
  PortStorage* port = find_port(state, found->second);
  if (port == nullptr || port->record.direction != direction ||
      !payload_identity_exactly_equal(port->record.payload_identity, payload_identity)) {
    return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_PORT_TYPE_MISMATCH"));
  }
  return found->second;
}

[[nodiscard]] bool creates_directed_cycle(GraphState& state, VertexId source_vertex,
                                          VertexId target_vertex) {
  if (!source_vertex || !target_vertex || source_vertex.value > state.vertex_adjacency_.size() ||
      target_vertex.value > state.vertex_adjacency_.size()) {
    return true;
  }
  const std::size_t source_index = static_cast<std::size_t>(source_vertex.value - 1U);
  const std::size_t target_index = static_cast<std::size_t>(target_vertex.value - 1U);
  if (source_index == target_index)
    return true;
  if (state.cycle_visit_epoch_ == std::numeric_limits<std::uint64_t>::max()) {
    std::fill(state.cycle_visit_marks_.begin(), state.cycle_visit_marks_.end(), 0U);
    state.cycle_visit_epoch_ = 1U;
  } else {
    ++state.cycle_visit_epoch_;
  }
  const std::uint64_t visit_epoch = state.cycle_visit_epoch_;
  std::vector<std::size_t> pending{target_index};
  while (!pending.empty()) {
    const std::size_t current = pending.back();
    pending.pop_back();
    if (current == source_index)
      return true;
    if (current >= state.cycle_visit_marks_.size() ||
        state.cycle_visit_marks_[current] == visit_epoch) {
      continue;
    }
    state.cycle_visit_marks_[current] = visit_epoch;
    const auto& outgoing = state.vertex_adjacency_[current];
    pending.insert(pending.end(), outgoing.begin(), outgoing.end());
  }
  return false;
}

}  // namespace

void reserve_adjacency(GraphState& state, VertexId source) {
  assert(source && source.value <= state.vertex_adjacency_.size());
  auto& outgoing = state.vertex_adjacency_[static_cast<std::size_t>(source.value - 1U)];
  outgoing.reserve(outgoing.size() + 1U);
}

void append_adjacency(GraphState& state, VertexId source, VertexId target) {
  assert(source && source.value <= state.vertex_adjacency_.size());
  assert(target && target.value <= state.vertex_adjacency_.size());
  state.vertex_adjacency_[static_cast<std::size_t>(source.value - 1U)].push_back(
      static_cast<std::size_t>(target.value - 1U));
}

namespace {

[[nodiscard]] std::optional<PayloadOptionsRecord> boundary_tensor_payload(const GraphState& state,
                                                                          GraphPortId port) {
  for (const auto& topic : state.external_topics_) {
    if (topic.record.graph_port == port && topic.record.tensor) {
      const PortStorage* const boundary_port = find_port(state, port);
      const bool descriptor_schema =
          boundary_port != nullptr &&
          boundary_port->record.payload_identity.kind == PayloadIdentityKind::kSchema &&
          boundary_port->record.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
      return PayloadOptionsRecord{
          .schema = topic.record.schema,
          .max_inline_bytes = topic.record.tensor->max_inline_bytes,
          .max_tensor_byte_span =
              descriptor_schema ? topic.record.tensor->max_byte_span : std::nullopt,
          .interface_tensor_emit_policy = topic.record.tensor->foreign_emit_policy,
      };
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<TensorContractRecord> effective_tensor_contract(
    const PortStorage& port) {
  return resolve_output_tensor_contract(port.record.setup.tensor_contract,
                                        port.record.graph_override.tensor_contract);
}

[[nodiscard]] std::optional<TensorRepresentation> effective_tensor_layout(const PortStorage& port) {
  const auto contract = effective_tensor_contract(port);
  if (contract) {
    return tensor_representation(*contract);
  }
  return port.record.graph_override.tensor_layout ? port.record.graph_override.tensor_layout
                                                  : port.record.setup.tensor_layout;
}

[[nodiscard]] bool tensor_layouts_compatible(
    const std::optional<TensorRepresentation>& source,
    const std::optional<TensorRepresentation>& target) noexcept {
  return compare_tensor_layouts(source, target);
}

[[nodiscard]] Error connection_error(const GraphState& state, const PortStorage& source,
                                     const PortStorage& target, std::string_view token,
                                     const TensorMismatches& mismatches) {
  const auto endpoint = [&](const PortStorage& port) {
    const auto owner = std::ranges::find(state.vertices_, port.record.owner, &VertexRecord::id);
    const auto path = owner == state.vertices_.end() ? std::string{"unknown"}
                                                     : authored_operator_path(state, *owner);
    return diagnostic_display_text(path) + "." + diagnostic_display_text(port.record.name);
  };
  std::string message =
      fmt::format("{}: cannot connect {} -> {}", token, endpoint(source), endpoint(target));
  if (token == "GRAPH_TENSOR_LAYOUT_EDGE_CONFLICT") {
    message +=
        fmt::format("; source spec={}, target spec={}",
                    source.record.graph_override.tensor_contract ? "graph override" : "setup",
                    target.record.graph_override.tensor_contract ? "graph override" : "setup");
  }
  for (std::size_t index = 0U; index < std::min(mismatches.size(), std::size_t{16U}); ++index) {
    const auto& mismatch = mismatches[index];
    message += fmt::format("; {}: provided={}, required={}",
                           diagnostic_display_text(mismatch.field),
                           diagnostic_display_text(mismatch.provided),
                           diagnostic_display_text(mismatch.required));
  }
  if (mismatches.size() > 16U) {
    message += "; additional mismatches omitted [truncated]";
  }
  return Error{ErrorCode::kIncompatible, std::move(message)};
}

template <typename T>
[[nodiscard]] const T* optional_value_pointer(const std::optional<T>& value) noexcept {
  return value.has_value() ? &value.value() : nullptr;
}

[[nodiscard]] bool declares_tensor_semantics(const PortStorage& port) noexcept {
  return port.record.payload_identity.kind == PayloadIdentityKind::kTensor ||
         port.record.setup.tensor_layout.has_value() || !port.record.setup.tensor_fields.empty();
}

[[nodiscard]] bool tensor_field_layouts_compatible(const PortStorage& source,
                                                   const PortStorage& target,
                                                   TensorMismatches& mismatches) {
  const auto& source_fields = source.record.setup.tensor_fields;
  const auto& target_fields = target.record.setup.tensor_fields;
  bool compatible = true;
  for (const auto& target_field : target_fields) {
    const auto source_field =
        std::find_if(source_fields.begin(), source_fields.end(), [&](const auto& candidate) {
          return candidate.field_id == target_field.field_id &&
                 candidate.tensor_index == target_field.tensor_index;
        });
    if (source_field == source_fields.end()) {
      continue;
    }
    const auto first = mismatches.size();
    const bool layout = compare_tensor_layouts(
        source_field->tensor_layout, target_field.tensor_layout, &mismatches);
    const bool bounds = compare_tensor_bounds(optional_value_pointer(source_field->tensor_contract),
                                              optional_value_pointer(target_field.tensor_contract),
                                              &mismatches);
    compatible = compatible && layout && bounds;
    for (std::size_t index = first; index < mismatches.size(); ++index) {
      mismatches[index].field = fmt::format("field[{}].tensor[{}].{}",
                                            target_field.field_id,
                                            target_field.tensor_index,
                                            mismatches[index].field);
    }
  }
  return compatible;
}

void append_tensor_payload_mismatches(const PayloadOptionsRecord& producer,
                                      const PayloadOptionsRecord& boundary,
                                      TensorMismatches* mismatches) {
  const auto policy_name = [](InterfaceTensorEmitPolicy policy) {
    switch (policy) {
      case InterfaceTensorEmitPolicy::kRequireExportable:
        return "require exportable";
      case InterfaceTensorEmitPolicy::kCopyIfNeeded:
        return "copy if needed";
      case InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded:
        return "inline or copy";
    }
    return "invalid";
  };
  const auto bound_name = [](const auto& value) {
    return value ? std::to_string(*value) : std::string{"not specified"};
  };
  if (producer.interface_tensor_emit_policy != boundary.interface_tensor_emit_policy) {
    tensor_mismatch(mismatches,
                    "tensor emit policy",
                    policy_name(producer.interface_tensor_emit_policy),
                    policy_name(boundary.interface_tensor_emit_policy));
  }
  if (producer.max_inline_bytes != boundary.max_inline_bytes) {
    tensor_mismatch(mismatches,
                    "max_inline_bytes",
                    bound_name(producer.max_inline_bytes),
                    bound_name(boundary.max_inline_bytes));
  }
  if (producer.max_tensor_byte_span != boundary.max_tensor_byte_span) {
    tensor_mismatch(mismatches,
                    "max_tensor_byte_span",
                    bound_name(producer.max_tensor_byte_span),
                    bound_name(boundary.max_tensor_byte_span));
  }
}

[[nodiscard]] expected<PayloadOptionsRecord, Error> effective_tensor_payload(
    const GraphState& state, const PortStorage& source, const PortStorage& target,
    std::optional<PayloadOptionsRecord> source_override = std::nullopt) {
  const auto source_contract = effective_tensor_contract(source);
  const auto target_contract = effective_tensor_contract(target);
  TensorMismatches mismatches;
  const bool layout = compare_tensor_layouts(
      effective_tensor_layout(source), effective_tensor_layout(target), &mismatches);
  const bool bounds = compare_tensor_bounds(source_contract ? &*source_contract : nullptr,
                                            target_contract ? &*target_contract : nullptr,
                                            &mismatches);
  if (!layout || !bounds) {
    return make_unexpected(
        connection_error(state, source, target, "GRAPH_TENSOR_LAYOUT_EDGE_CONFLICT", mismatches));
  }
  if (!tensor_field_layouts_compatible(source, target, mismatches)) {
    return make_unexpected(connection_error(
        state, source, target, "GRAPH_TENSOR_FIELD_LAYOUT_EDGE_CONFLICT", mismatches));
  }
  std::optional<PayloadOptionsRecord> producer = std::move(source_override);
  if (!producer) {
    producer = source.record.graph_override.payload_options
                   ? source.record.graph_override.payload_options
                   : source.record.setup.payload_options;
  }
  std::optional<PayloadOptionsRecord> boundary = boundary_tensor_payload(state, target.record.id);
  if (!boundary)
    boundary = boundary_tensor_payload(state, source.record.id);

  if (producer && boundary &&
      (producer->interface_tensor_emit_policy != boundary->interface_tensor_emit_policy ||
       producer->max_inline_bytes != boundary->max_inline_bytes ||
       producer->max_tensor_byte_span != boundary->max_tensor_byte_span)) {
    append_tensor_payload_mismatches(*producer, *boundary, &mismatches);
    return make_unexpected(connection_error(
        state, source, target, "GRAPH_TENSOR_PAYLOAD_POLICY_CONFLICT", mismatches));
  }
  if (producer)
    return *producer;
  if (boundary)
    return *boundary;
  return PayloadOptionsRecord{};
}

using EdgePayloadUpdates = std::vector<std::pair<std::size_t, std::optional<PayloadOptionsRecord>>>;

[[nodiscard]] expected<EdgePayloadUpdates, Error> recompute_incident_tensor_payloads(
    const GraphState& state, GraphPortId changed_port, const PortStorage& candidate) {
  EdgePayloadUpdates updates;
  for (std::size_t index = 0; index < state.edges_.size(); ++index) {
    const auto& edge = state.edges_[index];
    if (edge.source != changed_port && edge.target != changed_port)
      continue;
    const PortStorage* stored_source = find_port(state, edge.source);
    const PortStorage* stored_target = find_port(state, edge.target);
    if (stored_source == nullptr || stored_target == nullptr) {
      return make_unexpected(graph_error(ErrorCode::kFailure, "GRAPH_EDGE_PORT_MISSING"));
    }
    const PortStorage& source = edge.source == changed_port ? candidate : *stored_source;
    const PortStorage& target = edge.target == changed_port ? candidate : *stored_target;
    const bool tensor_edge = declares_tensor_semantics(source) || declares_tensor_semantics(target);
    if (!tensor_edge) {
      updates.emplace_back(index, std::nullopt);
      continue;
    }
    auto normalized = effective_tensor_payload(state, source, target);
    if (!normalized)
      return make_unexpected(std::move(normalized.error()));
    updates.emplace_back(index, std::move(normalized.value()));
  }
  return updates;
}

template <typename PortBaseT>
[[nodiscard]] expected<GraphPortId, Error> register_interface_port(
    GraphState& state, const PortBaseT& port, PortDirection direction,
    PayloadIdentityRecord payload_identity, std::uint64_t serialized_payload_capacity_bytes) {
  const void* owner = GraphPortOwnershipAccess::owner(port);
  if (owner != nullptr && owner != &state) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CROSS_GRAPH_PORT"));
  }
  const auto found = state.port_by_declaration_.find(&port);
  if (found != state.port_by_declaration_.end()) {
    PortStorage* existing = find_port(state, found->second);
    if (existing == nullptr || existing->record.owner_kind != PortOwnerKind::kGraphInterface ||
        existing->record.direction != direction ||
        !payload_identity_exactly_equal(existing->record.payload_identity, payload_identity) ||
        existing->record.serialized_payload_capacity_bytes != serialized_payload_capacity_bytes) {
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_INTERFACE_PORT_CONFLICT"));
    }
    return found->second;
  }
  auto name = validate_name(port.name(), "GRAPH_INTERFACE_PORT_NAME_INVALID");
  if (!name)
    return make_unexpected(std::move(name.error()));
  auto identity = validate_payload_identity(payload_identity);
  if (!identity)
    return make_unexpected(std::move(identity.error()));
  const bool schema = payload_identity.kind == PayloadIdentityKind::kSchema;
  const bool py_object = is_py_object_payload_identity(payload_identity);
  if ((schema && !py_object) != (serialized_payload_capacity_bytes != 0U) ||
      serialized_payload_capacity_bytes > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_SCHEMA_CAPACITY_INVALID"));
  }
  auto catalog = validate_pod_catalog(state, payload_identity);
  if (!catalog)
    return make_unexpected(std::move(catalog.error()));
  if (state.interface_port_names_.contains(std::string(port.name()))) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_INTERFACE_NAME_DUPLICATE"));
  }
  const VertexId vertex = state.next_vertex_id();
  const GraphPortId port_id = state.next_port_id();
  state.vertices_.push_back(VertexRecord{
      .id = vertex,
      .graph = state.graph_record_.id,
      .kind = VertexKind::kGraphInterface,
      .name = std::string(port.name()),
  });
  state.vertex_adjacency_.emplace_back();
  state.cycle_visit_marks_.push_back(0U);
  GraphMutationRollback::checkpoint(state);
  state.ports_.push_back(PortStorage{
      .record =
          PortRecord{
              .id = port_id,
              .owner = vertex,
              .owner_kind = PortOwnerKind::kGraphInterface,
              .direction = direction,
              .name = std::string(port.name()),
              .payload_identity = std::move(payload_identity),
              .serialized_payload_capacity_bytes = serialized_payload_capacity_bytes,
          },
      .handle_identity = std::make_shared<std::byte>(),
      .declaration = &port,
  });
  GraphMutationRollback::checkpoint(state);
  state.port_by_declaration_.emplace(&port, port_id);
  GraphMutationRollback::checkpoint(state);
  state.interface_port_names_.emplace(port.name());
  GraphMutationRollback::checkpoint(state);
  record_pod_identity(state, state.ports_.back().record.payload_identity);
  if (state.ports_.back().record.payload_identity.pod)
    GraphMutationRollback::checkpoint(state);
  GraphPortOwnershipAccess::claim(port, &state);
  GraphMutationRollback::checkpoint(state);
  return port_id;
}

}  // namespace

// Flow authoring validates and commits every related graph index atomically.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<FlowRegistration, Error> graph_add_flow(
    const std::shared_ptr<GraphState>& state, const OutputBase* source, const InputBase* target,
    const PayloadIdentityRecord& source_identity, const PayloadIdentityRecord& target_identity,
    ConnectionOptions options, std::optional<TensorProjectionRecord> tensor_projection) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  if (source == nullptr || target == nullptr) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_FLOW_PORT_INVALID"));
  }
  auto options_valid = validate_connection_options(options);
  if (!options_valid) {
    return make_unexpected(std::move(options_valid.error()));
  }
  const void* source_owner = GraphPortOwnershipAccess::owner(*source);
  const void* target_owner = GraphPortOwnershipAccess::owner(*target);
  const std::lock_guard lock(state->mutex_);
  const auto port_location = [&](const void* declaration, std::string_view fallback) {
    const auto registered = state->port_by_declaration_.find(declaration);
    const auto* port = registered != state->port_by_declaration_.end()
                           ? find_port(*state, registered->second)
                           : nullptr;
    if (port != nullptr) {
      const auto owner = std::ranges::find(state->vertices_, port->record.owner, &VertexRecord::id);
      if (owner != state->vertices_.end()) {
        return diagnostic_display_text(authored_operator_path(*state, *owner)) + "." +
               diagnostic_display_text(port->record.name);
      }
    }
    return fmt::format("unregistered port '{}'", diagnostic_display_text(fallback));
  };
  const auto contextual_error = [&](Error error) {
    error.message += fmt::format(": attempted connection {} -> {}",
                                 port_location(source, source->name()),
                                 port_location(target, target->name()));
    return make_unexpected(std::move(error));
  };
  const auto owns_embedded_state =
      [&](const auto& self, const GraphState& candidate, const void* owner) -> bool {
    for (const auto& embedded : candidate.embedded_graphs_) {
      if (embedded.state.get() == owner ||
          (embedded.state != nullptr && self(self, *embedded.state, owner))) {
        return true;
      }
    }
    return false;
  };
  // Embedding copies a child root interface into the parent declaration map so
  // parent authoring can connect that public boundary. Other declarations from
  // the embedded hierarchy intentionally fall through to GRAPH_PORT_NOT_FOUND;
  // a declaration owned by an unrelated Graph remains a cross-Graph error.
  const auto local_or_embedded_interface = [&](const void* declaration, const void* owner) {
    if (owner == nullptr || owner == state.get()) {
      return true;
    }
    const auto found = state->port_by_declaration_.find(declaration);
    if (found == state->port_by_declaration_.end()) {
      return owns_embedded_state(owns_embedded_state, *state, owner);
    }
    const PortStorage* port = find_port(*state, found->second);
    return port != nullptr && port->record.owner_kind == PortOwnerKind::kGraphInterface;
  };
  if (!local_or_embedded_interface(source, source_owner) ||
      !local_or_embedded_interface(target, target_owner)) {
    return contextual_error(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CROSS_GRAPH_PORT"));
  }
  auto source_id =
      find_port_by_declaration(*state, source, PortDirection::kOutput, source_identity);
  if (!source_id)
    return contextual_error(std::move(source_id.error()));
  auto target_id = find_port_by_declaration(*state, target, PortDirection::kInput, target_identity);
  if (!target_id)
    return contextual_error(std::move(target_id.error()));
  const PortStorage* source_port = find_port(*state, source_id.value());
  const PortStorage* target_port = find_port(*state, target_id.value());
  if (tensor_projection) {
    if (source_identity.kind != PayloadIdentityKind::kSchema ||
        target_identity.kind != PayloadIdentityKind::kTensor) {
      return contextual_error(graph_error(ErrorCode::kIncompatible, "GRAPH_PORT_TYPE_MISMATCH"));
    }
  } else if (!payload_identity_exactly_equal(source_identity, target_identity)) {
    return make_unexpected(connection_error(*state,
                                            *source_port,
                                            *target_port,
                                            "GRAPH_PORT_TYPE_MISMATCH",
                                            {{"payload type",
                                              graph_payload_identity_name(source_identity),
                                              graph_payload_identity_name(target_identity)}}));
  }
  if (std::any_of(state->edges_.begin(), state->edges_.end(), [&](const EdgeRecord& edge) {
        return edge.target == target_id.value();
      })) {
    return make_unexpected(connection_error(
        *state,
        *source_port,
        *target_port,
        "GRAPH_INPUT_FANIN_UNSUPPORTED",
        {{"input producers", "an additional producer", "one producer; input already connected"}}));
  }
  if (tensor_projection) {
    // Narrowing a payload is the consumer's decision, so the Tensor input must have said it
    // admits one. Serialized capacity is a schema fact and has no counterpart on a Tensor port.
    const PortStorage* consumer = target_port;
    for (std::size_t remaining = state->ports_.size();
         consumer != nullptr && consumer->record.owner_kind == PortOwnerKind::kGraphInterface;
         --remaining) {
      const auto binding = std::ranges::find_if(state->interface_bindings_, [&](const auto& alias) {
        return alias.public_port == consumer->record.id && alias.direction == PortDirection::kInput;
      });
      if (remaining == 0U || binding == state->interface_bindings_.end()) {
        consumer = nullptr;
        break;
      }
      consumer = find_port(*state, binding->internal_port);
    }
    if (consumer == nullptr || !consumer->record.setup.accepts_projected_tensor) {
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_PROJECTED_TENSOR_NOT_ACCEPTED"));
    }
  } else if (source_port->record.serialized_payload_capacity_bytes !=
             target_port->record.serialized_payload_capacity_bytes) {
    return make_unexpected(connection_error(
        *state,
        *source_port,
        *target_port,
        "GRAPH_SCHEMA_CAPACITY_MISMATCH",
        {{"serialized payload capacity",
          std::to_string(source_port->record.serialized_payload_capacity_bytes),
          std::to_string(target_port->record.serialized_payload_capacity_bytes)}}));
  }
  if ((source_owner != state.get() &&
       source_port->record.owner_kind != PortOwnerKind::kGraphInterface) ||
      (target_owner != state.get() &&
       target_port->record.owner_kind != PortOwnerKind::kGraphInterface)) {
    return contextual_error(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CROSS_GRAPH_PORT"));
  }
  if (options.memory_conversion && !declares_tensor_semantics(*source_port) &&
      !declares_tensor_semantics(*target_port)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_MEMORY_CONVERSION"));
  }
  if (creates_directed_cycle(*state, source_port->record.owner, target_port->record.owner)) {
    return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_DIRECT_CYCLE"));
  }
  std::optional<PayloadOptionsRecord> effective_payload;
  if (declares_tensor_semantics(*source_port) || declares_tensor_semantics(*target_port)) {
    auto normalized = effective_tensor_payload(*state, *source_port, *target_port);
    if (!normalized)
      return make_unexpected(std::move(normalized.error()));
    effective_payload = std::move(normalized.value());
  }
  GraphMutationRollback rollback(*state);
  const EdgeId edge = state->next_edge_id();
  auto handle_identity = std::make_shared<std::byte>();
  state->edge_handle_identities_.reserve(state->edge_handle_identities_.size() + 1U);
  // Allocate before recording the edge so rollback cannot confuse this flow with a prior parallel
  // flow between the same vertices.
  reserve_adjacency(*state, source_port->record.owner);
  state->edges_.push_back(EdgeRecord{
      .id = edge,
      .source = source_id.value(),
      .target = target_id.value(),
      .add_flow_options = copy_connection_options(options),
      .effective_tensor_payload = std::move(effective_payload),
      .tensor_projection = tensor_projection,
  });
  state->edge_handle_identities_.push_back(handle_identity);
  append_adjacency(*state, source_port->record.owner, target_port->record.owner);
  state->mark_mutation();
  GraphMutationRollback::checkpoint(*state);
  rollback.commit();
  graph_log_noexcept([&] { HOLOSCAN_LOG_TRACE("Registered Graph flow {}", edge.value); });
  return FlowRegistration{
      .edge = edge,
      .source = source_id.value(),
      .target = target_id.value(),
      .handle_identity = std::move(handle_identity),
  };
}

expected<void, Error> graph_bind_input_interface(const std::shared_ptr<GraphState>& state,
                                                 const InputBase* public_input,
                                                 const InputBase* internal_input,
                                                 PayloadIdentityRecord payload_identity) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (public_input == nullptr || internal_input == nullptr || public_input == internal_input) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_INTERFACE_PORT_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  auto internal =
      find_port_by_declaration(*state, internal_input, PortDirection::kInput, payload_identity);
  if (!internal)
    return make_unexpected(std::move(internal.error()));
  const PortStorage* internal_port = find_port(*state, internal.value());
  if (GraphPortOwnershipAccess::owner(*internal_input) != state.get() &&
      internal_port->record.owner_kind != PortOwnerKind::kGraphInterface) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CROSS_GRAPH_PORT"));
  }
  GraphMutationRollback rollback(*state);
  auto public_id = register_interface_port(*state,
                                           *public_input,
                                           PortDirection::kInput,
                                           std::move(payload_identity),
                                           internal_port->record.serialized_payload_capacity_bytes);
  if (!public_id)
    return make_unexpected(std::move(public_id.error()));
  if (std::any_of(state->interface_bindings_.begin(),
                  state->interface_bindings_.end(),
                  [&](const InterfaceBindingRecord& binding) {
                    return binding.public_port == public_id.value();
                  })) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_INTERFACE_BINDING_DUPLICATE"));
  }
  const PortStorage* public_port = find_port(*state, public_id.value());
  reserve_adjacency(*state, public_port->record.owner);
  state->interface_bindings_.push_back(InterfaceBindingRecord{
      .public_port = public_id.value(),
      .internal_port = internal.value(),
      .direction = PortDirection::kInput,
  });
  append_adjacency(*state, public_port->record.owner, internal_port->record.owner);
  state->mark_mutation();
  GraphMutationRollback::checkpoint(*state);
  rollback.commit();
  return {};
}

expected<void, Error> graph_bind_output_interface(const std::shared_ptr<GraphState>& state,
                                                  const OutputBase* internal_output,
                                                  const OutputBase* public_output,
                                                  PayloadIdentityRecord payload_identity) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (public_output == nullptr || internal_output == nullptr || public_output == internal_output) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_INTERFACE_PORT_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  auto internal =
      find_port_by_declaration(*state, internal_output, PortDirection::kOutput, payload_identity);
  if (!internal)
    return make_unexpected(std::move(internal.error()));
  const PortStorage* internal_port = find_port(*state, internal.value());
  if (GraphPortOwnershipAccess::owner(*internal_output) != state.get() &&
      internal_port->record.owner_kind != PortOwnerKind::kGraphInterface) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CROSS_GRAPH_PORT"));
  }
  GraphMutationRollback rollback(*state);
  auto public_id = register_interface_port(*state,
                                           *public_output,
                                           PortDirection::kOutput,
                                           std::move(payload_identity),
                                           internal_port->record.serialized_payload_capacity_bytes);
  if (!public_id)
    return make_unexpected(std::move(public_id.error()));
  if (std::any_of(state->interface_bindings_.begin(),
                  state->interface_bindings_.end(),
                  [&](const InterfaceBindingRecord& binding) {
                    return binding.public_port == public_id.value();
                  })) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_INTERFACE_BINDING_DUPLICATE"));
  }
  const PortStorage* public_port = find_port(*state, public_id.value());
  reserve_adjacency(*state, internal_port->record.owner);
  state->interface_bindings_.push_back(InterfaceBindingRecord{
      .public_port = public_id.value(),
      .internal_port = internal.value(),
      .direction = PortDirection::kOutput,
  });
  append_adjacency(*state, internal_port->record.owner, public_port->record.owner);
  state->mark_mutation();
  GraphMutationRollback::checkpoint(*state);
  rollback.commit();
  return {};
}

expected<GraphPortId, Error> graph_find_port(const std::shared_ptr<GraphState>& state,
                                             const void* declaration, PortDirection direction,
                                             const PayloadIdentityRecord& payload_identity) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  const std::lock_guard lock(state->mutex_);
  return find_port_by_declaration(*state, declaration, direction, payload_identity);
}

expected<std::shared_ptr<const void>, Error> graph_port_registration_identity(
    const std::shared_ptr<GraphState>& state, GraphPortId port) {
  if (!state) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  const PortStorage* found = find_port(*state, port);
  if (found == nullptr || !found->handle_identity) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_PORT_NOT_FOUND"));
  }
  return found->handle_identity;
}

const void* graph_port_declaration(const std::shared_ptr<GraphState>& state, GraphPortId port,
                                   PortDirection direction) noexcept {
  const std::lock_guard lock(state->mutex_);
  const PortStorage* found = find_port(*state, port);
  if (found == nullptr || found->record.direction != direction)
    return nullptr;
  return found->declaration;
}

const void* graph_port_declaration(const std::shared_ptr<GraphState>& state, GraphPortId port,
                                   PortDirection direction, const void* handle_identity) noexcept {
  const std::lock_guard lock(state->mutex_);
  const PortStorage* found = find_port(*state, port);
  if (found == nullptr || found->record.direction != direction ||
      found->handle_identity.get() != handle_identity) {
    return nullptr;
  }
  return found->declaration;
}

const void* graph_port_declaration(const std::shared_ptr<GraphState>& state, GraphPortId port,
                                   PortDirection direction,
                                   const PayloadIdentityRecord& payload_identity) noexcept {
  const std::lock_guard lock(state->mutex_);
  const PortStorage* found = find_port(*state, port);
  if (found == nullptr || found->record.direction != direction ||
      !payload_identity_exactly_equal(found->record.payload_identity, payload_identity)) {
    return nullptr;
  }
  return found->declaration;
}

expected<FlowRegistration, Error> graph_testing_flow_registration(
    const std::shared_ptr<GraphState>& state, GraphPortId source, GraphPortId target,
    const void* source_identity, const void* target_identity) {
  if (!state || source_identity == nullptr || target_identity == nullptr) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  const auto* from = find_port(*state, source);
  const auto* to = find_port(*state, target);
  if (from == nullptr || to == nullptr || from->handle_identity.get() != source_identity ||
      to->handle_identity.get() != target_identity ||
      from->record.direction != PortDirection::kOutput ||
      to->record.direction != PortDirection::kInput) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  std::optional<FlowRegistration> found;
  for (std::size_t index = 0U; index < state->edges_.size(); ++index) {
    const auto& edge = state->edges_[index];
    if (edge.source != source || edge.target != target) {
      continue;
    }
    if (found || index >= state->edge_handle_identities_.size()) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_FLOW_AMBIGUOUS"));
    }
    found = FlowRegistration{edge.id, source, target, state->edge_handle_identities_[index]};
  }
  if (!found) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_FLOW_NOT_FOUND"));
  }
  return std::move(*found);
}

bool graph_flow_valid(const std::shared_ptr<GraphState>& state, EdgeId edge, GraphPortId source,
                      GraphPortId target, const void* handle_identity) noexcept {
  if (!state || !edge || handle_identity == nullptr)
    return false;
  const std::lock_guard lock(state->mutex_);
  const EdgeRecord* record = find_edge(*state, edge);
  const std::size_t index = static_cast<std::size_t>(edge.value - 1U);
  return record != nullptr && index < state->edge_handle_identities_.size() &&
         state->edge_handle_identities_[index].get() == handle_identity &&
         record->source == source && record->target == target;
}

namespace {

template <typename Mutation>
expected<void, Error> configure_port(const std::shared_ptr<GraphState>& state, GraphPortId id,
                                     const void* handle_identity, Mutation&& mutation) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  PortStorage* port = find_port(*state, id);
  if (port == nullptr) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  if (port->handle_identity.get() != handle_identity) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  auto result = std::forward<Mutation>(mutation)(*port);
  if (!result)
    return result;
  state->mark_mutation();
  return {};
}

template <typename Mutation>
expected<void, Error> configure_edge(const std::shared_ptr<GraphState>& state, EdgeId id,
                                     const void* handle_identity, Mutation&& mutation) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  EdgeRecord* edge = find_edge(*state, id);
  const std::size_t index = id ? static_cast<std::size_t>(id.value - 1U) : 0U;
  if (edge == nullptr || index >= state->edge_handle_identities_.size()) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  if (state->edge_handle_identities_[index].get() != handle_identity) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  std::forward<Mutation>(mutation)(*edge);
  state->mark_mutation();
  return {};
}

}  // namespace

expected<void, Error> graph_configure_input_queue(const std::shared_ptr<GraphState>& state,
                                                  GraphPortId id, const void* handle_identity,
                                                  std::uint32_t depth) {
  return configure_port(
      state, id, handle_identity, [depth](PortStorage& port) -> expected<void, Error> {
        if (port.record.direction != PortDirection::kInput || depth == 0U) {
          return make_unexpected(
              graph_error(ErrorCode::kInvalidArgument, "GRAPH_INPUT_QUEUE_INVALID"));
        }
        port.record.graph_override.queue_depth = depth;
        return {};
      });
}

expected<void, Error> graph_configure_input_tensor_spec(const std::shared_ptr<GraphState>& state,
                                                        GraphPortId id, const void* handle_identity,
                                                        const TensorInputSpec& spec) {
  auto normalized = normalize_tensor_input_spec(spec);
  if (!normalized) {
    return make_unexpected(std::move(normalized.error()));
  }
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  PortStorage* port = find_port(*state, id);
  if (port == nullptr || port->handle_identity.get() != handle_identity ||
      port->record.direction != PortDirection::kInput) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_INPUT_CONFIG_INVALID"));
  }
  if (!declares_tensor_semantics(*port)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_LAYOUT_MUTATION"));
  }
  PortStorage candidate = *port;
  candidate.record.graph_override.tensor_layout = tensor_representation(*normalized);
  candidate.record.graph_override.tensor_contract = *normalized;
  candidate.record.graph_override.memory_conversion = normalized->memory_conversion;
  auto edge_updates = recompute_incident_tensor_payloads(*state, id, candidate);
  if (!edge_updates) {
    return make_unexpected(std::move(edge_updates.error()));
  }
  GraphMutationRollback::checkpoint(*state);
  port->record.graph_override.tensor_layout = candidate.record.graph_override.tensor_layout;
  port->record.graph_override.tensor_contract = std::move(*normalized);
  port->record.graph_override.memory_conversion = candidate.record.graph_override.memory_conversion;
  for (auto& [index, effective] : edge_updates.value()) {
    state->edges_[index].effective_tensor_payload = std::move(effective);
  }
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_configure_port_alias(const std::shared_ptr<GraphState>& state,
                                                 GraphPortId id, const void* handle_identity,
                                                 std::string_view alias) {
  auto valid = validate_name(alias, "GRAPH_PORT_ALIAS_INVALID");
  if (!valid)
    return valid;
  return configure_port(
      state, id, handle_identity, [alias](PortStorage& port) -> expected<void, Error> {
        port.record.graph_override.diagnostic_alias = std::string(alias);
        return {};
      });
}

expected<void, Error> graph_configure_output_emits(const std::shared_ptr<GraphState>& state,
                                                   GraphPortId id, const void* handle_identity,
                                                   std::uint32_t count) {
  return configure_port(
      state, id, handle_identity, [count](PortStorage& port) -> expected<void, Error> {
        if (port.record.direction != PortDirection::kOutput || count == 0U) {
          return make_unexpected(
              graph_error(ErrorCode::kInvalidArgument, "GRAPH_OUTPUT_BOUND_INVALID"));
        }
        port.record.graph_override.max_emits_per_compute = count;
        return {};
      });
}

expected<void, Error> graph_configure_output_history(const std::shared_ptr<GraphState>& state,
                                                     GraphPortId id, const void* handle_identity,
                                                     std::uint32_t depth) {
  return configure_port(
      state, id, handle_identity, [depth](PortStorage& port) -> expected<void, Error> {
        if (port.record.direction != PortDirection::kOutput || depth == 0U) {
          return make_unexpected(
              graph_error(ErrorCode::kInvalidArgument, "GRAPH_OUTPUT_CONFIG_INVALID"));
        }
        port.record.graph_override.late_joiner_history_depth = depth;
        return {};
      });
}

[[nodiscard]] expected<TensorContractRecord, Error> select_output_tensor_spec(
    const PortStorage& port, const TensorOutputSpec& spec,
    expected<TensorContractRecord, Error> normalized,
    expected<TensorContractRecord, Error> derived_refinement) {
  const bool field_scoped_derivation =
      std::any_of(port.record.setup.tensor_fields.begin(),
                  port.record.setup.tensor_fields.end(),
                  [](const auto& field) {
                    return field.tensor_byte_span_expression.has_value() ||
                           field.tensor_memory_kind_expression.has_value();
                  });
  if (field_scoped_derivation) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_SCOPE_REQUIRED"));
  }
  const bool derived_byte_span = port.record.setup.tensor_byte_span_expression.has_value();
  const bool derived_memory_kind = port.record.setup.tensor_memory_kind_expression.has_value();
  const bool derived_output = derived_byte_span || derived_memory_kind;
  if (derived_memory_kind && spec.representation.memory_kind) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_MEMORY_KIND_SOURCE_CONFLICT"));
  }
  if (derived_byte_span && spec.storage == TensorOutputStorage::kPassThrough) {
    if (!derived_refinement) {
      return make_unexpected(std::move(derived_refinement.error()));
    }
    normalized = std::move(derived_refinement);
  } else if (!normalized) {
    return make_unexpected(std::move(normalized.error()));
  }
  if (!derived_output) {
    return std::move(normalized.value());
  }

  const auto& setup_contract = port.record.setup.tensor_contract;
  const auto effective = resolve_output_tensor_contract(
      setup_contract, std::optional<TensorContractRecord>{normalized.value()});
  const bool compatible_refinement =
      normalized->storage == TensorOutputStorage::kPassThrough && !spec.bounds && setup_contract &&
      effective &&
      tensor_layouts_compatible(std::optional{setup_contract->representation},
                                std::optional{effective->representation}) &&
      (!effective->max_rank || !setup_contract->max_rank ||
       *effective->max_rank <= *setup_contract->max_rank);
  if (!compatible_refinement) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument,
                                       "GRAPH_TENSOR_DERIVED_ALLOCATION_OVERRIDE_INVALID"));
  }
  return std::move(normalized.value());
}

expected<void, Error> graph_configure_output_tensor_spec(const std::shared_ptr<GraphState>& state,
                                                         GraphPortId id,
                                                         const void* handle_identity,
                                                         const TensorOutputSpec& spec) {
  auto normalized = normalize_tensor_output_spec(spec);
  auto derived_refinement = normalize_tensor_output_spec_refinement_with_derived_byte_span(spec);
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  PortStorage* port = find_port(*state, id);
  if (port == nullptr || port->handle_identity.get() != handle_identity ||
      port->record.direction != PortDirection::kOutput) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_OUTPUT_CONFIG_INVALID"));
  }
  if (!declares_tensor_semantics(*port)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_LAYOUT_MUTATION"));
  }
  auto selected =
      select_output_tensor_spec(*port, spec, std::move(normalized), std::move(derived_refinement));
  if (!selected) {
    return make_unexpected(std::move(selected.error()));
  }
  TensorContractRecord normalized_contract = std::move(selected.value());
  if (!port->record.graph_override.tensor_contract && port->record.graph_override.payload_options &&
      payload_options_carry_tensor_transfer(*port->record.graph_override.payload_options)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT"));
  }
  PortStorage candidate = *port;
  candidate.record.graph_override.tensor_layout = tensor_representation(normalized_contract);
  candidate.record.graph_override.tensor_contract = normalized_contract;
  if (normalized_contract.transfer) {
    const auto& transfer = normalized_tensor_transfer(normalized_contract);
    PayloadOptionsRecord options{
        .max_inline_bytes =
            transfer.max_inline_bytes
                ? std::optional{static_cast<std::uint32_t>(*transfer.max_inline_bytes)}
                : std::nullopt,
        .max_tensor_byte_span = transfer.max_staging_byte_span,
        .interface_tensor_emit_policy = transfer.foreign_emit_policy,
    };
    if (candidate.record.payload_identity.schema) {
      options.schema = candidate.record.payload_identity.schema;
    }
    candidate.record.graph_override.payload_options = std::move(options);
  } else {
    candidate.record.graph_override.payload_options.reset();
  }
  if (normalized_contract.storage == TensorOutputStorage::kRuntimePool) {
    const TensorPoolBounds bounds = normalized_tensor_pool_bounds(normalized_contract);
    const bool supports_singular_tensor_allocation =
        candidate.record.payload_identity.kind == PayloadIdentityKind::kTensor ||
        (candidate.record.payload_identity.payload_kind == PayloadKind::kDescriptorBacked &&
         candidate.record.descriptor_tensor_field_count == 1U);
    if (!supports_singular_tensor_allocation || !candidate.setup_tensor_allocation ||
        bounds.max_rank > candidate.setup_tensor_allocation->max_rank ||
        bounds.max_byte_span > candidate.setup_tensor_allocation->max_byte_span) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_ALLOCATION_NOT_TIGHTENING"));
    }
    candidate.record.graph_override.tensor_allocation = bounds;
  } else {
    candidate.record.graph_override.tensor_allocation.reset();
  }
  auto edge_updates = recompute_incident_tensor_payloads(*state, id, candidate);
  if (!edge_updates) {
    return make_unexpected(std::move(edge_updates.error()));
  }
  GraphMutationRollback::checkpoint(*state);
  port->record.graph_override.tensor_layout = candidate.record.graph_override.tensor_layout;
  port->record.graph_override.tensor_contract = std::move(normalized_contract);
  port->record.graph_override.payload_options = candidate.record.graph_override.payload_options;
  port->record.graph_override.tensor_allocation = candidate.record.graph_override.tensor_allocation;
  for (auto& [index, effective] : edge_updates.value()) {
    state->edges_[index].effective_tensor_payload = std::move(effective);
  }
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_configure_output_payload(const std::shared_ptr<GraphState>& state,
                                                     GraphPortId id, const void* handle_identity,
                                                     PayloadOptions options) {
  PayloadOptionsRecord copied = holoscan::detail::copy_payload_options(options);
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  PortStorage* port = find_port(*state, id);
  if (port == nullptr || port->handle_identity.get() != handle_identity ||
      port->record.direction != PortDirection::kOutput) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_OUTPUT_CONFIG_INVALID"));
  }
  const auto layout = effective_tensor_layout(*port);
  if ((port->record.graph_override.tensor_contract ||
       port->record.setup.tensor_byte_span_expression ||
       port->record.setup.tensor_memory_kind_expression) &&
      payload_options_carry_tensor_transfer(copied)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT"));
  }
  auto valid = validate_payload_options(copied, port->record.payload_identity, layout);
  if (!valid)
    return valid;
  if (port->record.graph_override.tensor_contract && port->record.graph_override.payload_options) {
    copied.max_inline_bytes = port->record.graph_override.payload_options->max_inline_bytes;
    copied.max_tensor_byte_span = port->record.graph_override.payload_options->max_tensor_byte_span;
    copied.interface_tensor_emit_policy =
        port->record.graph_override.payload_options->interface_tensor_emit_policy;
  }
  std::vector<std::pair<std::size_t, PayloadOptionsRecord>> normalized;
  if (declares_tensor_semantics(*port)) {
    for (std::size_t index = 0; index < state->edges_.size(); ++index) {
      auto& edge = state->edges_[index];
      if (edge.source != id)
        continue;
      const PortStorage* target = find_port(*state, edge.target);
      auto effective = effective_tensor_payload(*state, *port, *target, copied);
      if (!effective)
        return make_unexpected(std::move(effective.error()));
      normalized.emplace_back(index, std::move(effective.value()));
    }
  }
  using PayloadOverride = decltype(port->record.graph_override.payload_options);
  using EffectivePayload = decltype(state->edges_[0].effective_tensor_payload);
  static_assert(std::is_nothrow_move_assignable_v<PayloadOverride>);
  static_assert(std::is_nothrow_move_assignable_v<EffectivePayload>);
  GraphMutationRollback::checkpoint(*state);
  port->record.graph_override.payload_options = std::move(copied);
  for (auto& [index, effective] : normalized)
    state->edges_[index].effective_tensor_payload = std::move(effective);
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_configure_flow_policy(const std::shared_ptr<GraphState>& state,
                                                  EdgeId id, const void* handle_identity,
                                                  QueuePolicy policy) {
  if (!closed_enum(policy)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_QUEUE_POLICY_INVALID"));
  }
  return configure_edge(state, id, handle_identity, [policy](EdgeRecord& edge) {
    edge.graph_override.queue_policy = policy;
  });
}

expected<void, Error> graph_configure_flow_depth(const std::shared_ptr<GraphState>& state,
                                                 EdgeId id, const void* handle_identity,
                                                 std::uint32_t depth) {
  if (depth == 0U) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_FLOW_BOUND_INVALID"));
  }
  return configure_edge(state, id, handle_identity, [depth](EdgeRecord& edge) {
    edge.graph_override.queue_depth = depth;
  });
}

expected<void, Error> graph_configure_flow_topic(const std::shared_ptr<GraphState>& state,
                                                 EdgeId id, const void* handle_identity,
                                                 std::string_view name) {
  auto valid = validate_name(name, "GRAPH_FLOW_TOPIC_NAME_INVALID");
  if (!valid)
    return valid;
  return configure_edge(state, id, handle_identity, [name](EdgeRecord& edge) {
    edge.graph_override.topic_name = std::string(name);
  });
}

expected<void, Error> graph_configure_flow_memory(const std::shared_ptr<GraphState>& state,
                                                  EdgeId id, const void* handle_identity,
                                                  MemoryConversionPolicy policy) {
  if (!closed_enum(policy)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_MEMORY_POLICY_INVALID"));
  }
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  EdgeRecord* edge = find_edge(*state, id);
  const std::size_t index = id ? static_cast<std::size_t>(id.value - 1U) : 0U;
  if (edge == nullptr || index >= state->edge_handle_identities_.size() ||
      state->edge_handle_identities_[index].get() != handle_identity) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_FLOW_NOT_FOUND"));
  }
  const PortStorage* source = find_port(*state, edge->source);
  const PortStorage* target = find_port(*state, edge->target);
  if (source == nullptr || target == nullptr ||
      (!declares_tensor_semantics(*source) && !declares_tensor_semantics(*target))) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_MEMORY_CONVERSION"));
  }
  edge->graph_override.memory_conversion = policy;
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_configure_flow_inflight(const std::shared_ptr<GraphState>& state,
                                                    EdgeId id, const void* handle_identity,
                                                    std::uint32_t count) {
  if (count == 0U) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_FLOW_BOUND_INVALID"));
  }
  return configure_edge(state, id, handle_identity, [count](EdgeRecord& edge) {
    edge.graph_override.max_inflight_samples = count;
  });
}

namespace {

[[nodiscard]] expected<void, Error> validate_topic_uri(std::string_view uri) {
  if (!uri.starts_with(kHoloMqTopicPrefix) || uri.size() == kHoloMqTopicPrefix.size() ||
      uri.size() > 4096U || !valid_utf8(uri) || edge_whitespace(uri) ||
      std::any_of(uri.begin(), uri.end(), [](char value) {
        const auto byte = static_cast<unsigned char>(value);
        return byte == 0U || byte < 0x20U || byte == 0x7FU;
      })) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_URI_INVALID"));
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_clock_health(const ClockHealthOptions& health,
                                                          bool imported) {
  if (!closed_enum(health.capability) || !closed_enum(health.requirement)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "ENDPOINT_CLOCK_HEALTH_ENUM_INVALID"));
  }
  const bool ordinary = health.capability == ClockHealthCapability::kUnavailable &&
                        health.requirement == ClockHealthRequirement::kNotRequired &&
                        !health.max_report_age;
  const bool monitored = imported && health.capability == ClockHealthCapability::kStatusReports &&
                         health.requirement == ClockHealthRequirement::kFreshHealthyReport &&
                         health.max_report_age && health.max_report_age->count() > 0;
  if (!ordinary && !monitored) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID"));
  }
  return {};
}

[[nodiscard]] TensorBoundaryContractRecord copy_tensor_boundary_contract_record(
    const TensorContractRecord& tensor) {
  const auto& transfer = tensor.transfer.value_or(TensorTransferPolicy{});
  return TensorBoundaryContractRecord{
      .memory_kind = tensor.representation.memory_kind,
      .dtype = tensor.representation.dtype,
      .rank = tensor.representation.rank,
      .max_rank = tensor.max_rank,
      .max_byte_span = tensor.max_byte_span,
      .axes = tensor.axes,
      // Normalization guarantees a concrete stride policy for graph Tensor contracts.
      .strides = *tensor.representation.strides,  // NOLINT(bugprone-unchecked-optional-access)
      .foreign_emit_policy = transfer.foreign_emit_policy,
      .max_inline_bytes =
          transfer.max_inline_bytes
              ? std::optional{static_cast<std::uint32_t>(*transfer.max_inline_bytes)}
              : std::nullopt,
      .max_staging_byte_span = transfer.max_staging_byte_span,
      .memory_conversion = tensor.memory_conversion,
      .origins = tensor.origins,
  };
}

[[nodiscard]] expected<std::optional<TensorBoundaryContractRecord>, Error> copy_tensor_requirements(
    const TopicOptions& options, const PayloadIdentityRecord& identity) {
  if (identity.kind != PayloadIdentityKind::kTensor) {
    if (options.tensor) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_OPTIONS_INVALID"));
    }
    return std::nullopt;
  }
  if (!options.tensor)
    return std::nullopt;
  TensorBoundaryContract boundary = *options.tensor;
  if (!boundary.memory_conversion) {
    boundary.memory_conversion = options.memory_conversion;
  } else if (options.memory_conversion && boundary.memory_conversion != options.memory_conversion) {
    return make_unexpected(
        graph_error(ErrorCode::kIncompatible, "GRAPH_TENSOR_MEMORY_POLICY_CONFLICT"));
  }
  auto normalized = normalize_tensor_boundary_contract(boundary);
  if (!normalized) {
    return make_unexpected(std::move(normalized.error()));
  }
  return std::optional<TensorBoundaryContractRecord>{
      copy_tensor_boundary_contract_record(*normalized)};
}

// Descriptor-field identity, layout, and transfer checks form one ordered schema transaction.
[[nodiscard]] expected<std::vector<DescriptorTensorTopicFieldRecord>, Error>
copy_descriptor_tensor_requirements(  // NOLINT(readability-function-cognitive-complexity)
    const TopicOptions& options, const PayloadIdentityRecord& identity,
    const std::vector<AuthoredDescriptorTensorTopicField>& authored_fields,
    const std::uint16_t descriptor_tensor_field_count) {
  const bool descriptor_schema = identity.kind == PayloadIdentityKind::kSchema &&
                                 identity.payload_kind == PayloadKind::kDescriptorBacked;
  if (!descriptor_schema) {
    if (descriptor_tensor_field_count != 0U || !authored_fields.empty()) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_DESCRIPTOR_TENSOR_TOPIC_SCHEMA_INVALID"));
    }
    return std::vector<DescriptorTensorTopicFieldRecord>{};
  }
  if (options.tensor || options.memory_conversion) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument,
                                       "GRAPH_DESCRIPTOR_TENSOR_TOPIC_FIELD_SCOPE_REQUIRED"));
  }
  if (descriptor_tensor_field_count == 0U ||
      authored_fields.size() != descriptor_tensor_field_count) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument,
                                       "GRAPH_DESCRIPTOR_TENSOR_TOPIC_INVENTORY_INCOMPLETE"));
  }

  std::vector<DescriptorTensorTopicFieldRecord> result;
  result.reserve(authored_fields.size());
  std::vector<std::uint16_t> field_ids;
  std::vector<std::uint16_t> tensor_indices;
  field_ids.reserve(authored_fields.size());
  tensor_indices.reserve(authored_fields.size());
  for (const auto& authored : authored_fields) {
    if (!authored.selector_valid || authored.tensor_index >= descriptor_tensor_field_count) {
      return make_unexpected(descriptor_topic_field_graph_error(
          ErrorCode::kInvalidArgument, "GRAPH_DESCRIPTOR_TENSOR_TOPIC_SELECTOR_INVALID", authored));
    }
    if (std::find(field_ids.begin(), field_ids.end(), authored.field_id) != field_ids.end() ||
        std::find(tensor_indices.begin(), tensor_indices.end(), authored.tensor_index) !=
            tensor_indices.end()) {
      return make_unexpected(descriptor_topic_field_graph_error(
          ErrorCode::kDuplicateName, "GRAPH_DESCRIPTOR_TENSOR_TOPIC_FIELD_DUPLICATE", authored));
    }
    if (!authored.contract) {
      return make_unexpected(descriptor_topic_field_graph_error(
          ErrorCode::kInvalidArgument, "GRAPH_DESCRIPTOR_TENSOR_TOPIC_CONTRACT_MISSING", authored));
    }
    auto normalized = normalize_tensor_boundary_contract(*authored.contract);
    if (!normalized) {
      Error error = std::move(normalized.error());
      const std::string_view token = error.message.empty()
                                         ? "GRAPH_DESCRIPTOR_TENSOR_TOPIC_CONTRACT_INVALID"
                                         : std::string_view{error.message};
      error = descriptor_topic_field_graph_error(error.code, token, authored);
      return make_unexpected(std::move(error));
    }
    field_ids.push_back(authored.field_id);
    tensor_indices.push_back(authored.tensor_index);
    result.push_back(DescriptorTensorTopicFieldRecord{
        .field_id = authored.field_id,
        .tensor_index = authored.tensor_index,
        .tensor = copy_tensor_boundary_contract_record(*normalized),
    });
  }
  std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
    return left.field_id < right.field_id;
  });
  for (std::size_t index = 0; index < result.size(); ++index) {
    if (result[index].tensor_index != index) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_DESCRIPTOR_TENSOR_TOPIC_ORDER_INVALID"));
    }
  }
  return result;
}

[[nodiscard]] PortConfigurationRecord descriptor_topic_port_configuration(
    const std::span<const DescriptorTensorTopicFieldRecord> fields, const PortDirection direction) {
  PortConfigurationRecord result;
  result.tensor_fields.reserve(fields.size());
  for (const auto& field : fields) {
    const auto& tensor = field.tensor;
    result.tensor_fields.push_back(TensorFieldConfigurationRecord{
        .field_id = field.field_id,
        .tensor_index = field.tensor_index,
        .tensor_layout =
            TensorRepresentation{
                .memory_kind = tensor.memory_kind,
                .dtype = tensor.dtype,
                .rank = tensor.rank,
                .strides = tensor.strides,
            },
        .memory_conversion = tensor.memory_conversion,
        .tensor_allocation =
            TensorPoolBounds{
                .max_rank = tensor.max_rank.value_or(tensor.rank.value_or(0U)),
                .max_byte_span = static_cast<std::size_t>(tensor.max_byte_span.value_or(0U)),
            },
        .payload_options =
            TensorFieldTransferRecord{
                .max_inline_bytes = tensor.max_inline_bytes,
                .max_tensor_byte_span =
                    tensor.max_staging_byte_span.value_or(tensor.max_byte_span.value_or(0U)),
                .interface_tensor_emit_policy = tensor.foreign_emit_policy,
            },
    });
  }
  if (result.tensor_fields.size() == 1U) {
    const auto& field = result.tensor_fields.front();
    result.tensor_layout = field.tensor_layout;
    // The field construction above always installs payload options.
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    result.payload_options = PayloadOptionsRecord{
        .max_inline_bytes = field.payload_options->max_inline_bytes,
        .max_tensor_byte_span = field.payload_options->max_tensor_byte_span,
        .interface_tensor_emit_policy = field.payload_options->interface_tensor_emit_policy,
    };
    // NOLINTEND(bugprone-unchecked-optional-access)
    if (direction == PortDirection::kInput) {
      result.memory_conversion = field.memory_conversion;
    } else {
      result.tensor_allocation = field.tensor_allocation;
    }
    result.tensor_fields.clear();
  }
  return result;
}

[[nodiscard]] expected<void, Error> validate_endpoint_bounds(const EndpointBounds& bounds,
                                                             bool imported) {
  if ((imported && bounds.subscriber) || (!imported && bounds.publisher) ||
      (bounds.publisher && bounds.subscriber)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_DIRECTION_BOUNDS_INVALID"));
  }
  if (bounds.publisher && bounds.publisher->max_loaned_samples &&
      *bounds.publisher->max_loaned_samples == 0U) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_PUBLISHER_BOUND_INVALID"));
  }
  if (bounds.subscriber &&
      ((bounds.subscriber->queue_depth && *bounds.subscriber->queue_depth == 0U) ||
       (bounds.subscriber->max_borrowed_samples &&
        *bounds.subscriber->max_borrowed_samples == 0U))) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_SUBSCRIBER_BOUND_INVALID"));
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_topic_schema(
    const TopicOptions& options, const PayloadIdentityRecord& identity,
    std::optional<SchemaIdentityRecord>& copied) {
  copied = copy_schema(options.schema);
  if (copied) {
    auto valid = validate_schema(*copied);
    if (!valid)
      return valid;
  }
  if (identity.kind == PayloadIdentityKind::kSchema) {
    if (copied && !schema_equal(copied, identity.schema)) {
      return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_TOPIC_SCHEMA_MISMATCH"));
    }
    copied = identity.schema;
  } else if (identity.kind != PayloadIdentityKind::kTensor && copied) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_SCHEMA_IRRELEVANT"));
  }
  return {};
}

template <typename PortBaseT>
// External-topic registration keeps ordinal consumption, deep copying, contract validation, and
// publication in one fail-closed path so failed declarations cannot partially enter a snapshot.
expected<GraphPortId, Error>
register_external_topic(  // NOLINT(readability-function-cognitive-complexity)
    const std::shared_ptr<GraphState>& state, std::string_view uri, TopicOptions options,
    std::unique_ptr<PortBaseT> port, PayloadIdentityRecord payload_identity,
    std::uint64_t serialized_payload_capacity_bytes,
    // Retain the ownership boundary established by the exported import/export entry points.
    std::vector<AuthoredDescriptorTensorTopicField>
        authored_tensor_fields,  // NOLINT(performance-unnecessary-value-param)
    std::uint16_t descriptor_tensor_field_count, bool imported) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  if (!port) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_PORT_INVALID"));
  }

  const std::unique_lock lock(state->mutex_);
  if (state->next_boundary_ordinal_ == std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(
        graph_error(ErrorCode::kResourceExhausted, "GRAPH_TOPIC_ORDINAL_EXHAUSTED"));
  }
  // Consume the mixed import/export declaration ordinal before
  // semantic validation. The exceptional-only guard distinguishes a normal typed rejection
  // from allocation/copy failure anywhere before the complete mutation transaction exists.
  const std::uint32_t ordinal = state->next_boundary_ordinal_++;
  const ExceptionalBoundaryOrdinalRollback ordinal_rollback{state->next_boundary_ordinal_, ordinal};
  GraphMutationRollback::checkpoint(*state);

  auto valid_uri = validate_topic_uri(uri);
  if (!valid_uri)
    return make_unexpected(std::move(valid_uri.error()));
  auto identity_valid = validate_payload_identity(payload_identity);
  if (!identity_valid)
    return make_unexpected(std::move(identity_valid.error()));
  const bool schema = payload_identity.kind == PayloadIdentityKind::kSchema;
  const bool py_object = is_py_object_payload_identity(payload_identity);
  // Process-local Python objects intentionally pass Graph authoring so compilation can report the
  // more specific boundary-policy diagnostic for an external topic.
  if ((schema && !py_object) != (serialized_payload_capacity_bytes != 0U) ||
      serialized_payload_capacity_bytes > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_SCHEMA_CAPACITY_INVALID"));
  }
  if (payload_identity.kind == PayloadIdentityKind::kScalar &&
      payload_identity.scalar->kind == ScalarIdentityKind::kLocalEnum) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_EXTERNAL_ENUM_IDENTITY_REQUIRED"));
  }
  std::optional<std::string> endpoint_name;
  if (options.endpoint_name) {
    auto valid_name = validate_endpoint_name(*options.endpoint_name);
    if (!valid_name)
      return make_unexpected(std::move(valid_name.error()));
    endpoint_name = std::string(*options.endpoint_name);
  }
  if (options.attach.max_wait && options.attach.max_wait->count() <= 0) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_ATTACH_BOUND_INVALID"));
  }
  if (!closed_enum(options.attach.peer) ||
      (options.memory_conversion && !closed_enum(*options.memory_conversion))) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_ENUM_INVALID"));
  }
  if (options.capture_time && !options.capture_time->clock.is_default() &&
      options.capture_time->clock.name().empty()) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CAPTURE_CLOCK_INVALID"));
  }
  const ClockHealthOptions health = options.clock_health.value_or(ClockHealthOptions{});
  auto valid_health = validate_clock_health(health, imported);
  if (!valid_health)
    return make_unexpected(std::move(valid_health.error()));
  auto valid_bounds = validate_endpoint_bounds(options.bounds, imported);
  if (!valid_bounds)
    return make_unexpected(std::move(valid_bounds.error()));
  auto tensor = copy_tensor_requirements(options, payload_identity);
  if (!tensor)
    return make_unexpected(std::move(tensor.error()));
  auto tensor_fields = copy_descriptor_tensor_requirements(
      options, payload_identity, authored_tensor_fields, descriptor_tensor_field_count);
  if (!tensor_fields)
    return make_unexpected(std::move(tensor_fields.error()));
  if (options.tensor && !options.memory_conversion && options.tensor->memory_conversion) {
    options.memory_conversion = options.tensor->memory_conversion;
  }
  std::optional<SchemaIdentityRecord> topic_schema;
  auto valid_schema = validate_topic_schema(options, payload_identity, topic_schema);
  if (!valid_schema)
    return make_unexpected(std::move(valid_schema.error()));
  if (options.memory_conversion && payload_identity.kind != PayloadIdentityKind::kTensor) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_MEMORY_CONVERSION_IRRELEVANT"));
  }

  if (endpoint_name && state->explicit_endpoint_names_.contains(*endpoint_name)) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_ENDPOINT_NAME_DUPLICATE"));
  }
  if (GraphPortOwnershipAccess::owner(*port) != nullptr) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_ALREADY_OWNED"));
  }
  auto catalog = validate_pod_catalog(*state, payload_identity);
  if (!catalog)
    return make_unexpected(std::move(catalog.error()));
  ExternalTopicStorage storage;
  GraphMutationRollback rollback(*state);
  rollback.restore_boundary_ordinal(ordinal);
  const VertexId vertex = state->next_vertex_id();
  const GraphPortId port_id = state->next_port_id();
  const ExternalTopicId topic_id = state->next_external_topic_id();
  const PortDirection direction = imported ? PortDirection::kOutput : PortDirection::kInput;
  const PortOwnerKind owner_kind =
      imported ? PortOwnerKind::kImportedTopic : PortOwnerKind::kExportedTopic;
  ExternalTopicRecord record{
      .id = topic_id,
      .vertex = vertex,
      .graph_port = port_id,
      .imported = imported,
      .uri = std::string(uri),
      .endpoint_name = endpoint_name,
      .generated_name_requested = !endpoint_name,
      .declaration_ordinal = ordinal,
      .payload_identity = std::move(payload_identity),
      .serialized_payload_capacity_bytes = serialized_payload_capacity_bytes,
      .schema = std::move(topic_schema),
      .memory_conversion = options.memory_conversion,
      .capture_time = std::move(options.capture_time),
      .clock_health = health,
      .clock_health_authored = options.clock_health.has_value(),
      .attach = options.attach,
      .tensor = std::move(tensor.value()),
      .tensor_fields = std::move(tensor_fields.value()),
      .bounds = options.bounds,
  };
  storage.record = std::move(record);
  auto boundary_tensor_configuration =
      descriptor_topic_port_configuration(storage.record.tensor_fields, direction);
  if constexpr (std::same_as<PortBaseT, OutputBase>) {
    storage.output = std::move(port);
  } else {
    storage.input = std::move(port);
  }
  const PortBaseT* owned_port = nullptr;
  if constexpr (std::same_as<PortBaseT, OutputBase>) {
    owned_port = storage.output.get();
  } else {
    owned_port = storage.input.get();
  }
  state->vertices_.push_back(VertexRecord{
      .id = vertex,
      .graph = state->graph_record_.id,
      .kind = imported ? VertexKind::kImportedTopic : VertexKind::kExportedTopic,
      .name = endpoint_name.value_or(std::string(uri)),
  });
  state->vertex_adjacency_.emplace_back();
  state->cycle_visit_marks_.push_back(0U);
  GraphMutationRollback::checkpoint(*state);
  state->ports_.push_back(PortStorage{
      .record =
          PortRecord{
              .id = port_id,
              .owner = vertex,
              .owner_kind = owner_kind,
              .direction = direction,
              .name = "external",
              .payload_identity = storage.record.payload_identity,
              .serialized_payload_capacity_bytes = serialized_payload_capacity_bytes,
              .descriptor_tensor_field_count = descriptor_tensor_field_count,
              .setup = std::move(boundary_tensor_configuration),
          },
      .handle_identity = std::make_shared<std::byte>(),
      .declaration = owned_port,
  });
  GraphMutationRollback::checkpoint(*state);
  state->port_by_declaration_.emplace(owned_port, port_id);
  GraphMutationRollback::checkpoint(*state);
  GraphPortOwnershipAccess::claim(*owned_port, state.get());
  GraphMutationRollback::checkpoint(*state);
  state->external_topics_.push_back(std::move(storage));
  GraphMutationRollback::checkpoint(*state);
  if (endpoint_name) {
    state->explicit_endpoint_names_.emplace(*endpoint_name);
    GraphMutationRollback::checkpoint(*state);
  }
  record_pod_identity(*state, state->external_topics_.back().record.payload_identity);
  if (state->external_topics_.back().record.payload_identity.pod)
    GraphMutationRollback::checkpoint(*state);
  state->mark_mutation();
  rollback.commit();
  return port_id;
}

}  // namespace

expected<GraphPortId, Error> graph_register_import(
    const std::shared_ptr<GraphState>& state, std::string_view uri, TopicOptions options,
    std::unique_ptr<OutputBase> port, PayloadIdentityRecord payload_identity,
    std::uint64_t serialized_payload_capacity_bytes,
    std::vector<AuthoredDescriptorTensorTopicField> tensor_fields,
    std::uint16_t descriptor_tensor_field_count) {
  return register_external_topic(state,
                                 uri,
                                 std::move(options),
                                 std::move(port),
                                 std::move(payload_identity),
                                 serialized_payload_capacity_bytes,
                                 std::move(tensor_fields),
                                 descriptor_tensor_field_count,
                                 true);
}

expected<GraphPortId, Error> graph_register_export(
    const std::shared_ptr<GraphState>& state, std::string_view uri, TopicOptions options,
    std::unique_ptr<InputBase> port, PayloadIdentityRecord payload_identity,
    std::uint64_t serialized_payload_capacity_bytes,
    std::vector<AuthoredDescriptorTensorTopicField> tensor_fields,
    std::uint16_t descriptor_tensor_field_count) {
  return register_external_topic(state,
                                 uri,
                                 std::move(options),
                                 std::move(port),
                                 std::move(payload_identity),
                                 serialized_payload_capacity_bytes,
                                 std::move(tensor_fields),
                                 descriptor_tensor_field_count,
                                 false);
}

expected<TensorOutputPortConfigurator*, Error> graph_tensor_configurator(
    const std::shared_ptr<GraphState>& state, GraphPortId id) {
  const std::lock_guard lock(state->mutex_);
  const PortStorage* port = find_port(*state, id);
  if (port == nullptr || !port->handle_identity) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_PORT_NOT_FOUND"));
  }
  for (auto& configurator : state->tensor_configurators_) {
    if (configurator.port_ == id &&
        configurator.handle_identity_.get() == port->handle_identity.get()) {
      return &configurator;
    }
  }
  try {
    GraphMutationRollback::checkpoint(*state);
    state->tensor_configurators_.push_back(
        TensorOutputPortConfigurator{state, id, port->handle_identity});
  } catch (const std::bad_alloc&) {
    // Preserve the nonthrowing expected path even while the allocator is
    // exhausted: constructing an optional diagnostic string could throw here.
    return make_unexpected(Error{.code = ErrorCode::kResourceExhausted});
  }
  return &state->tensor_configurators_.back();
}

}  // namespace holoscan::detail
