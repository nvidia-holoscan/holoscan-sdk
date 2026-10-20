/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Graph authoring hash and the published Graph snapshot.

#include <holoscan/core/graph.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "graph_state.hpp"

namespace holoscan::detail {
namespace {

using HashBuilder = holoscan::detail::StableHashBuilder;

template <typename T>
void hash_optional(HashBuilder& hash, const std::optional<T>& value) {
  hash.add(value.has_value());
  if (value)
    hash.add(*value);
}

void hash_bytes(HashBuilder& hash, std::span<const std::byte> bytes) {
  hash.add(static_cast<std::uint64_t>(bytes.size()));
  for (const std::byte value : bytes)
    hash.add_byte(static_cast<std::uint8_t>(value));
}

void hash_schema(HashBuilder& hash, const std::optional<SchemaIdentityRecord>& schema) {
  hash.add(schema.has_value());
  if (!schema)
    return;
  hash.add(schema->canonical_name);
  hash.add(schema->compatibility_epoch);
  hash_bytes(hash, schema->family_id.value);
  hash.add(
      std::string_view{schema->file_identifier.value.data(), schema->file_identifier.value.size()});
}

void hash_payload_identity(HashBuilder& hash, const PayloadIdentityRecord& identity) {
  hash.add(identity.kind);
  hash.add(identity.payload_kind);
  hash.add(identity.transport_profile_tag);
  hash.add(identity.scalar.has_value());
  if (identity.scalar) {
    hash.add(identity.scalar->kind);
    hash.add(identity.scalar->native_object_size);
  }
  hash_schema(hash, identity.schema);
  hash.add(identity.pod.has_value());
  if (!identity.pod)
    return;
  hash.add(identity.pod->canonical_name);
  hash.add(identity.pod->version);
  hash.add(identity.pod->object_size);
  hash.add(identity.pod->object_alignment);
  hash.add(static_cast<std::uint64_t>(identity.pod->fields.size()));
  for (const auto& field : identity.pod->fields) {
    hash.add(field.name);
    hash.add(field.byte_offset);
    hash.add(field.scalar_kind);
    hash.add(field.element_count);
  }
}

void hash_tensor_layout(HashBuilder& hash, const std::optional<TensorRepresentation>& layout) {
  hash.add(layout.has_value());
  if (!layout)
    return;
  hash_optional(hash, layout->memory_kind);
  hash.add(layout->dtype.has_value());
  if (layout->dtype) {
    hash.add(layout->dtype->code);
    hash.add(layout->dtype->bits);
    hash.add(layout->dtype->lanes);
  }
  hash_optional(hash, layout->rank);
  hash_optional(hash, layout->strides);
}

void hash_tensor_contract(HashBuilder& hash, const std::optional<TensorContractRecord>& contract) {
  hash.add(contract.has_value());
  if (!contract) {
    return;
  }
  hash.add(contract->role);
  hash_optional(hash, contract->representation.memory_kind);
  hash.add(contract->representation.dtype.has_value());
  if (contract->representation.dtype) {
    hash.add(contract->representation.dtype->code);
    hash.add(contract->representation.dtype->bits);
    hash.add(contract->representation.dtype->lanes);
  }
  hash_optional(hash, contract->representation.rank);
  hash_optional(hash, contract->representation.strides);
  hash_optional(hash, contract->max_rank);
  hash_optional(hash, contract->max_byte_span);
  hash.add(static_cast<std::uint64_t>(contract->axes.size()));
  for (const auto& axis : contract->axes) {
    hash.add(axis.max_extent);
    hash.add(axis.max_abs_stride_bytes);
  }
  hash.add(contract->transfer.has_value());
  if (contract->transfer) {
    hash.add(contract->transfer->foreign_emit_policy);
    hash_optional(hash, contract->transfer->max_inline_bytes);
    hash_optional(hash, contract->transfer->max_staging_byte_span);
  }
  hash_optional(hash, contract->memory_conversion);
  hash.add(contract->storage);
}

void hash_tensor_boundary_contract(HashBuilder& hash,
                                   const TensorBoundaryContractRecord& contract) {
  hash_optional(hash, contract.memory_kind);
  hash.add(contract.dtype.has_value());
  if (contract.dtype) {
    hash.add(contract.dtype->code);
    hash.add(contract.dtype->bits);
    hash.add(contract.dtype->lanes);
  }
  hash_optional(hash, contract.rank);
  hash_optional(hash, contract.max_rank);
  hash_optional(hash, contract.max_byte_span);
  hash.add(static_cast<std::uint64_t>(contract.axes.size()));
  for (const auto& axis : contract.axes) {
    hash.add(axis.max_extent);
    hash.add(axis.max_abs_stride_bytes);
  }
  hash.add(contract.strides);
  hash.add(contract.foreign_emit_policy);
  hash_optional(hash, contract.max_inline_bytes);
  hash_optional(hash, contract.max_staging_byte_span);
  hash_optional(hash, contract.memory_conversion);
  hash.add(contract.origins.memory_kind);
  hash.add(contract.origins.dtype);
  hash.add(contract.origins.rank);
  hash.add(contract.origins.strides);
  hash.add(contract.origins.max_rank);
  hash.add(contract.origins.max_byte_span);
  hash.add(contract.origins.max_staging_byte_span);
}

void hash_payload_options(HashBuilder& hash, const std::optional<PayloadOptionsRecord>& options) {
  hash.add(options.has_value());
  if (!options)
    return;
  hash_schema(hash, options->schema);
  hash_optional(hash, options->max_inline_bytes);
  hash_optional(hash, options->max_tensor_byte_span);
  hash.add(options->interface_tensor_emit_policy);
}

void hash_tensor_memory_kind_expression(
    HashBuilder& hash, const std::optional<TensorMemoryKindExpressionRecord>& expression) {
  if (!expression) {
    return;
  }
  hash.add(std::string_view{"tensor-memory-kind-expression-v2"});
  hash.add(expression->form);
  hash.add(expression->parameter_field);
  for (const std::byte byte : expression->parameter_full_id.value) {
    hash.add_byte(std::to_integer<std::uint8_t>(byte));
  }
  hash.add(static_cast<std::uint64_t>(expression->parameter_member_full_ids.size()));
  for (const auto& member_full_id : expression->parameter_member_full_ids) {
    for (const std::byte byte : member_full_id.value) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
  }
  hash.add(expression->parameter_path);
  hash.add(expression->parameter_schema_id);
  hash.add(expression->parameter_schema_version);
  hash.add(static_cast<std::uint64_t>(expression->mapping.size()));
  for (const auto& item : expression->mapping) {
    hash.add(item.source.kind);
    hash.add(item.source.bit_width);
    hash.add(item.source.raw_bits);
    hash.add(item.memory_kind);
  }
}

void hash_tensor_byte_span_node(HashBuilder& hash, const TensorByteSpanExpressionNodeRecord& node) {
  hash.add(node.operation);
  hash.add(node.left);
  hash.add(node.right);
  hash.add(node.value);
  hash.add(node.depth);
  hash.add(node.parameter_field);
  for (const std::byte byte : node.parameter_full_id.value) {
    hash.add_byte(std::to_integer<std::uint8_t>(byte));
  }
  hash.add(node.parameter_path);
  hash.add(node.parameter_schema_id);
  hash.add(node.parameter_schema_version);
  hash.add(node.input_name);
  if (!node.projection_id.empty()) {
    hash.add(std::string_view{"tensor-byte-span-projection-v2"});
    hash.add(node.projection_id);
    hash.add(static_cast<std::uint64_t>(node.projection_dependencies.size()));
    for (const auto& dependency : node.projection_dependencies) {
      hash.add(dependency.parameter_field);
      for (const std::byte byte : dependency.parameter_full_id.value) {
        hash.add_byte(std::to_integer<std::uint8_t>(byte));
      }
      hash.add(static_cast<std::uint64_t>(dependency.parameter_member_full_ids.size()));
      for (const auto& member_full_id : dependency.parameter_member_full_ids) {
        for (const std::byte byte : member_full_id.value) {
          hash.add_byte(std::to_integer<std::uint8_t>(byte));
        }
      }
      hash.add(dependency.parameter_path);
    }
  }
}

void hash_tensor_fields(HashBuilder& hash,
                        const std::vector<TensorFieldConfigurationRecord>& fields) {
  if (fields.empty()) {
    return;
  }
  hash.add(std::string_view{"tensor-fields-v1"});
  hash.add(static_cast<std::uint64_t>(fields.size()));
  for (const auto& field : fields) {
    hash.add(field.field_id);
    hash.add(field.tensor_index);
    hash_tensor_layout(hash, field.tensor_layout);
    hash_tensor_contract(hash, field.tensor_contract);
    hash_optional(hash, field.memory_conversion);
    hash.add(field.tensor_allocation.has_value());
    if (field.tensor_allocation) {
      hash.add(field.tensor_allocation->max_rank);
      hash.add(static_cast<std::uint64_t>(field.tensor_allocation->max_byte_span));
    }
    hash.add(field.payload_options.has_value());
    if (field.payload_options) {
      hash_optional(hash, field.payload_options->max_inline_bytes);
      hash_optional(hash, field.payload_options->max_tensor_byte_span);
      hash.add(field.payload_options->interface_tensor_emit_policy);
    }
    if (field.tensor_byte_span_expression) {
      hash.add(std::string_view{"tensor-field-byte-span-expression-v1"});
      hash.add(static_cast<std::uint64_t>(field.tensor_byte_span_expression->nodes.size()));
      for (const auto& node : field.tensor_byte_span_expression->nodes) {
        hash_tensor_byte_span_node(hash, node);
      }
    }
    hash_tensor_memory_kind_expression(hash, field.tensor_memory_kind_expression);
  }
}

void hash_port_configuration(HashBuilder& hash, const PortConfigurationRecord& config) {
  hash_optional(hash, config.queue_depth);
  hash_optional(hash, config.max_emits_per_compute);
  hash_optional(hash, config.late_joiner_history_depth);
  hash.add(config.rate_kind);
  hash.add(config.declared_nominal_rate.has_value());
  if (config.declared_nominal_rate) {
    hash.add(config.declared_nominal_rate->numerator_hz);
    hash.add(config.declared_nominal_rate->denominator_hz);
  }
  hash.add(config.declared_expected_rate.has_value());
  if (config.declared_expected_rate) {
    hash.add(config.declared_expected_rate->numerator_hz);
    hash.add(config.declared_expected_rate->denominator_hz);
  }
  hash.add(config.follows_input.has_value());
  if (config.follows_input) {
    hash.add(config.follows_input->value);
  }
  hash.add(config.follows_ratio.has_value());
  if (config.follows_ratio) {
    hash.add(config.follows_ratio->numerator);
    hash.add(config.follows_ratio->denominator);
  }
  hash.add(config.lifecycle_binding_stage.has_value());
  if (config.lifecycle_binding_stage) {
    hash.add(*config.lifecycle_binding_stage);
  }
  hash.add(config.declared_max_nominal_rate.has_value());
  if (config.declared_max_nominal_rate) {
    hash.add(config.declared_max_nominal_rate->numerator_hz);
    hash.add(config.declared_max_nominal_rate->denominator_hz);
  }
  hash.add(config.diagnostic_alias.has_value());
  if (config.diagnostic_alias)
    hash.add(*config.diagnostic_alias);
  hash_tensor_layout(hash, config.tensor_layout);
  hash_tensor_contract(hash, config.tensor_contract);
  hash_optional(hash, config.memory_conversion);
  hash.add(config.tensor_allocation.has_value());
  if (config.tensor_allocation) {
    hash.add(config.tensor_allocation->max_rank);
    hash.add(static_cast<std::uint64_t>(config.tensor_allocation->max_byte_span));
  }
  if (config.tensor_byte_span_expression) {
    hash.add(std::string_view{"tensor-byte-span-expression-v1"});
    hash.add(static_cast<std::uint64_t>(config.tensor_byte_span_expression->nodes.size()));
    for (const auto& node : config.tensor_byte_span_expression->nodes) {
      hash_tensor_byte_span_node(hash, node);
    }
  }
  hash_tensor_memory_kind_expression(hash, config.tensor_memory_kind_expression);
  hash_payload_options(hash, config.payload_options);
  hash_tensor_fields(hash, config.tensor_fields);
  if (config.accepts_projected_tensor) {
    hash.add(std::string_view{"accepts-projected-tensor-v1"});
  }
}

void hash_connection(HashBuilder& hash, const ConnectionOptionsRecord& options) {
  hash.add(options.queue_policy);
  hash_optional(hash, options.queue_depth);
  hash.add(options.topic_name.has_value());
  if (options.topic_name)
    hash.add(*options.topic_name);
  hash_optional(hash, options.memory_conversion);
  hash_optional(hash, options.max_inflight_samples);
}

void hash_connection_override(HashBuilder& hash, const ConnectionOverrideRecord& options) {
  hash_optional(hash, options.queue_policy);
  hash_optional(hash, options.queue_depth);
  hash.add(options.topic_name.has_value());
  if (options.topic_name)
    hash.add(*options.topic_name);
  hash_optional(hash, options.memory_conversion);
  hash_optional(hash, options.max_inflight_samples);
}

void hash_clock_key(HashBuilder& hash, const ClockKey& key) {
  hash.add(key.is_default());
  hash.add(key.name());
}

// The canonical hash serializer remains visibly ordered with the GraphView record layout. Splitting
// the record sequence across helpers would obscure the compatibility-sensitive field order.
GraphAuthoringHash compute_graph_hash(  // NOLINT(readability-function-cognitive-complexity)
    const GraphSnapshotData& snapshot) {
  HashBuilder hash;
  hash.add("holoscan.graph.authoring.v1");
  hash.add(static_cast<std::uint64_t>(snapshot.graphs.size()));
  for (const auto& graph : snapshot.graphs) {
    hash.add(graph.id.value);
    hash.add(graph.parent.has_value());
    if (graph.parent)
      hash.add(graph.parent->value);
    hash.add(graph.name);
    hash.add(graph.path.has_value());
    if (graph.path) {
      hash.add(static_cast<std::uint64_t>(graph.path->segments().size()));
      for (const auto& segment : graph.path->segments())
        hash.add(segment);
    }
  }
  hash.add(static_cast<std::uint64_t>(snapshot.vertices.size()));
  for (const auto& vertex : snapshot.vertices) {
    hash.add(vertex.id.value);
    hash.add(vertex.graph.value);
    hash.add(vertex.kind);
    hash.add(vertex.name);
    hash.add(vertex.operator_host);
    hash.add(vertex.serialization_domain);
    // Keep hashes of graphs without the new opt-in byte-for-byte stable relative to host metadata.
    if (vertex.cuda_graph_capture_eligible) {
      hash.add("cuda_graph_capture_eligible");
    }
  }
  hash.add(static_cast<std::uint64_t>(snapshot.ports.size()));
  for (const auto& port : snapshot.ports) {
    hash.add(port.id.value);
    hash.add(port.owner.value);
    hash.add(port.owner_kind);
    hash.add(port.direction);
    hash.add(port.optional);
    hash.add(port.name);
    hash_payload_identity(hash, port.payload_identity);
    hash.add(port.serialized_payload_capacity_bytes);
    hash.add(port.generic_loan_supported);
    hash.add(port.generic_loan_slot_size_bytes);
    hash_port_configuration(hash, port.setup);
    hash_port_configuration(hash, port.graph_override);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.edges.size()));
  for (const auto& edge : snapshot.edges) {
    hash.add(edge.id.value);
    hash.add(edge.source.value);
    hash.add(edge.target.value);
    hash_connection(hash, edge.add_flow_options);
    hash_connection_override(hash, edge.graph_override);
    hash_payload_options(hash, edge.effective_tensor_payload);
    if (edge.tensor_projection) {
      hash.add(std::string_view{"tensor-projection-v1"});
      hash.add(edge.tensor_projection->field_id);
      hash.add(edge.tensor_projection->tensor_index);
    }
  }
  hash.add(static_cast<std::uint64_t>(snapshot.interface_bindings.size()));
  for (const auto& binding : snapshot.interface_bindings) {
    hash.add(binding.public_port.value);
    hash.add(binding.internal_port.value);
    hash.add(binding.direction);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.external_topics.size()));
  for (const auto& topic : snapshot.external_topics) {
    hash.add(topic.id.value);
    hash.add(topic.vertex.value);
    hash.add(topic.graph_port.value);
    hash.add(topic.imported);
    hash.add(topic.uri);
    hash.add(topic.endpoint_name.has_value());
    if (topic.endpoint_name)
      hash.add(*topic.endpoint_name);
    hash.add(topic.generated_name_requested);
    hash.add(topic.declaration_ordinal);
    hash_payload_identity(hash, topic.payload_identity);
    hash.add(topic.serialized_payload_capacity_bytes);
    hash_schema(hash, topic.schema);
    hash_optional(hash, topic.memory_conversion);
    hash.add(topic.capture_time.has_value());
    if (topic.capture_time)
      hash_clock_key(hash, topic.capture_time->clock);
    hash.add(topic.clock_health.capability);
    hash.add(topic.clock_health.requirement);
    hash.add(topic.clock_health.max_report_age.has_value());
    if (topic.clock_health.max_report_age)
      hash.add(topic.clock_health.max_report_age->count());
    hash.add(topic.clock_health_authored);
    hash.add(topic.attach.peer);
    hash.add(topic.attach.max_wait.has_value());
    if (topic.attach.max_wait)
      hash.add(topic.attach.max_wait->count());
    hash.add(topic.tensor.has_value());
    if (topic.tensor) {
      hash_optional(hash, topic.tensor->memory_kind);
      hash.add(topic.tensor->dtype.has_value());
      if (topic.tensor->dtype) {
        hash.add(topic.tensor->dtype->code);
        hash.add(topic.tensor->dtype->bits);
        hash.add(topic.tensor->dtype->lanes);
      }
      hash_optional(hash, topic.tensor->rank);
      hash_optional(hash, topic.tensor->max_rank);
      hash_optional(hash, topic.tensor->max_byte_span);
      hash.add(static_cast<std::uint64_t>(topic.tensor->axes.size()));
      for (const auto& axis : topic.tensor->axes) {
        hash.add(axis.max_extent);
        hash.add(axis.max_abs_stride_bytes);
      }
      hash.add(topic.tensor->strides);
      hash.add(topic.tensor->foreign_emit_policy);
      hash_optional(hash, topic.tensor->max_inline_bytes);
    }
    if (!topic.tensor_fields.empty()) {
      hash.add(std::string_view{"descriptor-topic-tensor-fields-v1"});
      hash.add(static_cast<std::uint64_t>(topic.tensor_fields.size()));
      for (const auto& field : topic.tensor_fields) {
        hash.add(field.field_id);
        hash.add(field.tensor_index);
        hash_tensor_boundary_contract(hash, field.tensor);
      }
    }
    hash.add(topic.bounds.publisher.has_value());
    if (topic.bounds.publisher)
      hash_optional(hash, topic.bounds.publisher->max_loaned_samples);
    hash.add(topic.bounds.subscriber.has_value());
    if (topic.bounds.subscriber) {
      hash_optional(hash, topic.bounds.subscriber->queue_depth);
      hash_optional(hash, topic.bounds.subscriber->max_borrowed_samples);
    }
  }
  hash.add(static_cast<std::uint64_t>(snapshot.readiness_sources.size()));
  for (const auto& source : snapshot.readiness_sources) {
    hash.add(source.owner.value);
    hash.add(source.declaration_ordinal);
    hash.add(source.kind);
    hash.add(source.name);
    hash.add(source.requested_capacity);
    hash.add(source.requested_sender_reference_capacity);
    hash_clock_key(hash, source.requested_clock);
    hash.add(source.wake_requirement);
    hash.add(source.referenced);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.regions.size()));
  for (const auto& region : snapshot.regions) {
    hash.add(region.id.value);
    hash.add(region.graph.value);
    hash.add(region.name);
    hash.add(static_cast<std::uint64_t>(region.members.size()));
    for (const VertexId member : region.members)
      hash.add(member.value);
    hash.add(static_cast<std::uint64_t>(region.graph_members.size()));
    for (const GraphId member : region.graph_members)
      hash.add(member.value);
    hash.add(static_cast<std::uint64_t>(region.constraints.size()));
    for (const auto& constraint : region.constraints) {
      hash.add(constraint.kind());
      hash.add(constraint.latency().count());
    }
    // Preserve the v1 identity slot formerly occupied by the removed optimization-hint list.
    hash.add(std::uint64_t{0U});
    hash_optional(hash, region.profile);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.partitions.size()));
  for (const auto& partition : snapshot.partitions) {
    hash.add(partition.id.value);
    hash.add(partition.graph.value);
    hash.add(partition.name);
    hash.add(static_cast<std::uint64_t>(partition.members.size()));
    for (const VertexId member : partition.members)
      hash.add(member.value);
    hash.add(partition.include_all);
    hash.add(static_cast<std::uint64_t>(partition.graph_members.size()));
    for (const GraphId member : partition.graph_members)
      hash.add(member.value);
  }
  hash.add(snapshot.automatic_partition_remaining);
  hash.add(static_cast<std::uint64_t>(snapshot.clocks.size()));
  for (const auto& clock : snapshot.clocks) {
    hash.add(clock.graph.value);
    hash_clock_key(hash, clock.key);
    hash.add(clock.shared);
    hash.add(clock.is_default);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.clock_capabilities.size()));
  for (const auto& capability : snapshot.clock_capabilities) {
    hash.add(capability.graph.value);
    hash_clock_key(hash, capability.key);
    hash.add(capability.can_read_now);
    hash.add(capability.can_wait_until);
    hash.add(capability.steady);
    hash.add(capability.independent_progress);
    hash.add(capability.finite_deadline);
    hash.add(capability.external_driver_admitted);
    hash.add(capability.evidence_id);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.clock_imports.size()));
  for (const auto& clock_import : snapshot.clock_imports) {
    hash.add(clock_import.graph.value);
    hash_clock_key(hash, clock_import.key);
    hash.add(clock_import.is_default);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.clock_bindings.size()));
  for (const auto& binding : snapshot.clock_bindings) {
    hash.add(binding.child_graph.value);
    hash_clock_key(hash, binding.child_clock);
    hash.add(binding.parent_graph.value);
    hash_clock_key(hash, binding.parent_clock);
  }
  auto clock_conversions = snapshot.clock_conversions;
  std::sort(
      clock_conversions.begin(), clock_conversions.end(), [](const auto& lhs, const auto& rhs) {
        return std::tuple{lhs.from.is_default(),
                          lhs.graph.value,
                          lhs.from.name(),
                          lhs.to.is_default(),
                          lhs.to.name(),
                          lhs.conversion.scale.numerator,
                          lhs.conversion.scale.denominator,
                          lhs.conversion.offset_ns} < std::tuple{rhs.from.is_default(),
                                                                 rhs.graph.value,
                                                                 rhs.from.name(),
                                                                 rhs.to.is_default(),
                                                                 rhs.to.name(),
                                                                 rhs.conversion.scale.numerator,
                                                                 rhs.conversion.scale.denominator,
                                                                 rhs.conversion.offset_ns};
      });
  hash.add(static_cast<std::uint64_t>(clock_conversions.size()));
  for (const auto& conversion : clock_conversions) {
    hash.add(conversion.graph.value);
    hash_clock_key(hash, conversion.from);
    hash_clock_key(hash, conversion.to);
    hash.add(conversion.conversion.scale.numerator);
    hash.add(conversion.conversion.scale.denominator);
    hash.add(conversion.conversion.offset_ns);
  }
  auto lifecycle_dependencies = snapshot.lifecycle_dependencies;
  std::sort(lifecycle_dependencies.begin(),
            lifecycle_dependencies.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.graph.value, lhs.stage, lhs.before.value, lhs.after.value} <
                     std::tuple{rhs.graph.value, rhs.stage, rhs.before.value, rhs.after.value};
            });
  hash.add(static_cast<std::uint64_t>(lifecycle_dependencies.size()));
  for (const auto& dependency : lifecycle_dependencies) {
    hash.add(dependency.graph.value);
    hash.add(dependency.before.value);
    hash.add(dependency.after.value);
    hash.add(dependency.stage);
  }
  auto lifecycle_callbacks = snapshot.lifecycle_callbacks;
  std::sort(
      lifecycle_callbacks.begin(), lifecycle_callbacks.end(), [](const auto& lhs, const auto& rhs) {
        return std::tuple{lhs.owner.value, lhs.stage, lhs.required, lhs.method, lhs.origin} <
               std::tuple{rhs.owner.value, rhs.stage, rhs.required, rhs.method, rhs.origin};
      });
  hash.add(static_cast<std::uint64_t>(lifecycle_callbacks.size()));
  for (const auto& callback : lifecycle_callbacks) {
    hash.add(callback.owner.value);
    hash.add(callback.stage);
    hash.add(callback.required);
    hash.add(callback.method);
    hash.add(callback.origin);
  }
  auto lifecycle_facts = snapshot.lifecycle_facts;
  std::sort(lifecycle_facts.begin(), lifecycle_facts.end(), [](const auto& lhs, const auto& rhs) {
    return std::tuple{lhs.owner.value, lhs.stage, lhs.kind, lhs.stable_name} <
           std::tuple{rhs.owner.value, rhs.stage, rhs.kind, rhs.stable_name};
  });
  hash.add(static_cast<std::uint64_t>(lifecycle_facts.size()));
  for (const auto& fact : lifecycle_facts) {
    hash.add(fact.owner.value);
    hash.add(fact.stage);
    hash.add(fact.kind);
    hash.add(fact.stable_name);
  }
  auto lifecycle_permissions = snapshot.lifecycle_restart_permissions;
  std::sort(lifecycle_permissions.begin(),
            lifecycle_permissions.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.origin,
                                lhs.requester.has_value(),
                                lhs.requester ? lhs.requester->value : 0U,
                                lhs.target.value} <
                     std::tuple{rhs.origin,
                                rhs.requester.has_value(),
                                rhs.requester ? rhs.requester->value : 0U,
                                rhs.target.value};
            });
  hash.add(static_cast<std::uint64_t>(lifecycle_permissions.size()));
  for (const auto& permission : lifecycle_permissions) {
    hash.add(permission.origin);
    hash.add(permission.requester.has_value());
    if (permission.requester)
      hash.add(permission.requester->value);
    hash.add(permission.target.value);
  }
  return GraphAuthoringHash{.value = hash.finish().value};
}

}  // namespace

std::shared_ptr<const GraphSnapshotData> graph_snapshot(const std::shared_ptr<GraphState>& state) {
  auto prepared = state->prepare_observation();
  if (!prepared)
    throw RuntimeError(prepared.error());
  const std::lock_guard lock(state->mutex_);
  auto snapshot = std::make_shared<GraphSnapshotData>();
  snapshot->generation = state->generation_;
  snapshot->graphs.push_back(state->graph_record_);
  snapshot->graphs.insert(snapshot->graphs.end(),
                          state->embedded_graph_records_.begin(),
                          state->embedded_graph_records_.end());
  snapshot->vertices = state->vertices_;
  snapshot->ports.reserve(state->ports_.size());
  for (const auto& port : state->ports_)
    snapshot->ports.push_back(port.record);
  snapshot->edges = state->edges_;
  snapshot->interface_bindings = state->interface_bindings_;
  snapshot->external_topics.reserve(state->external_topics_.size());
  for (const auto& topic : state->external_topics_) {
    snapshot->external_topics.push_back(topic.record);
  }
  snapshot->readiness_sources = state->readiness_sources_;
  snapshot->regions = state->regions_;
  snapshot->partitions = state->partitions_;
  snapshot->automatic_partition_remaining = state->automatic_partition_remaining_;
  snapshot->clocks.reserve(state->clocks_.size());
  snapshot->clock_capabilities.reserve(state->clocks_.size());
  for (const auto& clock : state->clocks_) {
    snapshot->clocks.push_back(clock.record);
    snapshot->clock_capabilities.push_back(clock.capabilities);
  }
  snapshot->clock_imports = state->clock_imports_;
  snapshot->clock_bindings = state->clock_bindings_;
  snapshot->clock_conversions = state->clock_conversions_;
  snapshot->lifecycle_dependencies = state->lifecycle_dependencies_;
  snapshot->lifecycle_callbacks = state->lifecycle_callbacks_;
  snapshot->lifecycle_facts = state->lifecycle_facts_;
  snapshot->lifecycle_restart_permissions = state->lifecycle_restart_permissions_;
  const GraphSnapshotData canonical = canonicalize_graph_snapshot(*snapshot);
  snapshot->identity = GraphAuthoringIdentity{
      .graph = state->instance_id_,
      .version = snapshot->generation,
      .hash = compute_graph_hash(canonical),
  };
  return snapshot;
}

}  // namespace holoscan::detail
