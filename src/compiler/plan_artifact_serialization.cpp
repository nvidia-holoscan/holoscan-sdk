/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/plan_artifact_serialization.hpp"
#include "compiler/plan_artifact_serialization_internal.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <fmt/format.h>

#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/schema/schema_package.hpp>

#include "compiler/closed_island_activation_plan.hpp"
#include "compiler/cuda_execution_plan_validation.hpp"
#include "compiler/cuda_graph_region_lowering.hpp"
#include "compiler/inline_partition_execution_plan.hpp"
#include "compiler/optional_input_plan_validation.hpp"
#include "compiler/parameter_plan_lowering.hpp"
#include "compiler/partition_authority_validation.hpp"
#include "compiler/partition_runner_plan.hpp"
#include "compiler/plan_artifact_rate_serialization.hpp"
#include "compiler/process_local_channel_plan_validation.hpp"
#include "core/canonical_text_encoding.hpp"
#include "core/diagnostic_registry.hpp"
#include "core/memory_kind_validation.hpp"
#include "core/source_id_internal.hpp"
#include "core/tensor_allocation_record_validation.hpp"

namespace holoscan::detail {
namespace {

using plan_artifact_serialization_internal::append_array;
using plan_artifact_serialization_internal::append_bool;
using plan_artifact_serialization_internal::append_logical_tensor_requirements;
using plan_artifact_serialization_internal::append_number;
using plan_artifact_serialization_internal::append_optional_profile;
using plan_artifact_serialization_internal::append_optional_string;
using plan_artifact_serialization_internal::append_optional_tensor_layout;
using plan_artifact_serialization_internal::append_tensor_allocation_bounds;
using plan_artifact_serialization_internal::append_tensor_axes;

constexpr std::string_view kPlanSchema = "holoscan.execution_plan.private.v6";

[[nodiscard]] Error artifact_error(ErrorCode code, std::string_view token) {
  return Error{.code = code, .message = std::string(token)};
}

[[nodiscard]] bool has_rejected_tensor_layout(const ExecutionPlanState& state) noexcept {
  return !state.admitted &&
         std::any_of(
             state.diagnostics.begin(), state.diagnostics.end(), [](const auto& diagnostic) {
               return diagnostic.severity == holoscan::DiagnosticSeverity::kError &&
                      diagnostic.token == "PHYSICAL_TENSOR_LAYOUT_MISMATCH";
             });
}

[[nodiscard]] bool nonzero_parameter_hash(const ParamSchemaHash& hash) noexcept {
  return std::any_of(
      hash.value.begin(), hash.value.end(), [](std::byte value) { return value != std::byte{}; });
}

[[nodiscard]] bool projection_dependency_less(
    const TensorByteSpanProjectionDependencyRecord& left,
    const TensorByteSpanProjectionDependencyRecord& right) noexcept {
  if (left.parameter_member_full_ids != right.parameter_member_full_ids) {
    return std::lexicographical_compare(left.parameter_member_full_ids.begin(),
                                        left.parameter_member_full_ids.end(),
                                        right.parameter_member_full_ids.begin(),
                                        right.parameter_member_full_ids.end(),
                                        [](ParamSchemaHash left_id, ParamSchemaHash right_id) {
                                          return left_id.value < right_id.value;
                                        });
  }
  return std::tuple{left.parameter_field, left.parameter_full_id.value, left.parameter_path} <
         std::tuple{right.parameter_field, right.parameter_full_id.value, right.parameter_path};
}

[[nodiscard]] bool valid_tensor_byte_span_projection_evidence(const ExecutionPlanState& state) {
  for (const auto& record : state.tensor_byte_span_derivations) {
    for (const auto& node : record.nodes) {
      const bool projection = node.operation == TensorByteSpanOperation::kParameterProjection;
      if (!projection) {
        if (!node.projection_id.empty() || !node.projection_dependencies.empty()) {
          return false;
        }
        continue;
      }
      if (!valid_schema_text(node.projection_id) ||
          node.projection_id.size() > kMaximumTensorByteSpanProjectionIdBytes ||
          !valid_schema_text(node.parameter_schema_id) || node.parameter_schema_version == 0U ||
          node.projection_dependencies.empty() ||
          node.projection_dependencies.size() > kMaximumTensorByteSpanExpressionDependencies ||
          node.left != 0U || node.right != 0U || node.operand != 0U || node.parameter_field != 0U ||
          node.input || node.resolved_value == 0U ||
          node.resolved_value > std::numeric_limits<std::size_t>::max() ||
          !std::is_sorted(node.projection_dependencies.begin(),
                          node.projection_dependencies.end(),
                          projection_dependency_less)) {
        return false;
      }
      for (std::size_t index = 0U; index < node.projection_dependencies.size(); ++index) {
        const auto& dependency = node.projection_dependencies[index];
        if (dependency.parameter_field == 0U ||
            !nonzero_parameter_hash(dependency.parameter_full_id) ||
            dependency.parameter_member_full_ids.empty() ||
            dependency.parameter_member_full_ids.back() != dependency.parameter_full_id ||
            std::any_of(dependency.parameter_member_full_ids.begin(),
                        dependency.parameter_member_full_ids.end(),
                        [](const auto& member_full_id) {
                          return !nonzero_parameter_hash(member_full_id);
                        }) ||
            !valid_schema_text(dependency.parameter_path) ||
            (index != 0U && dependency == node.projection_dependencies[index - 1U])) {
          return false;
        }
      }
    }
  }
  return true;
}

[[nodiscard]] auto memory_kind_derivation_key(
    const TensorMemoryKindDerivationPlanRecord& record) noexcept {
  return std::tuple{record.output.value(), record.field_id, record.tensor_index};
}

[[nodiscard]] bool valid_tensor_memory_kind_derivations(const ExecutionPlanState& state) {
  for (std::size_t index = 0U; index < state.tensor_memory_kind_derivations.size(); ++index) {
    const auto& record = state.tensor_memory_kind_derivations[index];
    if (!record.output || record.plan_generation != state.plan_generation ||
        record.run_generation != 0U || record.parameter_schema_id.empty() ||
        record.parameter_schema_version == 0U || record.parameter_field == 0U ||
        record.parameter_full_id == ParamSchemaHash() || record.parameter_path.empty() ||
        record.parameter_member_full_ids.empty() ||
        record.parameter_member_full_ids.back() != record.parameter_full_id ||
        std::any_of(record.parameter_member_full_ids.begin(),
                    record.parameter_member_full_ids.end(),
                    [](const auto& member) { return member == ParamSchemaHash(); }) ||
        record.field_id.has_value() != record.tensor_index.has_value() ||
        !valid_concrete_memory_kind(record.resolved_memory_kind) ||
        record.origin != TensorContractValueOrigin::kCompilerDerivedMemoryKind ||
        (record.update_disposition !=
             TensorMemoryKindDerivationUpdateDisposition::kFrozenCompileTime &&
         record.update_disposition !=
             TensorMemoryKindDerivationUpdateDisposition::kRecompileRequired)) {
      return false;
    }
    if (index != 0U &&
        !(memory_kind_derivation_key(state.tensor_memory_kind_derivations[index - 1U]) <
          memory_kind_derivation_key(record))) {
      return false;
    }
    if (record.form == TensorMemoryKindExpressionForm::kDirect) {
      if (!record.mapping.empty()) {
        return false;
      }
    } else if (record.form == TensorMemoryKindExpressionForm::kMapped) {
      if (record.mapping.empty() || record.mapping.size() > kMaximumTensorMemoryKindMappingCases ||
          std::none_of(record.mapping.begin(), record.mapping.end(), [&](const auto& row) {
            return row.memory_kind == record.resolved_memory_kind;
          })) {
        return false;
      }
      for (std::size_t row_index = 0U; row_index < record.mapping.size(); ++row_index) {
        const auto& row = record.mapping[row_index];
        if (row.source.kind == ParamScalarKind::kNone || row.source.bit_width == 0U ||
            !valid_concrete_memory_kind(row.memory_kind)) {
          return false;
        }
        if (row_index != 0U) {
          const auto& previous = record.mapping[row_index - 1U];
          if (!(std::tuple{
                    previous.source.kind, previous.source.bit_width, previous.source.raw_bits} <
                std::tuple{row.source.kind, row.source.bit_width, row.source.raw_bits})) {
            return false;
          }
        }
      }
    } else {
      return false;
    }
  }
  return true;
}

template <typename Digest>
[[nodiscard]] Digest digest_bytes(std::string_view bytes) noexcept {
  // REQ: COMPILE-002 - digest canonical bytes, never C++ object representation.
  StableHashBuilder hash;
  for (const char byte : bytes) {
    hash.add_byte(static_cast<std::uint8_t>(byte));
  }
  Digest result;
  result.value = hash.finish().value;
  return result;
}

template <typename Records, typename Predicate, typename AppendRecord>
void append_filtered_array(std::string& output, const Records& records, Predicate include_record,
                           AppendRecord append_record) {
  output.push_back('[');
  bool first = true;
  for (const auto& record : records) {
    if (!include_record(record)) {
      continue;
    }
    if (!first) {
      output.push_back(',');
    }
    first = false;
    append_record(output, record);
  }
  output.push_back(']');
}

template <typename Bytes>
void append_hex(std::string& output, const Bytes& bytes) {
  output.push_back('"');
  append_lowercase_hex(output, std::span<const std::byte>{bytes}, HexPrefix::kNone);
  output.push_back('"');
}

template <typename Bytes>
void append_sha256(std::string& output, const Bytes& bytes) {
  output.push_back('"');
  append_lowercase_hex(output, std::span<const std::byte>{bytes}, HexPrefix::kSha256);
  output.push_back('"');
}

[[nodiscard]] constexpr std::string_view profile_token(Profile profile) noexcept {
  switch (profile) {
    case Profile::kStandard:
      return "standard";
    case Profile::kRobotics:
      return "robotics";
    case Profile::kRealTime:
      return "real_time";
    case Profile::kRegulated:
      return "regulated";
    case Profile::kResearch:
      return "research";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view scheduling_policy_token(SchedulingPolicy policy) noexcept {
  switch (policy) {
    case SchedulingPolicy::kOther:
      return "other";
    case SchedulingPolicy::kFIFO:
      return "fifo";
    case SchedulingPolicy::kRoundRobin:
      return "round_robin";
    case SchedulingPolicy::kDeadline:
      return "deadline";
  }
  return "invalid";
}

void append_thread_placement(std::string& json, const ThreadPlacement& placement) {
  json += "{\"affinity\":";
  if (placement.affinity) {
    json.push_back('[');
    for (std::size_t index = 0U; index < placement.affinity->cores.size(); ++index) {
      if (index != 0U) {
        json.push_back(',');
      }
      append_number(json, placement.affinity->cores[index]);
    }
    json.push_back(']');
  } else {
    json += "null";
  }
  json += ",\"irq_isolation\":";
  append_number(json, static_cast<std::uint32_t>(placement.irq_isolation));
  json += ",\"probe_failure_policy\":";
  append_number(json, static_cast<std::uint32_t>(placement.probe_failure_policy));
  json += ",\"scheduling\":{\"deadline\":";
  if (placement.scheduling.deadline) {
    const auto& deadline = *placement.scheduling.deadline;
    json += "{\"deadline_ns\":";
    append_number(json, deadline.deadline_ns);
    json += ",\"period_ns\":";
    append_number(json, deadline.period_ns);
    json += ",\"runtime_ns\":";
    append_number(json, deadline.runtime_ns);
    json.push_back('}');
  } else {
    json += "null";
  }
  json += ",\"policy\":";
  append_json_string(json, scheduling_policy_token(placement.scheduling.policy));
  json += ",\"priority\":";
  if (placement.scheduling.priority) {
    append_number(json, *placement.scheduling.priority);
  } else {
    json += "null";
  }
  json += "},\"wake_mode\":";
  append_number(json, static_cast<std::uint32_t>(placement.wake_mode));
  json.push_back('}');
}

void append_operator_ref(std::string& output, OperatorRef ref) {
  output += "{\"id\":";
  append_number(output, ref.id);
  output += ",\"plan_generation\":";
  append_number(output, ref.plan_generation);
  output += ",\"run_generation\":";
  append_number(output, ref.run_generation);
  output.push_back('}');
}

void append_parameter_storage(std::string& output, const ParamStorageTraitsSummary& storage) {
  output += "{\"fixed_object_extent\":";
  append_bool(output, storage.fixed_object_extent);
  output += ",\"has_portable_codec\":";
  append_bool(output, storage.has_portable_codec);
  output += ",\"inline_bytes\":";
  append_number(output, storage.inline_bytes);
  output += ",\"owns_dynamic_storage\":";
  append_bool(output, storage.owns_dynamic_storage);
  output += ",\"supports_bounded_materialization\":";
  append_bool(output, storage.supports_bounded_materialization);
  output.push_back('}');
}

void append_parameter_type_shape(std::string& output, const ParamFieldTypeShape& shape) {
  output += "{\"bit_width\":";
  append_number(output, shape.bit_width);
  output += ",\"duration_period_denominator\":";
  append_number(output, shape.duration_period_denominator);
  output += ",\"duration_period_numerator\":";
  append_number(output, shape.duration_period_numerator);
  output += ",\"element_or_rep_bit_width\":";
  append_number(output, shape.element_or_rep_bit_width);
  output += ",\"element_or_rep_type\":";
  append_number(output, static_cast<std::uint32_t>(shape.element_or_rep_type));
  output += ",\"has_stable_shape\":";
  append_bool(output, shape.has_stable_shape);
  output += ",\"optional\":";
  append_bool(output, shape.optional);
  output += ",\"type\":";
  append_number(output, static_cast<std::uint32_t>(shape.type));
  output.push_back('}');
}

void append_lifecycle_capacity_values(std::string& output,
                                      const LifecycleTransactionCapacityValues& values) {
  output += "{\"max_audit_records\":";
  append_number(output, values.max_audit_records);
  output += ",\"max_candidate_generations\":";
  append_number(output, values.max_candidate_generations);
  output += ",\"max_events\":";
  append_number(output, values.max_events);
  output += ",\"max_fenced_operators\":";
  append_number(output, values.max_fenced_operators);
  output += ",\"max_pending_requests\":";
  append_number(output, values.max_pending_requests);
  output += ",\"max_pending_transactions\":";
  append_number(output, values.max_pending_transactions);
  output += ",\"max_quiesce_records\":";
  append_number(output, values.max_quiesce_records);
  output += ",\"max_receipts\":";
  append_number(output, values.max_receipts);
  output += ",\"max_rollback_records\":";
  append_number(output, values.max_rollback_records);
  output.push_back('}');
}

void append_clock_plan_records(std::string& output, std::span<const ClockPlanRecord> records) {
  output.push_back('[');
  bool first = true;
  for (const ClockPlanRecord& record : records) {
    if (!first) {
      output.push_back(',');
    }
    first = false;
    output += "{\"graph\":";
    append_number(output, record.graph.value);
    output += ",\"id\":";
    append_number(output, record.id.value);
    output += ",\"is_default\":";
    append_bool(output, record.is_default);
    output += ",\"can_read_now\":";
    append_bool(output, record.can_read_now);
    output += ",\"can_wait_until\":";
    append_bool(output, record.can_wait_until);
    output += ",\"steady\":";
    append_bool(output, record.steady);
    output += ",\"independent_progress\":";
    append_bool(output, record.independent_progress);
    output += ",\"finite_deadline\":";
    append_bool(output, record.finite_deadline);
    output += ",\"external_driver_admitted\":";
    append_bool(output, record.external_driver_admitted);
    output += ",\"capability_evidence_id\":";
    append_json_string(output, record.capability_evidence_id);
    output += ",\"key\":";
    append_json_string(output, record.key.name());
    output += ",\"shared\":";
    append_bool(output, record.shared);
    output.push_back('}');
  }
  output.push_back(']');
}

void append_payload_identity(std::string& output, const PayloadIdentityRecord& identity) {
  output += "{\"kind\":";
  append_number(output, static_cast<std::uint32_t>(identity.kind));
  output += ",\"payload_kind\":";
  append_number(output, static_cast<std::uint32_t>(identity.payload_kind));
  output += ",\"transport_profile_tag\":";
  append_number(output, identity.transport_profile_tag);
  output += ",\"pod\":";
  if (identity.pod) {
    output += "{\"alignment\":";
    append_number(output, identity.pod->object_alignment);
    output += ",\"fields\":";
    append_array(output, identity.pod->fields, [](std::string& json, const auto& field) {
      json += "{\"count\":";
      append_number(json, field.element_count);
      json += ",\"name\":";
      append_json_string(json, field.name);
      json += ",\"offset\":";
      append_number(json, field.byte_offset);
      json += ",\"scalar_kind\":";
      append_number(json, static_cast<std::uint32_t>(field.scalar_kind));
      json.push_back('}');
    });
    output += ",\"name\":";
    append_json_string(output, identity.pod->canonical_name);
    output += ",\"size\":";
    append_number(output, identity.pod->object_size);
    output += ",\"version\":";
    append_number(output, identity.pod->version);
    output.push_back('}');
  } else {
    output += "null";
  }
  output += ",\"scalar_kind\":";
  if (identity.scalar) {
    append_number(output, static_cast<std::uint32_t>(identity.scalar->kind));
  } else {
    output += "null";
  }
  output += ",\"scalar_native_object_size\":";
  if (identity.scalar) {
    append_number(output, identity.scalar->native_object_size);
  } else {
    output += "null";
  }
  output += ",\"schema\":";
  if (identity.schema) {
    output += "{\"canonical_name\":";
    append_json_string(output, identity.schema->canonical_name);
    output += ",\"compatibility_epoch\":";
    append_number(output, identity.schema->compatibility_epoch);
    output += ",\"family_id\":";
    append_hex(output, identity.schema->family_id.value);
    output += ",\"file_identifier\":";
    append_json_string(output,
                       std::string_view{identity.schema->file_identifier.value.data(),
                                        identity.schema->file_identifier.value.size()});
    output.push_back('}');
  } else {
    output += "null";
  }
  output.push_back('}');
}

void append_optional_tensor_layout_value(std::string& output,
                                         const std::optional<TensorRepresentation>& layout) {
  if (!layout) {
    output += "null";
    return;
  }
  output += "{\"dtype\":";
  if (layout->dtype) {
    output += "{\"bits\":";
    append_number(output, layout->dtype->bits);
    output += ",\"code\":";
    append_number(output, layout->dtype->code);
    output += ",\"lanes\":";
    append_number(output, layout->dtype->lanes);
    output.push_back('}');
  } else {
    output += "null";
  }
  output += ",\"memory_kind\":";
  if (layout->memory_kind) {
    append_number(output, static_cast<std::uint32_t>(*layout->memory_kind));
  } else {
    output += "null";
  }
  output += ",\"rank\":";
  if (layout->rank) {
    append_number(output, *layout->rank);
  } else {
    output += "null";
  }
  output += ",\"strides\":";
  if (layout->strides) {
    append_number(output, static_cast<std::uint32_t>(*layout->strides));
  } else {
    output += "null";
  }
  output.push_back('}');
}

void append_tensor_axes_value(std::string& output, const std::span<const TensorAxisBounds> axes) {
  append_array(output, axes, [](std::string& json, const TensorAxisBounds& axis) {
    json += "{\"max_abs_stride_bytes\":";
    append_number(json, axis.max_abs_stride_bytes);
    json += ",\"max_extent\":";
    append_number(json, axis.max_extent);
    json.push_back('}');
  });
}

[[nodiscard]] std::string_view tensor_contract_origin_name(
    const TensorContractValueOrigin origin) noexcept {
  switch (origin) {
    case TensorContractValueOrigin::kAbsent:
      return "absent";
    case TensorContractValueOrigin::kAuthored:
      return "authored";
    case TensorContractValueOrigin::kDerivedExactRank:
      return "derived_exact_rank";
    case TensorContractValueOrigin::kDerivedMaxByteSpan:
      return "derived_max_byte_span";
    case TensorContractValueOrigin::kDefaultedBoundary:
      return "defaulted_boundary";
    case TensorContractValueOrigin::kDerivedShapeBounds:
      return "derived_shape_bounds";
    case TensorContractValueOrigin::kCompilerDerivedByteSpan:
      return "compiler_derived_byte_span";
    case TensorContractValueOrigin::kCompilerDerivedMemoryKind:
      return "compiler_derived_memory_kind";
  }
  return "absent";
}

void append_optional_tensor_contract(std::string& output,
                                     const std::optional<TensorContractRecord>& contract,
                                     const bool include_provenance) {
  if (!contract) {
    output += "null";
    return;
  }
  output += "{\"bounds\":{\"axes\":";
  append_tensor_axes(output, contract->axes);
  output += ",\"max_byte_span\":";
  if (contract->max_byte_span) {
    append_number(output, *contract->max_byte_span);
  } else {
    output += "null";
  }
  output += ",\"max_rank\":";
  if (contract->max_rank) {
    append_number(output, *contract->max_rank);
  } else {
    output += "null";
  }
  output += "},\"memory_conversion\":";
  if (contract->memory_conversion) {
    append_number(output, static_cast<std::uint32_t>(*contract->memory_conversion));
  } else {
    output += "null";
  }
  output += ",\"representation\":";
  append_optional_tensor_layout(output, std::optional{contract->representation});
  output += ",\"role\":";
  append_number(output, static_cast<std::uint32_t>(contract->role));
  output += ",\"storage\":";
  append_number(output, static_cast<std::uint32_t>(contract->storage));
  output += ",\"transfer\":";
  if (contract->transfer) {
    output += "{\"foreign_emit_policy\":";
    append_number(output, static_cast<std::uint32_t>(contract->transfer->foreign_emit_policy));
    output += ",\"max_inline_bytes\":";
    if (contract->transfer->max_inline_bytes) {
      append_number(output, *contract->transfer->max_inline_bytes);
    } else {
      output += "null";
    }
    output += ",\"max_staging_byte_span\":";
    if (contract->transfer->max_staging_byte_span) {
      append_number(output, *contract->transfer->max_staging_byte_span);
    } else {
      output += "null";
    }
    output.push_back('}');
  } else {
    output += "null";
  }
  if (include_provenance) {
    output += ",\"origins\":{\"dtype\":";
    append_json_string(output, tensor_contract_origin_name(contract->origins.dtype));
    output += ",\"max_byte_span\":";
    append_json_string(output, tensor_contract_origin_name(contract->origins.max_byte_span));
    output += ",\"max_rank\":";
    append_json_string(output, tensor_contract_origin_name(contract->origins.max_rank));
    output += ",\"max_staging_byte_span\":";
    append_json_string(output,
                       tensor_contract_origin_name(contract->origins.max_staging_byte_span));
    output += ",\"memory_kind\":";
    append_json_string(output, tensor_contract_origin_name(contract->origins.memory_kind));
    output += ",\"rank\":";
    append_json_string(output, tensor_contract_origin_name(contract->origins.rank));
    output += ",\"strides\":";
    append_json_string(output, tensor_contract_origin_name(contract->origins.strides));
    output.push_back('}');
  }
  output.push_back('}');
}

void append_logical_tensor_requirements_value(std::string& output,
                                              const TensorBoundaryContractRecord& logical,
                                              const bool include_provenance) {
  output += "{\"role\":\"external_boundary\",\"representation\":{\"memory_kind\":";
  if (logical.memory_kind) {
    append_number(output, static_cast<std::uint32_t>(*logical.memory_kind));
  } else {
    output += "null";
  }
  output += ",\"dtype\":";
  if (logical.dtype) {
    output += "{\"bits\":";
    append_number(output, logical.dtype->bits);
    output += ",\"code\":";
    append_number(output, logical.dtype->code);
    output += ",\"lanes\":";
    append_number(output, logical.dtype->lanes);
    output.push_back('}');
  } else {
    output += "null";
  }
  output += ",\"rank\":";
  if (logical.rank) {
    append_number(output, *logical.rank);
  } else {
    output += "null";
  }
  output += ",\"strides\":";
  append_number(output, static_cast<std::uint32_t>(logical.strides));
  output += "},\"bounds\":{\"max_rank\":";
  if (logical.max_rank) {
    append_number(output, *logical.max_rank);
  } else {
    output += "null";
  }
  output += ",\"max_byte_span\":";
  if (logical.max_byte_span) {
    append_number(output, *logical.max_byte_span);
  } else {
    output += "null";
  }
  output += ",\"axes\":";
  append_tensor_axes(output, logical.axes);
  output += "},\"transfer\":{\"foreign_emit_policy\":";
  append_number(output, static_cast<std::uint32_t>(logical.foreign_emit_policy));
  output += ",\"max_inline_bytes\":";
  if (logical.max_inline_bytes) {
    append_number(output, *logical.max_inline_bytes);
  } else {
    output += "null";
  }
  output += ",\"max_staging_byte_span\":";
  if (logical.max_staging_byte_span) {
    append_number(output, *logical.max_staging_byte_span);
  } else {
    output += "null";
  }
  output += "},\"memory_conversion\":";
  if (logical.memory_conversion) {
    append_number(output, static_cast<std::uint32_t>(*logical.memory_conversion));
  } else {
    output += "null";
  }
  if (!include_provenance) {
    output.push_back('}');
    return;
  }
  output += ",\"origins\":{\"memory_kind\":";
  append_json_string(output, tensor_contract_origin_name(logical.origins.memory_kind));
  output += ",\"dtype\":";
  append_json_string(output, tensor_contract_origin_name(logical.origins.dtype));
  output += ",\"rank\":";
  append_json_string(output, tensor_contract_origin_name(logical.origins.rank));
  output += ",\"strides\":";
  append_json_string(output, tensor_contract_origin_name(logical.origins.strides));
  output += ",\"max_rank\":";
  append_json_string(output, tensor_contract_origin_name(logical.origins.max_rank));
  output += ",\"max_byte_span\":";
  append_json_string(output, tensor_contract_origin_name(logical.origins.max_byte_span));
  output += ",\"max_staging_byte_span\":";
  append_json_string(output, tensor_contract_origin_name(logical.origins.max_staging_byte_span));
  output.push_back('}');
  output.push_back('}');
}

void append_descriptor_tensor_fields(std::string& output,
                                     std::span<const DescriptorTensorTopicFieldRecord> fields,
                                     const bool include_provenance) {
  append_array(output, fields, [&](std::string& json, const auto& field) {
    json += "{\"field_id\":";
    append_number(json, field.field_id);
    json += ",\"tensor\":";
    append_logical_tensor_requirements(json, field.tensor, include_provenance);
    json += ",\"tensor_index\":";
    append_number(json, field.tensor_index);
    json.push_back('}');
  });
}

void append_optional_algorithm(std::string& output, const std::optional<AlgorithmKey>& algorithm) {
  if (!algorithm.has_value()) {
    output += "null";
    return;
  }
  output += "{\"id\":";
  append_json_string(output, algorithm->id);
  output += ",\"version\":";
  append_number(output, algorithm->version);
  output.push_back('}');
}

void append_optional_maturity(std::string& output, std::optional<AlgorithmMaturity> maturity) {
  if (!maturity.has_value()) {
    output += "null";
    return;
  }
  append_number(output, static_cast<std::uint32_t>(*maturity));
}

void append_optional_profile_value(std::string& output, std::optional<Profile> profile) {
  if (!profile.has_value()) {
    output += "null";
    return;
  }
  append_json_string(output, profile_token(*profile));
}

void append_optional_string_value(std::string& output, const std::optional<std::string>& value) {
  if (!value) {
    output += "null";
    return;
  }
  append_json_string(output, *value);
}

void append_optional_time_point(std::string& output, const std::optional<TimePoint>& value) {
  if (!value) {
    output += "null";
    return;
  }
  output += "{\"clock\":";
  append_number(output, value->clock_id.value);
  output += ",\"timestamp_ns\":";
  append_number(output, value->timestamp_ns);
  output.push_back('}');
}

[[nodiscard]] constexpr bool valid_operator_completion_kind(
    const OperatorCompletionRuleKind kind) noexcept {
  switch (kind) {
    case OperatorCompletionRuleKind::kComputeCalls:
      return true;
  }
  return false;
}

[[nodiscard]] constexpr bool valid_operator_completion_trigger_kind(
    const TemporalRecordTriggerKind kind) noexcept {
  switch (kind) {
    case TemporalRecordTriggerKind::kClock:
    case TemporalRecordTriggerKind::kReadable:
    case TemporalRecordTriggerKind::kNotified:
    case TemporalRecordTriggerKind::kDoorbell:
    case TemporalRecordTriggerKind::kSpin:
      return true;
    case TemporalRecordTriggerKind::kAnyInput:
    case TemporalRecordTriggerKind::kEach:
    case TemporalRecordTriggerKind::kAligned:
    case TemporalRecordTriggerKind::kWindow:
    case TemporalRecordTriggerKind::kPredicate:
      return false;
  }
  return false;
}

[[nodiscard]] constexpr std::optional<ReadinessSourceKind> operator_completion_readiness_kind(
    const TemporalRecordTriggerKind kind) noexcept {
  switch (kind) {
    case TemporalRecordTriggerKind::kReadable:
      return ReadinessSourceKind::kReadable;
    case TemporalRecordTriggerKind::kNotified:
      return ReadinessSourceKind::kNotification;
    case TemporalRecordTriggerKind::kDoorbell:
      return ReadinessSourceKind::kDoorbell;
    case TemporalRecordTriggerKind::kSpin:
      return ReadinessSourceKind::kSpin;
    case TemporalRecordTriggerKind::kClock:
    case TemporalRecordTriggerKind::kAnyInput:
    case TemporalRecordTriggerKind::kEach:
    case TemporalRecordTriggerKind::kAligned:
    case TemporalRecordTriggerKind::kWindow:
    case TemporalRecordTriggerKind::kPredicate:
      return std::nullopt;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<VertexId> operator_completion_owner_vertex(
    const ExecutionPlanState& state, const OperatorInstanceId owner) {
  const ExecutableOperatorFactoryRecord* matched = nullptr;
  for (const auto& candidate : state.executable_factories.operators) {
    if (candidate.operator_id != owner) {
      continue;
    }
    if (matched != nullptr || !candidate.vertex) {
      return std::nullopt;
    }
    matched = &candidate;
  }
  if (matched != nullptr) {
    return matched->vertex;
  }
  if (!state.executable_factories.operators.empty()) {
    return std::nullopt;
  }

  // Low-level physical-artifact fixtures intentionally omit both the canonical Graph and
  // executable factories. In that representation only, their compact owner values are authored
  // in the same identity space as the retained physical-port projection.
  if (!state.graph.vertices().empty()) {
    return std::nullopt;
  }
  const VertexId fixture_vertex{owner};
  const bool fixture_port =
      std::any_of(state.ports.begin(), state.ports.end(), [&](const auto& port) {
        return port.owner == fixture_vertex && port.owner_kind == PortOwnerKind::kOperator;
      });
  return fixture_port ? std::optional<VertexId>{fixture_vertex} : std::nullopt;
}

template <typename Ports>
[[nodiscard]] bool operator_completion_port_shape_supported(const Ports& ports,
                                                            const VertexId owner) {
  bool has_output = false;
  for (const auto& port : ports) {
    if (port.owner != owner || port.owner_kind != PortOwnerKind::kOperator) {
      continue;
    }
    switch (port.direction) {
      case PortDirection::kInput:
        return false;
      case PortDirection::kOutput:
        has_output = true;
        break;
    }
  }
  return has_output;
}

[[nodiscard]] bool operator_completion_port_shape_supported(const ExecutionPlanState& state,
                                                            const VertexId owner) {
  if (!state.graph.vertices().empty()) {
    return operator_completion_port_shape_supported(state.graph.ports(), owner);
  }
  return operator_completion_port_shape_supported(state.ports, owner);
}

[[nodiscard]] bool operator_completion_source_supported(const ExecutionPlanState& state,
                                                        const TemporalTriggerPlanRecord& trigger,
                                                        const VertexId owner) {
  if (trigger.kind == TemporalRecordTriggerKind::kClock) {
    return true;
  }
  const std::optional<ReadinessSourceKind> expected_kind =
      operator_completion_readiness_kind(trigger.kind);
  if (!expected_kind) {
    return false;
  }

  const ReadinessPlanRecord* matched = nullptr;
  for (const auto& source : state.readiness) {
    if (source.source_id != trigger.readiness_source) {
      continue;
    }
    if (matched != nullptr) {
      return false;
    }
    matched = &source;
  }
  return matched != nullptr && matched->plan_generation == state.plan_generation &&
         matched->owner == owner && matched->kind == *expected_kind && matched->referenced &&
         matched->materialized;
}

[[nodiscard]] bool operator_completion_trigger_supported(const ExecutionPlanState& state,
                                                         const OperatorCompletionPlanRecord& rule,
                                                         const VertexId owner) {
  const TemporalTriggerPlanRecord* matched = nullptr;
  std::size_t owner_trigger_count = 0U;
  for (const auto& trigger : state.temporal_triggers) {
    if (trigger.owner.id != rule.owner.id) {
      continue;
    }
    ++owner_trigger_count;
    matched = &trigger;
  }
  return owner_trigger_count == 1U && matched != nullptr && matched->owner == rule.owner &&
         matched->origin == TemporalRecordOrigin::kAuthored &&
         valid_operator_completion_trigger_kind(matched->kind) &&
         static_cast<bool>(matched->readiness_source) &&
         operator_completion_source_supported(state, *matched, owner);
}

[[nodiscard]] bool valid_operator_completion_plan(const ExecutionPlanState& state) {
  OperatorInstanceId previous_owner{};
  for (const auto& record : state.operator_completion_rules) {
    const std::optional<VertexId> owner_vertex =
        operator_completion_owner_vertex(state, record.owner.id);
    if (!record.owner.plan_bound() || record.owner.plan_generation != state.plan_generation ||
        !valid_operator_completion_kind(record.kind) || record.owner.id <= previous_owner ||
        !owner_vertex || !operator_completion_port_shape_supported(state, *owner_vertex) ||
        !operator_completion_trigger_supported(state, record, *owner_vertex)) {
      return false;
    }
    previous_owner = record.owner.id;
  }
  return true;
}

[[nodiscard]] bool valid_partitioning_mode(PartitioningMode mode) noexcept {
  switch (mode) {
    case PartitioningMode::kAutomatic:
    case PartitioningMode::kConstrainedAutomatic:
    case PartitioningMode::kManual:
      return true;
  }
  return false;
}

[[nodiscard]] bool valid_partition_inspection_plan(const ExecutionPlanState& state) {
  if (!valid_partitioning_mode(state.partitioning_plan.mode)) {
    return false;
  }
  // Rejected planning attempts retain bounded proposal/decision evidence, but publish no accepted
  // inspection authority. Preserve that artifact contract while applying the shared exact-cover
  // validator to every accepted plan consumed by both finalization and Runtime.
  return state.partitioning_plan.disposition != PlanDisposition::kAccepted ||
         valid_accepted_partition_authority(state);
}

void append_profile_provenance(std::string& output, const ProfileManifest& manifest,
                               bool include_provenance) {
  if (!include_provenance) {
    return;
  }
  output += ",\"producer_build_id\":";
  append_json_string(output, manifest.producer_build_id);
}

void append_temporal_channel_and_diagnostic_sections(std::string& output,
                                                     const ExecutionPlanState& state,
                                                     const bool include_provenance) {
  output += ",\"aligned_triggers\":";
  append_array(output, state.aligned_triggers, [](std::string& json, const auto& record) {
    json += "{\"accepted_once_consumption\":";
    append_bool(json, record.accepted_once_consumption);
    json += ",\"algorithm\":";
    append_json_string(json, record.algorithm);
    json += ",\"checked_direct_conversion\":";
    append_bool(json, record.checked_direct_conversion);
    json += ",\"checked_spread_subtraction\":";
    append_bool(json, record.checked_spread_subtraction);
    json += ",\"closure_proofs\":";
    append_array(json, record.closure_proofs, [](std::string& values, auto proof) {
      append_number(values, static_cast<std::uint32_t>(proof));
    });
    json += ",\"comparison_clock\":";
    append_number(json, record.comparison_clock.value);
    json += ",\"exact_window_matches\":";
    append_bool(json, record.exact_window_matches);
    json += ",\"fifo_heads_only\":";
    append_bool(json, record.fifo_heads_only);
    json += ",\"formation_failure_identity\":";
    append_json_string(json, record.formation_failure_identity);
    json += ",\"inclusive_window_ns\":";
    append_number(json, record.inclusive_window_ns);
    json += ",\"members\":";
    append_array(json, record.members, [](std::string& members, const auto& member) {
      members += "{\"cache_capacity\":";
      append_number(members, member.cache_capacity);
      members += ",\"capture_time_required\":";
      append_bool(members, member.capture_time_required);
      members += ",\"conversion_id\":";
      append_number(members, member.conversion_id);
      members += ",\"event_clock\":";
      append_number(members, member.event_clock.value);
      members += ",\"input\":";
      append_number(members, member.input.value());
      members += ",\"member_ordinal\":";
      append_number(members, member.member_ordinal);
      members += ",\"nondecreasing_normalized_time\":";
      append_bool(members, member.nondecreasing_normalized_time);
      members += ",\"queue_capacity\":";
      append_number(members, member.queue_capacity);
      members.push_back('}');
    });
    json += ",\"miss_witness_identity\":";
    append_json_string(json, record.miss_witness_identity);
    json += ",\"missing_requires_finite_proof\":";
    append_bool(json, record.missing_requires_finite_proof);
    json += ",\"no_scan_ahead\":";
    append_bool(json, record.no_scan_ahead);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"public_watermark_protocol\":";
    append_bool(json, record.public_watermark_protocol);
    json += ",\"reference_is_maximum_head\":";
    append_bool(json, record.reference_is_maximum_head);
    json += ",\"retire_all_minimum_heads\":";
    append_bool(json, record.retire_all_minimum_heads);
    json += ",\"retirement_order\":";
    append_json_string(json, record.retirement_order);
    json += ",\"silence_is_not_closure\":";
    append_bool(json, record.silence_is_not_closure);
    json.push_back('}');
  });
  output += ",\"bracketed_windows\":";
  append_array(output, state.bracketed_windows, [](std::string& json, const auto& record) {
    json += "{\"anchor\":";
    append_number(json, record.anchor.value());
    json += ",\"cache_capacity\":";
    append_number(json, record.cache_capacity);
    json += ",\"close_condition\":";
    append_json_string(json, record.close_condition);
    json += ",\"conversion_id\":";
    append_number(json, record.conversion_id);
    json += ",\"event_clock\":";
    append_number(json, record.event_clock.value);
    json += ",\"inclusive_interior\":";
    append_bool(json, record.inclusive_interior);
    json += ",\"input\":";
    append_number(json, record.input.value());
    json += ",\"insufficient_history_reason\":";
    append_json_string(json, record.insufficient_history_reason);
    json += ",\"insufficient_history_witness_identity\":";
    append_json_string(json, record.insufficient_history_witness_identity);
    json += ",\"lease_lifetime_evidence\":";
    append_json_string(json, record.lease_lifetime_evidence);
    json += ",\"lookahead\":";
    append_number(json, record.lookahead);
    json += ",\"lookbehind\":";
    append_number(json, record.lookbehind);
    json += ",\"max_samples\":";
    append_number(json, record.max_samples);
    json += ",\"missing\":";
    append_number(json, static_cast<std::uint32_t>(record.missing));
    json += ",\"ordering\":";
    append_json_string(json, record.ordering);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"payload_identity\":";
    append_payload_identity(json, record.payload_identity);
    json += ",\"producer_epoch_identity_schema\":";
    append_json_string(json, record.producer_epoch_identity_schema);
    json += ",\"producer_max_emits_per_compute\":";
    append_number(json, record.producer_max_emits_per_compute);
    json += ",\"queue_capacity\":";
    append_number(json, record.queue_capacity);
    json += ",\"retain_overlap\":";
    append_bool(json, record.retain_overlap);
    json += ",\"retained_overlap_capacity\":";
    append_number(json, record.retained_overlap_capacity);
    json += ",\"sample_limit_reason\":";
    append_json_string(json, record.sample_limit_reason);
    json += ",\"sample_limit_witness_identity\":";
    append_json_string(json, record.sample_limit_witness_identity);
    json += ",\"semantic_reference\":";
    append_json_string(json, record.semantic_reference);
    json += ",\"source_sequence_identity_schema\":";
    append_json_string(json, record.source_sequence_identity_schema);
    json += ",\"strict_outer_samples\":";
    append_bool(json, record.strict_outer_samples);
    json += ",\"timeout_reason\":";
    append_json_string(json, record.timeout_reason);
    json += ",\"timeout_witness_identity\":";
    append_json_string(json, record.timeout_witness_identity);
    json += ",\"activation_time_policy\":";
    append_json_string(json, record.activation_time_policy);
    json.push_back('}');
  });
  output += ",\"channel_locality\":";
  append_array(output, state.channel_locality, [](std::string& json, const auto& record) {
    json += "{\"accepted\":";
    append_number(json, static_cast<std::uint32_t>(record.accepted));
    json += ",\"edge\":";
    append_number(json, record.edge.value);
    json += ",\"requested\":";
    append_number(json, static_cast<std::uint32_t>(record.requested));
    json += ",\"requires_runtime_attach\":";
    append_bool(json, record.requires_runtime_attach);
    json.push_back('}');
  });
  output += ",\"channel_publishers\":";
  append_array(output, state.channel_publishers, [](std::string& json, const auto& record) {
    json += "{\"max_loaned_samples\":";
    append_number(json, record.max_loaned_samples);
    json += ",\"replacement_attribution_required\":";
    append_bool(json, record.replacement_attribution_required);
    json += ",\"service_id\":";
    append_json_string(json, record.service_id);
    json += ",\"topic\":";
    append_number(json, record.topic.value);
    json.push_back('}');
  });
  output += ",\"channel_services\":";
  append_array(
      output, state.channel_services, [include_provenance](std::string& json, const auto& record) {
        json += "{\"event_wake\":";
        append_bool(json, record.event_wake);
        json += ",\"max_subscriber_borrowed_samples\":";
        append_number(json, record.max_subscriber_borrowed_samples);
        json += ",\"max_subscriber_queue_depth\":";
        append_number(json, record.max_subscriber_queue_depth);
        json += ",\"payload_identity\":";
        append_payload_identity(json, record.payload_identity);
        json += ",\"publisher_history_depth\":";
        append_number(json, record.publisher_history_depth);
        json += ",\"queue_policy\":";
        append_number(json, static_cast<std::uint32_t>(record.queue_policy));
        json += ",\"service_id\":";
        append_json_string(json, record.service_id);
        json += ",\"tensor_emit_policy\":";
        append_number(json, static_cast<std::uint32_t>(record.tensor_emit_policy));
        if (!record.tensor_fields.empty()) {
          json += ",\"tensor_fields\":";
          append_descriptor_tensor_fields(json, record.tensor_fields, include_provenance);
        }
        json += ",\"tensor_payload\":";
        append_bool(json, record.tensor_payload);
        json += ",\"topic\":";
        append_number(json, record.topic.value);
        json.push_back('}');
      });
  output += ",\"channel_subscribers\":";
  append_array(output, state.channel_subscribers, [](std::string& json, const auto& record) {
    json += "{\"event_wake\":";
    append_bool(json, record.event_wake);
    json += ",\"max_borrowed_samples\":";
    append_number(json, record.max_borrowed_samples);
    json += ",\"requested_queue_depth\":";
    append_number(json, record.requested_queue_depth);
    json += ",\"service_id\":";
    append_json_string(json, record.service_id);
    json += ",\"topic\":";
    append_number(json, record.topic.value);
    json.push_back('}');
  });
  output += ",\"clock_catalog\":{\"clocks\":";
  append_clock_plan_records(output, state.clocks);
  output += ",\"plan_generation\":";
  append_number(output, state.plan_generation);
  output += ",\"root_default_clock\":";
  append_number(output, state.root_default_clock.value);
  output += "},\"clock_conversions\":";
  append_array(
      output, state.clock_conversions, [](std::string& json, const ClockConversionRecord& record) {
        json += "{\"denominator\":";
        append_number(json, record.scale.denominator);
        json += ",\"from\":";
        append_number(json, record.from.value);
        json += ",\"graph\":";
        append_number(json, record.graph.value);
        json += ",\"id\":";
        append_number(json, record.id);
        json += ",\"numerator\":";
        append_number(json, record.scale.numerator);
        json += ",\"offset_ns\":";
        append_number(json, record.offset_ns);
        json += ",\"overflow\":";
        append_json_string(json,
                           record.overflow == ClockOverflowPolicy::kError ? "error" : "unknown");
        json += ",\"plan_generation\":";
        append_number(json, record.plan_generation);
        json += ",\"rounding\":";
        append_json_string(
            json,
            record.rounding == ClockRoundingMode::kNearestTiesToEven ? "ties_to_even" : "unknown");
        json += ",\"to\":";
        append_number(json, record.to.value);
        json.push_back('}');
      });
  output += ",\"clock_import_resolutions\":";
  append_array(output, state.clock_import_resolutions, [](std::string& json, const auto& record) {
    json += "{\"child_clock\":";
    append_json_string(json, record.child_clock.name());
    json += ",\"child_graph\":";
    append_number(json, record.child_graph.value);
    json += ",\"is_default\":";
    append_bool(json, record.is_default);
    json += ",\"parent_clock\":";
    append_json_string(json, record.parent_clock.name());
    json += ",\"parent_graph\":";
    append_number(json, record.parent_graph.value);
    json += ",\"resolved_clock\":";
    append_number(json, record.resolved_clock.value);
    json.push_back('}');
  });
  output += ",\"clock_triggers\":";
  append_array(output, state.clock_triggers, [](std::string& json, const auto& record) {
    json += "{\"arithmetic_failure_identity\":";
    append_json_string(json, record.arithmetic_failure_identity);
    json += ",\"checked_terminal_no_wrap\":";
    append_bool(json, record.checked_terminal_no_wrap);
    json += ",\"clock\":";
    append_number(json, record.clock.value);
    json += ",\"coalesced_range_capacity\":";
    append_number(json, record.coalesced_range_capacity);
    json += ",\"coalesced_range_identity\":";
    append_json_string(json, record.coalesced_range_identity);
    json += ",\"epoch_discontinuity\":";
    append_number(json, static_cast<std::uint32_t>(record.epoch_discontinuity));
    json += ",\"exact_now_eligible\":";
    append_bool(json, record.exact_now_eligible);
    json += ",\"explicit_phase\":";
    append_optional_time_point(json, record.explicit_phase);
    json += ",\"explicit_phase_preserved_on_restart\":";
    append_bool(json, record.explicit_phase_preserved_on_restart);
    json += ",\"failed_rollback_restores_watermark\":";
    append_bool(json, record.failed_rollback_restores_watermark);
    json += ",\"global_start_barrier\":";
    append_bool(json, record.global_start_barrier);
    json += ",\"greatest_due_constant_time\":";
    append_bool(json, record.greatest_due_constant_time);
    json += ",\"initial_handled_index\":";
    if (record.initial_handled_index) {
      append_number(json, *record.initial_handled_index);
    } else {
      json += "null";
    }
    json += ",\"initial_tick_index\":";
    append_number(json, record.initial_tick_index);
    json += ",\"omitted_phase_index_zero_immediate\":";
    append_bool(json, record.omitted_phase_index_zero_immediate);
    json += ",\"omitted_phase_recaptured_on_restart\":";
    append_bool(json, record.omitted_phase_recaptured_on_restart);
    json += ",\"one_candidate_per_observation\":";
    append_bool(json, record.one_candidate_per_observation);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"period_ns\":";
    append_number(json, record.period_ns);
    json += ",\"phase_conversion_id\":";
    append_number(json, record.phase_conversion_id);
    json += ",\"phase_origin\":";
    append_number(json, static_cast<std::uint32_t>(record.phase_origin));
    json += ",\"scheduled_tick_is_reference\":";
    append_bool(json, record.scheduled_tick_is_reference);
    json += ",\"skipped_indices_have_no_candidate\":";
    append_bool(json, record.skipped_indices_have_no_candidate);
    json += ",\"stale_anchor_identity\":";
    append_json_string(json, record.stale_anchor_identity);
    json += ",\"stop_blocks_future_freeze\":";
    append_bool(json, record.stop_blocks_future_freeze);
    json += ",\"tick_policy\":";
    append_number(json, static_cast<std::uint32_t>(record.tick_policy));
    json.push_back('}');
  });
  output += ",\"deadlines\":";
  append_array(output, state.deadlines, [](std::string& json, const auto& record) {
    json += "{\"action\":";
    append_number(json, static_cast<std::uint32_t>(record.action));
    json += ",\"automatic_output\":";
    append_bool(json, record.automatic_output);
    json += ",\"budget_ns\":";
    append_number(json, record.budget_ns);
    json += ",\"callback_count_if_late\":";
    append_number(json, record.callback_count_if_late);
    json += ",\"callback_state_rollback\":";
    append_bool(json, record.callback_state_rollback);
    json += ",\"checked_addition\":";
    append_bool(json, record.checked_addition);
    json += ",\"consumes_candidate_tag\":";
    append_bool(json, record.consumes_candidate_tag);
    json += ",\"conversion_id\":";
    append_number(json, record.conversion_id);
    json += ",\"duration_measurement_separate\":";
    append_bool(json, record.duration_measurement_separate);
    json += ",\"exact_boundary_is_on_time\":";
    append_bool(json, record.exact_boundary_is_on_time);
    json += ",\"final_invocation_gate\":";
    append_bool(json, record.final_invocation_gate);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"source_clock\":";
    append_number(json, record.source_clock.value);
    json += ",\"source_coordinate\":";
    append_number(json, static_cast<std::uint32_t>(record.source_coordinate));
    json += ",\"source_coordinate_identity\":";
    append_json_string(json, record.source_coordinate_identity);
    json += ",\"violation_witness_identity\":";
    append_json_string(json, record.violation_witness_identity);
    json.push_back('}');
  });
  output += ",\"diagnostics\":";
  append_array(output, state.diagnostics, [](std::string& json, const Diagnostic& diagnostic) {
    json += "{\"cause_id\":";
    if (diagnostic.cause_id) {
      append_number(json, diagnostic.cause_id->value);
    } else {
      json += "null";
    }
    json += ",\"endpoint\":";
    append_json_string(json, diagnostic.endpoint);
    json += ",\"id\":";
    append_number(json, diagnostic.id.value);
    json += ",\"json_pointer\":";
    append_json_string(json, diagnostic.json_pointer);
    json += ",\"provided\":";
    append_json_string(json, diagnostic.provided);
    json += ",\"required\":";
    append_json_string(json, diagnostic.required);
    json += ",\"service\":";
    append_json_string(json, diagnostic.service);
    json += ",\"severity\":";
    append_number(json, static_cast<std::uint32_t>(diagnostic.severity));
    json += ",\"source\":{\"graph\":";
    append_number(json, diagnostic.source.graph.value);
    json += ",\"graph_generation\":";
    append_number(json, diagnostic.source.graph_generation);
    json += ",\"port\":";
    append_number(json, diagnostic.source.port.value());
    json += ",\"vertex\":";
    append_number(json, diagnostic.source.vertex.value);
    json.push_back('}');
    json += ",\"token\":";
    append_json_string(json, diagnostic.token);
    json.push_back('}');
  });
  output += ",\"endpoint_catalog\":";
  output += state.endpoint_catalog.json().empty() ? "{}" : state.endpoint_catalog.json();
  output += ",\"endpoint_clock_health\":";
  append_array(output, state.endpoint_clock_health, [](std::string& json, const auto& record) {
    const auto append_duration = [](std::string& target, const auto& duration) {
      if (duration.has_value()) {
        append_number(target, duration->count());
      } else {
        target += "null";
      }
    };
    json += "{\"admitted_jitter_margin_ns\":";
    append_duration(json, record.admitted_jitter_margin);
    json += ",\"capability\":";
    append_number(json, static_cast<std::uint32_t>(record.capability));
    json += ",\"capture_clock\":";
    append_number(json, record.capture_clock.value);
    json += ",\"capture_conversion_id\":";
    append_number(json, record.capture_conversion_id);
    json += ",\"clock_health_update_witness_id\":";
    append_json_string(json, record.clock_health_update_witness_id);
    json += ",\"conversion_generation\":";
    append_number(json, record.conversion_generation);
    json += ",\"degraded_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.degraded_disposition));
    json += ",\"endpoint\":";
    append_number(json, record.endpoint.value());
    json += ",\"fixed_report_ingress_capacity\":";
    append_number(json, record.fixed_report_ingress_capacity);
    json += ",\"fixed_update_capacity\":";
    append_number(json, record.fixed_update_capacity);
    json += ",\"health_age_clock\":";
    append_number(json, record.health_age_clock.value);
    json += ",\"health_update_sequence_id\":";
    append_json_string(json, record.health_update_sequence_id);
    json += ",\"ingress_capacity_origin\":";
    append_number(json, static_cast<std::uint32_t>(record.ingress_capacity_origin));
    json += ",\"max_monitor_period_ns\":";
    append_duration(json, record.max_monitor_period);
    json += ",\"max_observation_to_call_delay_ns\":";
    append_duration(json, record.max_observation_to_call_delay);
    json += ",\"max_report_age_ns\":";
    append_duration(json, record.max_report_age);
    json += ",\"max_report_delivery_delay_ns\":";
    append_duration(json, record.max_report_delivery_delay);
    json += ",\"monitor_evidence_reference\":";
    append_json_string(json, record.monitor_evidence_reference);
    json += ",\"monitor_path_kind\":";
    append_number(json, static_cast<std::uint32_t>(record.monitor_path_kind));
    json += ",\"no_report_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.no_report_disposition));
    json += ",\"per_dispatch_report_budget\":";
    append_number(json, record.per_dispatch_report_budget);
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"producer_epoch_source\":";
    append_number(json, static_cast<std::uint32_t>(record.producer_epoch_source));
    json += ",\"publisher_operation_sequence_id\":";
    append_json_string(json, record.publisher_operation_sequence_id);
    json += ",\"report_sequence_id\":";
    append_json_string(json, record.report_sequence_id);
    json += ",\"requirement\":";
    append_number(json, static_cast<std::uint32_t>(record.requirement));
    json += ",\"stale_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.stale_disposition));
    json += ",\"trace_loss_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.trace_loss_disposition));
    json += ",\"unhealthy_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.unhealthy_disposition));
    json += ",\"unspecified_reason_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.unspecified_reason_disposition));
    json.push_back('}');
  });
  output += ",\"clock_health_reason_schema\":[";
  constexpr std::array<std::string_view, 10> kClockHealthReasonTokens{
      "none",
      "synchronization_lock_lost",
      "phase_error_limit_exceeded",
      "rate_error_limit_exceeded",
      "calibration_expired",
      "clock_stalled",
      "timestamp_regression",
      "timestamp_source_fault",
      "monitor_fault",
      "unspecified",
  };
  for (std::size_t index = 0U; index < kClockHealthReasonTokens.size(); ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    output += "{\"id\":";
    append_number(output, index);
    output += ",\"token\":";
    append_json_string(output, kClockHealthReasonTokens[index]);
    output.push_back('}');
  }
  output.push_back(']');
}

void append_graph_and_lifecycle_sections(std::string& output, const ExecutionPlanState& state) {
  const auto capture_eligible = [](const VertexRecord& vertex) {
    return vertex.cuda_graph_capture_eligible;
  };
  if (std::any_of(state.graph.vertices().begin(), state.graph.vertices().end(), capture_eligible)) {
    output += ",\"cuda_graph_capture_eligible_operators\":";
    append_filtered_array(output,
                          state.graph.vertices(),
                          capture_eligible,
                          [](std::string& json, const VertexRecord& vertex) {
                            json += "{\"graph\":";
                            append_number(json, vertex.graph.value);
                            json += ",\"name\":";
                            append_json_string(json, vertex.name);
                            json += ",\"vertex\":";
                            append_number(json, vertex.id.value);
                            json.push_back('}');
                          });
  }
  const auto cuda_graph_region = [](const RegionRecord& region) {
    return std::any_of(region.constraints.begin(), region.constraints.end(), [](const auto& item) {
      return item.kind() == RegionConstraintKind::kCudaGraph;
    });
  };
  if (std::any_of(state.graph.regions().begin(), state.graph.regions().end(), cuda_graph_region)) {
    output += ",\"cuda_graph_regions\":";
    append_filtered_array(
        output,
        state.graph.regions(),
        cuda_graph_region,
        [](std::string& json, const RegionRecord& region) {
          json += "{\"constraint_kind\":\"cuda_graph\",\"graph\":";
          append_number(json, region.graph.value);
          json += ",\"graph_members\":";
          append_array(json, region.graph_members, [](std::string& values, GraphId graph) {
            append_number(values, graph.value);
          });
          json += ",\"id\":";
          append_number(json, region.id.value);
          json += ",\"members\":";
          append_array(json, region.members, [](std::string& values, VertexId vertex) {
            append_number(values, vertex.value);
          });
          json += ",\"name\":";
          append_json_string(json, region.name);
          json.push_back('}');
        });
  }
  if (!state.cuda_graph_capture_plans.empty()) {
    const auto append_capture_edge = [](std::string& json, const CUDAGraphCaptureEdgeRecord& edge) {
      json += "{\"edge\":";
      append_number(json, edge.edge.value);
      json += ",\"source\":";
      append_number(json, edge.source.value);
      json += ",\"source_port\":";
      append_number(json, edge.source_port.value());
      json += ",\"target\":";
      append_number(json, edge.target.value);
      json += ",\"target_port\":";
      append_number(json, edge.target_port.value());
      json.push_back('}');
    };
    output += ",\"cuda_graph_capture_plans\":";
    append_array(
        output, state.cuda_graph_capture_plans, [&](std::string& json, const auto& capture_plan) {
          json += "{\"contracts\":";
          append_array(json, capture_plan.contracts, [](std::string& values, const auto& contract) {
            values += "{\"owner\":";
            append_operator_ref(values, contract.owner);
            values += ",\"synthesized_default\":";
            append_bool(values, contract.synthesized_default);
            values += ",\"trigger\":";
            append_number(values, static_cast<std::uint32_t>(contract.trigger));
            values += ",\"trigger_input\":";
            append_number(values, contract.trigger_input.value());
            values.push_back('}');
          });
          json += ",\"egress\":";
          if (capture_plan.egress) {
            append_capture_edge(json, *capture_plan.egress);
          } else {
            json += "null";
          }
          json += ",\"graph\":";
          append_number(json, capture_plan.graph.value);
          json += ",\"ingress\":";
          if (capture_plan.ingress) {
            append_capture_edge(json, *capture_plan.ingress);
          } else {
            json += "null";
          }
          json += ",\"internal_edges\":";
          append_array(json, capture_plan.internal_edges, append_capture_edge);
          json += ",\"members\":";
          append_array(json, capture_plan.members, [](std::string& values, const auto& member) {
            values += "{\"operator_ref\":";
            append_operator_ref(values, member.operator_ref);
            values += ",\"vertex\":";
            append_number(values, member.vertex.value);
            values.push_back('}');
          });
          json += ",\"region\":";
          append_number(json, capture_plan.region.value);
          json += ",\"region_name\":";
          append_json_string(json, capture_plan.region_name);
          json += ",\"sink\":";
          append_operator_ref(json, capture_plan.sink);
          json += ",\"source\":";
          append_operator_ref(json, capture_plan.source);
          json.push_back('}');
        });
  }
  output += ",\"graph_generation\":";
  append_number(output, state.graph_identity.version);
  output += ",\"graph_hash\":";
  append_hex(output, state.graph_identity.hash.value);
  output += ",\"generic_output_loans\":";
  append_array(output, state.generic_output_loans, [](std::string& json, const auto& record) {
    json += "{\"aggregate_byte_capacity\":";
    append_number(json, record.aggregate_byte_capacity);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"fallback_rejected\":";
    append_bool(json, record.fallback_rejected);
    json += ",\"materialize_before\":";
    append_number(json, static_cast<std::uint32_t>(record.materialize_before));
    json += ",\"max_emits_per_compute\":";
    append_number(json, record.max_emits_per_compute);
    json += ",\"max_live_loans\":";
    append_number(json, record.max_live_loans);
    json += ",\"output\":";
    append_number(json, record.output.value());
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"pool_id\":";
    append_json_string(json, record.pool_id);
    json += ",\"pre_submit_preserves_same_activation_loan\":";
    append_bool(json, record.pre_submit_preserves_same_activation_loan);
    json += ",\"retirement_capacity\":";
    append_number(json, record.retirement_capacity);
    json += ",\"runtime_retry_enabled\":";
    append_bool(json, record.runtime_retry_enabled);
    json += ",\"run_generation\":";
    append_number(json, record.run_generation);
    json += ",\"slot_count\":";
    append_number(json, record.slot_count);
    json += ",\"slot_size_bytes\":";
    append_number(json, record.slot_size_bytes);
    json += ",\"trait_evidence\":";
    append_number(json, static_cast<std::uint32_t>(record.trait_evidence));
    json.push_back('}');
  });
  output += ",\"imported_generation_capacities\":";
  append_array(
      output, state.imported_generation_capacities, [](std::string& json, const auto& record) {
        json += "{\"attach_max_wait_ns\":";
        if (record.attach_max_wait_ns) {
          append_number(json, *record.attach_max_wait_ns);
        } else {
          json += "null";
        }
        json += ",\"candidate_reservation_capacity\":";
        append_number(json, record.candidate_reservation_capacity);
        json += ",\"effective_profile\":";
        append_optional_profile(json, record.effective_profile);
        json += ",\"endpoint_max_generations\":";
        append_number(json, record.endpoint_max_generations);
        json += ",\"endpoint_or_service_id\":";
        append_json_string(json, record.endpoint_or_service_id);
        json += ",\"fallback_rejected\":";
        append_bool(json, record.fallback_rejected);
        json += ",\"no_wrap_aba_generation\":";
        append_bool(json, record.no_wrap_aba_generation);
        json += ",\"peer_requirement\":";
        append_number(json, static_cast<std::uint32_t>(record.peer_requirement));
        json += ",\"quarantine_record_capacity\":";
        append_number(json, record.quarantine_record_capacity);
        json += ",\"release_record_capacity\":";
        append_number(json, record.release_record_capacity);
        json += ",\"restart_capable_minimum_two\":";
        append_bool(json, record.restart_capable_minimum_two);
        json += ",\"runtime_max_generations\":";
        append_number(json, record.runtime_max_generations);
        json += ",\"runtime_scope_id\":";
        append_json_string(json, record.runtime_scope_id);
        json += ",\"service_max_generations\":";
        append_number(json, record.service_max_generations);
        json += ",\"topic\":";
        append_number(json, record.topic.value);
        json.push_back('}');
      });
  output += ",\"input_selections\":";
  append_array(output, state.input_selections, [](std::string& json, const auto& record) {
    json += "{\"batch_selection_mode\":";
    append_number(json, static_cast<std::uint32_t>(record.batch_selection_mode));
    json += ",\"cache_capacity\":";
    append_number(json, record.cache_capacity);
    json += ",\"cardinality\":";
    append_number(json, static_cast<std::uint32_t>(record.cardinality));
    json += ",\"conversion_id\":";
    append_number(json, record.conversion_id);
    json += ",\"input\":";
    append_number(json, record.input.value());
    json += ",\"lookahead\":";
    append_number(json, record.lookahead);
    json += ",\"lookbehind\":";
    append_number(json, record.lookbehind);
    json += ",\"max_age_ns\":";
    append_number(json, record.max_age_ns);
    json += ",\"max_selected_samples\":";
    append_number(json, record.max_selected_samples);
    json += ",\"max_selected_samples_origin\":";
    append_number(json, static_cast<std::uint32_t>(record.max_selected_samples_origin));
    json += ",\"missing\":";
    append_number(json, static_cast<std::uint32_t>(record.missing));
    json += ",\"reference_clock\":";
    append_number(json, record.reference_clock.value);
    json += ",\"reuse_max_age_ns\":";
    append_number(json, record.reuse_max_age_ns);
    json += ",\"selector\":";
    append_number(json, static_cast<std::uint32_t>(record.selector));
    json += ",\"unwired_safe_default\":";
    append_bool(json, record.unwired_safe_default);
    json.push_back('}');
  });
  output += ",\"interface_port_bindings\":";
  append_array(output, state.interface_port_bindings, [](std::string& json, const auto& record) {
    json += "{\"direction\":";
    append_number(json, static_cast<std::uint32_t>(record.direction));
    json += ",\"internal_port\":";
    append_number(json, record.internal_port.value());
    json += ",\"public_port\":";
    append_number(json, record.public_port.value());
    json.push_back('}');
  });
  output += ",\"lifecycle_checkpoints\":";
  append_array(output, state.lifecycle_checkpoints, [](std::string& json, const auto& record) {
    json += "{\"callback\":";
    append_bool(json, record.callback_present);
    json += ",\"level\":";
    append_number(json, record.stable_topological_level);
    json += ",\"operator\":";
    append_operator_ref(json, record.operator_instance);
    json += ",\"required\":";
    append_bool(json, record.required);
    json += ",\"stage\":";
    append_number(json, static_cast<std::uint32_t>(record.stage));
    json.push_back('}');
  });
  output += ",\"lifecycle_dependencies\":";
  append_array(output, state.lifecycle_dependencies, [](std::string& json, const auto& record) {
    json += "{\"after\":";
    append_operator_ref(json, record.after);
    json += ",\"before\":";
    append_operator_ref(json, record.before);
    json += ",\"key\":";
    append_json_string(json, record.dependency_key);
    json += ",\"kind\":";
    append_number(json, static_cast<std::uint32_t>(record.kind));
    json += ",\"stage\":";
    append_number(json, static_cast<std::uint32_t>(record.stage));
    json.push_back('}');
  });
  output += ",\"lifecycle_mutable_fields\":";
  append_array(output, state.lifecycle_mutable_fields, [](std::string& json, const auto& record) {
    json += "{\"field\":";
    append_number(json, record.field);
    json += ",\"from_stage\":";
    append_number(json, static_cast<std::uint32_t>(record.from_stage));
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json.push_back('}');
  });
  output += ",\"lifecycle_restart_closures\":";
  append_array(output, state.lifecycle_restart_closures, [](std::string& json, const auto& record) {
    json += "{\"closure\":";
    append_array(json, record.closure, [](std::string& values, const OperatorRef& member) {
      append_operator_ref(values, member);
    });
    json += ",\"required_fenced_operators\":";
    append_number(json, record.required_fenced_operators);
    json += ",\"required_quiesce_records\":";
    append_number(json, record.required_quiesce_records);
    json += ",\"required_rollback_records\":";
    append_number(json, record.required_rollback_records);
    json += ",\"stage_resolutions\":";
    append_array(json, record.stage_resolutions, [](std::string& stages, const auto& stage) {
      stages += "{\"effective_stage\":";
      append_number(stages, static_cast<std::uint32_t>(stage.effective_stage));
      stages += ",\"reason\":";
      append_number(stages, static_cast<std::uint32_t>(stage.reason));
      stages += ",\"required_quiesce_records\":";
      append_number(stages, stage.required_quiesce_records);
      stages += ",\"required_rollback_records\":";
      append_number(stages, stage.required_rollback_records);
      stages += ",\"required_structural_event_records\":";
      append_number(stages, stage.required_structural_event_records);
      stages += ",\"requested_stage\":";
      append_number(stages, static_cast<std::uint32_t>(stage.requested_stage));
      stages.push_back('}');
    });
    json += ",\"target\":";
    append_operator_ref(json, record.target);
    json += ",\"valid_requested_stage_mask\":";
    append_number(json, record.valid_requested_stage_mask);
    json.push_back('}');
  });
  output += ",\"lifecycle_restart_permissions\":";
  append_array(
      output, state.lifecycle_restart_permissions, [](std::string& json, const auto& record) {
        json += "{\"requester\":";
        append_operator_ref(json, record.requester);
        json += ",\"target\":";
        append_operator_ref(json, record.target);
        json.push_back('}');
      });
  output += ",\"lifecycle_restart_targets\":";
  append_array(output, state.lifecycle_restart_targets, [](std::string& json, const auto& record) {
    json += "{\"allowed_from_stage_mask\":";
    append_number(json, record.allowed_from_stage_mask);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"external_control_allowed\":";
    append_bool(json, record.external_control_allowed);
    json += ",\"parameter_origin_allowed\":";
    append_bool(json, record.parameter_origin_allowed);
    json += ",\"profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"reason\":";
    append_number(json, record.disposition_reason);
    json += ",\"scope\":";
    append_json_string(json, record.effective_scope_id);
    json += ",\"self_allowed\":";
    append_bool(json, record.self_allowed);
    json += ",\"target\":";
    append_operator_ref(json, record.target);
    json.push_back('}');
  });
  output += ",\"lifecycle_transaction_capacity\":";
  if (state.lifecycle_transaction_capacity) {
    const auto& capacity = *state.lifecycle_transaction_capacity;
    output += "{\"configured\":";
    append_lifecycle_capacity_values(output, capacity.configured);
    output += ",\"derivation_reason\":";
    append_number(output, capacity.derivation_reason);
    output += ",\"parameter_origin_enabled\":";
    append_bool(output, capacity.parameter_origin_enabled);
    output += ",\"required_generation_slots_per_operator\":";
    append_number(output, capacity.required_generation_slots_per_operator);
    output += ",\"required_minimum\":";
    append_lifecycle_capacity_values(output, capacity.required_minimum);
    output += ",\"required_parameter_arena_bytes\":";
    append_number(output, capacity.required_parameter_arena_bytes);
    output += ",\"required_parameter_audit_records\":";
    append_number(output, capacity.required_parameter_audit_records);
    output += ",\"resolved_parameter_update_policy\":";
    append_number(output, static_cast<std::uint32_t>(capacity.resolved_parameter_update_policy));
    output.push_back('}');
  } else {
    output += "null";
  }
  output += ",\"missing_aggregates\":";
  append_array(output, state.missing_aggregates, [](std::string& json, const auto& record) {
    json += "{\"absent_callback_count\":";
    append_number(json, record.absent_callback_count);
    json += ",\"absent_without_safe_default_no_fires\":";
    append_bool(json, record.absent_without_safe_default_no_fires);
    json += ",\"aggregate_identity\":";
    append_json_string(json, record.aggregate_identity);
    json += ",\"bracketed_window_all_or_nothing\":";
    append_bool(json, record.bracketed_window_all_or_nothing);
    json += ",\"degraded_callback_count\":";
    append_number(json, record.degraded_callback_count);
    json += ",\"degraded_commits_destructive_selection\":";
    append_bool(json, record.degraded_commits_destructive_selection);
    json += ",\"evaluate_complete_set\":";
    append_bool(json, record.evaluate_complete_set);
    json += ",\"input_state_identity\":";
    append_json_string(json, record.input_state_identity);
    json += ",\"inputs\":";
    append_array(json, record.inputs, [](std::string& inputs, const auto& input) {
      inputs += "{\"input\":";
      append_number(inputs, input.input.value());
      inputs += ",\"missing\":";
      append_number(inputs, static_cast<std::uint32_t>(input.missing));
      inputs += ",\"ordinal\":";
      append_number(inputs, input.ordinal);
      inputs += ",\"selector\":";
      append_number(inputs, static_cast<std::uint32_t>(input.selector));
      inputs.push_back('}');
    });
    json += ",\"no_fire_preserves_provisional_selection\":";
    append_bool(json, record.no_fire_preserves_provisional_selection);
    json += ",\"nominal_callback_count\":";
    append_number(json, record.nominal_callback_count);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"phase\":";
    append_number(json, static_cast<std::uint32_t>(record.phase));
    json += ",\"safe_default_dominates\":";
    append_bool(json, record.safe_default_dominates);
    json += ",\"selection_reason_identity\":";
    append_json_string(json, record.selection_reason_identity);
    json.push_back('}');
  });
  if (!state.operator_completion_rules.empty()) {
    output += ",\"operator_completion_rules\":";
    append_array(output,
                 state.operator_completion_rules,
                 [](std::string& json, const OperatorCompletionPlanRecord& record) {
                   json += "{\"kind\":";
                   append_number(json, static_cast<std::uint32_t>(record.kind));
                   json += ",\"limit\":";
                   append_number(json, record.limit);
                   json += ",\"owner\":";
                   append_operator_ref(json, record.owner);
                   json.push_back('}');
                 });
  }
  output += ",\"operator_parameters\":";
  append_array(output, state.operator_parameters, [](std::string& json, const auto& record) {
    json += "{\"contract_hash\":";
    append_hex(json, record.contract_hash.value);
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"fixed_scalar_generation_strategy\":";
    append_bool(json, record.fixed_scalar_generation_strategy);
    json += ",\"initial_generation\":";
    append_number(json, record.initial_generation);
    json += ",\"lifecycle_updates_enabled\":";
    append_bool(json, record.lifecycle_updates_enabled);
    json += ",\"metadata_hash\":";
    append_hex(json, record.metadata_hash.value);
    json += ",\"no_params\":";
    append_bool(json, record.no_params);
    json += ",\"object_alignment\":";
    append_number(json, record.object_alignment);
    json += ",\"object_size\":";
    append_number(json, record.object_size);
    json += ",\"operator_path\":";
    append_json_string(json, record.operator_path);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"runtime_updates_enabled\":";
    append_bool(json, record.runtime_updates_enabled);
    json += ",\"schema_availability\":";
    append_number(json, static_cast<std::uint32_t>(record.schema_availability));
    json += ",\"schema_coverage\":";
    append_number(json, static_cast<std::uint32_t>(record.schema_coverage));
    json += ",\"schema_id\":";
    append_json_string(json, record.schema_id);
    json += ",\"schema_proof\":";
    append_number(json, static_cast<std::uint32_t>(record.schema_proof));
    json += ",\"schema_version\":";
    append_number(json, record.schema_version);
    json += ",\"storage\":";
    append_parameter_storage(json, record.storage);
    json += ",\"structural_hash\":";
    append_hex(json, record.structural_hash.value);
    json.push_back('}');
  });
  output += ",\"parameter_derived_terms\":";
  append_array(output, state.parameter_derived_terms, [](std::string& json, const auto& record) {
    json += "{\"field\":";
    append_number(json, record.field);
    json += ",\"field_full_id\":";
    append_hex(json, record.field_full_id.value);
    json += ",\"field_path\":";
    append_json_string(json, record.field_path);
    json += ",\"initial_value_ns\":";
    append_number(json, record.initial_value_ns);
    json += ",\"input\":";
    append_number(json, record.input.value());
    json += ",\"kind\":";
    append_number(json, static_cast<std::uint32_t>(record.kind));
    json += ",\"maximum_value_ns\":";
    append_number(json, record.maximum_value_ns);
    json += ",\"minimum_value_ns\":";
    append_number(json, record.minimum_value_ns);
    json += ",\"mutability\":";
    append_number(json, static_cast<std::uint32_t>(record.mutability));
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"schema_id\":";
    append_json_string(json, record.schema_id);
    json += ",\"schema_version\":";
    append_number(json, record.schema_version);
    json += ",\"term_id\":";
    append_json_string(json, record.term_id);
    json += ",\"update_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.update_disposition));
    json.push_back('}');
  });
  output += ",\"parameter_fields\":";
  append_array(output, state.parameter_fields, [](std::string& json, const auto& record) {
    json += "{\"canonical_name\":";
    append_json_string(json, record.canonical_name);
    json += ",\"earliest_invalidated_stage\":";
    append_number(json, static_cast<std::uint32_t>(record.earliest_invalidated_stage));
    json += ",\"enum_choices\":";
    append_array(json, record.enum_choices, [](std::string& enum_json, const auto& choice) {
      enum_json += "{\"name\":";
      append_json_string(enum_json, choice.name);
      enum_json += ",\"value\":{\"bit_width\":";
      append_number(enum_json, choice.value.bit_width);
      enum_json += ",\"kind\":";
      append_number(enum_json, static_cast<std::uint32_t>(choice.value.kind));
      enum_json += ",\"raw_bits\":";
      append_number(enum_json, choice.value.raw_bits);
      enum_json += "}}";
    });
    json += ",\"external_access\":";
    append_number(json, static_cast<std::uint32_t>(record.external_access));
    json += ",\"field\":";
    append_number(json, record.field);
    json += ",\"full_id\":";
    append_hex(json, record.full_id.value);
    json += ",\"logical_type\":";
    append_number(json, static_cast<std::uint32_t>(record.logical_type));
    json += ",\"mutability\":";
    append_number(json, static_cast<std::uint32_t>(record.mutability));
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"persistence\":";
    append_number(json, static_cast<std::uint32_t>(record.persistence));
    json += ",\"sensitivity\":";
    append_number(json, static_cast<std::uint32_t>(record.sensitivity));
    json += ",\"storage\":";
    append_parameter_storage(json, record.storage);
    json += ",\"type_shape\":";
    append_parameter_type_shape(json, record.type_shape);
    json += ",\"update_disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.update_disposition));
    json += ",\"value_route\":";
    append_number(json, static_cast<std::uint32_t>(record.value_route));
    json += ",\"visibility\":";
    append_number(json, static_cast<std::uint32_t>(record.visibility));
    json.push_back('}');
  });
  output += ",\"parameter_runtime_capacity\":";
  if (state.parameter_runtime_capacity) {
    const auto& capacity = *state.parameter_runtime_capacity;
    output += "{\"arena_strategy\":";
    append_number(output, static_cast<std::uint32_t>(capacity.arena_strategy));
    output += ",\"audit_capacity\":";
    append_number(output, capacity.audit_capacity);
    output += ",\"compute_ingress_capacity\":";
    append_number(output, capacity.compute_ingress_capacity);
    output += ",\"enabled_operator_count\":";
    append_number(output, capacity.enabled_operator_count);
    output += ",\"event_capacity\":";
    append_number(output, capacity.event_capacity);
    output += ",\"generation_slots_per_operator\":";
    append_number(output, capacity.generation_slots_per_operator);
    output += ",\"lifecycle_updates_enabled\":";
    append_bool(output, capacity.lifecycle_updates_enabled);
    output += ",\"max_arena_alignment\":";
    append_number(output, capacity.max_arena_alignment);
    output += ",\"max_compute_patches_per_request\":";
    append_number(output, capacity.max_compute_patches_per_request);
    output += ",\"max_event_cursors\":";
    append_number(output, capacity.max_event_cursors);
    output += ",\"max_pending_updates_per_operator\":";
    append_number(output, capacity.max_pending_updates_per_operator);
    output += ",\"parameter_arena_bytes\":";
    append_number(output, capacity.parameter_arena_bytes);
    output += ",\"required_lifecycle_capacity\":";
    append_lifecycle_capacity_values(output, capacity.required_lifecycle_capacity);
    output += ",\"resolved_update_policy\":";
    append_number(output, static_cast<std::uint32_t>(capacity.resolved_update_policy));
    output += ",\"runtime_updates_enabled\":";
    append_bool(output, capacity.runtime_updates_enabled);
    output.push_back('}');
  } else {
    output += "null";
  }
  plan_artifact_rate_serialization_internal::append_rate_plan_sections(output, state);
}

// Canonical artifact order is clearer as one linear section writer; optional fields and record
// lambdas inflate the metric without introducing control-flow coupling.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void append_topology_and_predicate_sections(std::string& output, const ExecutionPlanState& state,
                                            const bool include_provenance) {
  output += ",\"payload_edges\":";
  append_array(output, state.payload_edges, [](std::string& json, const auto& record) {
    json += "{\"cache_capacity\":";
    append_number(json, record.cache_capacity);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"edge\":";
    append_number(json, record.edge.value);
    json += ",\"fanout\":";
    append_number(json, record.fanout);
    json += ",\"locality\":";
    append_number(json, static_cast<std::uint32_t>(record.locality));
    json += ",\"max_inflight_samples\":";
    append_number(json, record.max_inflight_samples);
    json += ",\"payload_identity\":";
    append_payload_identity(json, record.payload_identity);
    json += ",\"producer_max_emits_per_compute\":";
    append_number(json, record.producer_max_emits_per_compute);
    json += ",\"queue_depth\":";
    append_number(json, record.queue_depth);
    json += ",\"queue_policy\":";
    append_number(json, static_cast<std::uint32_t>(record.queue_policy));
    json += ",\"reason\":";
    append_number(json, record.reason.value);
    json += ",\"source\":";
    append_number(json, record.source.value());
    json += ",\"target\":";
    append_number(json, record.target.value());
    json.push_back('}');
  });
  plan_artifact_rate_serialization_internal::append_pool_capacity_plan_sections(output, state);
  output += ",\"ports\":";
  append_array(output, state.ports, [include_provenance](std::string& json, const auto& record) {
    json += "{\"direction\":";
    append_number(json, static_cast<std::uint32_t>(record.direction));
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"generic_loan_slot_size_bytes\":";
    append_number(json, record.generic_loan_slot_size_bytes);
    json += ",\"generic_loan_supported\":";
    append_bool(json, record.generic_loan_supported);
    json += ",\"id\":";
    append_number(json, record.id.value());
    json += ",\"late_joiner_history_depth\":";
    append_number(json, record.late_joiner_history_depth);
    json += ",\"local_name\":";
    append_json_string(json, record.local_name);
    json += ",\"max_emits_per_compute\":";
    append_number(json, record.max_emits_per_compute);
    json += ",\"optional\":";
    append_bool(json, record.optional);
    json += ",\"owner\":";
    append_number(json, record.owner.value);
    json += ",\"owner_kind\":";
    append_number(json, static_cast<std::uint32_t>(record.owner_kind));
    json += ",\"payload_identity\":";
    append_payload_identity(json, record.payload_identity);
    json += ",\"queue_depth\":";
    append_number(json, record.queue_depth);
    json += ",\"serialized_payload_capacity_bytes\":";
    append_number(json, record.serialized_payload_capacity_bytes);
    json += ",\"tensor_layout\":";
    append_optional_tensor_layout(json, record.tensor_layout);
    json += ",\"tensor_contract\":";
    append_optional_tensor_contract(json, record.tensor_contract, include_provenance);
    json.push_back('}');
  });
  output += ",\"publication_outcomes\":";
  append_array(output, state.publication_outcomes, [](std::string& json, const auto& record) {
    json += "{\"output\":";
    append_number(json, record.output.value());
    json += ",\"post_submit_assigns_sequence\":";
    append_bool(json, record.post_submit_assigns_sequence);
    json += ",\"post_submit_consumes_authority\":";
    append_bool(json, record.post_submit_consumes_authority);
    json += ",\"post_submit_error\":";
    append_number(json, static_cast<std::uint32_t>(record.post_submit_error));
    json += ",\"post_submit_partial_error\":";
    append_number(json, static_cast<std::uint32_t>(record.post_submit_partial_error));
    json += ",\"pre_submit_assigns_sequence\":";
    append_bool(json, record.pre_submit_assigns_sequence);
    json += ",\"pre_submit_authority\":";
    append_number(json, static_cast<std::uint32_t>(record.pre_submit_authority));
    json += ",\"pre_submit_error\":";
    append_number(json, static_cast<std::uint32_t>(record.pre_submit_error));
    json += ",\"retain_claims_until_retirement\":";
    append_bool(json, record.retain_claims_until_retirement);
    json += ",\"runtime_retry_enabled\":";
    append_bool(json, record.runtime_retry_enabled);
    json += ",\"tensor\":";
    append_bool(json, record.tensor);
    json.push_back('}');
  });
  output += ",\"partition_decisions\":";
  append_array(output, state.partition_decisions, [](std::string& json, const auto& record) {
    json += "{\"decision\":";
    append_number(json, static_cast<std::uint32_t>(record.decision));
    json += ",\"partition\":";
    append_number(json, record.partition.value);
    json += ",\"reasons\":";
    append_array(json, record.reasons, [](std::string& values, auto reason) {
      append_number(values, static_cast<std::uint32_t>(reason));
    });
    json += ",\"vertex\":";
    append_number(json, record.vertex.value);
    json.push_back('}');
  });
  output += ",\"partitioning\":{";
  output += "\"accepted_algorithm\":";
  append_optional_algorithm(output, state.partitioning_plan.accepted_algorithm);
  output += ",\"authored_partition_count\":";
  append_number(output, state.partitioning_plan.authored_partition_count);
  output += ",\"automatic_partition_count\":";
  append_number(output, state.partitioning_plan.automatic_partition_count);
  output += ",\"config_digest\":";
  append_hex(output, state.partitioning_plan.config_digest.value);
  output += ",\"disposition\":";
  append_number(output, static_cast<std::uint32_t>(state.partitioning_plan.disposition));
  output += ",\"interface_version\":";
  append_number(output, state.partitioning_plan.interface_version);
  output += ",\"maturity\":";
  append_optional_maturity(output, state.partitioning_plan.maturity);
  output += ",\"mode\":";
  append_number(output, static_cast<std::uint32_t>(state.partitioning_plan.mode));
  output += ",\"reason\":";
  append_number(output, state.partitioning_plan.reason.value);
  output += ",\"requested_algorithm\":";
  append_optional_algorithm(output, state.partitioning_plan.requested_algorithm);
  output += ",\"selection_source\":";
  append_number(output, static_cast<std::uint32_t>(state.partitioning_plan.selection_source));
  output += "},\"partitions\":";
  append_array(output, state.execution_partitions, [](std::string& json, const auto& record) {
    json += "{\"authored_name\":";
    if (record.authored_name) {
      append_json_string(json, *record.authored_name);
    } else {
      json += "null";
    }
    json += ",\"graph_members\":";
    append_array(json, record.graph_members, [](std::string& values, GraphId graph) {
      append_number(values, graph.value);
    });
    json += ",\"id\":";
    append_number(json, record.id.value);
    json += ",\"lane\":";
    append_number(json, record.execution_lane);
    json += ",\"members\":";
    append_array(json, record.members, [](std::string& values, VertexId vertex) {
      append_number(values, vertex.value);
    });
    json += ",\"operator_paths\":";
    append_array(json, record.operator_paths, [](std::string& values, const auto& path) {
      append_json_string(values, path);
    });
    json.push_back('}');
  });
  output += ",\"predicate_inputs\":";
  append_array(output, state.predicate_inputs, [](std::string& json, const auto& record) {
    json += "{\"argument_ordinal\":";
    append_number(json, record.argument_ordinal);
    json += ",\"cache_capacity\":";
    append_number(json, record.cache_capacity);
    json += ",\"capture_time_required\":";
    append_bool(json, record.capture_time_required);
    json += ",\"conversion_id\":";
    append_number(json, record.conversion_id);
    json += ",\"implicit_driver\":";
    append_bool(json, record.implicit_driver);
    json += ",\"input\":";
    append_number(json, record.input.value());
    json += ",\"max_age_ns\":";
    append_number(json, record.max_age_ns);
    json += ",\"missing\":";
    append_number(json, static_cast<std::uint32_t>(record.missing));
    json += ",\"payload_identity\":";
    append_payload_identity(json, record.payload_identity);
    json += ",\"predicate_id\":";
    append_json_string(json, record.predicate_id);
    json += ",\"reference_clock\":";
    append_number(json, record.reference_clock.value);
    json += ",\"reuse_max_age_ns\":";
    append_number(json, record.reuse_max_age_ns);
    json += ",\"selection_reason_schema\":";
    append_json_string(json, record.selection_reason_schema);
    json += ",\"selection_state_schema\":";
    append_json_string(json, record.selection_state_schema);
    json += ",\"selector\":";
    append_number(json, static_cast<std::uint32_t>(record.selector));
    json.push_back('}');
  });
  output += ",\"predicate_callables\":";
  append_array(output, state.predicate_callables, [&](std::string& json, const auto& record) {
    json += "{\"alignment_bytes\":";
    append_number(json, record.alignment_bytes);
    json += ",\"bounded_call\":";
    append_bool(json, record.bounded_call);
    json += ",\"build_evidence\":";
    append_json_string(json, record.build_evidence);
    json += ",\"capability_evidence_references\":";
    append_array(
        json,
        record.capability_evidence_references,
        [](std::string& values, const std::string& value) { append_json_string(values, value); });
    if (include_provenance) {
      json += ",\"capture_provenance\":";
      append_json_string(json, record.capture_provenance);
    }
    json += ",\"criticality_constraints\":";
    append_array(
        json, record.criticality_constraints, [](std::string& values, const std::string& value) {
          append_json_string(values, value);
        });
    json += ",\"diagnostic_ids\":";
    append_array(json, record.diagnostic_ids, [](std::string& values, const std::string& value) {
      append_json_string(values, value);
    });
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"evidence_revocation_epoch\":";
    append_number(json, record.evidence_revocation_epoch);
    json += ",\"evidence_validity_epoch\":";
    append_number(json, record.evidence_validity_epoch);
    json += ",\"implementation_fingerprint\":";
    append_json_string(json, record.implementation_fingerprint);
    json += ",\"is_noexcept\":";
    append_bool(json, record.is_noexcept);
    json += ",\"kind\":";
    append_number(json, static_cast<std::uint32_t>(record.kind));
    json += ",\"argument_contract\":";
    append_json_string(json, record.argument_contract);
    if (include_provenance) {
      json += ",\"binding_provenance\":";
      append_json_string(json, record.binding_provenance);
    }
    json += ",\"dispatch_model\":";
    append_json_string(json, record.dispatch_model);
    json += ",\"no_dynamic_allocation\":";
    append_bool(json, record.no_dynamic_allocation);
    json += ",\"owned_lifetime\":";
    append_bool(json, record.owned_lifetime);
    json += ",\"possible_dynamic_allocation\":";
    append_bool(json, record.possible_dynamic_allocation);
    json += ",\"predicate_id\":";
    append_json_string(json, record.predicate_id);
    json += ",\"predicate_key\":";
    append_optional_string(json, record.predicate_key);
    json += ",\"predicate_key_version\":";
    if (record.predicate_key_version) {
      json += "{\"major\":";
      append_number(json, record.predicate_key_version->major);
      json += ",\"minor\":";
      append_number(json, record.predicate_key_version->minor);
      json += ",\"patch\":";
      append_number(json, record.predicate_key_version->patch);
      json.push_back('}');
    } else {
      json += "null";
    }
    json += ",\"receiver_role\":";
    append_json_string(json, record.receiver_role);
    json += ",\"requires_python_gil\":";
    append_bool(json, record.requires_python_gil);
    json += ",\"result_kind\":";
    append_number(json, static_cast<std::uint32_t>(record.result_kind));
    json += ",\"self_rebind\":";
    append_bool(json, record.self_rebind);
    json += ",\"size_bytes\":";
    append_number(json, record.size_bytes);
    json += ",\"toolchain_evidence\":";
    append_json_string(json, record.toolchain_evidence);
    json += ",\"type_erased\":";
    append_bool(json, record.type_erased);
    json.push_back('}');
  });
  output += ",\"predicate_triggers\":";
  append_array(output, state.predicate_triggers, [](std::string& json, const auto& record) {
    json += "{\"candidate_state_is_transactional\":";
    append_bool(json, record.candidate_state_is_transactional);
    json += ",\"clock\":";
    append_number(json, record.clock.value);
    json += ",\"compute_adopts_evaluator_snapshot\":";
    append_bool(json, record.compute_adopts_evaluator_snapshot);
    json += ",\"config_alignment_bytes\":";
    append_number(json, record.config_alignment_bytes);
    json += ",\"config_digest\":";
    append_optional_string(json, record.config_digest);
    json += ",\"config_schema\":";
    append_optional_string(json, record.config_schema);
    json += ",\"config_size_bytes\":";
    append_number(json, record.config_size_bytes);
    json += ",\"config_type_digest\":";
    append_optional_string(json, record.config_type_digest);
    json += ",\"consumes_reserved_candidate_tag\":";
    append_bool(json, record.consumes_reserved_candidate_tag);
    json += ",\"driver\":";
    append_number(json, record.driver.value());
    json += ",\"evaluation_witness_identity\":";
    append_json_string(json, record.evaluation_witness_identity);
    json += ",\"error_witness_identity\":";
    append_json_string(json, record.error_witness_identity);
    json += ",\"evaluator_error_policy\":";
    append_json_string(json, record.evaluator_error_policy);
    json += ",\"firing\":";
    append_number(json, static_cast<std::uint32_t>(record.firing));
    json += ",\"has_config\":";
    append_bool(json, record.has_config);
    json += ",\"graph_generation\":";
    append_number(json, record.graph_generation);
    json += ",\"id\":";
    append_json_string(json, record.id);
    json += ",\"identity\":";
    append_json_string(json, record.identity);
    json += ",\"identity_scope\":";
    append_number(json, static_cast<std::uint32_t>(record.identity_scope));
    json += ",\"initial_truth\":";
    append_bool(json, record.initial_truth);
    json += ",\"missing_or_error_preserves_truth\":";
    append_bool(json, record.missing_or_error_preserves_truth);
    json += ",\"missing_precedence\":";
    append_json_string(json, record.missing_precedence);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"period_ns\":";
    append_number(json, record.period_ns);
    json += ",\"phase\":";
    append_optional_time_point(json, record.phase);
    json += ",\"replay\":";
    append_number(json, static_cast<std::uint32_t>(record.replay));
    json += ",\"replay_evidence\":";
    append_json_string(json, record.replay_evidence);
    json += ",\"reset_policy\":";
    append_number(json, static_cast<std::uint32_t>(record.reset_policy));
    json += ",\"reset_reasons\":";
    append_array(json, record.reset_reasons, [](std::string& values, const std::string& value) {
      append_json_string(values, value);
    });
    json += ",\"reset_witness_identity\":";
    append_json_string(json, record.reset_witness_identity);
    json += ",\"semantic_version\":";
    if (record.semantic_version) {
      json += "{\"major\":";
      append_number(json, record.semantic_version->major);
      json += ",\"minor\":";
      append_number(json, record.semantic_version->minor);
      json += ",\"patch\":";
      append_number(json, record.semantic_version->patch);
      json.push_back('}');
    } else {
      json += "null";
    }
    json += ",\"source\":";
    append_number(json, static_cast<std::uint32_t>(record.source));
    json += ",\"state_alignment_bytes\":";
    append_number(json, record.state_alignment_bytes);
    json += ",\"state_generation\":";
    append_number(json, record.state_generation);
    json += ",\"state_initialization\":";
    append_json_string(json, record.state_initialization);
    json += ",\"state_schema\":";
    append_json_string(json, record.state_schema);
    json += ",\"state_size_bytes\":";
    append_number(json, record.state_size_bytes);
    json += ",\"state_type_digest\":";
    append_json_string(json, record.state_type_digest);
    json += ",\"safe_default_witness_identity\":";
    append_json_string(json, record.safe_default_witness_identity);
    json += ",\"suppression_witness_identity\":";
    append_json_string(json, record.suppression_witness_identity);
    json += ",\"successful_evaluation_commits_truth\":";
    append_bool(json, record.successful_evaluation_commits_truth);
    json += ",\"transition_witness_identity\":";
    append_json_string(json, record.transition_witness_identity);
    json += ",\"reaction_fire_witness_identity\":";
    append_json_string(json, record.reaction_fire_witness_identity);
    json.push_back('}');
  });
  output += ",\"process_roles\":";
  append_array(output, state.process_roles, [](std::string& json, const auto& record) {
    json += "{\"accepted_isolation\":";
    if (record.accepted_isolation) {
      append_number(json, static_cast<std::uint32_t>(*record.accepted_isolation));
    } else {
      json += "null";
    }
    json += ",\"accepted_launch_group\":";
    append_json_string(json, record.accepted_launch_group);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"id\":";
    append_number(json, record.id.value);
    json += ",\"members\":";
    append_array(json, record.members, [](std::string& values, VertexId vertex) {
      append_number(values, vertex.value);
    });
    json += ",\"reason\":";
    append_number(json, record.reason.value);
    json += ",\"require_locality_probe_success\":";
    append_bool(json, record.require_locality_probe_success);
    json += ",\"requested_isolation\":";
    append_number(json, static_cast<std::uint32_t>(record.requested_isolation));
    json += ",\"requested_launch_group\":";
    append_json_string(json, record.requested_launch_group);
    json += ",\"role\":";
    append_json_string(json, record.role);
    json += ",\"runtime_seal_required\":";
    append_bool(json, record.runtime_seal_required);
    json.push_back('}');
  });
  output += ",\"process_selectors\":";
  append_array(output, state.process_selectors, [](std::string& json, const auto& record) {
    json += "{\"authored_value\":";
    append_json_string(json, record.authored_value);
    json += ",\"canonical_value\":";
    append_json_string(json, record.canonical_value);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"external_topic\":";
    if (record.external_topic) {
      append_number(json, record.external_topic->value);
    } else {
      json += "null";
    }
    json += ",\"graph\":";
    if (record.graph) {
      append_number(json, record.graph->value);
    } else {
      json += "null";
    }
    json += ",\"kind\":";
    append_number(json, static_cast<std::uint32_t>(record.kind));
    json += ",\"members\":";
    append_array(json, record.members, [](std::string& values, VertexId vertex) {
      append_number(values, vertex.value);
    });
    json += ",\"ordinal\":";
    append_number(json, record.ordinal);
    json += ",\"process_role\":";
    append_number(json, record.process_role.value);
    json += ",\"reason\":";
    append_number(json, record.reason.value);
    json += ",\"region\":";
    if (record.region) {
      append_number(json, record.region->value);
    } else {
      json += "null";
    }
    json += ",\"vertex\":";
    if (record.vertex) {
      append_number(json, record.vertex->value);
    } else {
      json += "null";
    }
    json.push_back('}');
  });
  output += ",\"vertex_process_placements\":";
  append_array(output, state.vertex_process_placements, [](std::string& json, const auto& record) {
    json += "{\"isolation\":";
    append_number(json, static_cast<std::uint32_t>(record.isolation));
    json += ",\"launch_group\":";
    append_json_string(json, record.launch_group);
    json += ",\"process_group\":";
    append_number(json, record.process_group.value);
    json += ",\"process_role\":";
    if (record.process_role) {
      append_number(json, record.process_role->value);
    } else {
      json += "null";
    }
    json += ",\"runtime_seal_required\":";
    append_bool(json, record.runtime_seal_required);
    json += ",\"vertex\":";
    append_number(json, record.vertex.value);
    json.push_back('}');
  });
}

void append_profile_and_schedule_sections(std::string& output, const ExecutionPlanState& state,
                                          bool include_provenance) {
  output += ",\"profile\":{";
  output += "\"disposition\":";
  append_number(output, static_cast<std::uint32_t>(state.profile_manifest.disposition));
  output += ",\"derived_capacities\":";
  append_array(
      output, state.profile_manifest.derived_capacities, [](std::string& json, const auto& record) {
        json += "{\"capacity_id\":";
        append_json_string(json, record.capacity_id);
        json += ",\"limit\":";
        append_number(json, record.limit);
        json += ",\"object_id\":";
        append_json_string(json, record.object_id);
        json += ",\"unit\":";
        append_json_string(json, record.unit);
        json.push_back('}');
      });
  output += ",\"effective\":";
  append_optional_profile(output, state.profile_manifest.effective_graph_profile);
  output += ",\"evidence_references\":";
  append_array(output,
               state.profile_manifest.evidence_references,
               [](std::string& json, const auto& record) {
                 json += "{\"id\":";
                 append_json_string(json, record.id);
                 json += ",\"kind\":";
                 append_json_string(json, record.kind);
                 json += ",\"version\":";
                 append_number(json, record.version);
                 json.push_back('}');
               });
  output += ",\"explicit_fallbacks\":";
  append_array(
      output, state.profile_manifest.explicit_fallbacks, [](std::string& json, const auto& record) {
        json += "{\"evidence_id\":";
        append_json_string(json, record.evidence_id);
        json += ",\"fallback_id\":";
        append_json_string(json, record.fallback_id);
        json += ",\"feature_id\":";
        append_json_string(json, record.feature_id);
        json += ",\"reason_id\":";
        append_json_string(json, record.reason_id);
        json.push_back('}');
      });
  output += ",\"feature_dispositions\":";
  append_array(output,
               state.profile_manifest.feature_dispositions,
               [](std::string& json, const auto& record) {
                 json += "{\"disposition\":";
                 append_number(json, static_cast<std::uint32_t>(record.disposition));
                 json += ",\"evidence_id\":";
                 append_json_string(json, record.evidence_id);
                 json += ",\"feature_id\":";
                 append_json_string(json, record.feature_id);
                 json += ",\"rule_id\":";
                 append_json_string(json, record.rule_id);
                 json.push_back('}');
               });
  output += ",\"generated_endpoint_names\":";
  append_array(output,
               state.profile_manifest.generated_endpoint_names,
               [](std::string& json, const auto& record) {
                 json += "{\"algorithm\":";
                 append_json_string(json, record.algorithm);
                 json += ",\"disposition\":";
                 append_number(json, static_cast<std::uint32_t>(record.disposition));
                 json += ",\"evidence_id\":";
                 append_json_string(json, record.evidence_id);
                 json += ",\"final_name\":";
                 append_json_string(json, record.final_name);
                 json += ",\"source\":";
                 append_number(json, static_cast<std::uint32_t>(record.source));
                 json += ",\"version\":";
                 append_number(json, record.version);
                 json.push_back('}');
               });
  output += ",\"materialization_obligations\":";
  append_array(output,
               state.profile_manifest.materialization_obligations,
               [](std::string& json, const auto& record) {
                 json += "{\"cleanup_stage_mask\":";
                 append_number(json, record.cleanup_stage_mask);
                 json += ",\"complete_before_stage\":";
                 append_number(json, static_cast<std::uint32_t>(record.complete_before_stage));
                 json += ",\"evidence_id\":";
                 append_json_string(json, record.evidence_id);
                 json += ",\"max_cleanup_actions\":";
                 append_number(json, record.max_cleanup_actions);
                 json += ",\"object_id\":";
                 append_json_string(json, record.object_id);
                 json += ",\"strategy_id\":";
                 append_json_string(json, record.strategy_id);
                 json.push_back('}');
               });
  output += ",\"policy_digest\":";
  append_sha256(output, state.profile_manifest.profile_policy_id.digest.value);
  output += ",\"policy_major\":";
  append_number(output, state.profile_manifest.profile_policy_id.major);
  output += ",\"policy_minor\":";
  append_number(output, state.profile_manifest.profile_policy_id.minor);
  output += ",\"plan_schema_version\":";
  append_json_string(output, state.profile_manifest.plan_schema_version);
  append_profile_provenance(output, state.profile_manifest, include_provenance);
  output += ",\"regions\":";
  append_array(
      output, state.profile_manifest.effective_regions, [](std::string& json, const auto& record) {
        json += "{\"effective\":";
        if (record.effective_profile) {
          append_json_string(json, profile_token(*record.effective_profile));
        } else {
          json += "null";
        }
        json += ",\"id\":";
        append_number(json, record.region_id.value);
        json += ",\"origin\":";
        append_number(json, static_cast<std::uint32_t>(record.origin));
        json += ",\"requested\":";
        if (record.requested_profile) {
          append_json_string(json, profile_token(*record.requested_profile));
        } else {
          json += "null";
        }
        json.push_back('}');
      });
  output += ",\"requested\":";
  append_optional_profile(output, state.profile_manifest.requested_profile);
  output += ",\"required_capabilities\":";
  append_array(output,
               state.profile_manifest.required_capabilities,
               [](std::string& json, const auto& record) {
                 json += "{\"capability_id\":";
                 append_json_string(json, record.capability_id);
                 json += ",\"evidence_id\":";
                 append_json_string(json, record.evidence_id);
                 json += ",\"version\":";
                 append_number(json, record.version);
                 json.push_back('}');
               });
  output += ",\"runtime_abi_version\":";
  append_json_string(output, state.profile_manifest.runtime_abi_version);
  output += ",\"scope_resolutions\":";
  append_array(
      output, state.profile_manifest.scope_resolutions, [](std::string& json, const auto& record) {
        json += "{\"contributing_regions\":";
        append_array(json, record.contributing_regions, [](std::string& values, RegionId region) {
          append_number(values, region.value);
        });
        json += ",\"effective_profile\":";
        append_json_string(json, profile_token(record.effective_profile));
        json += ",\"object_id\":";
        append_json_string(json, record.object_id);
        json += ",\"object_kind\":";
        append_json_string(json, record.object_kind);
        json.push_back('}');
      });
  output += ",\"target_triple\":";
  append_json_string(output, state.profile_manifest.target_triple);
  output += "},\"readable_triggers\":";
  append_array(output, state.readable_triggers, [](std::string& json, const auto& record) {
    json += "{\"automatic_retry\":";
    append_bool(json, record.automatic_retry);
    json += ",\"bind_at_arm\":";
    append_bool(json, record.bind_at_arm);
    json += ",\"disable_before_stop\":";
    append_bool(json, record.disable_before_stop);
    json += ",\"effective_capacity\":";
    append_number(json, record.effective_capacity);
    json += ",\"enable_after_global_start\":";
    append_bool(json, record.enable_after_global_start);
    json += ",\"error_creates_candidate\":";
    append_bool(json, record.error_creates_candidate);
    json += ",\"fatal_error\":";
    append_number(json, static_cast<std::uint32_t>(record.fatal_error));
    json += ",\"initial_source_generation\":";
    append_number(json, record.initial_source_generation);
    json += ",\"initial_source_sequence\":";
    append_number(json, record.initial_source_sequence);
    json += ",\"level_readable\":";
    append_bool(json, record.level_readable);
    json += ",\"max_outstanding_candidates\":";
    append_number(json, record.max_outstanding_candidates);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"readiness_witness_identity\":";
    append_json_string(json, record.readiness_witness_identity);
    json += ",\"rearm_only_if_still_readable\":";
    append_bool(json, record.rearm_only_if_still_readable);
    json += ",\"restart_advances_generation\":";
    append_bool(json, record.restart_advances_generation);
    json += ",\"source\":";
    append_number(json, source_id_value(record.source));
    json += ",\"source_error_identity\":";
    append_json_string(json, record.source_error_identity);
    json += ",\"stale_generation_identity\":";
    append_json_string(json, record.stale_generation_identity);
    json += ",\"stale_generation_rejected\":";
    append_bool(json, record.stale_generation_rejected);
    json += ",\"unbind_at_release\":";
    append_bool(json, record.unbind_at_release);
    json += ",\"witness_clock\":";
    append_number(json, record.witness_clock.value);
    json.push_back('}');
  });
  output += ",\"readiness\":";
  append_array(output, state.readiness, [](std::string& json, const ReadinessPlanRecord& record) {
    json += "{\"capacity\":";
    append_number(json, record.accepted_capacity);
    json += ",\"declaration_ordinal\":";
    append_number(json, record.declaration_ordinal);
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"kind\":";
    append_number(json, static_cast<std::uint32_t>(record.kind));
    json += ",\"materialized\":";
    append_bool(json, record.materialized);
    json += ",\"name\":";
    append_json_string(json, record.name);
    json += ",\"owner\":";
    append_number(json, record.owner.value);
    json += ",\"owner_path\":";
    append_json_string(json, record.owner_path);
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"referenced\":";
    append_bool(json, record.referenced);
    json += ",\"requested_capacity\":";
    append_number(json, record.requested_capacity);
    json += ",\"requested_clock\":{\"default\":";
    append_bool(json, record.requested_clock.is_default());
    json += ",\"name\":";
    append_json_string(json, record.requested_clock.name());
    json += "},\"requested_sender_capacity\":";
    append_number(json, record.requested_sender_reference_capacity);
    json += ",\"resolved_clock\":";
    append_number(json, record.resolved_clock.value);
    json += ",\"sender_capacity\":";
    append_number(json, record.accepted_sender_reference_capacity);
    json += ",\"source_id\":";
    append_number(json, source_id_value(record.source_id));
    json += ",\"initial_source_generation\":";
    append_number(json, record.initial_source_generation);
    json += ",\"initial_source_sequence\":";
    append_number(json, record.initial_source_sequence);
    json += ",\"disposition_reason\":";
    append_number(json, record.disposition_reason.value);
    json += ",\"wake_requirement\":";
    append_number(json, static_cast<std::uint32_t>(record.wake_requirement));
    json.push_back('}');
  });
  output += ",\"schedule_edges\":";
  append_array(output, state.schedule_edges, [](std::string& json, const auto& record) {
    json += "{\"destination\":";
    append_number(json, record.destination.value);
    json += ",\"edge\":";
    append_number(json, record.edge.value);
    json += ",\"source\":";
    append_number(json, record.source.value);
    json.push_back('}');
  });
  output += ",\"scheduler\":{";
  output += "\"activation_overlap\":";
  append_bool(output, state.scheduler_plan.activation_overlap);
  output += ",\"max_concurrent_activations\":";
  append_number(output, state.scheduler_plan.max_concurrent_activations);
  output += ",\"execution_lane_count\":";
  append_number(output, state.scheduler_plan.execution_lane_count);
  output += ",\"intra_path_parallelism\":";
  append_number(output, state.scheduler_plan.intra_path_parallelism);
  output += ",\"ready_policy\":";
  append_json_string(output, state.scheduler_plan.ready_policy.id);
  output += ",\"ready_policy_version\":";
  append_number(output, state.scheduler_plan.ready_policy.version);
  output += ",\"reason\":";
  append_number(output, state.scheduler_plan.reason.value);
  output += ",\"worker_count\":";
  append_number(output, state.scheduler_plan.worker_count);
  output.push_back('}');
  output += ",\"temporal_triggers\":";
  append_array(output, state.temporal_triggers, [](std::string& json, const auto& record) {
    json += "{\"bounded_state_capacity\":";
    append_number(json, record.bounded_state_capacity);
    json += ",\"conversion_id\":";
    append_number(json, record.conversion_id);
    json += ",\"deadline_ns\":";
    if (record.deadline_ns) {
      append_number(json, *record.deadline_ns);
    } else {
      json += "null";
    }
    json += ",\"declared_inputs\":";
    append_array(json, record.declared_inputs, [](std::string& values, PortId input) {
      append_number(values, input.value());
    });
    json += ",\"kind\":";
    append_number(json, static_cast<std::uint32_t>(record.kind));
    json += ",\"origin\":";
    append_number(json, static_cast<std::uint32_t>(record.origin));
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"period_ns\":";
    append_number(json, record.period_ns);
    json += ",\"phase\":";
    append_optional_time_point(json, record.phase);
    json += ",\"presence_guaranteed_inputs\":";
    append_array(json, record.presence_guaranteed_inputs, [](std::string& values, PortId input) {
      append_number(values, input.value());
    });
    json += ",\"readiness_source\":";
    append_number(json, source_id_value(record.readiness_source));
    json += ",\"reference_clock\":";
    append_number(json, record.reference_clock.value);
    json += ",\"replay_identity\":";
    append_json_string(json, record.replay_identity);
    json += ",\"trigger_clock\":";
    append_number(json, record.trigger_clock.value);
    json += ",\"trigger_input\":";
    append_number(json, record.trigger_input.value());
    json += ",\"violation\":";
    if (record.violation) {
      append_number(json, static_cast<std::uint32_t>(*record.violation));
    } else {
      json += "null";
    }
    json += ",\"witness_identity\":";
    append_json_string(json, record.witness_identity);
    json.push_back('}');
  });
}

void append_placement_lane_and_window_sections(std::string& output,
                                               const ExecutionPlanState& state) {
  output += ",\"partition_placements\":";
  append_array(output, state.partition_placements, [](std::string& json, const auto& record) {
    json += "{\"accepted\":";
    if (record.accepted) {
      append_thread_placement(json, *record.accepted);
    } else {
      json += "null";
    }
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"execution_lane\":";
    append_number(json, record.execution_lane);
    json += ",\"authored_name\":";
    append_json_string(json, record.authored_name);
    json += ",\"partition\":";
    append_number(json, record.partition.value);
    json += ",\"reason\":";
    append_number(json, record.reason.value);
    json += ",\"requested\":";
    append_thread_placement(json, record.requested);
    json.push_back('}');
  });
  output += ",\"placement_probes\":";
  append_array(output, state.placement_probes, [](std::string& json, const auto& record) {
    json += "{\"evidence_id\":";
    append_json_string(json, record.evidence_id);
    json += ",\"probe_id\":";
    append_json_string(json, record.probe_id);
    json += ",\"reason\":";
    append_number(json, record.reason.value);
    json += ",\"partition\":";
    append_number(json, record.partition.value);
    json += ",\"status\":";
    append_number(json, static_cast<std::uint32_t>(record.status));
    json.push_back('}');
  });
  output += ",\"execution_lanes\":";
  append_array(output, state.execution_lanes, [](std::string& json, const auto& record) {
    json += "{\"accepted_placement\":";
    append_thread_placement(json, record.accepted_placement);
    json += ",\"execution_lane\":";
    append_number(json, record.execution_lane);
    json += ",\"partition\":";
    append_number(json, record.partition.value);
    json.push_back('}');
  });
  output += ",\"window_triggers\":";
  append_array(output, state.window_triggers, [](std::string& json, const auto& record) {
    json += "{\"advance_on_every_anchor\":";
    append_bool(json, record.advance_on_every_anchor);
    json += ",\"anchor\":";
    append_number(json, record.anchor.value());
    json += ",\"bracketed_inputs\":";
    append_array(json, record.bracketed_inputs, [](std::string& values, PortId input) {
      append_number(values, input.value());
    });
    json += ",\"close_witness_identity\":";
    append_json_string(json, record.close_witness_identity);
    json += ",\"cancelled_witness_identity\":";
    append_json_string(json, record.cancelled_witness_identity);
    json += ",\"capability_evidence_id\":";
    append_json_string(json, record.capability_evidence_id);
    json += ",\"external_driver_used\":";
    append_bool(json, record.external_driver_used);
    json += ",\"first_anchor_prime_only\":";
    append_bool(json, record.first_anchor_prime_only);
    json += ",\"end_of_stream_witness_identity\":";
    append_json_string(json, record.end_of_stream_witness_identity);
    json += ",\"max_pending_windows\":";
    append_number(json, record.max_pending_windows);
    json += ",\"max_wait_ns\":";
    append_number(json, record.max_wait_ns);
    json += ",\"owner\":";
    append_operator_ref(json, record.owner);
    json += ",\"open_witness_identity\":";
    append_json_string(json, record.open_witness_identity);
    json += ",\"pending_limit_witness_identity\":";
    append_json_string(json, record.pending_limit_witness_identity);
    json += ",\"presence_guaranteed_inputs\":";
    append_array(json, record.presence_guaranteed_inputs, [](std::string& values, PortId input) {
      append_number(values, input.value());
    });
    json += ",\"primed_witness_identity\":";
    append_json_string(json, record.primed_witness_identity);
    json += ",\"producer_epoch_identity_schema\":";
    append_json_string(json, record.producer_epoch_identity_schema);
    json += ",\"reset_on_producer_epoch\":";
    append_bool(json, record.reset_on_producer_epoch);
    json += ",\"reset_on_timestamp_regression\":";
    append_bool(json, record.reset_on_timestamp_regression);
    json += ",\"timeout_witness_identity\":";
    append_json_string(json, record.timeout_witness_identity);
    json += ",\"timestamp_regression_witness_identity\":";
    append_json_string(json, record.timestamp_regression_witness_identity);
    json += ",\"wait_clock\":";
    append_number(json, record.wait_clock.value);
    json.push_back('}');
  });
}

}  // namespace

namespace plan_artifact_serialization_internal {

void append_optional_profile(std::string& output, std::optional<Profile> profile) {
  append_optional_profile_value(output, profile);
}

void append_optional_string(std::string& output, const std::optional<std::string>& value) {
  append_optional_string_value(output, value);
}

void append_optional_tensor_layout(std::string& output,
                                   const std::optional<TensorRepresentation>& layout) {
  append_optional_tensor_layout_value(output, layout);
}

void append_tensor_axes(std::string& output, const std::span<const TensorAxisBounds> axes) {
  append_tensor_axes_value(output, axes);
}

void append_logical_tensor_requirements(std::string& output,
                                        const TensorBoundaryContractRecord& logical,
                                        const bool include_provenance) {
  append_logical_tensor_requirements_value(output, logical, include_provenance);
}

}  // namespace plan_artifact_serialization_internal

// Canonical serialization emits the complete public semantic plan schema in one order. Private
// Runtime construction facts are validated above but remain derived, process-local state.
expected<std::string, Error> plan_artifact_json(const ExecutionPlanState& state,
                                                bool include_provenance) {
  // REQ: COMPILE-002 - canonical semantics, not diagnostic build provenance,
  // form the digest preimage. The complete artifact retains both.
  // REQ: COMPILE-009 - serialization reads immutable records and performs no runtime/OS mutation.
  if (!validate_tensor_field_allocation_order(state.tensor_field_allocations)) {
    return make_unexpected(artifact_error(ErrorCode::kInvalidArgument,
                                          "PHYSICAL_TENSOR_FIELD_ALLOCATION_ORDER_INVALID"));
  }
  if (!valid_tensor_byte_span_projection_evidence(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "TENSOR_BYTE_SPAN_PROJECTION_PLAN_INVALID"));
  }
  if (!valid_tensor_memory_kind_derivations(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "TENSOR_MEMORY_KIND_DERIVATION_PLAN_INVALID"));
  }
  if (!plan_artifact_rate_serialization_internal::rate_plan_records_valid(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "RATE_PLAN_RECORDS_INVALID"));
  }
  if (!plan_artifact_rate_serialization_internal::pool_capacity_plan_records_valid(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "POOL_CAPACITY_PLAN_INVALID"));
  }
  if (!valid_partition_inspection_plan(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "PARTITION_INSPECTION_PLAN_INVALID"));
  }
  if (!valid_operator_completion_plan(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "OPERATOR_COMPLETION_PLAN_INVALID"));
  }
  const bool rejected_tensor_layout = has_rejected_tensor_layout(state);
  if (!valid_process_local_channel_plan(state) && !rejected_tensor_layout) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID"));
  }
  if (!valid_closed_island_activation_plan(state) && !rejected_tensor_layout) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "CLOSED_ISLAND_ACTIVATION_PLAN_INVALID"));
  }
  if (!valid_inline_partition_execution_plan(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "INLINE_PARTITION_EXECUTION_PLAN_INVALID"));
  }
  if (!valid_optional_input_plan(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "TEMPORAL_INPUT_OPTIONAL_INVALID"));
  }
  if (!valid_partition_runner_plan(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "PARTITION_RESOURCE_REALIZATION_FAILED"));
  }
  if (!valid_cuda_graph_capture_plans(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "CUDA_GRAPH_CAPTURE_PLAN_INVALID"));
  }
  if (auto cuda_plan = validate_cuda_execution_plan_records(state); !cuda_plan) {
    return make_unexpected(std::move(cuda_plan).error());
  }
  if (state.admitted && !valid_external_channel_plan(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "EXTERNAL_CHANNEL_PLAN_INVALID"));
  }
  if (state.admitted && !valid_cross_process_channel_plan(state)) {
    return make_unexpected(
        artifact_error(ErrorCode::kInvalidArgument, "CROSS_PROCESS_CHANNEL_PLAN_INVALID"));
  }
  std::string output{"{\"admitted\":"};
  append_bool(output, state.admitted);
  // REQ: COMPILE-015, COMPILE-016, COMPILE-017 - temporal record families are
  // separate canonical arrays whose pointer-free joins contribute to identity.
  append_temporal_channel_and_diagnostic_sections(output, state, include_provenance);
  append_graph_and_lifecycle_sections(output, state);
  append_topology_and_predicate_sections(output, state, include_provenance);
  append_profile_and_schedule_sections(output, state, include_provenance);
  plan_artifact_serialization_internal::append_tensor_and_cuda_sections(
      output, state, include_provenance);
  append_placement_lane_and_window_sections(output, state);
  output += ",\"schema\":";
  append_json_string(output, kPlanSchema);
  output.push_back('}');

  return output;
}

expected<void, Error> finalize_plan_artifact(ExecutionPlanState& state) {
  auto semantic = plan_artifact_json(state, false);
  if (!semantic) {
    return make_unexpected(std::move(semantic).error());
  }
  state.digest = digest_bytes<PlanDigest>(semantic.value());

  auto complete = plan_artifact_json(state, true);
  if (!complete) {
    return make_unexpected(std::move(complete).error());
  }
  state.json = std::move(complete).value();
  return {};
}

}  // namespace holoscan::detail
