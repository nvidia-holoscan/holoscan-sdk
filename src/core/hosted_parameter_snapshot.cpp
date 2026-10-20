/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/hosted_parameter_snapshot.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>

namespace holoscan {
namespace detail {

struct HostedParameterSnapshotState {
  std::string stable_id;
  std::string description;
  std::vector<HostedParameterField> values;
  std::vector<ParamFieldDescriptor> fields;
  ParamSchemaDescriptor schema;
};

namespace {

[[noreturn]] void reject(std::string_view token, std::string_view field = {}) {
  std::string message{token};
  if (!field.empty()) {
    message.append(" field_name=");
    message.append(field);
  }
  throw RuntimeError{Error{ErrorCode::kInvalidArgument, std::move(message)}};
}

[[nodiscard]] bool supported_shape(const ParamFieldTypeShape& shape) noexcept {
  if (shape.optional || !shape.has_stable_shape) {
    return false;
  }
  switch (shape.type) {
    case ParamLogicalType::kBool:
      return shape.bit_width == 1U;
    case ParamLogicalType::kSignedInteger:
    case ParamLogicalType::kUnsignedInteger:
      return shape.bit_width == 8U || shape.bit_width == 16U || shape.bit_width == 32U ||
             shape.bit_width == 64U;
    case ParamLogicalType::kFloatingPoint:
      return shape.bit_width == 32U || shape.bit_width == 64U;
    case ParamLogicalType::kSequence:
      return shape.element_or_rep_type == ParamLogicalType::kSignedInteger ||
             shape.element_or_rep_type == ParamLogicalType::kUnsignedInteger;
    case ParamLogicalType::kString:
    case ParamLogicalType::kFilesystemPath:
    case ParamLogicalType::kDuration:
    case ParamLogicalType::kObject:
      return false;
  }
  return false;
}

[[nodiscard]] bool scalar_matches(ParamScalarValue value, ParamLogicalType type,
                                  std::uint16_t width) noexcept {
  switch (type) {
    case ParamLogicalType::kBool:
      return value.kind == ParamScalarKind::kBool && value.bit_width == 1U && value.raw_bits <= 1U;
    case ParamLogicalType::kSignedInteger:
      return value.kind == ParamScalarKind::kSignedInteger && value.bit_width == width;
    case ParamLogicalType::kUnsignedInteger:
      return value.kind == ParamScalarKind::kUnsignedInteger && value.bit_width == width;
    case ParamLogicalType::kFloatingPoint:
      return value.kind == ParamScalarKind::kFloatingPoint && value.bit_width == width;
    case ParamLogicalType::kString:
    case ParamLogicalType::kSequence:
    case ParamLogicalType::kFilesystemPath:
    case ParamLogicalType::kDuration:
    case ParamLogicalType::kObject:
      return false;
  }
  return false;
}

[[nodiscard]] ParamConstraintScalar constraint_value(ParamScalarValue value) noexcept {
  switch (value.kind) {
    case ParamScalarKind::kBool:
    case ParamScalarKind::kUnsignedInteger:
      return ParamConstraintScalar{.kind = ParamConstraintScalarKind::kUnsigned,
                                   .unsigned_value = value.raw_bits};
    case ParamScalarKind::kSignedInteger: {
      std::uint64_t bits = value.raw_bits;
      if (value.bit_width < 64U && value.bit_width > 0U &&
          (bits & (std::uint64_t{1} << (value.bit_width - 1U))) != 0U) {
        bits |= ~((std::uint64_t{1} << value.bit_width) - 1U);
      }
      return ParamConstraintScalar{.kind = ParamConstraintScalarKind::kSigned,
                                   .signed_value = std::bit_cast<std::int64_t>(bits)};
    }
    case ParamScalarKind::kFloatingPoint: {
      const double decoded = value.bit_width == 32U
                                 ? static_cast<double>(std::bit_cast<float>(
                                       static_cast<std::uint32_t>(value.raw_bits)))
                                 : std::bit_cast<double>(value.raw_bits);
      return ParamConstraintScalar{.kind = ParamConstraintScalarKind::kFloating,
                                   .floating_value = decoded};
    }
    case ParamScalarKind::kDurationNanoseconds:
      return ParamConstraintScalar{.kind = ParamConstraintScalarKind::kSigned,
                                   .signed_value = std::bit_cast<std::int64_t>(value.raw_bits)};
    case ParamScalarKind::kNone:
      return {};
  }
  return {};
}

[[nodiscard]] bool constraint_less(ParamConstraintScalar left,
                                   ParamConstraintScalar right) noexcept {
  if (left.kind != right.kind) {
    return false;
  }
  switch (left.kind) {
    case ParamConstraintScalarKind::kSigned:
      return left.signed_value < right.signed_value;
    case ParamConstraintScalarKind::kUnsigned:
      return left.unsigned_value < right.unsigned_value;
    case ParamConstraintScalarKind::kFloating:
      return left.floating_value < right.floating_value;
    case ParamConstraintScalarKind::kNone:
      return false;
  }
  return false;
}

[[nodiscard]] bool constraint_valid(ParamConstraintScalar value,
                                    ParamConstraintScalarKind expected) noexcept {
  return value.kind == expected &&
         (expected != ParamConstraintScalarKind::kFloating || std::isfinite(value.floating_value));
}

[[nodiscard]] ParamValidationResult validate_one(const HostedParameterField& value,
                                                 const ParamFieldDescriptor& field) noexcept {
  if (field.logical_type == ParamLogicalType::kSequence &&
      value.values.size() > field.constraints.max_elements) {
    return {ParamValidationCode::kCardinalityExceeded, field.id, field.canonical_name, {}};
  }
  for (const ParamScalarValue scalar : value.values) {
    const ParamConstraintScalar converted = constraint_value(scalar);
    if (converted.kind == ParamConstraintScalarKind::kNone ||
        (converted.kind == ParamConstraintScalarKind::kFloating &&
         !std::isfinite(converted.floating_value))) {
      return {ParamValidationCode::kOutOfRange, field.id, field.canonical_name, {}};
    }
    if (field.constraints.has_range && (constraint_less(converted, field.constraints.range_min) ||
                                        constraint_less(field.constraints.range_max, converted))) {
      return {ParamValidationCode::kOutOfRange, field.id, field.canonical_name, {}};
    }
    if (field.constraints.allowed_value_count != 0U) {
      bool found = false;
      for (std::size_t index = 0U; index < field.constraints.allowed_value_count; ++index) {
        found = found || scalar_equal(converted, field.constraints.allowed_values[index]);
      }
      if (!found) {
        return {ParamValidationCode::kNotAllowed, field.id, field.canonical_name, {}};
      }
    }
  }
  return {};
}

[[nodiscard]] const HostedParameterField* hosted_field(const void* aggregate,
                                                       std::size_t index) noexcept {
  if (aggregate == nullptr) {
    return nullptr;
  }
  const auto& snapshot = *static_cast<const HostedParameterSnapshot*>(aggregate);
  return snapshot.state && index < snapshot.state->values.size() ? &snapshot.state->values[index]
                                                                 : nullptr;
}

bool read_hosted_scalar(const void* aggregate, std::size_t index,
                        ParamScalarValue& value) noexcept {
  const HostedParameterField* field = hosted_field(aggregate, index);
  if (field == nullptr || field->type_shape.type == ParamLogicalType::kSequence ||
      field->values.size() != 1U) {
    return false;
  }
  value = field->values.front();
  return true;
}

std::span<const ParamScalarValue> read_hosted_sequence(const void* aggregate,
                                                       std::size_t index) noexcept {
  const HostedParameterField* field = hosted_field(aggregate, index);
  if (field == nullptr || field->type_shape.type != ParamLogicalType::kSequence) {
    return {};
  }
  return field->values;
}

ParamValidationResult validate_hosted_field(const void* aggregate,
                                            const ParamFieldDescriptor& field) noexcept {
  const HostedParameterField* value =
      hosted_field(aggregate, field.local_operations.hosted_field_index);
  return value == nullptr ? ParamValidationResult{ParamValidationCode::kOutOfRange,
                                                  field.id,
                                                  field.canonical_name,
                                                  {}}
                          : validate_one(*value, field);
}

ParamFieldId compact_field_id(ParamSchemaHash full_id) noexcept {
  ParamFieldId result{};
  for (std::size_t index = 0U; index < sizeof(result); ++index) {
    result = (result << 8U) | std::to_integer<std::uint8_t>(full_id.value[index]);
  }
  return result;
}

}  // namespace
}  // namespace detail

HostedParameterSnapshot::operator bool() const noexcept {
  return schema() != nullptr;
}

const ParamTypeDescriptor& HostedParameterSnapshot::type() const noexcept {
  return param_type_descriptor<HostedParameterSnapshot>();
}

const ParamSchemaDescriptor* HostedParameterSnapshot::schema() const noexcept {
  return state ? &state->schema : nullptr;
}

bool HostedParameterSnapshot::read_scalar(const ParamFieldDescriptor& field,
                                          ParamScalarValue& value) const noexcept {
  const std::size_t index = field.local_operations.hosted_field_index;
  return state && index < state->values.size() &&
         field.local_member_token == &state->values[index] &&
         field.local_operations.read_hosted_member_as_scalar != nullptr &&
         field.local_operations.read_hosted_member_as_scalar(this, index, value);
}

std::span<const ParamScalarValue> HostedParameterSnapshot::read_scalar_sequence(
    const ParamFieldDescriptor& field) const noexcept {
  const std::size_t index = field.local_operations.hosted_field_index;
  if (!state || index >= state->values.size() ||
      field.local_member_token != &state->values[index] ||
      field.local_operations.read_hosted_member_as_scalar_sequence == nullptr) {
    return {};
  }
  return field.local_operations.read_hosted_member_as_scalar_sequence(this, index);
}

HostedParameterSnapshotBuilder::HostedParameterSnapshotBuilder(std::string stable_id,
                                                               std::string description)
    : stable_id_(std::move(stable_id)), description_(std::move(description)) {}

HostedParameterSnapshotBuilder& HostedParameterSnapshotBuilder::field(HostedParameterField value) {
  fields_.push_back(std::move(value));
  return *this;
}

// Construction intentionally keeps field validation, descriptor publication, and canonical hashes
// in one pass so the three projections cannot drift from the accepted field inventory.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
HostedParameterSnapshot HostedParameterSnapshotBuilder::build() const {
  if (!detail::valid_schema_text(stable_id_) ||
      (!description_.empty() && !detail::valid_utf8(description_))) {
    detail::reject("HOSTED_PARAMETER_SCHEMA_INVALID");
  }
  auto state = std::make_shared<detail::HostedParameterSnapshotState>();
  state->stable_id = stable_id_;
  state->description = description_;
  state->values = fields_;
  state->fields.resize(fields_.size());

  for (std::size_t index = 0U; index < state->values.size(); ++index) {
    const HostedParameterField& value = state->values[index];
    if (!detail::valid_schema_text(value.name) ||
        (!value.description.empty() && !detail::valid_utf8(value.description)) ||
        !detail::supported_shape(value.type_shape) ||
        (value.mutability != ParamMutability::kCompileTime &&
         value.mutability != ParamMutability::kRecompile)) {
      detail::reject("HOSTED_PARAMETER_FIELD_INVALID", value.name);
    }
    if (std::ranges::any_of(std::span{state->values}.first(index),
                            [&](const auto& prior) { return prior.name == value.name; })) {
      detail::reject("HOSTED_PARAMETER_FIELD_DUPLICATE", value.name);
    }
    const bool sequence = value.type_shape.type == ParamLogicalType::kSequence;
    if ((!sequence && value.values.size() != 1U) ||
        (sequence && (value.constraints.max_elements == 0U ||
                      value.values.size() > value.constraints.max_elements))) {
      detail::reject("HOSTED_PARAMETER_VALUE_INVALID", value.name);
    }
    const ParamLogicalType scalar_type =
        sequence ? value.type_shape.element_or_rep_type : value.type_shape.type;
    const std::uint16_t scalar_width =
        sequence ? value.type_shape.element_or_rep_bit_width : value.type_shape.bit_width;
    if (!std::ranges::all_of(value.values, [&](ParamScalarValue scalar) {
          return detail::scalar_matches(scalar, scalar_type, scalar_width);
        })) {
      detail::reject("HOSTED_PARAMETER_VALUE_INVALID", value.name);
    }
    const ParamConstraintScalarKind constraint_kind =
        scalar_type == ParamLogicalType::kFloatingPoint   ? ParamConstraintScalarKind::kFloating
        : scalar_type == ParamLogicalType::kSignedInteger ? ParamConstraintScalarKind::kSigned
                                                          : ParamConstraintScalarKind::kUnsigned;
    if (value.constraints.allowed_value_count > value.constraints.allowed_values.size() ||
        (value.constraints.has_range &&
         (!detail::constraint_valid(value.constraints.range_min, constraint_kind) ||
          !detail::constraint_valid(value.constraints.range_max, constraint_kind) ||
          detail::constraint_less(value.constraints.range_max, value.constraints.range_min)))) {
      detail::reject("HOSTED_PARAMETER_CONSTRAINT_INVALID", value.name);
    }
    for (std::size_t allowed_index = 0U; allowed_index < value.constraints.allowed_value_count;
         ++allowed_index) {
      if (!detail::constraint_valid(value.constraints.allowed_values[allowed_index],
                                    constraint_kind)) {
        detail::reject("HOSTED_PARAMETER_CONSTRAINT_INVALID", value.name);
      }
    }

    detail::StableHashBuilder id_hash;
    id_hash.add("holoscan.param.field.v1");
    id_hash.add(state->stable_id);
    id_hash.add(value.name);
    const ParamSchemaHash full_id = id_hash.finish();
    ParamFieldDescriptor& field = state->fields[index];
    field.id = detail::compact_field_id(full_id);
    field.full_id = full_id;
    field.canonical_name = value.name;
    field.logical_type = value.type_shape.type;
    field.type_shape = value.type_shape;
    field.mutability = value.mutability;
    field.earliest_invalidated_stage = LifecycleStage::kConfigure;
    field.description = value.description;
    field.constraints = value.constraints;
    field.storage = ParamStorageTraitsSummary{
        .owns_dynamic_storage = sequence,
        .fixed_object_extent = !sequence,
        .has_portable_codec = false,
        .supports_bounded_materialization = true,
        .inline_bytes = sequence ? 0U : static_cast<std::size_t>((scalar_width + 7U) / 8U),
    };
    field.local_member_token = &state->values[index];
    if (sequence) {
      field.local_operations.read_hosted_member_as_scalar_sequence = &detail::read_hosted_sequence;
    } else {
      field.local_operations.read_hosted_member_as_scalar = &detail::read_hosted_scalar;
    }
    field.local_operations.hosted_field_index = index;
    field.validate = &detail::validate_hosted_field;
    if (!detail::validate_one(value, field).ok()) {
      detail::reject("HOSTED_PARAMETER_VALUE_INVALID", value.name);
    }
  }

  detail::StableHashBuilder structural;
  detail::StableHashBuilder contract;
  detail::StableHashBuilder metadata;
  structural.add("holoscan.param.structural.v1");
  contract.add("holoscan.param.contract.v1");
  metadata.add("holoscan.param.metadata.v1");
  structural.add(state->stable_id);
  structural.add(std::uint32_t{1});
  structural.add(ParamSchemaCoverage::kComplete);
  contract.add(state->stable_id);
  contract.add(std::uint32_t{1});
  contract.add(ParamSchemaCoverage::kComplete);
  metadata.add(state->stable_id);
  metadata.add(std::uint32_t{1});
  metadata.add(ParamSchemaCoverage::kComplete);
  metadata.add(state->description);
  for (const ParamFieldDescriptor& field : state->fields) {
    detail::add_structural_field(structural, field);
    detail::add_contract_field(contract, field);
    detail::add_contract_field(metadata, field);
    metadata.add(field.policy_evidence.persistence_explicit);
    metadata.add(field.policy_evidence.external_access_explicit);
    metadata.add(field.policy_evidence.sensitivity_explicit);
    metadata.add(field.description);
    metadata.add(field.unit);
  }
  state->schema = ParamSchemaDescriptor{
      .stable_id = state->stable_id,
      .version = 1U,
      .description = state->description,
      .structural_hash = structural.finish(),
      .contract_hash = contract.finish(),
      .metadata_hash = metadata.finish(),
      .coverage = ParamSchemaCoverage::kComplete,
      .proof = ParamSchemaProof::kAuthorAsserted,
      .local_type = &param_type_descriptor<HostedParameterSnapshot>(),
      .fields = state->fields,
  };
  return HostedParameterSnapshot{.state = std::move(state)};
}

}  // namespace holoscan
