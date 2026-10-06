/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/authoring/parameter_materializer.hpp>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fmt/format.h>

namespace holoscan::authoring {
namespace {

constexpr std::size_t kMaximumRenderedIdentityBytes = 96U;

[[nodiscard]] Error parameter_error(std::string_view token, std::string_view field = {},
                                    ErrorCode code = ErrorCode::kInvalidArgument) {
  std::string message =
      field.empty() ? std::string{token} : fmt::format("{} field_name={}", token, field);
  return Error{code, std::move(message)};
}

[[nodiscard]] Error parameter_error_noexcept(
    std::string_view token, std::string_view field = {},
    ErrorCode code = ErrorCode::kInvalidArgument) noexcept {
  try {
    return parameter_error(token, field, code);
  } catch (...) {
    return Error{code};
  }
}

[[nodiscard]] constexpr std::string_view validation_code_name(ParamValidationCode code) noexcept {
  switch (code) {
    case ParamValidationCode::kOk:
      return "ok";
    case ParamValidationCode::kOutOfRange:
      return "out_of_range";
    case ParamValidationCode::kNotAllowed:
      return "not_allowed";
    case ParamValidationCode::kCardinalityExceeded:
      return "cardinality_exceeded";
    case ParamValidationCode::kInvalidEncoding:
      return "invalid_encoding";
    case ParamValidationCode::kValidatorRejected:
      return "validator_rejected";
  }
  return "unknown";
}

void append_bounded_identity(std::string& message, std::string_view identity) {
  if (identity.size() <= kMaximumRenderedIdentityBytes) {
    message.append(identity);
    return;
  }
  holoscan::detail::StableHashBuilder hash;
  hash.add("holoscan.param.validation-diagnostic.identity/v1");
  hash.add(identity);
  const ParamSchemaHash digest = hash.finish();
  constexpr std::string_view kHexDigits{"0123456789abcdef"};
  std::size_t prefix_bytes = kMaximumRenderedIdentityBytes;
  while (prefix_bytes != 0U && !holoscan::detail::valid_utf8(identity.substr(0U, prefix_bytes))) {
    --prefix_bytes;
  }
  message.append(identity.substr(0U, prefix_bytes));
  fmt::format_to(std::back_inserter(message), "...;bytes={};sha256=", identity.size());
  for (const std::byte byte : digest.value) {
    const auto value = static_cast<std::uint8_t>(byte);
    message.push_back(kHexDigits[value >> 4U]);
    message.push_back(kHexDigits[value & 0x0FU]);
  }
}

[[nodiscard]] Error validation_error(const ParamValidationResult& validation,
                                     std::string_view field_name = {}) {
  std::string message = fmt::format("PARAM_VALIDATION_FAILED category={} field_id={}",
                                    validation_code_name(validation.code),
                                    validation.field_id);
  const std::string_view rendered_field = field_name.empty() ? validation.field_name : field_name;
  if (!rendered_field.empty()) {
    message.append(" field_name=");
    append_bounded_identity(message, rendered_field);
  }
  if (!validation.validator_id.empty()) {
    message.append(" validator_id=");
    append_bounded_identity(message, validation.validator_id);
  }
  return Error{ErrorCode::kInvalidArgument, std::move(message)};
}

[[nodiscard]] bool valid_integer_width(std::uint16_t width) noexcept {
  return width == 8U || width == 16U || width == 32U || width == 64U;
}

[[nodiscard]] bool decode_signed(ParamScalarValue value, std::int64_t& result) noexcept {
  if (value.kind != ParamScalarKind::kSignedInteger || !valid_integer_width(value.bit_width)) {
    return false;
  }
  if (value.bit_width == 64U) {
    result = std::bit_cast<std::int64_t>(value.raw_bits);
    return true;
  }
  const std::uint64_t mask = (std::uint64_t{1} << value.bit_width) - 1U;
  if ((value.raw_bits & ~mask) != 0U) {
    return false;
  }
  const std::uint64_t sign = std::uint64_t{1} << (value.bit_width - 1U);
  result = std::bit_cast<std::int64_t>((value.raw_bits & sign) != 0U ? value.raw_bits | ~mask
                                                                     : value.raw_bits);
  return true;
}

[[nodiscard]] bool decode_unsigned(ParamScalarValue value, std::uint64_t& result) noexcept {
  if (value.kind != ParamScalarKind::kUnsignedInteger || !valid_integer_width(value.bit_width)) {
    return false;
  }
  if (value.bit_width != 64U && value.raw_bits >= (std::uint64_t{1} << value.bit_width)) {
    return false;
  }
  result = value.raw_bits;
  return true;
}

[[nodiscard]] bool fits_signed(std::int64_t value, std::uint16_t width) noexcept {
  if (!valid_integer_width(width)) {
    return false;
  }
  if (width == 64U) {
    return true;
  }
  const std::int64_t limit = std::int64_t{1} << (width - 1U);
  return value >= -limit && value < limit;
}

[[nodiscard]] bool fits_unsigned(std::uint64_t value, std::uint16_t width) noexcept {
  return valid_integer_width(width) && (width == 64U || value < (std::uint64_t{1} << width));
}

[[nodiscard]] ParamScalarValue signed_value(std::int64_t value, std::uint16_t width) noexcept {
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
  const std::uint64_t mask =
      width == 64U ? std::numeric_limits<std::uint64_t>::max() : (std::uint64_t{1} << width) - 1U;
  return ParamScalarValue{.kind = ParamScalarKind::kSignedInteger,
                          .bit_width = static_cast<std::uint8_t>(width),
                          .raw_bits = bits & mask};
}

[[nodiscard]] ParamScalarValue unsigned_value(std::uint64_t value, std::uint16_t width) noexcept {
  return ParamScalarValue{.kind = ParamScalarKind::kUnsignedInteger,
                          .bit_width = static_cast<std::uint8_t>(width),
                          .raw_bits = value};
}

[[nodiscard]] bool coerce_integer(ParamScalarValue input, ParamLogicalType target,
                                  std::uint16_t width, ParamScalarValue& output) noexcept {
  std::int64_t signed_input{};
  std::uint64_t unsigned_input{};
  if (target == ParamLogicalType::kSignedInteger) {
    if (decode_signed(input, signed_input) && fits_signed(signed_input, width)) {
      output = signed_value(signed_input, width);
      return true;
    }
    if (decode_unsigned(input, unsigned_input) &&
        unsigned_input <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) &&
        fits_signed(static_cast<std::int64_t>(unsigned_input), width)) {
      output = signed_value(static_cast<std::int64_t>(unsigned_input), width);
      return true;
    }
    return false;
  }
  if (decode_unsigned(input, unsigned_input) && fits_unsigned(unsigned_input, width)) {
    output = unsigned_value(unsigned_input, width);
    return true;
  }
  if (decode_signed(input, signed_input) && signed_input >= 0 &&
      fits_unsigned(static_cast<std::uint64_t>(signed_input), width)) {
    output = unsigned_value(static_cast<std::uint64_t>(signed_input), width);
    return true;
  }
  return false;
}

[[nodiscard]] bool coerce_floating(ParamScalarValue input, std::uint16_t width,
                                   ParamScalarValue& output) noexcept {
  double value{};
  if (input.kind == ParamScalarKind::kFloatingPoint && input.bit_width == 64U) {
    value = std::bit_cast<double>(input.raw_bits);
  } else if (input.kind == ParamScalarKind::kFloatingPoint && input.bit_width == 32U &&
             input.raw_bits <= std::numeric_limits<std::uint32_t>::max()) {
    value = std::bit_cast<float>(static_cast<std::uint32_t>(input.raw_bits));
  } else {
    std::int64_t signed_input{};
    std::uint64_t unsigned_input{};
    if (decode_signed(input, signed_input)) {
      value = static_cast<double>(signed_input);
    } else if (decode_unsigned(input, unsigned_input)) {
      value = static_cast<double>(unsigned_input);
    } else {
      return false;
    }
  }
  if (width == 32U) {
    const float narrowed = static_cast<float>(value);
    if (std::isfinite(value) && !std::isfinite(narrowed)) {
      return false;
    }
    output = ParamScalarValue::from(narrowed);
    return true;
  }
  if (width == 64U) {
    output = ParamScalarValue::from(value);
    return true;
  }
  return false;
}

[[nodiscard]] bool coerce_scalar(ParamScalarValue input, ParamLogicalType logical_type,
                                 std::uint16_t bit_width, ParamScalarValue& output) noexcept {
  switch (logical_type) {
    case ParamLogicalType::kBool:
      if (input.kind == ParamScalarKind::kBool && input.bit_width == 1U && input.raw_bits <= 1U) {
        output = input;
        return true;
      }
      return false;
    case ParamLogicalType::kSignedInteger:
    case ParamLogicalType::kUnsignedInteger:
      return coerce_integer(input, logical_type, bit_width, output);
    case ParamLogicalType::kFloatingPoint:
      return coerce_floating(input, bit_width, output);
    case ParamLogicalType::kDuration: {
      std::int64_t nanoseconds{};
      std::uint64_t unsigned_nanoseconds{};
      if (input.kind == ParamScalarKind::kDurationNanoseconds && input.bit_width == 64U) {
        output = input;
        return true;
      }
      if (decode_signed(input, nanoseconds)) {
        output = ParamScalarValue{.kind = ParamScalarKind::kDurationNanoseconds,
                                  .bit_width = 64,
                                  .raw_bits = std::bit_cast<std::uint64_t>(nanoseconds)};
        return true;
      }
      if (decode_unsigned(input, unsigned_nanoseconds) &&
          unsigned_nanoseconds <=
              static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        output = ParamScalarValue{.kind = ParamScalarKind::kDurationNanoseconds,
                                  .bit_width = 64,
                                  .raw_bits = unsigned_nanoseconds};
        return true;
      }
      return false;
    }
    case ParamLogicalType::kString:
    case ParamLogicalType::kSequence:
    case ParamLogicalType::kFilesystemPath:
    case ParamLogicalType::kObject:
      return false;
  }
  return false;
}

[[nodiscard]] bool coerce_scalar(ParamScalarValue input, const ParamFieldDescriptor& field,
                                 ParamScalarValue& output) noexcept {
  return coerce_scalar(input, field.logical_type, field.type_shape.bit_width, output);
}

[[nodiscard]] bool coerce_sequence_scalar(ParamScalarValue input, const ParamFieldDescriptor& field,
                                          ParamScalarValue& output) noexcept {
  return coerce_scalar(input,
                       field.type_shape.element_or_rep_type,
                       field.type_shape.element_or_rep_bit_width,
                       output);
}

[[nodiscard]] constexpr bool is_scalar_logical_type(ParamLogicalType type) noexcept {
  switch (type) {
    case ParamLogicalType::kBool:
    case ParamLogicalType::kSignedInteger:
    case ParamLogicalType::kUnsignedInteger:
    case ParamLogicalType::kFloatingPoint:
    case ParamLogicalType::kDuration:
      return true;
    case ParamLogicalType::kString:
    case ParamLogicalType::kSequence:
    case ParamLogicalType::kFilesystemPath:
    case ParamLogicalType::kObject:
      return false;
  }
  return false;
}

[[nodiscard]] const ParamEnumChoice* find_enum_choice(const ParamFieldDescriptor& field,
                                                      std::string_view name) noexcept {
  for (std::size_t index = 0; index < field.enum_choices.count; ++index) {
    if (field.enum_choices.values[index].name == name) {
      return &field.enum_choices.values[index];
    }
  }
  return nullptr;
}

[[nodiscard]] bool assignment_scalar(const ParameterValue& value, const ParamFieldDescriptor& field,
                                     bool sequence_element, ParamScalarValue& output) noexcept {
  ParamScalarValue input;
  if (value.kind == ParameterValueKind::kScalar) {
    input = value.scalar;
  } else if (value.kind == ParameterValueKind::kString && field.enum_choices.count != 0U) {
    const ParamEnumChoice* choice = find_enum_choice(field, value.string);
    if (choice == nullptr) {
      return false;
    }
    input = choice->value;
  } else {
    return false;
  }
  return sequence_element ? coerce_sequence_scalar(input, field, output)
                          : coerce_scalar(input, field, output);
}

[[nodiscard]] bool apply_sequence(void* aggregate, const ParamFieldDescriptor& field,
                                  const ParameterValue& value) {
  const auto sequence = value.sequence_values();
  if (field.local_operations.replace_member_from_scalar_sequence != nullptr) {
    std::vector<ParamScalarValue> elements;
    elements.reserve(sequence.size());
    for (const ParameterValue& element : sequence) {
      ParamScalarValue coerced;
      if (!assignment_scalar(element, field, true, coerced)) {
        return false;
      }
      elements.push_back(coerced);
    }
    return field.local_operations.replace_member_from_scalar_sequence(aggregate, elements);
  }
  if (field.local_operations.replace_member_from_string_sequence != nullptr) {
    std::vector<std::string_view> elements;
    elements.reserve(sequence.size());
    for (const ParameterValue& element : sequence) {
      if (element.kind != ParameterValueKind::kString) {
        return false;
      }
      elements.push_back(element.string);
    }
    field.local_operations.replace_member_from_string_sequence(aggregate, elements);
    return true;
  }
  return false;
}

[[nodiscard]] bool find_validation_path(const void* aggregate, const ParamSchemaDescriptor& schema,
                                        ParamFieldId field_id, std::string& result) {
  for (const ParamFieldDescriptor& field : schema.fields) {
    if (field.validate == nullptr || field.validate(aggregate, field).ok()) {
      continue;
    }
    if (field.id == field_id) {
      result = field.canonical_name;
      return true;
    }
    if (field.local_operations.nested_schema == nullptr ||
        field.local_operations.nested_member == nullptr) {
      continue;
    }
    const ParamSchemaDescriptor* nested_schema = field.local_operations.nested_schema();
    const void* nested_member = field.local_operations.nested_member(aggregate);
    std::string nested_path;
    if (nested_schema != nullptr && nested_member != nullptr &&
        find_validation_path(nested_member, *nested_schema, field_id, nested_path)) {
      result = detail::parameter_field_path(field.canonical_name, nested_path);
      return true;
    }
  }
  return false;
}

[[nodiscard]] expected<void, Error> apply_assignments(
    void* aggregate, const ParamSchemaDescriptor& schema,
    std::span<const ParameterAssignment> assignments, std::string_view prefix) {
  std::unordered_set<const ParamFieldDescriptor*> assigned_fields;
  assigned_fields.reserve(assignments.size());
  for (const ParameterAssignment& assignment : assignments) {
    const auto lookup = detail::lookup_parameter_field(schema, assignment.name);
    if (lookup.ambiguous) {
      return make_unexpected(parameter_error(
          "AUTHORING_PARAMETER_AMBIGUOUS", detail::parameter_field_path(prefix, assignment.name)));
    }
    const ParamFieldDescriptor* const field = lookup.field;
    if (field == nullptr) {
      return make_unexpected(parameter_error(
          "AUTHORING_PARAMETER_UNKNOWN", detail::parameter_field_path(prefix, assignment.name)));
    }
    std::optional<std::string> canonical_path;
    const auto path = [&]() -> const std::string& {
      if (!canonical_path.has_value()) {
        canonical_path.emplace(detail::parameter_field_path(prefix, field->canonical_name));
      }
      return *canonical_path;
    };
    if (!assigned_fields.insert(field).second) {
      return make_unexpected(parameter_error("AUTHORING_PARAMETER_DUPLICATE", path()));
    }

    try {
      bool replaced{};
      bool kind_matches = true;
      bool kind_supported = true;
      switch (assignment.value.kind) {
        case ParameterValueKind::kUnset:
          return make_unexpected(parameter_error("AUTHORING_PARAMETER_VALUE_UNSET", path()));
        case ParameterValueKind::kNone:
          if (field->local_operations.clear_optional_member == nullptr) {
            return make_unexpected(parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH", path()));
          }
          field->local_operations.clear_optional_member(aggregate);
          replaced = true;
          break;
        case ParameterValueKind::kAssignments:
          kind_matches = field->logical_type == ParamLogicalType::kObject;
          kind_supported = field->local_operations.mutable_nested_member != nullptr &&
                           field->local_operations.nested_schema != nullptr;
          if (kind_matches && kind_supported) {
            const std::string& nested_path = path();
            void* const nested_member = field->local_operations.mutable_nested_member(aggregate);
            const ParamSchemaDescriptor* const nested_schema =
                field->local_operations.nested_schema();
            if (nested_member != nullptr && nested_schema != nullptr) {
              auto nested = apply_assignments(nested_member,
                                              *nested_schema,
                                              assignment.value.nested_assignments(),
                                              nested_path);
              if (!nested) {
                return make_unexpected(std::move(nested).error());
              }
              replaced = true;
            }
          }
          break;
        case ParameterValueKind::kScalar:
          kind_matches = is_scalar_logical_type(field->logical_type);
          kind_supported = field->local_operations.replace_member_from_scalar != nullptr;
          if (kind_matches && kind_supported) {
            ParamScalarValue coerced;
            replaced = assignment_scalar(assignment.value, *field, false, coerced) &&
                       field->local_operations.replace_member_from_scalar(aggregate, coerced);
          }
          break;
        case ParameterValueKind::kString:
          kind_matches = field->logical_type == ParamLogicalType::kString ||
                         field->logical_type == ParamLogicalType::kFilesystemPath ||
                         field->enum_choices.count != 0U;
          kind_supported = field->local_operations.replace_member_from_scalar != nullptr ||
                           field->local_operations.replace_member_from_string != nullptr;
          if (kind_matches && field->local_operations.replace_member_from_scalar != nullptr) {
            ParamScalarValue coerced;
            replaced = assignment_scalar(assignment.value, *field, false, coerced) &&
                       field->local_operations.replace_member_from_scalar(aggregate, coerced);
          } else if (kind_matches &&
                     field->local_operations.replace_member_from_string != nullptr) {
            field->local_operations.replace_member_from_string(aggregate, assignment.value.string);
            replaced = true;
          }
          break;
        case ParameterValueKind::kSequence:
          kind_matches = field->logical_type == ParamLogicalType::kSequence;
          kind_supported = field->local_operations.replace_member_from_scalar_sequence != nullptr ||
                           field->local_operations.replace_member_from_string_sequence != nullptr;
          if (kind_matches && kind_supported && field->constraints.max_elements != 0U &&
              assignment.value.sequence_values().size() > field->constraints.max_elements) {
            return make_unexpected(validation_error(
                ParamValidationResult{
                    ParamValidationCode::kCardinalityExceeded, field->id, field->canonical_name},
                path()));
          }
          if (kind_matches && kind_supported) {
            replaced = apply_sequence(aggregate, *field, assignment.value);
          }
          break;
      }
      if (!replaced) {
        const std::string_view token = kind_matches && !kind_supported
                                           ? "AUTHORING_PARAMETER_KIND_UNSUPPORTED"
                                           : "AUTHORING_PARAMETER_TYPE_MISMATCH";
        return make_unexpected(parameter_error(token, path()));
      }
    } catch (const std::bad_alloc&) {
      return make_unexpected(parameter_error_noexcept(
          "AUTHORING_PARAMETER_ALLOCATION_FAILED", path(), ErrorCode::kResourceExhausted));
    } catch (...) {
      return make_unexpected(parameter_error_noexcept(
          "AUTHORING_PARAMETER_ASSIGNMENT_FAILED", path(), ErrorCode::kFailure));
    }
  }

  if (schema.local_type == nullptr || schema.local_type->validate_aggregate == nullptr) {
    return make_unexpected(parameter_error("AUTHORING_PARAMETER_VALIDATOR_UNAVAILABLE", prefix));
  }
  const ParamValidationResult validation = schema.local_type->validate_aggregate(aggregate);
  if (!validation.ok()) {
    std::string relative_path;
    (void)find_validation_path(aggregate, schema, validation.field_id, relative_path);
    std::string canonical_path = relative_path;
    if (!prefix.empty()) {
      canonical_path = relative_path.empty() ? std::string{prefix}
                                             : detail::parameter_field_path(prefix, relative_path);
    }
    return make_unexpected(validation_error(validation, canonical_path));
  }
  return {};
}

}  // namespace

MaterializedParameters::~MaterializedParameters() {
  reset();
}

MaterializedParameters::MaterializedParameters(MaterializedParameters&& other) noexcept
    : type_(std::exchange(other.type_, nullptr)), value_(std::exchange(other.value_, nullptr)) {}

MaterializedParameters& MaterializedParameters::operator=(MaterializedParameters&& other) noexcept {
  if (this != &other) {
    reset();
    type_ = std::exchange(other.type_, nullptr);
    value_ = std::exchange(other.value_, nullptr);
  }
  return *this;
}

void MaterializedParameters::reset() noexcept {
  if (value_ != nullptr) {
    type_->destroy(value_);
    ::operator delete(value_, std::align_val_t{type_->object_alignment});
  }
  type_ = nullptr;
  value_ = nullptr;
}

expected<MaterializedParameters, Error> materialize_parameters(
    const ParamTypeDescriptor& type, const ParamSchemaDescriptor* schema,
    std::span<const ParameterAssignment> assignments) {
  try {
    if (type.default_construct == nullptr || type.destroy == nullptr || type.object_size == 0U ||
        type.object_alignment == 0U) {
      return make_unexpected(parameter_error("AUTHORING_PARAMETER_TYPE_INVALID"));
    }
    if (!assignments.empty() && schema == nullptr) {
      return make_unexpected(parameter_error("AUTHORING_PARAMETER_SCHEMA_UNAVAILABLE"));
    }
    if (schema != nullptr && schema->local_type != &type) {
      return make_unexpected(parameter_error("AUTHORING_PARAMETER_SCHEMA_MISMATCH"));
    }

    void* storage = nullptr;
    try {
      storage = ::operator new(type.object_size, std::align_val_t{type.object_alignment});
      type.default_construct(storage);
    } catch (const std::bad_alloc&) {
      if (storage != nullptr) {
        ::operator delete(storage, std::align_val_t{type.object_alignment});
      }
      return make_unexpected(parameter_error_noexcept(
          "AUTHORING_PARAMETER_ALLOCATION_FAILED", {}, ErrorCode::kResourceExhausted));
    } catch (...) {
      if (storage != nullptr) {
        ::operator delete(storage, std::align_val_t{type.object_alignment});
      }
      return make_unexpected(parameter_error_noexcept(
          "AUTHORING_PARAMETER_CONSTRUCTION_FAILED", {}, ErrorCode::kFailure));
    }
    MaterializedParameters result{type, storage};
    if (schema != nullptr) {
      auto applied = apply_assignments(storage, *schema, assignments, {});
      if (!applied) {
        return make_unexpected(std::move(applied).error());
      }
    }

    if (schema == nullptr) {
      if (type.validate_aggregate == nullptr) {
        return make_unexpected(parameter_error("AUTHORING_PARAMETER_VALIDATOR_UNAVAILABLE"));
      }
      const ParamValidationResult validation = type.validate_aggregate(storage);
      if (!validation.ok()) {
        return make_unexpected(validation_error(validation));
      }
    }
    return result;
  } catch (const std::bad_alloc&) {
    return make_unexpected(parameter_error_noexcept(
        "AUTHORING_PARAMETER_ALLOCATION_FAILED", {}, ErrorCode::kResourceExhausted));
  } catch (...) {
    return make_unexpected(parameter_error_noexcept(
        "AUTHORING_PARAMETER_MATERIALIZATION_FAILED", {}, ErrorCode::kFailure));
  }
}

}  // namespace holoscan::authoring
