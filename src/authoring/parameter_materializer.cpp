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
#include <string>
#include <unordered_set>
#include <utility>

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

[[nodiscard]] Error validation_error(const ParamValidationResult& validation) {
  std::string message = fmt::format("PARAM_VALIDATION_FAILED category={} field_id={}",
                                    validation_code_name(validation.code),
                                    validation.field_id);
  if (!validation.field_name.empty()) {
    message.append(" field_name=");
    append_bounded_identity(message, validation.field_name);
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

[[nodiscard]] bool coerce_scalar(ParamScalarValue input, const ParamFieldDescriptor& field,
                                 ParamScalarValue& output) noexcept {
  switch (field.logical_type) {
    case ParamLogicalType::kBool:
      if (input.kind == ParamScalarKind::kBool && input.bit_width == 1U && input.raw_bits <= 1U) {
        output = input;
        return true;
      }
      return false;
    case ParamLogicalType::kSignedInteger:
    case ParamLogicalType::kUnsignedInteger:
      return coerce_integer(input, field.logical_type, field.type_shape.bit_width, output);
    case ParamLogicalType::kFloatingPoint:
      return coerce_floating(input, field.type_shape.bit_width, output);
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

struct FieldLookup {
  const ParamFieldDescriptor* field{};
  bool ambiguous{};
};

[[nodiscard]] FieldLookup find_field(const ParamSchemaDescriptor& schema,
                                     std::string_view name) noexcept {
  FieldLookup result;
  for (const auto& field : schema.fields) {
    if (field.canonical_name == name || (!field.alias.empty() && field.alias == name)) {
      if (result.field != nullptr) {
        result.ambiguous = true;
        return result;
      }
      result.field = &field;
    }
  }
  return result;
}

[[nodiscard]] expected<void, Error> apply_assignments(
    void* storage, const ParamSchemaDescriptor& schema,
    std::span<const ParameterAssignment> assignments) {
  std::unordered_set<const ParamFieldDescriptor*> assigned_fields;
  assigned_fields.reserve(assignments.size());

  for (const ParameterAssignment& assignment : assignments) {
    const FieldLookup lookup = find_field(schema, assignment.name);
    if (lookup.ambiguous) {
      return make_unexpected(parameter_error("AUTHORING_PARAMETER_AMBIGUOUS", assignment.name));
    }
    const ParamFieldDescriptor* const field = lookup.field;
    if (field == nullptr) {
      return make_unexpected(parameter_error("AUTHORING_PARAMETER_UNKNOWN", assignment.name));
    }
    if (!assigned_fields.insert(field).second) {
      return make_unexpected(parameter_error("AUTHORING_PARAMETER_DUPLICATE", assignment.name));
    }

    bool replaced = false;
    if (assignment.value.kind == ParameterValueKind::kScalar &&
        field->local_operations.replace_member_from_scalar != nullptr) {
      ParamScalarValue coerced{};
      replaced = coerce_scalar(assignment.value.scalar, *field, coerced) &&
                 field->local_operations.replace_member_from_scalar(storage, coerced);
    } else if (assignment.value.kind == ParameterValueKind::kString &&
               field->local_operations.replace_member_from_string != nullptr) {
      try {
        field->local_operations.replace_member_from_string(storage, assignment.value.string);
        replaced = true;
      } catch (const std::bad_alloc&) {
        return make_unexpected(parameter_error_noexcept("AUTHORING_PARAMETER_ALLOCATION_FAILED",
                                                        assignment.name,
                                                        ErrorCode::kResourceExhausted));
      } catch (...) {
        return make_unexpected(parameter_error_noexcept(
            "AUTHORING_PARAMETER_ASSIGNMENT_FAILED", assignment.name, ErrorCode::kFailure));
      }
    }
    if (!replaced) {
      if (field->logical_type == ParamLogicalType::kSequence ||
          field->logical_type == ParamLogicalType::kFilesystemPath ||
          field->logical_type == ParamLogicalType::kObject) {
        return make_unexpected(
            parameter_error("AUTHORING_PARAMETER_KIND_UNSUPPORTED", assignment.name));
      }
      return make_unexpected(parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH", assignment.name));
    }
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
      auto assignment_result = apply_assignments(storage, *schema, assignments);
      if (!assignment_result) {
        return make_unexpected(std::move(assignment_result.error()));
      }
    }

    if (type.validate_aggregate == nullptr) {
      return make_unexpected(parameter_error("AUTHORING_PARAMETER_VALIDATOR_UNAVAILABLE"));
    }
    const ParamValidationResult validation = type.validate_aggregate(storage);
    if (!validation.ok()) {
      return make_unexpected(validation_error(validation));
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
