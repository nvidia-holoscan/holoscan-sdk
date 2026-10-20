/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/parameter_schema.hpp>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema_types.hpp>

namespace holoscan::detail {

ParamValidationResult validate_param_scalar(const ParamConstraintScalar observed,
                                            const ParamFieldDescriptor& descriptor) noexcept {
  if (descriptor.constraints.has_range) {
    if (!scalar_finite(observed) || scalar_less(observed, descriptor.constraints.range_min) ||
        scalar_less(descriptor.constraints.range_max, observed)) {
      return ParamValidationResult{
          ParamValidationCode::kOutOfRange, descriptor.id, descriptor.canonical_name};
    }
  }
  if (descriptor.constraints.allowed_value_count != 0U) {
    bool found = false;
    for (std::size_t index = 0; index < descriptor.constraints.allowed_value_count; ++index) {
      found = found || scalar_equal(observed, descriptor.constraints.allowed_values[index]);
    }
    if (!found) {
      return ParamValidationResult{
          ParamValidationCode::kNotAllowed, descriptor.id, descriptor.canonical_name};
    }
  }
  return {};
}

ParamValidationResult validate_param_string(const std::string_view value,
                                            const ParamFieldDescriptor& descriptor) noexcept {
  // Reject cheap byte bounds before scanning UTF-8. Besides bounding cold-path work, this freezes
  // cardinality as the deterministic first failure when both facts are invalid.
  if (descriptor.constraints.max_utf8_bytes != 0U &&
      value.size() > descriptor.constraints.max_utf8_bytes) {
    return ParamValidationResult{
        ParamValidationCode::kCardinalityExceeded, descriptor.id, descriptor.canonical_name};
  }
  if (descriptor.string_encoding == ParamStringEncoding::kUtf8 && !valid_utf8(value)) {
    return ParamValidationResult{
        ParamValidationCode::kInvalidEncoding, descriptor.id, descriptor.canonical_name};
  }
  return {};
}

ParamValidationResult validate_param_string_sequence(
    const std::span<const std::string> values, const ParamFieldDescriptor& descriptor) noexcept {
  if (descriptor.constraints.max_elements != 0U &&
      values.size() > descriptor.constraints.max_elements) {
    return ParamValidationResult{
        ParamValidationCode::kCardinalityExceeded, descriptor.id, descriptor.canonical_name};
  }
  if (descriptor.constraints.max_element_utf8_bytes != 0U) {
    for (const std::string& element : values) {
      if (element.size() > descriptor.constraints.max_element_utf8_bytes) {
        return ParamValidationResult{
            ParamValidationCode::kCardinalityExceeded, descriptor.id, descriptor.canonical_name};
      }
    }
  }
  if (descriptor.element_string_encoding == ParamStringEncoding::kUtf8) {
    for (const std::string& element : values) {
      if (!valid_utf8(element)) {
        return ParamValidationResult{
            ParamValidationCode::kInvalidEncoding, descriptor.id, descriptor.canonical_name};
      }
    }
  }
  return {};
}

}  // namespace holoscan::detail
