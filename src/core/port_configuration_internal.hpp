/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

#include <holoscan/core/graph_ids.hpp>
#include <holoscan/core/graph_records.hpp>

#include "operator_spec_compiler_access.hpp"

namespace holoscan::detail {

/// Copy one setup declaration into its canonical Graph representation.
[[nodiscard]] inline PortConfigurationRecord setup_port_configuration(
    const OperatorSpecCompilerAccess::PortMetadata& metadata,
    std::optional<GraphPortId> follows_input = std::nullopt) {
  PortConfigurationRecord result;
  result.rate_kind = metadata.rate_kind;
  result.declared_nominal_rate = metadata.nominal_rate;
  result.declared_expected_rate = metadata.expected_rate;
  result.follows_input = follows_input;
  result.follows_ratio = metadata.follows_ratio;
  result.lifecycle_binding_stage = metadata.lifecycle_binding_stage;
  result.declared_max_nominal_rate = metadata.max_nominal_rate;
  if (metadata.direction == PortDirection::kInput) {
    result.queue_depth = metadata.queue_depth;
    result.memory_conversion = metadata.memory_conversion;
    result.payload_options = metadata.payload_options;
    result.accepts_projected_tensor = metadata.accepts_projected_tensor;
  } else {
    result.max_emits_per_compute = metadata.max_emits_per_compute;
    result.late_joiner_history_depth = metadata.late_joiner_history_depth;
    result.tensor_allocation = metadata.tensor_allocation;
    result.payload_options = metadata.payload_options;
  }
  if (!metadata.diagnostic_alias.empty()) {
    result.diagnostic_alias = std::string(metadata.diagnostic_alias);
  }
  result.tensor_layout = metadata.tensor_layout;
  result.tensor_contract = metadata.tensor_contract;
  result.tensor_byte_span_expression = metadata.tensor_byte_span_expression;
  result.tensor_memory_kind_expression = metadata.tensor_memory_kind_expression;
  if (metadata.descriptor_tensor_field_count == 1U && metadata.tensor_fields.size() == 1U) {
    const auto& field = metadata.tensor_fields.front().configuration;
    result.tensor_layout = field.tensor_layout;
    result.tensor_contract = field.tensor_contract;
    result.tensor_byte_span_expression = field.tensor_byte_span_expression;
    result.tensor_memory_kind_expression = field.tensor_memory_kind_expression;
    if (field.payload_options) {
      result.payload_options = PayloadOptionsRecord{
          .max_inline_bytes = field.payload_options->max_inline_bytes
                                  ? std::optional{static_cast<std::uint32_t>(
                                        *field.payload_options->max_inline_bytes)}
                                  : std::nullopt,
          .max_tensor_byte_span = field.payload_options->max_tensor_byte_span,
          .interface_tensor_emit_policy = field.payload_options->interface_tensor_emit_policy,
      };
    }
    if (metadata.direction == PortDirection::kInput) {
      result.memory_conversion = field.memory_conversion;
    } else {
      result.tensor_allocation = field.tensor_allocation;
    }
    return result;
  }
  result.tensor_fields.reserve(metadata.tensor_fields.size());
  for (const auto& field : metadata.tensor_fields) {
    result.tensor_fields.push_back(field.configuration);
  }
  std::sort(result.tensor_fields.begin(),
            result.tensor_fields.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.field_id < rhs.field_id; });
  return result;
}

}  // namespace holoscan::detail
