/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <span>
#include <string>

#include "compiler/execution_plan_state.hpp"

namespace holoscan::detail::plan_artifact_serialization_internal {

template <typename T>
void append_number(std::string& output, T value) {
  output += std::to_string(value);
}

inline void append_bool(std::string& output, bool value) {
  output += value ? "true" : "false";
}

template <typename Records, typename AppendRecord>
void append_array(std::string& output, const Records& records, AppendRecord append_record) {
  output.push_back('[');
  bool first = true;
  for (const auto& record : records) {
    if (!first) {
      output.push_back(',');
    }
    first = false;
    append_record(output, record);
  }
  output.push_back(']');
}

void append_optional_profile(std::string& output, std::optional<Profile> profile);
void append_optional_string(std::string& output, const std::optional<std::string>& value);
void append_optional_tensor_layout(std::string& output,
                                   const std::optional<TensorRepresentation>& layout);
void append_tensor_axes(std::string& output, std::span<const TensorAxisBounds> axes);
void append_tensor_allocation_bounds(std::string& output, const TensorPoolBounds& bounds);
void append_logical_tensor_requirements(std::string& output,
                                        const TensorBoundaryContractRecord& logical,
                                        bool include_provenance);

void append_tensor_and_cuda_sections(std::string& output, const ExecutionPlanState& state,
                                     bool include_provenance);

}  // namespace holoscan::detail::plan_artifact_serialization_internal
