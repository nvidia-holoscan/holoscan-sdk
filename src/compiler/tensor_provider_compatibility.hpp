/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/tensor_contract.hpp>

#include "../core/plan_record_contracts.hpp"
#include "../core/tensor_contract_comparison.hpp"

namespace holoscan::detail {

[[nodiscard]] inline bool tensor_provider_satisfies_input(
    MemoryKind memory_kind, const TensorPoolBounds& provider_bounds,
    std::span<const TensorAxisBounds> provider_axes,
    const std::optional<TensorRepresentation>& provider_layout,
    const std::optional<TensorRepresentation>& resolved_input_layout,
    const std::optional<TensorContractRecord>& input_contract,
    TensorMismatches* mismatches = nullptr) {
  const std::optional<TensorRepresentation> input_layout =
      input_contract ? std::optional{input_contract->representation} : resolved_input_layout;
  if (!input_layout) {
    return true;
  }
  auto observed = provider_layout.value_or(TensorRepresentation{});
  observed.memory_kind = memory_kind;
  bool compatible = compare_tensor_layouts(observed, input_layout, mismatches);
  if (input_layout->dtype && !observed.dtype) {
    compatible = false;
    if (mismatches != nullptr) {
      tensor_mismatch(mismatches, "dtype", "unspecified", tensor_dtype_name(*input_layout->dtype));
    }
  }
  if (input_layout->rank && !observed.rank) {
    compatible = tensor_mismatch(mismatches, "rank", "unspecified", *input_layout->rank);
  }
  if (input_layout->strides && !observed.strides) {
    compatible = tensor_mismatch(
        mismatches, "strides", "unspecified", tensor_stride_name(*input_layout->strides));
  }
  if (!input_contract) {
    return compatible;
  }
  if (input_contract->max_rank && provider_bounds.max_rank > *input_contract->max_rank) {
    compatible = tensor_mismatch(
        mismatches, "max_rank", provider_bounds.max_rank, *input_contract->max_rank);
  }
  if (input_contract->max_byte_span &&
      provider_bounds.max_byte_span > *input_contract->max_byte_span) {
    compatible = tensor_mismatch(
        mismatches, "max_byte_span", provider_bounds.max_byte_span, *input_contract->max_byte_span);
  }
  if (provider_axes.size() < input_contract->axes.size()) {
    compatible = tensor_mismatch(
        mismatches, "axis_count", provider_axes.size(), input_contract->axes.size());
  }
  for (std::size_t axis = 0U; axis < std::min(provider_axes.size(), input_contract->axes.size());
       ++axis) {
    const auto record = [&](const char* constraint, auto provided, auto required) {
      if (mismatches == nullptr) {
        return false;
      }
      const auto field = fmt::format("axes[{}].{}", axis, constraint);
      return tensor_mismatch(mismatches, field.c_str(), provided, required);
    };
    if (provider_axes[axis].max_extent > input_contract->axes[axis].max_extent) {
      compatible = record(
          "max_extent", provider_axes[axis].max_extent, input_contract->axes[axis].max_extent);
    }
    if (provider_axes[axis].max_abs_stride_bytes != 0U &&
        input_contract->axes[axis].max_abs_stride_bytes != 0U &&
        provider_axes[axis].max_abs_stride_bytes >
            input_contract->axes[axis].max_abs_stride_bytes) {
      compatible = record("max_abs_stride_bytes",
                          provider_axes[axis].max_abs_stride_bytes,
                          input_contract->axes[axis].max_abs_stride_bytes);
    }
  }
  return compatible;
}

[[nodiscard]] inline bool tensor_boundary_provider_satisfies_input(
    const TensorBoundaryContractRecord& provider,
    const std::optional<TensorRepresentation>& resolved_input_layout,
    const std::optional<TensorContractRecord>& input_contract,
    TensorMismatches* mismatches = nullptr) {
  return tensor_provider_satisfies_input(
      provider.memory_kind.value_or(MemoryKind::kUnknown),
      TensorPoolBounds{
          .max_rank = provider.max_rank.value_or(provider.rank.value_or(0U)),
          .max_byte_span = provider.max_byte_span.value_or(0U),
      },
      provider.axes,
      TensorRepresentation{
          .memory_kind = provider.memory_kind,
          .dtype = provider.dtype,
          .rank = provider.rank,
          .strides = provider.strides,
      },
      resolved_input_layout,
      input_contract,
      mismatches);
}

[[nodiscard]] inline bool tensor_interface_pool_satisfies_input(
    const TensorInterfacePoolPlanRecord& provider,
    const std::optional<TensorRepresentation>& resolved_input_layout,
    const std::optional<TensorContractRecord>& input_contract,
    TensorMismatches* mismatches = nullptr) {
  return tensor_boundary_provider_satisfies_input(
      provider.logical, resolved_input_layout, input_contract, mismatches);
}

}  // namespace holoscan::detail
