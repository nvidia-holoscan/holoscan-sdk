/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <optional>
#include <span>

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/tensor_contract.hpp>

#include "../core/plan_record_contracts.hpp"

namespace holoscan::detail {

[[nodiscard]] inline bool tensor_provider_satisfies_input(
    MemoryKind memory_kind, const TensorPoolBounds& provider_bounds,
    std::span<const TensorAxisBounds> provider_axes,
    const std::optional<TensorRepresentation>& provider_layout,
    const std::optional<TensorRepresentation>& resolved_input_layout,
    const std::optional<TensorContractRecord>& input_contract) noexcept {
  const std::optional<TensorRepresentation> input_layout =
      input_contract ? std::optional{input_contract->representation} : resolved_input_layout;
  if (!input_layout) {
    return true;
  }
  if (input_layout->memory_kind && *input_layout->memory_kind != memory_kind) {
    return false;
  }
  if (input_layout->dtype &&
      (!provider_layout || !provider_layout->dtype ||
       !dl_data_type_exactly_equal(*input_layout->dtype, *provider_layout->dtype))) {
    return false;
  }
  if (input_layout->rank && (!provider_layout || !provider_layout->rank ||
                             *input_layout->rank != *provider_layout->rank)) {
    return false;
  }
  if (input_layout->strides && (!provider_layout || !provider_layout->strides ||
                                *input_layout->strides != *provider_layout->strides)) {
    return false;
  }
  if (!input_contract) {
    return true;
  }
  if ((input_contract->max_rank && provider_bounds.max_rank > *input_contract->max_rank) ||
      (input_contract->max_byte_span &&
       provider_bounds.max_byte_span > *input_contract->max_byte_span) ||
      provider_axes.size() < input_contract->axes.size()) {
    return false;
  }
  for (std::size_t axis = 0U; axis < input_contract->axes.size(); ++axis) {
    if (provider_axes[axis].max_extent > input_contract->axes[axis].max_extent ||
        (provider_axes[axis].max_abs_stride_bytes != 0U &&
         input_contract->axes[axis].max_abs_stride_bytes != 0U &&
         provider_axes[axis].max_abs_stride_bytes >
             input_contract->axes[axis].max_abs_stride_bytes)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool tensor_boundary_provider_satisfies_input(
    const TensorBoundaryContractRecord& provider,
    const std::optional<TensorRepresentation>& resolved_input_layout,
    const std::optional<TensorContractRecord>& input_contract) noexcept {
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
      input_contract);
}

[[nodiscard]] inline bool tensor_interface_pool_satisfies_input(
    const TensorInterfacePoolPlanRecord& provider,
    const std::optional<TensorRepresentation>& resolved_input_layout,
    const std::optional<TensorContractRecord>& input_contract) noexcept {
  return tensor_boundary_provider_satisfies_input(
      provider.logical, resolved_input_layout, input_contract);
}

}  // namespace holoscan::detail
