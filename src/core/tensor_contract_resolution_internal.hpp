/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

#include <holoscan/core/tensor_contract.hpp>

namespace holoscan::detail {

[[nodiscard]] inline std::optional<TensorContractRecord> resolve_output_tensor_contract(
    const std::optional<TensorContractRecord>& setup,
    const std::optional<TensorContractRecord>& graph_override) {
  if (!setup || setup->storage != TensorOutputStorage::kRuntimePool || !graph_override ||
      graph_override->storage != TensorOutputStorage::kPassThrough) {
    return graph_override ? graph_override : setup;
  }

  TensorContractRecord result = *setup;
  const TensorContractRecord& refinement = *graph_override;
  const auto replace =
      [](auto& target, auto& target_origin, const auto& source, const auto source_origin) {
        if (source) {
          target = source;
          target_origin = source_origin;
        }
      };
  replace(result.representation.memory_kind,
          result.origins.memory_kind,
          refinement.representation.memory_kind,
          refinement.origins.memory_kind);
  replace(result.representation.dtype,
          result.origins.dtype,
          refinement.representation.dtype,
          refinement.origins.dtype);
  replace(result.representation.rank,
          result.origins.rank,
          refinement.representation.rank,
          refinement.origins.rank);
  replace(result.representation.strides,
          result.origins.strides,
          refinement.representation.strides,
          refinement.origins.strides);
  replace(
      result.max_rank, result.origins.max_rank, refinement.max_rank, refinement.origins.max_rank);
  replace(result.max_byte_span,
          result.origins.max_byte_span,
          refinement.max_byte_span,
          refinement.origins.max_byte_span);
  if (!refinement.axes.empty()) {
    result.axes = refinement.axes;
  }
  if (refinement.transfer) {
    result.transfer = refinement.transfer;
    result.origins.max_staging_byte_span = refinement.origins.max_staging_byte_span;
  }
  if (refinement.memory_conversion) {
    result.memory_conversion = refinement.memory_conversion;
  }
  return result;
}

}  // namespace holoscan::detail
