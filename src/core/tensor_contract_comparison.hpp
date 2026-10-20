/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>
#include <fmt/format.h>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/memory_kind.hpp>
#include <holoscan/core/tensor_contract_types.hpp>

namespace holoscan::detail {

using TensorMismatches = std::vector<DiagnosticMismatch>;

template <typename Provided, typename Required>
inline bool tensor_mismatch(TensorMismatches* mismatches, const char* field,
                            const Provided& provided, const Required& required) {
  if (mismatches != nullptr && mismatches->size() < 16U) {
    mismatches->push_back({field, fmt::format("{}", provided), fmt::format("{}", required)});
  } else if (mismatches != nullptr && mismatches->size() == 16U) {
    mismatches->push_back({"diagnostics", "additional mismatches omitted", "at most 16 entries"});
  }
  return false;
}

inline const char* tensor_memory_name(MemoryKind kind) noexcept {
  switch (kind) {
    case MemoryKind::kUnknown:
      return "unknown";
    case MemoryKind::kHost:
      return "host";
    case MemoryKind::kPinnedHost:
      return "pinned_host";
    case MemoryKind::kCudaDevice:
      return "cuda_device";
    case MemoryKind::kCudaManaged:
      return "cuda_managed";
  }
  return "invalid";
}

inline const char* tensor_stride_name(TensorStridePolicy policy) noexcept {
  switch (policy) {
    case TensorStridePolicy::kContiguousOnly:
      return "contiguous";
    case TensorStridePolicy::kBoundedStrided:
      return "bounded_strided";
  }
  return "invalid";
}

inline std::string tensor_dtype_name(DLDataType dtype) {
  const char* name = nullptr;
  switch (dtype.code) {
    case kDLInt:
      name = "int";
      break;
    case kDLUInt:
      name = "uint";
      break;
    case kDLFloat:
      name = "float";
      break;
    case kDLBfloat:
      name = "bfloat";
      break;
    case kDLComplex:
      name = "complex";
      break;
    case kDLBool:
      name = "bool";
      break;
    default:
      return fmt::format("dtype(code={},bits={},lanes={})", dtype.code, dtype.bits, dtype.lanes);
  }
  return dtype.lanes == 1U ? fmt::format("{}{}", name, dtype.bits)
                           : fmt::format("{}{}x{}", name, dtype.bits, dtype.lanes);
}

inline bool compare_tensor_layouts(const std::optional<TensorRepresentation>& source,
                                   const std::optional<TensorRepresentation>& target,
                                   TensorMismatches* mismatches = nullptr) {
  if (!source || !target) {
    return true;
  }
  bool compatible = true;
  if (source->memory_kind && target->memory_kind && source->memory_kind != target->memory_kind) {
    compatible = tensor_mismatch(mismatches,
                                 "memory_kind",
                                 tensor_memory_name(*source->memory_kind),
                                 tensor_memory_name(*target->memory_kind));
  }
  if (source->rank && target->rank && source->rank != target->rank) {
    compatible = tensor_mismatch(mismatches, "rank", *source->rank, *target->rank);
  }
  if (source->strides && target->strides && source->strides != target->strides) {
    compatible = tensor_mismatch(mismatches,
                                 "strides",
                                 tensor_stride_name(*source->strides),
                                 tensor_stride_name(*target->strides));
  }
  if (source->dtype && target->dtype &&
      (source->dtype->code != target->dtype->code || source->dtype->bits != target->dtype->bits ||
       source->dtype->lanes != target->dtype->lanes)) {
    compatible = false;
    if (mismatches != nullptr) {
      tensor_mismatch(mismatches,
                      "dtype",
                      tensor_dtype_name(*source->dtype),
                      tensor_dtype_name(*target->dtype));
    }
  }
  return compatible;
}

inline bool compare_tensor_bounds(const TensorContractRecord* source,
                                  const TensorContractRecord* target,
                                  TensorMismatches* mismatches = nullptr) {
  if (source == nullptr || target == nullptr) {
    return true;
  }
  bool compatible = true;
  if (source->max_rank && target->max_rank && *source->max_rank > *target->max_rank) {
    compatible = tensor_mismatch(mismatches, "max_rank", *source->max_rank, *target->max_rank);
  }
  if (source->max_byte_span && target->max_byte_span &&
      *source->max_byte_span > *target->max_byte_span) {
    compatible = tensor_mismatch(
        mismatches, "max_byte_span", *source->max_byte_span, *target->max_byte_span);
  }
  for (std::size_t axis = 0U; axis < std::min(source->axes.size(), target->axes.size()); ++axis) {
    const auto record = [&](const char* constraint, auto provided, auto required) {
      if (mismatches == nullptr) {
        return false;
      }
      const auto field = fmt::format("axes[{}].{}", axis, constraint);
      return tensor_mismatch(mismatches, field.c_str(), provided, required);
    };
    if (source->axes[axis].max_extent > target->axes[axis].max_extent) {
      compatible =
          record("max_extent", source->axes[axis].max_extent, target->axes[axis].max_extent);
    }
    if (source->axes[axis].max_abs_stride_bytes != 0U &&
        target->axes[axis].max_abs_stride_bytes != 0U &&
        source->axes[axis].max_abs_stride_bytes > target->axes[axis].max_abs_stride_bytes) {
      compatible = record("max_abs_stride_bytes",
                          source->axes[axis].max_abs_stride_bytes,
                          target->axes[axis].max_abs_stride_bytes);
    }
  }
  return compatible;
}

}  // namespace holoscan::detail
