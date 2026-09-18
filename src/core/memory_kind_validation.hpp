/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace holoscan::detail {

[[nodiscard]] constexpr bool valid_memory_kind_value(MemoryKind memory_kind) noexcept {
  switch (memory_kind) {
    case MemoryKind::kUnknown:
    case MemoryKind::kHost:
    case MemoryKind::kPinnedHost:
    case MemoryKind::kCudaDevice:
    case MemoryKind::kCudaManaged:
      return true;
  }
  return false;
}

[[nodiscard]] constexpr bool valid_concrete_memory_kind(MemoryKind memory_kind) noexcept {
  return memory_kind != MemoryKind::kUnknown && valid_memory_kind_value(memory_kind);
}

[[nodiscard]] inline bool valid_direct_memory_kind_domain(
    const ParamFieldDescriptor& field) noexcept {
  if (field.enum_choices.count == 0U) {
    return false;
  }
  for (std::size_t index = 0U; index < field.enum_choices.count; ++index) {
    MemoryKind memory_kind{};
    if (!field.enum_choices.values[index].value.try_as(memory_kind) ||
        !valid_concrete_memory_kind(memory_kind)) {
      return false;
    }
  }
  return true;
}

static_assert(valid_memory_kind_value(MemoryKind::kUnknown));
static_assert(valid_concrete_memory_kind(MemoryKind::kHost));
static_assert(valid_concrete_memory_kind(MemoryKind::kPinnedHost));
static_assert(valid_concrete_memory_kind(MemoryKind::kCudaDevice));
static_assert(valid_concrete_memory_kind(MemoryKind::kCudaManaged));
static_assert(!valid_concrete_memory_kind(MemoryKind::kUnknown));
static_assert(!valid_memory_kind_value(static_cast<MemoryKind>(255U)));

}  // namespace holoscan::detail
