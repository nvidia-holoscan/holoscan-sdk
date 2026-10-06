/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace holoscan::python::detail {

namespace nb = nanobind;

template <typename ValueAt>
nb::tuple integer_tuple(std::size_t size, const ValueAt& value_at) {
  nb::tuple_builder builder(size);
  for (std::size_t index = 0U; index < size; ++index) {
    builder.put(value_at(index));
  }
  return builder.commit();
}

inline nb::tuple scaled_integer_tuple(std::span<const std::int64_t> values,
                                      std::int64_t scale = 1) {
  return integer_tuple(values.size(),
                       [values, scale](std::size_t index) { return values[index] * scale; });
}

}  // namespace holoscan::python::detail
