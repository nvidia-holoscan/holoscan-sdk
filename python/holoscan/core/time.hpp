/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <string_view>

#include <holoscan/core/tag.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan {

namespace nb = nanobind;

// Stores a Python-facing duration in the nanosecond unit used by Core authoring APIs.
struct PyDuration {
  std::int64_t nanoseconds{};
};

void init_time(nb::module_& module);

[[nodiscard]] std::int64_t python_duration_nanoseconds(
    nb::handle value, std::string_view field,
    std::string_view type_error_token = "AUTHORING_PARAMETER_TYPE_MISMATCH",
    std::string_view range_error_token = "PARAM_VALIDATION_FAILED category=out_of_range");

}  // namespace holoscan
