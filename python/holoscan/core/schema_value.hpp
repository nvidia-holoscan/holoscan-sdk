/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <holoscan/authoring/registry.hpp>

#include "native_operator.hpp"

namespace holoscan {

namespace nb = nanobind;

// Convert one Python scalar/container/Tensor tree into the language-neutral schema value model.
[[nodiscard]] authoring::ForeignSchemaValue python_schema_value(nb::handle value);
// Convert one language-neutral schema value tree into ordinary Python values and Tensor wrappers.
[[nodiscard]] nb::object python_schema_value(const authoring::ForeignSchemaValue& value);
// Materialize a generated Python schema class, or return the ordinary value tree when absent.
[[nodiscard]] nb::object materialize_python_schema(const authoring::ForeignSchemaValue& value,
                                                   const GilSafeObject& schema_type);

}  // namespace holoscan
