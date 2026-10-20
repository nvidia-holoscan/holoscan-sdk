/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include <holoscan/authoring/registry.hpp>

// Registration rows of the built-in payload types. make_registered_payload<T> dominates the compile
// time of each row, so the rows are built outside registry.cpp, in sources that compile in
// parallel. Each built-in type is instantiated in exactly one source, so every row's thunks see the
// same PythonSchemaNativeTraits<T> specializations (none, for the built-in types).
namespace holoscan::authoring::detail {

/// @brief Build the rows of the fixed-width integer payloads, holoscan.uint8 to holoscan.int64.
[[nodiscard]] std::vector<RegisteredPayloadType> make_builtin_integer_payloads();

/// @brief Build the rows of holoscan.boolean, holoscan.float32, holoscan.float64, and
/// holoscan.Tensor.
[[nodiscard]] std::vector<RegisteredPayloadType> make_builtin_other_payloads();

}  // namespace holoscan::authoring::detail
