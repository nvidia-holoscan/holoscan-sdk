/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <filesystem>

namespace holoscan {

namespace nb = nanobind;

// Converts any Python path-like object with Python's native filesystem codec.
[[nodiscard]] std::filesystem::path python_filesystem_path(nb::handle value);

}  // namespace holoscan
