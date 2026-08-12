/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/config/document.hpp>

#include <string_view>

namespace holoscan::config_io::detail {

/// Verify that the private reflect-cpp generic backend accepts Holoscan canonical JSON.
[[nodiscard]] config::ConfigResult<void> verify_reflect_generic(std::string_view canonical_json);

}  // namespace holoscan::config_io::detail
