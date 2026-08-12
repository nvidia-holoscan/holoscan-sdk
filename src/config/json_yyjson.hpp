/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include <holoscan/config/io.hpp>

#include "config/config_internal.hpp"

namespace holoscan::config_io::detail {

// Parse strict RFC 8259 syntax with the private yyjson provider, then convert the temporary DOM
// into the Holoscan-owned semantic tree. No yyjson object or type escapes this call.
[[nodiscard]] config::ConfigResult<config::ConfigValue> parse_json(
    std::string_view source, ParseOptions options,
    std::vector<config::detail::LosslessScalarSpan>* spans = nullptr,
    std::size_t source_offset = 0U);

}  // namespace holoscan::config_io::detail
