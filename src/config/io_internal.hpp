/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string_view>

#include <holoscan/config/io.hpp>

namespace holoscan::config_io::detail {

// Parse canonical JSON generated inside the SDK under serialization limits. This private seam may
// consume more than the public 16 MiB untrusted-input ceiling, but it preserves the same grammar,
// tree, string, and container bounds and never accepts caller-provided source bytes directly.
[[nodiscard]] HOLOSCAN_CONFIG_IO_EXPORT config::ConfigResult<config::Document>
parse_generated_canonical_json(std::string_view source, OutputLimits limits);

}  // namespace holoscan::config_io::detail
