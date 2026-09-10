/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <string_view>

namespace holoscan::detail::artifact_test {

[[nodiscard]] std::string normalized_artifact_json(std::string_view text);

}  // namespace holoscan::detail::artifact_test
