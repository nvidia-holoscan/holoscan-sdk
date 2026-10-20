// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/visualization/holoscan_operator_holoviz_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::visualization {

/// @brief Install the HolovizOp row into the process-local authoring registry.
///
/// The hand-authored row accepts the operator's structural layer and sink declarations without
/// adding visualization-specific value kinds to the Core registry ABI. Loading the operator
/// library registers the row automatically. This function reports that row-specific result and
/// is safe to call repeatedly from provider initialization.
/// @return Success, or the registration failure retained across repeated calls.
[[nodiscard]] HOLOSCAN_OPERATOR_HOLOVIZ_EXPORT expected<void, Error>
register_holoviz_operator_catalog();

}  // namespace holoscan::visualization
