// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/vision/holoscan_vision_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::vision {

/// @brief Install the Vision Operator rows into the process-local authoring registry.
///
/// The operation is process-idempotent and safe to call from every executable or language module
/// that links the Vision library. Each row is generated from the operator's canonical `Params`
/// schema and ordinary default-construction thunk.
/// @return Success, or the first registration failure retained across repeated calls.
HOLOSCAN_VISION_EXPORT expected<void, Error> register_vision_operator_catalog();

}  // namespace holoscan::vision
