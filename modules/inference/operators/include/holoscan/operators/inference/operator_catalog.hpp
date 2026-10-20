// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/operators/inference/holoscan_op_inference_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::ops {

/// @brief Return the InferenceOp catalog-registration status.
///
/// The hand-authored row accepts the operator's map-shaped structural configuration without adding
/// an inference-specific map kind to the Core registry ABI. Loading the Inference operator library
/// installs the row automatically. This process-idempotent function reports only that row's
/// retained registration result and is safe for language-binding initialization shims to call.
/// @return Success, or the Inference row's retained registration failure.
[[nodiscard]] HOLOSCAN_OP_INFERENCE_EXPORT expected<void, Error>
register_inference_operator_catalog();

}  // namespace holoscan::ops
