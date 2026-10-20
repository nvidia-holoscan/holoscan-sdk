/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>

namespace holomq {
namespace cuda {

/// Returns the CUDA device ordinal for the currently bound driver context, or empty when \c
/// cuCtxGetDevice fails.
[[nodiscard]] std::optional<int32_t> current_device_ordinal() noexcept;

}  // namespace cuda
}  // namespace holomq
