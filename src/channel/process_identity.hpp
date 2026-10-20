/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <holoscan/holoscan_core_export.h>

namespace holoscan::channel_internal {

/// Return the operating-system process identity used only for private same-process descriptors.
[[nodiscard]] HOLOSCAN_CORE_EXPORT std::uint64_t current_process_identity() noexcept;

/// Mint a nonzero opaque producer-incarnation token for one catalog attachment.
///
/// The value is identity only, not an ordering coordinate. Receivers fence on any change.
[[nodiscard]] HOLOSCAN_CORE_EXPORT std::uint64_t mint_producer_generation() noexcept;

}  // namespace holoscan::channel_internal
