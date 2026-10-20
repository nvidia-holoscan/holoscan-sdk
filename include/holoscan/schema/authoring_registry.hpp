/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_schema_export.h>

namespace holoscan::schema {

/// @brief Register the SDK-owned self-contained schema payload rows exactly once.
///
/// Shared-library loading performs the same registration automatically. Static consumers and
/// language modules may call this function to retain and initialize the schema catalog explicitly.
/// Registration failures are retained by `authoring::AuthoringRegistry::registration_status()`.
///
/// @return `true` when every SDK schema row was registered successfully.
[[nodiscard]] HOLOSCAN_SCHEMA_EXPORT bool register_sdk_schema_payloads() noexcept;

}  // namespace holoscan::schema
