/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace holoscan::doc::errors {

constexpr const char* kErrorCode = R"doc(Stable Holoscan SDK error category.)doc";
constexpr const char* kHoloscanError = R"doc(
Base exception for Holoscan SDK failures.

Every instance carries a stable ``code`` and ``token``. Branch on those fields
instead of matching the human-readable message.
)doc";

}  // namespace holoscan::doc::errors
