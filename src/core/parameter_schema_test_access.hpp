/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/core/parameter_schema.hpp>

namespace holoscan::detail {

/// @brief Supply qualified-proof state only to non-installed conformance tests.
class ParamSchemaTestAccess {
 public:
  [[nodiscard]] static constexpr ParamSchemaHeader proven(ParamSchemaHeader header) noexcept {
    header.proof_ = ParamSchemaProof::kProven;
    return header;
  }
};

}  // namespace holoscan::detail
