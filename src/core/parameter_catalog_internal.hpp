/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <utility>
#include <vector>

#include <holoscan/core/parameter_catalog.hpp>

namespace holoscan::detail {

/// @brief Construct immutable Parameter catalogs from compiler-validated records.
class ParameterCatalogAccess {
 public:
  [[nodiscard]] static ParameterCatalogView make(ParamCatalogDigest digest,
                                                 std::vector<OperatorParamRecord> operators,
                                                 std::vector<ParamFieldCatalogRecord> fields) {
    ParameterCatalogView result;
    result.digest_ = digest;
    result.operators_ = std::move(operators);
    result.fields_ = std::move(fields);
    return result;
  }
};

}  // namespace holoscan::detail
