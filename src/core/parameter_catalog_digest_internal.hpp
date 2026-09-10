/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <span>

#include <holoscan/core/parameter_catalog.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace holoscan::detail {

/// Add the canonical ordered Parameter catalog records to an existing hash builder.
///
/// Callers own the versioned digest domain and any plan/run projection prefix. This function owns
/// the common record-field encoding so compile-time and Runtime catalog identities cannot drift.
void add_parameter_catalog_records(StableHashBuilder& hash,
                                   std::span<const OperatorParamRecord> operators,
                                   std::span<const ParamFieldCatalogRecord> fields) noexcept;

}  // namespace holoscan::detail
