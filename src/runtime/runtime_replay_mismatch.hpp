/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <holoscan/core/graph_ids.hpp>
#include <holoscan/core/operator_ref.hpp>
#include <holoscan/core/parameter_schema_types.hpp>

namespace holoscan::detail {

/// @brief Classify the bounded declaration surface that failed cold Runtime replay.
///
/// Values are private diagnostic schema, not a public error-code domain. They deliberately avoid
/// free-form contract text, process addresses, RTTI, and compiler type spellings.
enum class RuntimeReplayMismatchCategory : std::uint8_t {
  kNone = 0,
  kDeclaration,
  kPorts,
  kReadiness,
  kLifecycle,
  kTemporalContract,
  kParameterContract,
};

/// @brief Retain one exact, pointer-free cold reconstruction mismatch witness.
///
/// The factory slot is one-based in the canonical executable-factory inventory. Expected and
/// observed are domain-separated SHA-256 fingerprints of the immutable plan side and replay side,
/// respectively. The record is produced only on a cold failure path and adds no dispatch work.
struct RuntimeReplayMismatchDetail {
  bool present{};
  VertexId vertex{};
  OperatorInstanceId operator_id{};
  std::uint32_t factory_slot{};
  RuntimeReplayMismatchCategory category{RuntimeReplayMismatchCategory::kNone};
  ParamSchemaHash expected{};
  ParamSchemaHash observed{};
};

}  // namespace holoscan::detail
