/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::detail {

enum class RuntimeSupportThreadPlacementPhase : std::uint8_t {
  kValidate,
  kApply,
  kReadback,
  kVerify,
};

/// Own cold failure context for one explicit Runtime support-thread affinity.
struct RuntimeSupportThreadPlacementFailure {
  bool present{};
  SupportThreadRole role{SupportThreadRole::kController};
  RuntimeSupportThreadPlacementPhase phase{RuntimeSupportThreadPlacementPhase::kValidate};
  Error error{ErrorCode::kSuccess};
  CPUAffinity requested;
  std::optional<CPUAffinity> observed;
};

[[nodiscard]] std::string_view runtime_support_thread_placement_phase_name(
    RuntimeSupportThreadPlacementPhase phase) noexcept;

[[nodiscard]] std::string runtime_support_thread_placement_required(
    const RuntimeSupportThreadPlacementFailure& failure);

[[nodiscard]] std::string runtime_support_thread_placement_provided(
    const RuntimeSupportThreadPlacementFailure& failure);

/// Apply and exactly read back one accepted Runtime support-thread affinity, when present.
///
/// When \p previous_affinity is non-null and the role has an accepted binding, capture the
/// calling thread's allowed CPU set before applying that binding. An absent binding leaves the
/// output empty and performs no platform operation.
[[nodiscard]] expected<void, Error> apply_support_thread_affinity(
    std::span<const SupportThreadAffinityPlanRecord> records, SupportThreadRole role,
    RuntimeSupportThreadPlacementFailure* failure = nullptr,
    std::optional<CPUAffinity>* previous_affinity = nullptr);

}  // namespace holoscan::detail
