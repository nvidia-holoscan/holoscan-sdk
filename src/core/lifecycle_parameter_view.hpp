/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>

#include <holoscan/core/expected.hpp>
#include <holoscan/core/lifecycle_types.hpp>
#include <holoscan/core/operator_ref.hpp>

namespace holoscan::detail {

/// Identify which immutable aggregate slot a lifecycle callback must observe.
enum class LifecycleParameterViewKind : std::uint8_t {
  kStartupGenerationZero,
  kOldPublished,
  kPrivateCandidate,
  kRetainedCurrent,
  kCurrentPublished,
};

/// Complete logical source and generation selected for one callback attempt.
struct LifecycleParameterViewSelection {
  LifecycleParameterViewKind kind{LifecycleParameterViewKind::kStartupGenerationZero};
  ParamGeneration generation{};

  friend constexpr bool operator==(LifecycleParameterViewSelection,
                                   LifecycleParameterViewSelection) noexcept = default;
};

/// Immutable generation facts retained around a lifecycle transition.
struct LifecycleParameterViewInputs {
  ParamGeneration old_published{};
  ParamGeneration current_published{};
  std::optional<ParamGeneration> private_candidate;
  std::optional<LifecycleParameterViewSelection> creating_view;
};

enum class LifecycleParameterViewSelectionError : std::uint8_t {
  kOutsideCallback,
  kMissingCreatingView,
  kInvalidTransitionPhase,
};

/// Select one complete logical parameter view without materializing or publishing storage.
[[nodiscard]] expected<LifecycleParameterViewSelection, LifecycleParameterViewSelectionError>
select_lifecycle_parameter_view(LifecycleTransitionPhase phase,
                                const LifecycleParameterViewInputs& inputs) noexcept;

}  // namespace holoscan::detail
