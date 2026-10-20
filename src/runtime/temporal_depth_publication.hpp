/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace holoscan::detail::runtime_internal {

/// Validate a complete Runtime selector inventory before publishing any candidate depth.
///
/// Ready-state and selector-slot types stay private to their callers. Keeping the two passes in
/// this source-private template lets focused tests exercise the exact publication algorithm
/// without exporting a test-only symbol from the Core DSO.
template <typename ReadyRunStates>
[[nodiscard]] inline bool commit_rate_candidate_temporal_depths(ReadyRunStates& states) noexcept {
  for (const auto& state : states) {
    for (const auto& selector : state.input_selectors) {
      if (selector.aligned_member) {
        continue;
      }
      if (!selector.selector.validate_rate_candidate_capacity_commit()) {
        return false;
      }
    }
    for (const auto& aligned : state.aligned_selectors) {
      if (!aligned.selector.validate_rate_candidate_capacities_commit()) {
        return false;
      }
    }
  }
  for (auto& state : states) {
    for (auto& selector : state.input_selectors) {
      if (selector.aligned_member) {
        continue;
      }
      selector.selector.commit_validated_rate_candidate_capacity();
    }
    for (auto& aligned : state.aligned_selectors) {
      aligned.selector.commit_validated_rate_candidate_capacities();
    }
  }
  return true;
}

}  // namespace holoscan::detail::runtime_internal
