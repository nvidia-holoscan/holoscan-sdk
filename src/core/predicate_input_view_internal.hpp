/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/core/predicate_input_view.hpp>

namespace holoscan::detail {

// Private Runtime/test construction seam. The process-local generation pointer is never plan,
// schema, diagnostic, or transport identity. Every admitted view carries live generation
// evidence. Generation zero is the invalid sentinel; no all-zero representation bypasses lifetime
// checking.
struct PredicateBindingAccess {
  template <PublishedPayload T>
  [[nodiscard]] static expected<PredicateInputView<T>, Error> input(
      PredicateInputBinding<T> binding) noexcept {
    const bool fresh = binding.selection.state == InputState::kSelected &&
                       binding.selection.reason == SelectionReason::kNone;
    const bool reused = binding.selection.state == InputState::kReused &&
                        binding.selection.reason != SelectionReason::kNone &&
                        binding.selection.reason != SelectionReason::kNotSelectedForActivation;
    const bool valid_generation = binding.expected_generation != 0 &&
                                  binding.active_generation != nullptr &&
                                  *binding.active_generation != 0 &&
                                  *binding.active_generation == binding.expected_generation;
    if (binding.sample == nullptr || (!fresh && !reused) || !valid_generation) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return PredicateInputView<T>{binding};
  }

  [[nodiscard]] static expected<PredicateEvaluationContext, Error> context(
      PredicateEvaluationBinding binding) noexcept {
    if (binding.expected_generation == 0 || binding.active_generation == nullptr ||
        *binding.active_generation == 0 ||
        *binding.active_generation != binding.expected_generation) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return PredicateEvaluationContext{binding};
  }
};

}  // namespace holoscan::detail
