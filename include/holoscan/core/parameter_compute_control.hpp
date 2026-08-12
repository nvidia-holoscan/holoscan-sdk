/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <span>
#include <type_traits>

#include <holoscan/core/parameter_update.hpp>

namespace holoscan {

namespace detail {
using ParamComputeSubmitFunction = ParamSubmitResult (*)(void* state,
                                                         const ParamInlinePatch* patches,
                                                         std::size_t patch_count) noexcept;
}  // namespace detail

/// @brief Borrow the fixed-size compute-originated parameter submission seam.
///
/// The facade does not own, retain, or extend the callback-lifetime binding. It is valid only while
/// the `ExecutionContext` that produced it is valid. It exposes no owning submission, read, wait,
/// lifecycle restart, or session-ingress-close authority.
class ParamComputeControl {
 public:
  /// @brief Construct an inert borrowed facade.
  constexpr ParamComputeControl() noexcept = default;

  /// @brief Try to copy fixed patches into preallocated update ingress.
  /// @param patches Fixed-size patches whose contents are copied before return.
  /// @return Immediate submitted or typed rejection result.
  /// @note This call is nonblocking, allocation-free, and `noexcept`. An empty span rejects. A
  /// successful enqueue returns `kSubmitted` with sequence zero because acceptance assigns the
  /// run-local sequence only after bounded ingress drain. `kQueueFull` can report either finite
  /// ingress capacity or transient ingress-lock contention. The call never changes the generation
  /// pinned by the current callback and never retains the patch span.
  [[nodiscard]] ParamSubmitResult try_submit_from_compute(
      std::span<const ParamInlinePatch> patches) const noexcept {
    if (state_ == nullptr || submit_ == nullptr) {
      return ParamSubmitResult{
          .status = ParamUpdateStatus::kUpdatesDisabled,
          .state = ParamUpdateState::kRejected,
      };
    }
    return submit_(state_, patches.data(), patches.size());
  }

 private:
  constexpr ParamComputeControl(void* state, detail::ParamComputeSubmitFunction submit) noexcept
      : state_(state), submit_(submit) {}

  void* state_{};
  detail::ParamComputeSubmitFunction submit_{};

  friend class ExecutionContext;
};

static_assert(std::is_standard_layout_v<ParamComputeControl>);
static_assert(std::is_trivially_copyable_v<ParamComputeControl>);

}  // namespace holoscan
