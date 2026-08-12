/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include "ready_policy.hpp"

namespace holoscan::detail::scheduling {

inline constexpr std::uint32_t kDeterministicSerialReadyPolicyVersion = 1U;

/// Return the stable algorithm identifier for the EA1 READY policy.
[[nodiscard]] constexpr std::string_view deterministic_serial_ready_policy_id() noexcept {
  return "holoscan.ready.deterministic_serial";
}

/// Return the owning public evidence key for the EA1 READY policy.
[[nodiscard]] AlgorithmKey deterministic_serial_ready_policy_algorithm();

/// Select one activation at a time using the compiler-frozen deterministic total order.
class DeterministicSerialReadyPolicy final : public ReadyPolicy {
 public:
  [[nodiscard]] ReadyPolicyDescriptor descriptor() const override;

  [[nodiscard]] expected<void, Error> initialize(const SchedulerPlanView& plan,
                                                 ReadyPolicyState& state) override;

  [[nodiscard]] ReadyCandidateIndex select(ReadySetView ready,
                                           ReadyPolicyState& state) noexcept override;

  void on_activation_started(const ReadyCandidateView& candidate,
                             ReadyPolicyState& state) noexcept override;

  void on_activation_completed(const ReadyCandidateView& candidate,
                               ReadyPolicyState& state) noexcept override;
};

/// Construct a fresh deterministic-serial policy for explicit catalog injection.
[[nodiscard]] std::unique_ptr<ReadyPolicy> make_deterministic_serial_ready_policy();

}  // namespace holoscan::detail::scheduling
