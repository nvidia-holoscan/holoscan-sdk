/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>

#include <holoscan/core/lifecycle.hpp>

namespace holoscan::detail {

struct DiagnosticCapture;

using LifecycleClockResolver = expected<ClockRef, Error> (*)(void*, const ClockKey&) noexcept;
using LifecycleNotificationResolver =
    expected<NotificationSender, Error> (*)(void*, NotificationSource&) noexcept;
using LifecycleTargetResolver = expected<OperatorRef, Error> (*)(void*, OperatorRef) noexcept;
using LifecyclePathResolver = expected<OperatorRef, Error> (*)(void*, const OperatorPath&) noexcept;
using LifecycleRateBinder = expected<void, Error> (*)(void*, const void*, Rate) noexcept;
using PortRateResolver = expected<PortRateView, Error> (*)(void*, const void*, bool) noexcept;

/// Callback-lifetime state populated by the lifecycle executor or private conformance fixture.
/// Process-local type tokens and operation pointers never enter a Graph snapshot or plan record.
struct LifecycleContextState {
  std::atomic_bool cancellation_requested{false};
  const std::atomic_bool* cancellation_source{};
  const void* params{};
  const void* params_type{};
  ParamGeneration params_generation{};
  LifecycleTransitionPhase phase{LifecycleTransitionPhase::kNone};
  void* service_state{};
  LifecycleClockResolver resolve_clock{};
  LifecycleNotificationResolver notification_sender{};
  LifecycleTargetResolver restart_target{};
  LifecyclePathResolver restart_path{};
  void* rate_state{};
  LifecycleRateBinder bind_nominal_rate{};
  PortRateResolver resolve_rate{};
  DiagnosticCapture* diagnostic_capture{};
};

}  // namespace holoscan::detail
