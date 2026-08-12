/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/lifecycle.hpp>

#include "lifecycle_context_internal.hpp"

namespace holoscan {

CancellationView LifecycleContext::cancellation() const noexcept {
  if (state_ == nullptr) {
    return CancellationView{};
  }
  return CancellationView{state_->cancellation_source != nullptr ? state_->cancellation_source
                                                                 : &state_->cancellation_requested};
}

const void* LifecycleContext::parameter_address(const void* exact_type) const noexcept {
  if (state_ == nullptr || state_->params == nullptr || state_->params_type != exact_type) {
    return nullptr;
  }
  return state_->params;
}

ParamGeneration LifecycleContext::params_generation() const noexcept {
  return state_ != nullptr ? state_->params_generation : ParamGeneration{};
}

LifecycleTransitionPhase LifecycleContext::transition_phase() const noexcept {
  return state_ != nullptr ? state_->phase : LifecycleTransitionPhase::kNone;
}

expected<ClockRef, Error> LifecycleContext::resolve_clock(
    ClockKey key) const noexcept {  // NOLINT(performance-unnecessary-value-param)
  if (state_ == nullptr || state_->resolve_clock == nullptr) {
    return make_unexpected(Error{ErrorCode::kLifecycleInvalidCallingContext});
  }
  return state_->resolve_clock(state_->service_state, key);
}

expected<NotificationSender, Error> LifecycleContext::notification_sender(
    NotificationSource& source) const noexcept {
  if (state_ == nullptr || state_->notification_sender == nullptr) {
    return make_unexpected(Error{ErrorCode::kLifecycleInvalidCallingContext});
  }
  return state_->notification_sender(state_->service_state, source);
}

expected<OperatorRef, Error> LifecycleContext::restart_target(OperatorRef plan_ref) const noexcept {
  if (state_ == nullptr || state_->restart_target == nullptr) {
    return make_unexpected(Error{ErrorCode::kLifecycleInvalidCallingContext});
  }
  return state_->restart_target(state_->service_state, plan_ref);
}

expected<OperatorRef, Error> LifecycleContext::restart_target(
    const OperatorPath& path) const noexcept {
  if (state_ == nullptr || state_->restart_path == nullptr) {
    return make_unexpected(Error{ErrorCode::kLifecycleInvalidCallingContext});
  }
  return state_->restart_path(state_->service_state, path);
}

}  // namespace holoscan
