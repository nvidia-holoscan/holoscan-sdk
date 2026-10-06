/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/lifecycle.hpp>

#include <utility>

#include "diagnostic_context.hpp"
#include "lifecycle_context_internal.hpp"

namespace holoscan {

void LifecycleContext::report_error(ErrorCode code, std::string_view message) noexcept {
  if (state_ != nullptr && state_->diagnostic_capture != nullptr &&
      state_->diagnostic_capture->count == 0U) {
    state_->diagnostic_capture->append(code, message);
  }
}

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

void LifecycleContext::record_callback_failure(Error error) noexcept {
  if (state_ != nullptr && state_->report_failure != nullptr) {
    state_->report_failure(state_->service_state, std::move(error));
  }
}

expected<void, Error> LifecycleContext::bind_nominal_rate_port(const void* output,
                                                               Rate value) noexcept {
  if (state_ == nullptr || state_->bind_nominal_rate == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kLifecycleInvalidCallingContext, "PORT_RATE_BIND_NOT_AUTHORIZED"});
  }
  if (output == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "PORT_RATE_BIND_NOT_AUTHORIZED"});
  }
  return state_->bind_nominal_rate(state_->rate_state, output, value);
}

expected<PortRateView, Error> LifecycleContext::rate_port(const void* declaration,
                                                          bool output) const noexcept {
  if (state_ == nullptr || state_->resolve_rate == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kLifecycleInvalidCallingContext, "PORT_RATE_VIEW_INVALID"});
  }
  if (declaration == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "PORT_RATE_VIEW_INVALID"});
  }
  return state_->resolve_rate(state_->rate_state, declaration, output);
}

}  // namespace holoscan
