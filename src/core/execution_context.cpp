/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/execution_context.hpp>

#include "execution_context_internal.hpp"

namespace holoscan {
namespace {
constexpr TraceContext kAbsentTrace{};
}  // namespace

namespace detail {

[[noreturn]] void parameter_type_contract_violation() noexcept {
  // An exact callback parameter-type mismatch is an always-on contract violation. Keep the
  // release-defined termination path bounded and data-free so diagnostics cannot weaken it.
  std::terminate();
}

}  // namespace detail

Tag ExecutionContext::tag() const noexcept {
  return state_ != nullptr ? state_->tag : Tag{};
}

::cuda::stream_ref ExecutionContext::cuda_stream() noexcept {
  return ::cuda::stream_ref{state_ != nullptr ? state_->cuda_stream : cudaStream_t{}};
}

ClockRef ExecutionContext::clock() const noexcept {
  return state_ != nullptr && state_->clock != nullptr ? *state_->clock : ClockRef{};
}

TagTime ExecutionContext::tag_time() const noexcept {
  return state_ != nullptr ? state_->tag_time : TagTime{};
}

TimePoint ExecutionContext::activation_time() const noexcept {
  return state_ != nullptr ? state_->activation_time : TimePoint{};
}

const TraceContext& ExecutionContext::trace() const noexcept {
  return state_ != nullptr ? state_->trace : kAbsentTrace;
}

TraceId ExecutionContext::trace_id() const noexcept {
  return trace().trace_id;
}

ActivationStatus ExecutionContext::activation() const noexcept {
  return state_ != nullptr
             ? state_->activation
             : ActivationStatus{ActivationReason::kNominal, OnViolation::kEmitWitness};
}

CancellationView ExecutionContext::cancellation() const noexcept {
  if (state_ == nullptr) {
    return CancellationView{};
  }
  return CancellationView{state_->cancellation_source != nullptr ? state_->cancellation_source
                                                                 : &state_->cancellation_requested};
}

OperatorRef ExecutionContext::self() const noexcept {
  return state_ != nullptr ? state_->self : OperatorRef{};
}

std::string_view ExecutionContext::operator_path() const noexcept {
  return state_ != nullptr ? state_->operator_path : std::string_view{};
}

const void* ExecutionContext::parameter_address(const void* exact_type) const noexcept {
  if (state_ == nullptr) {
    return nullptr;
  }
  if (state_->pinned_params != nullptr) {
    return state_->pinned_params->local_type_token == exact_type ? state_->pinned_params->value
                                                                 : nullptr;
  }
  return state_->params != nullptr && state_->params_type == exact_type ? state_->params : nullptr;
}

ParamGeneration ExecutionContext::params_generation() const noexcept {
  if (state_ == nullptr) {
    return {};
  }
  return state_->pinned_params != nullptr ? state_->pinned_params->generation
                                          : state_->params_generation;
}

ParamComputeControl ExecutionContext::param_control() const noexcept {
  if (state_ == nullptr) {
    return ParamComputeControl{};
  }
  return ParamComputeControl{state_->param_control_state, state_->submit_param_update};
}

expected<LifecycleRestartReceipt, Error> ExecutionContext::request_restart(
    LifecycleRestartOptions options) noexcept {
  if (!is_lifecycle_forward_stage(options.from_stage)) {
    return make_unexpected(Error{ErrorCode::kLifecycleInvalidRestartStage});
  }
  if (state_ == nullptr || state_->request_restart == nullptr) {
    return make_unexpected(Error{ErrorCode::kLifecycleTransactionsDisabled});
  }
  if (auto valid = detail::validate_restart_target(state_->self, state_->self); !valid) {
    return make_unexpected(Error{valid.error().code});
  }
  return state_->request_restart(state_->restart_state, state_->self, options);
}

expected<LifecycleRestartReceipt, Error> ExecutionContext::request_restart(
    OperatorRef target, LifecycleRestartOptions options) noexcept {
  if (!is_lifecycle_forward_stage(options.from_stage)) {
    return make_unexpected(Error{ErrorCode::kLifecycleInvalidRestartStage});
  }
  if (state_ == nullptr || state_->request_restart == nullptr) {
    return make_unexpected(Error{ErrorCode::kLifecycleTransactionsDisabled});
  }
  if (auto valid = detail::validate_restart_target(state_->self, target); !valid) {
    return make_unexpected(Error{valid.error().code});
  }
  return state_->request_restart(state_->restart_state, target, options);
}

expected<PortRateView, Error> ExecutionContext::rate_port(const void* declaration,
                                                          bool output) const noexcept {
  if (state_ == nullptr || state_->resolve_rate == nullptr || declaration == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return state_->resolve_rate(state_->rate_state, declaration, output);
}

void ExecutionContext::request_completion() noexcept {
  if (state_ != nullptr && state_->request_termination != nullptr) {
    state_->request_termination(state_->termination_request_state,
                                detail::TerminationRequest::kCompletion);
  }
}

void ExecutionContext::request_stop() noexcept {
  if (state_ != nullptr && state_->request_termination != nullptr) {
    state_->request_termination(state_->termination_request_state,
                                detail::TerminationRequest::kStop);
  }
}

}  // namespace holoscan
