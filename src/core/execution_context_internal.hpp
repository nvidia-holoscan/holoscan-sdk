/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <atomic>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace holoscan::detail {

using RestartRequestFunction = expected<LifecycleRestartReceipt, Error> (*)(
    void*, OperatorRef, LifecycleRestartOptions) noexcept;

/// @brief Couple every callback-visible Parameter fact in one immutable publication record.
///
/// Runtime pins this exact record before invoking a callback and retains its owning slot through
/// callback return. The legacy scalar fields in `ExecutionContextState` remain only for focused
/// synthetic fixtures until they migrate to the same production pin owner.
struct PinnedParameterHeader {
  const void* value{};             ///< Exact immutable aggregate address.
  const void* local_type_token{};  ///< Process-local defining-module type token.
  ParamGeneration generation{};    ///< Complete aggregate generation.
  ParamSchemaHash schema_hash{};   ///< Declared contract hash, or zero for opaque mode.
  std::uint32_t slot{};            ///< Bounded store slot, or zero for static `NoParams`.
};

[[nodiscard]] inline expected<void, Error> validate_restart_target(
    const OperatorRef self, const OperatorRef target) noexcept {
  if (!target.run_bound() || !self.run_bound() || target.plan_generation != self.plan_generation) {
    return make_unexpected(Error{ErrorCode::kStaleOperatorRef});
  }
  if (target.run_generation != self.run_generation) {
    return make_unexpected(Error{ErrorCode::kCrossSessionOperatorRef});
  }
  return {};
}

// Runtime and the private conformance fixture populate this bounded callback-lifetime record. The
// public facade retains only its address. Process-local parameter type tokens never enter a plan,
// diagnostic identity, schema, or transport record.
struct ExecutionContextState {
  Tag tag{};
  TagTime tag_time{};
  TimePoint activation_time{};
  ClockRef clock{};
  cudaStream_t cuda_stream{};
  TraceContext trace{};
  ActivationStatus activation{ActivationReason::kNominal, OnViolation::kEmitWitness};
  std::atomic_bool cancellation_requested{false};
  const std::atomic_bool* cancellation_source{};
  const void* params{};
  const void* params_type{};
  ParamGeneration params_generation{};
  const PinnedParameterHeader* pinned_params{};
  void* param_control_state{};
  ParamComputeSubmitFunction submit_param_update{};
  void* restart_state{};
  RestartRequestFunction request_restart{};
  OperatorRef self{};
};

}  // namespace holoscan::detail
