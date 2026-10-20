/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>

#include "core/domain/tensor_access.hpp"
#include "execution_context_internal.hpp"

namespace holoscan::detail {

/// @brief Invoke the private Operator callback ABI for the production runtime.
///
/// This non-installed accessor supplies no queue, scheduler, or public testing
/// surface. Runtime must finish declaration verification and activation
/// binding before calling `invoke()`.
class OperatorRuntimeAccess {
 public:
  /// @brief Invoke one Operator with callback-lifetime runtime state.
  /// @param op Fresh session-owned Operator instance.
  /// @param state Fully bound activation state retained through callback return.
  /// @return Operator callback result without retry or exception translation.
  [[nodiscard]] static expected<void, Error> invoke(OperatorBase& op,
                                                    ExecutionContextState& state) {
    const TensorConsumerScope tensor_scope{
        state.cuda_stream != nullptr ? std::optional{state.cuda_stream} : std::nullopt,
        state.cuda_stream_lifetime,
        state.cuda_stream_capturing};
    ExecutionContext context{&state};
    return op.invoke_compute(context);
  }

  /// @brief Return the exact process-local parameter descriptor for one Operator.
  /// @param op Fresh session-owned Operator instance.
  /// @return Stable descriptor associated with the concrete typed base.
  [[nodiscard]] static const ParamTypeDescriptor& parameter_type(const OperatorBase& op) noexcept {
    return op.defining_param_type();
  }

  /// @brief Return the process-local declared parameter schema for one Operator.
  /// @param op Fresh session-owned Operator instance.
  /// @return Static schema descriptor, or null for an opaque local aggregate.
  /// @note Runtime must verify this descriptor against pointer-free plan records before using any
  /// local field token or operation thunk.
  [[nodiscard]] static const ParamSchemaDescriptor* parameter_schema(
      const OperatorBase& op) noexcept {
    return op.defining_param_schema();
  }

  /// @brief Return the address of the parameter aggregate stored by one Operator instance.
  /// @param op Fresh session-owned Operator instance.
  /// @return Stable address valid for the Operator lifetime.
  /// @note The address is stable, but Runtime may replace the stored aggregate while callbacks
  /// are quiesced so a subsequent legacy lifecycle callback observes its selected phase view.
  [[nodiscard]] static const void* parameter_address(const OperatorBase& op) noexcept {
    return op.parameter_address_impl();
  }

  /// @brief Replace the session-owned Operator aggregate with a published runtime generation.
  /// @param op Fresh session-owned Operator whose callbacks are quiesced.
  /// @param value Exact aggregate value from the Operator's pinned generation store.
  /// @note The baseline runtime invokes this only for the admitted trivially copyable generation
  /// strategy immediately before a legacy start() or stop() callback. A rollback or restoration
  /// phase replaces the value again before invoking user code.
  static void replace_runtime_parameter_value(OperatorBase& op, const void* value) noexcept {
    op.replace_runtime_parameter_value_impl(value);
  }

  /// @brief Return whether one Operator uses the no-storage Parameter sentinel.
  /// @param op Fresh session-owned Operator instance.
  /// @return \c true only for an exact `Operator<NoParams>` base.
  /// @note The virtual property remains stable across shared-library boundaries,
  /// unlike comparing process-local type-token addresses emitted in separate DSOs.
  [[nodiscard]] static bool has_no_params(const OperatorBase& op) noexcept {
    return op.has_no_params_impl();
  }
};

}  // namespace holoscan::detail
